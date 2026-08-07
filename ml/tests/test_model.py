"""
Encoder/heads/depth-prior unit tests — Step 3
(`docs/13_Testing_Documentation.md` §6). Uses a small hand-built
`HeteroData` fixture rather than the real database, per the testing doc's
"given a small fixture HeteroData" wording -- these are unit tests, not
integration tests (`ml/tests/test_graph.py` already covers the real DB).

Run: pytest ml/tests/test_model.py -v
"""

from __future__ import annotations

import math

import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from ml.models.depth import (
    STRUCTURAL_DEPTH_PRIOR,
    STRUCTURAL_DEPTH_PRIOR_V2,
    readout_layer_for_task,
)
from ml.models.encoder import build_encoder
from ml.models.heads import FocalLoss, PredictionHead, alpha_from_positive_rate


def _fixture_hetero_data() -> HeteroData:
    """3 node types, 1 forward relation (+ its reverse via ToUndirected),
    small enough to eyeball by hand."""
    torch.manual_seed(0)
    data = HeteroData()
    data["Supplier"].x = torch.randn(5, 6)
    data["Component"].x = torch.randn(8, 4)
    data["Shipment"].x = torch.randn(6, 3)

    data["Supplier", "SUPPLIES", "Component"].edge_index = torch.tensor(
        [[0, 0, 1, 2, 3], [0, 1, 2, 3, 4]], dtype=torch.long
    )
    data["Shipment", "SHIPS_FROM", "Supplier"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 0]], dtype=torch.long
    )
    return ToUndirected()(data)


@pytest.fixture
def fixture_data():
    return _fixture_hetero_data()


@pytest.fixture
def in_dims(fixture_data):
    return {nt: fixture_data[nt].x.size(-1) for nt in fixture_data.node_types}


# ---------------------------------------------------------------------------
# Encoder forward pass
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("architecture", ["hgt", "graphsage", "gat", "rgcn", "rgcn_attn", "rgcn_relemb"])
def test_encoder_forward_pass_shape_and_no_nan(fixture_data, in_dims, architecture):
    metadata = fixture_data.metadata()
    encoder = build_encoder(architecture, metadata, in_dims, hidden=16, num_layers=4)
    layers = encoder(fixture_data.x_dict, fixture_data.edge_index_dict)

    assert len(layers) == 4
    for layer in layers:
        for node_type in fixture_data.node_types:
            h = layer[node_type]
            assert h.shape == (fixture_data[node_type].num_nodes, 16)
            assert not torch.isnan(h).any()
            assert not torch.isinf(h).any()


def test_hgt_alias_accepts_full_architecture_name(fixture_data, in_dims):
    metadata = fixture_data.metadata()
    encoder = build_encoder("heterogeneous_graph_transformer", metadata, in_dims, hidden=8, num_layers=1)
    layers = encoder(fixture_data.x_dict, fixture_data.edge_index_dict)
    assert len(layers) == 1


def test_unknown_architecture_raises(fixture_data, in_dims):
    with pytest.raises(ValueError, match="unknown architecture"):
        build_encoder("not-a-real-architecture", fixture_data.metadata(), in_dims)


def test_rgcn_parameter_count_reflects_basis_sharing(fixture_data, in_dims):
    """The whole RGCN architecture bet (`ml/models/rgcn_encoder.py`'s module
    docstring) rests on relation parameters costing `num_bases * hidden^2 +
    num_relations * num_bases` -- a SINGLE shared-basis pool reused at every
    layer -- not `num_relations * hidden^2` (a full dedicated matrix per
    relation, HGT's approach) and not scaled by `num_layers` (each layer
    reuses the same basis pool). Locks that in directly against the
    `rel_basis`/`rel_coeff` parameter tensors themselves, independent of
    lin_in/self-loop sizing."""
    hidden, num_bases = 16, 2
    _, edge_types = fixture_data.metadata()
    num_relations = len(edge_types)
    assert num_bases < num_relations  # otherwise basis sharing buys nothing to measure

    encoder = build_encoder("rgcn", fixture_data.metadata(), in_dims, hidden=hidden,
                             num_layers=4, num_bases=num_bases)

    # Exact formula: relation cost is basis pool + per-relation coefficients,
    # not a dedicated hidden x hidden matrix per relation.
    assert encoder.rel_basis.numel() == num_bases * hidden * hidden
    assert encoder.rel_coeff.numel() == num_relations * num_bases
    relational_params = encoder.rel_basis.numel() + encoder.rel_coeff.numel()
    naive_dedicated_per_relation = num_relations * hidden * hidden
    assert relational_params < naive_dedicated_per_relation

    # Shared across every layer, not multiplied by num_layers: rel_basis/
    # rel_coeff must be identical in size between a shallow and deep stack,
    # even though total parameter count still grows (self-loop is per-layer).
    shallow = build_encoder("rgcn", fixture_data.metadata(), in_dims, hidden=hidden,
                             num_layers=1, num_bases=num_bases)
    deep = build_encoder("rgcn", fixture_data.metadata(), in_dims, hidden=hidden,
                          num_layers=8, num_bases=num_bases)
    assert shallow.rel_basis.numel() == deep.rel_basis.numel() == num_bases * hidden * hidden
    assert shallow.rel_coeff.numel() == deep.rel_coeff.numel() == num_relations * num_bases
    shallow_total = sum(p.numel() for p in shallow.parameters())
    deep_total = sum(p.numel() for p in deep.parameters())
    assert deep_total > shallow_total  # self-loop params do grow with num_layers


def test_rgcn_attn_parameter_count_close_to_rgcn(fixture_data, in_dims):
    """Option 1's (`ml/models/rgcn_attn_encoder.py`) whole design bet: the
    shared attention scorer is the ONLY new learnable parameter relative to
    `RGCNEncoder` -- two `Linear(hidden, 1)` layers (`att_msg` + `att_dst`,
    weight+bias each) PER LAYER, nothing else (not per relation, not per
    node type). Locks in the exact size of that increase."""
    hidden, num_layers, num_bases = 16, 4, 2
    metadata = fixture_data.metadata()

    rgcn = build_encoder("rgcn", metadata, in_dims, hidden=hidden,
                          num_layers=num_layers, num_bases=num_bases)
    rgcn_attn = build_encoder("rgcn_attn", metadata, in_dims, hidden=hidden,
                               num_layers=num_layers, num_bases=num_bases)

    expected_increase = 2 * (hidden + 1) * num_layers  # att_msg + att_dst, weight+bias, per layer
    assert rgcn_attn.parameter_count() - rgcn.parameter_count() == expected_increase


def test_rgcn_attn_attention_sums_to_one_per_destination(fixture_data, in_dims, monkeypatch):
    """The property that distinguishes this design from `RGCNEncoder`'s
    per-relation-mean-then-sum scheme: attention weights over a destination
    node's incoming edges must sum to 1 across its ENTIRE incoming edge set,
    regardless of which relation each edge arrived through -- a joint
    softmax, not a per-relation one. `Supplier` in this fixture is fed by
    two distinct relation types after `ToUndirected` -- `SHIPS_FROM` (from
    Shipment, 6 edges) and `rev_SUPPLIES` (from Component, 5 edges) -- so
    this directly exercises the cross-relation joint softmax, not a single
    relation's own trivially-normalized weights."""
    import ml.models.rgcn_attn_encoder as rgcn_attn_module

    captured = []
    real_softmax = rgcn_attn_module.softmax

    def spy_softmax(logit_all, dst_all, num_nodes=None):
        alpha = real_softmax(logit_all, dst_all, num_nodes=num_nodes)
        captured.append((num_nodes, dst_all.clone(), alpha.clone()))
        return alpha

    monkeypatch.setattr(rgcn_attn_module, "softmax", spy_softmax)

    metadata = fixture_data.metadata()
    encoder = build_encoder("rgcn_attn", metadata, in_dims, hidden=16, num_layers=4, num_bases=2)
    encoder(fixture_data.x_dict, fixture_data.edge_index_dict)

    supplier_n = fixture_data["Supplier"].num_nodes  # 5, distinct from Component's 8 and Shipment's 6
    supplier_calls = [c for c in captured if c[0] == supplier_n]
    assert len(supplier_calls) == encoder.num_layers  # one joint softmax call per layer for Supplier

    _num_nodes, dst_all, alpha = supplier_calls[0]
    assert dst_all.numel() == 11  # SHIPS_FROM's 6 + rev_SUPPLIES's 5 -- genuinely cross-relation

    for node_idx in dst_all.unique():
        mask = dst_all == node_idx
        assert alpha[mask].sum().item() == pytest.approx(1.0, abs=1e-5)


def test_rgcn_relemb_parameter_count_formula(fixture_data, in_dims):
    """Option 3's (`ml/models/rgcn_relemb_encoder.py`) whole design bet: a
    relation-identity signal costs `num_relations * relation_embed_dim`
    (the embedding table, rows not matrices) + `relation_embed_dim *
    num_layers` (`att_rel`'s weights) + `num_layers` (`att_rel`'s biases) --
    NOT a second basis-decomposed attention pool (Option 2, not built).
    Locks in the exact parameter delta over `RGCNAttnEncoder` at matched
    hidden/num_bases."""
    hidden, num_layers, num_bases, relation_embed_dim = 16, 4, 2, 5
    metadata = fixture_data.metadata()
    _, edge_types = metadata
    num_relations = len(edge_types)

    rgcn_attn = build_encoder("rgcn_attn", metadata, in_dims, hidden=hidden,
                               num_layers=num_layers, num_bases=num_bases)
    rgcn_relemb = build_encoder("rgcn_relemb", metadata, in_dims, hidden=hidden,
                                 num_layers=num_layers, num_bases=num_bases,
                                 relation_embed_dim=relation_embed_dim)

    expected_increase = (
        num_relations * relation_embed_dim          # rel_embed table
        + relation_embed_dim * num_layers            # att_rel weights, per layer
        + num_layers                                  # att_rel biases, per layer
    )
    assert rgcn_relemb.parameter_count() - rgcn_attn.parameter_count() == expected_increase


def test_rgcn_relemb_attention_sums_to_one_per_destination(fixture_data, in_dims, monkeypatch):
    """Same cross-relation joint-softmax property as
    `test_rgcn_attn_attention_sums_to_one_per_destination`, adapted for
    `rgcn_relemb`: the extra `rel_term` additive offset in the logit must
    not change the softmax's normalization scope -- weights over a
    destination node's ENTIRE incoming edge set, across every relation
    feeding it, must still sum to 1."""
    import ml.models.rgcn_relemb_encoder as rgcn_relemb_module

    captured = []
    real_softmax = rgcn_relemb_module.softmax

    def spy_softmax(logit_all, dst_all, num_nodes=None):
        alpha = real_softmax(logit_all, dst_all, num_nodes=num_nodes)
        captured.append((num_nodes, dst_all.clone(), alpha.clone()))
        return alpha

    monkeypatch.setattr(rgcn_relemb_module, "softmax", spy_softmax)

    metadata = fixture_data.metadata()
    encoder = build_encoder("rgcn_relemb", metadata, in_dims, hidden=16, num_layers=4,
                             num_bases=2, relation_embed_dim=5)
    encoder(fixture_data.x_dict, fixture_data.edge_index_dict)

    supplier_n = fixture_data["Supplier"].num_nodes  # 5, distinct from Component's 8 and Shipment's 6
    supplier_calls = [c for c in captured if c[0] == supplier_n]
    assert len(supplier_calls) == encoder.num_layers  # one joint softmax call per layer for Supplier

    _num_nodes, dst_all, alpha = supplier_calls[0]
    assert dst_all.numel() == 11  # SHIPS_FROM's 6 + rev_SUPPLIES's 5 -- genuinely cross-relation

    for node_idx in dst_all.unique():
        mask = dst_all == node_idx
        assert alpha[mask].sum().item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Structural depth prior
# ---------------------------------------------------------------------------

def test_structural_depth_prior_matches_documented_layers():
    assert STRUCTURAL_DEPTH_PRIOR == {"delay": 2, "shortage": 3, "impact": 3}
    assert readout_layer_for_task("delay", num_layers=4) == 2
    assert readout_layer_for_task("shortage", num_layers=4) == 3
    assert readout_layer_for_task("impact", num_layers=4) == 3


def test_structural_depth_prior_clamps_to_available_layers():
    """At L=1,2 (the sweep's shallow ends), a task whose prior calls for a
    deeper layer than exists must clamp, not index out of range."""
    assert readout_layer_for_task("shortage", num_layers=1) == 1
    assert readout_layer_for_task("impact", num_layers=2) == 2
    assert readout_layer_for_task("delay", num_layers=1) == 1


def test_structural_depth_prior_v2_only_changes_impact():
    """Step A: the corrected prior changes impact (h3 -> h2) and leaves
    delay/shortage exactly as documented -- a controlled, single-variable
    change, not a wholesale re-derivation."""
    assert STRUCTURAL_DEPTH_PRIOR_V2["delay"] == STRUCTURAL_DEPTH_PRIOR["delay"]
    assert STRUCTURAL_DEPTH_PRIOR_V2["shortage"] == STRUCTURAL_DEPTH_PRIOR["shortage"]
    assert STRUCTURAL_DEPTH_PRIOR_V2["impact"] == 2
    assert STRUCTURAL_DEPTH_PRIOR["impact"] == 3  # original untouched

    assert readout_layer_for_task("impact", num_layers=4, prior=STRUCTURAL_DEPTH_PRIOR_V2) == 2
    assert readout_layer_for_task("impact", num_layers=4) == 3  # default still as-documented


def test_shared_depth_overrides_structural_prior_uniformly():
    """Step 4's L-sweep ablation: every task reads the SAME layer L,
    regardless of its own structural-prior depth."""
    for task in STRUCTURAL_DEPTH_PRIOR:
        assert readout_layer_for_task(task, num_layers=4, shared_depth=1) == 1
        assert readout_layer_for_task(task, num_layers=4, shared_depth=3) == 3


# ---------------------------------------------------------------------------
# Prediction head
# ---------------------------------------------------------------------------

def test_prediction_head_output_shape_and_param_count():
    d = 64
    head = PredictionHead(d)
    z = torch.randn(10, d)
    logits = head(z)
    assert logits.shape == (10,)

    # project_HADES.md §6.1: (d^2 + d) + (d + 1) = 4,225 params for d=64
    n_params = sum(p.numel() for p in head.parameters())
    assert n_params == (d * d + d) + (d + 1) == 4225


# ---------------------------------------------------------------------------
# Focal loss -- hand-computed reference value
# ---------------------------------------------------------------------------

def test_focal_loss_matches_hand_computed_value():
    """logit=0 (p=0.5), y=1, gamma=2, alpha=0.5:
    p_t = 0.5, ce = -log(0.5) = 0.693147..., alpha_t = 0.5
    loss = 0.5 * (1 - 0.5)^2 * 0.693147 = 0.0866434..."""
    loss_fn = FocalLoss(gamma=2.0, alpha=0.5)
    logits = torch.tensor([0.0])
    targets = torch.tensor([1.0])
    loss = loss_fn(logits, targets).item()
    expected = 0.5 * (0.5 ** 2) * math.log(2)
    assert loss == pytest.approx(expected, rel=1e-5)


def test_focal_loss_down_weights_easy_examples_more_than_hard_ones():
    """A well-classified example (high p for y=1) should contribute much
    less loss than a poorly-classified one, at the same alpha."""
    loss_fn = FocalLoss(gamma=2.0, alpha=0.5)
    easy = loss_fn(torch.tensor([5.0]), torch.tensor([1.0])).item()   # p ~= 0.993, correct
    hard = loss_fn(torch.tensor([-5.0]), torch.tensor([1.0])).item()  # p ~= 0.007, wrong
    assert easy < hard


def test_alpha_from_positive_rate_is_inverse_frequency_and_clamped():
    assert alpha_from_positive_rate(0.11) == pytest.approx(0.89)
    assert alpha_from_positive_rate(0.0) == 0.95   # clamped away from 1.0
    assert alpha_from_positive_rate(1.0) == 0.05   # clamped away from 0.0
