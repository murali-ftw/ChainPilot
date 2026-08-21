#!/usr/bin/env python3
"""
Layer 3 v3, PHASE 2 — GATE 1: entity-relevance diagnostics, five methods, one gate.

`layer3_neurosymbolic_build_prompt.md` §4. Five candidate relevance methods are put through
**one shared verification harness** so their numbers are commensurable, and the occlusion arm is
`ml/explain_prediction.py` + `ml/test_explanation_faithfulness.py` reused as the baseline rather
than rebuilt.

    1  occlusion        reuse `ml/explain_prediction.py::occlusion_attribution` unmodified
    2  grad_input       x_i * d(logit)/d(x_i), new -- taken on the LOGIT, deliberately
    3  learned_gate     g^(k)(v) = sigma(W_k h_v + b_k), new -- amortised per-prediction feature
                        mask trained to preserve the prediction under an L1 sparsity penalty
    4  group_perturb    new -- occlude a whole feature BLOCK, or delete a whole incident
                        RELATION, rather than one scalar
    5  operational_cf   new -- the SCM-valid, Gate-0-permitted co-parent edit; the only one of
                        the five that returns ENTITY relevance rather than feature relevance

**What is measured, per method, and why each of the five numbers is separate.**

*coverage* -- the fraction of predictions on which a relevance signal is computable at all. The
occlusion baseline's is **3.4%**, because 96.5% of `delay` predictions sit at p >= 0.999 where a
saturated sigmoid cannot move (`reports/decision_support_build.md` §3.2a). Coverage is reported
per method **and per population**, never averaged across them.

*saturation behaviour* -- every method is run twice: once on the non-saturated band (the
population the 0.166 baseline was measured on, so the comparison is like for like) and once on
the **high-confidence** population specifically -- the top decile of predicted probability. That
population is defined empirically rather than by the p >= 0.999 threshold the prior build used,
because Phase 0 measures the p >= 0.999 fraction on this pipeline at **0.0%** for all three
tasks: a fixed-threshold population would be empty and the question would go unanswered instead
of answered. This is the axis the prompt singles out, and it is
also why methods 2 and 3 operate on the logit: a prediction pinned at p = 0.99997 still has a
perfectly informative logit, and a method that reads probabilities is measuring float precision
there. Faithfulness is therefore scored in **both** spaces, and both are reported.

*faithfulness* -- ablate the cited factor and require direction, magnitude, and beating a
**matched, non-circular null**: the *same factor* ablated on *other entities drawn from the same
population*. `reports/decision_support_build.md` §3.2b measured the two obvious alternatives and
found both degenerate -- same-entity nulls are circular (pass by construction), and a pooled
random null collapses to ~0 because the pool is 96% saturated (pass rate 1.000). Neither is
reused. Chance for a 90th-percentile null is **0.100** by construction.

*agreement* -- against Gate 2's verified rule set, via `ml/gate2_symbolic.py::scm_tier`. On this
generator the causal parents of every label are latent, so the tiers are three, not two:
`scm_parent`, `scm_descendant`, `scm_unrelated`.

*reproducibility* -- 5 dataset seeds x 3 model-init seeds, judged against
`max(init-seed floor, dataset-seed floor)` from `ml/layer3_baseline.py::floors`.

**Cost control, stated because it bounds the numbers.** The null dominates: it is one forward
pass per pool draw per distinct cited factor. It is therefore computed **once per (snapshot,
population, factor) and shared across all five methods**, which is exact -- the null depends on
the factor and the pool, not on which method cited it -- and it is computed only for factors some
method actually cited. Three of the six test snapshots are scored, evenly spaced.

    python3 ml/gate1_relevance.py --tasks delay,impact --dseeds 42,43,44,45,46 --mseeds 0,1,2
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.ds_backbone import get_backbone, load_world, predict  # noqa: E402
from ml.explain_prediction import (  # noqa: E402
    _Shim, baseline_vector, occlusion_attribution, saturation_report, select_targets)
from ml.gate2_symbolic import scm_tier  # noqa: E402
from ml.models.depth import TASK_ENTITY_TYPE  # noqa: E402

METHODS = ("occlusion", "grad_input", "learned_gate", "group_perturb", "operational_cf")
SAT_HI, SAT_LO = 0.999, 1e-4
REL_TOL = 0.5          # `ml/test_explanation_faithfulness.py::REL_TOL`, unchanged
NULL_PCTL = 90         # ditto: chance is therefore 0.100
CHANCE = 1.0 - NULL_PCTL / 100.0

# Feature names by node type, in the loader's emitted column order. Asserted against the tensor
# width at run time and padded positionally if it disagrees -- the same discipline
# `ml/explain_prediction.py::feature_labels` uses, extended to the two entity types Gate 1 needs
# that it did not.
FEATURE_NAMES = {
    "Shipment": ["carrier_on_time_rate_90d", "days_to_eta", "days_since_dispatch",
                 "status_scheduled", "status_in_transit", "status_delivered", "status_delayed"],
    "Supplier": ["lead_time_days_z", "capacity_score_z"] + [f"country_{i}" for i in range(6)]
                + ["on_time_rate_30d", "on_time_rate_90d", "on_time_rate_180d", "trend_slope",
                   "lateness_variance", "days_since_last_late"],
    "Product": ["min_stock_ratio", "avg_stock_ratio", "total_stock", "total_reorder_threshold",
                "warehouse_count", "bom_component_count", "bom_mean_quantity_required"]
               + [f"category_{i}" for i in range(6)],
}

# Feature blocks for method 4. A block is a set of columns that describe ONE aspect of the
# entity; occluding them together asks "does this aspect matter", which a single-column
# occlusion cannot ask when the aspect is spread over a one-hot.
FEATURE_GROUPS = {
    "Shipment": {"carrier": ["carrier_on_time_rate_90d"],
                 "schedule": ["days_to_eta", "days_since_dispatch"],
                 "status": ["status_scheduled", "status_in_transit", "status_delivered",
                            "status_delayed"]},
    "Supplier": {"static": ["lead_time_days_z", "capacity_score_z"],
                 "geography": [f"country_{i}" for i in range(6)],
                 "history": ["on_time_rate_30d", "on_time_rate_90d", "on_time_rate_180d"],
                 "trend": ["trend_slope", "lateness_variance", "days_since_last_late"]},
    "Product": {"inventory": ["min_stock_ratio", "avg_stock_ratio", "total_stock",
                              "total_reorder_threshold", "warehouse_count"],
                "bom": ["bom_component_count", "bom_mean_quantity_required"],
                "category": [f"category_{i}" for i in range(6)]},
}


def names_for(data, entity_type: str) -> list[str]:
    width = data[entity_type].x.size(-1)
    known = FEATURE_NAMES.get(entity_type, [])
    if len(known) >= width:
        return known[:width]
    return known + [f"feature_{i}" for i in range(len(known), width)]


# --------------------------------------------------------------------------- model access

@torch.no_grad()
def predict_logit(model, bundle, task: str) -> np.ndarray:
    """The task's pre-sigmoid logit for every entity.

    Read directly rather than by inverting `predict`'s probability: at p = 1 - 1e-7 the
    probability has thrown away everything that distinguishes one saturated prediction from
    another, and the whole saturation question is about that discarded range.
    """
    model.eval()
    logits, _ = model(bundle.data.x_dict, bundle.data.edge_index_dict)
    return logits[task].float().cpu().numpy()


# --------------------------------------------------------------------------- factors

class Factor:
    """An ablatable object a relevance method can cite, plus how to ablate it.

    All five methods produce factors, and the verification harness treats them identically:
    ablate, measure the response of the target's prediction in probability and logit space, and
    compare against the same factor ablated on a matched pool. Making the factor the unit --
    rather than the feature index -- is what lets a feature, a feature block, a whole relation
    and a co-parent edge be scored on one scale.
    """

    def __init__(self, key: str, kind: str, label: str, cols=None, relation=None,
                 partner: int | None = None, scm_feature: str | None = None):
        self.key, self.kind, self.label = key, kind, label
        self.cols = cols or []
        self.relation = relation
        self.partner = partner
        self.scm_feature = scm_feature

    @contextlib.contextmanager
    def applied(self, bundle, target: int, entity_type: str, bvec: torch.Tensor,
                sup_index_of=None):
        data = bundle.data
        if self.kind in ("feature", "group"):
            x = data[entity_type].x
            saved = x[target, self.cols].clone()
            try:
                for c in self.cols:
                    x[target, c] = bvec[c]
                yield
            finally:
                x[target, self.cols] = saved
        elif self.kind == "relation":
            saved = _drop_incident(data, self.relation, entity_type, target)
            try:
                yield
            finally:
                _restore(data, saved)
        elif self.kind == "coparent":
            saved = _drop_coparent_link(data, target, self.partner)
            try:
                yield
            finally:
                _restore(data, saved)
        else:
            raise ValueError(self.kind)


def _mirror_of(data, et: tuple):
    src, rel, dst = et
    for cand in ((dst, f"rev_{rel}", src), (dst, rel, src)):
        if cand in data.edge_types and cand != et:
            return cand
    return None


def _drop_incident(data, relation: tuple, entity_type: str, target: int) -> list:
    """Delete every edge of `relation` incident on `target`, in both directions.

    Mirrored, for the reason `ml/counterfactual_edit.py` documents: the loader applies
    `ToUndirected()`, and editing one direction leaves the graph internally inconsistent in a way
    nothing catches until the numbers are wrong.
    """
    saved = []
    for et in [relation, _mirror_of(data, relation)]:
        if et is None or et not in data.edge_types:
            continue
        ei = data[et].edge_index
        row = (0 if et[0] == entity_type else (1 if et[2] == entity_type else None))
        if row is None:
            continue
        keep = (ei[row] != target)
        saved.append((et, ei.clone()))
        data[et].edge_index = ei[:, keep]
    return saved


def _drop_coparent_link(data, target: int, partner: int) -> list:
    """Remove the shared-component links that make `target` and `partner` co-parents.

    This is the graph-side expression of `do(coparents[s] -= {p})` -- the same lever Gate 3's
    `remove_supplier`/`add_dual_source` move and the one Gate 0 Tier C tested for
    identifiability. Only the *partner's* SUPPLIES edges into the shared components are removed,
    so the target's own sourcing is untouched and the edit is the co-parent relationship itself.
    """
    et = ("Supplier", "SUPPLIES", "Component")
    if et not in data.edge_types:
        return []
    ei = data[et].edge_index
    mine = set(ei[1][ei[0] == target].tolist())
    drop = (ei[0] == partner) & torch.tensor([int(c) in mine for c in ei[1].tolist()],
                                             dtype=torch.bool)
    if not bool(drop.any()):
        return []
    saved = []
    for rel in [et, _mirror_of(data, et)]:
        if rel is None or rel not in data.edge_types:
            continue
        cur = data[rel].edge_index
        # `ToUndirected` emits the reverse relation column-for-column from the forward one, so
        # the same mask applies to both. The width check is the guard: if that ever stops
        # holding, the twin is left alone rather than sliced by a mask that does not describe it.
        if cur.size(1) != drop.numel():
            continue
        saved.append((rel, cur.clone()))
        data[rel].edge_index = cur[:, ~drop]
    return saved


def _restore(data, saved: list) -> None:
    for et, ei in saved:
        data[et].edge_index = ei


# --------------------------------------------------------------------------- the five methods

def m_occlusion(model, bundle, targets, task, names, bvec) -> np.ndarray:
    """`ml/explain_prediction.py::occlusion_attribution`, called unmodified."""
    phi, got = occlusion_attribution(model, bundle, targets, task, "median")
    # `ml/explain_prediction.py` carries real column names only for Shipment and falls back to
    # positional ones elsewhere, so the agreement check is on the width always and on the names
    # only where both modules claim to know them. A width mismatch means the loader moved and
    # every attribution below would be mislabelled.
    assert len(got) == len(names), (
        f"feature width disagrees: explain_prediction {len(got)} vs gate1 {len(names)}")
    named = [i for i, g in enumerate(got) if not g.startswith("feature_")]
    assert all(got[i] == names[i] for i in named), \
        "loader feature order moved under explain_prediction"
    return phi


def m_grad_input(model, bundle, targets, task, names, bvec) -> np.ndarray:
    """`x_i * d(logit)/d(x_i)`, on the LOGIT rather than the probability.

    The distinction is the whole reason this method is in the comparison. `d(p)/d(x)` carries a
    factor `p(1-p)`, which at p = 0.9999 is 1e-4 and at p = 1 - 1e-7 is 1e-7: a probability-space
    gradient reports "nothing matters" for exactly the predictions the model is most confident
    about, which is the same failure occlusion has, arrived at by a different route. On the logit
    the saturation factor is absent.

    Backbone parameters stay frozen; the graph is built for the INPUT tensor only.
    """
    et = TASK_ENTITY_TYPE[task]
    x_dict = {k: v.clone() for k, v in bundle.data.x_dict.items()}
    x_dict[et] = x_dict[et].detach().requires_grad_(True)
    model.eval()
    logits, _ = model(x_dict, bundle.data.edge_index_dict)
    sel = logits[task][torch.as_tensor(targets, dtype=torch.long)].sum()
    g, = torch.autograd.grad(sel, x_dict[et])
    idx = torch.as_tensor(targets, dtype=torch.long)
    contrib = (g[idx] * (bundle.data[et].x[idx] - bvec.unsqueeze(0))).detach().cpu().numpy()
    return contrib


class GateHead(torch.nn.Module):
    """`g^(k)(v) = sigma(W_k h_v + b_k)` -- one linear map per node type, per task."""

    def __init__(self, d_in: int, n_feat: int):
        super().__init__()
        self.lin = torch.nn.Linear(d_in, n_feat)

    def forward(self, h):
        return torch.sigmoid(self.lin(h))


def train_gate(model, train_bundles, task: str, bvec: torch.Tensor, seed: int,
               steps: int = 40, lam: float = 0.05, lr: float = 5e-2) -> GateHead:
    """Fit the relevance gate to keep the prediction while switching features off.

        L = mean( (logit(f(x_masked)) - logit(f(x)))^2 ) + lam * mean(g)

    with `x_masked[v] = g_v * x_v + (1 - g_v) * baseline`. The L1 term is what makes `g` a
    relevance statement rather than the constant 1: without it the trivial optimum is to keep
    every feature. Trained on TRAIN snapshots and applied to test, so the gate never sees the
    predictions it is asked to explain.

    Deliberately amortised (a function of the frozen embedding `h_v`) rather than optimised
    per prediction: the proposal this build is testing specifies `sigma(W_k h_i + b_k)`, and an
    amortised gate is the only form of it that can be evaluated on a held-out entity at all.
    """
    et = TASK_ENTITY_TYPE[task]
    torch.manual_seed(seed)
    d_in = None
    with torch.no_grad():
        _, layers = model(train_bundles[0].data.x_dict, train_bundles[0].data.edge_index_dict)
        d_in = layers[_depth_of(model, task)][et].size(-1)
    head = GateHead(d_in, train_bundles[0].data[et].x.size(-1))
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    live = [n for n, p in model.named_parameters() if p.requires_grad]
    assert not live, f"backbone not frozen: {live[:3]}"

    b = train_bundles[len(train_bundles) // 2]        # one mid-window snapshot; see docstring
    with torch.no_grad():
        base_logit, layers = model(b.data.x_dict, b.data.edge_index_dict)
        base_logit = base_logit[task].detach()
        h = layers[_depth_of(model, task)][et].detach()
    x0 = b.data[et].x.detach()
    for _ in range(steps):
        opt.zero_grad()
        g = head(h)
        xm = g * x0 + (1 - g) * bvec.unsqueeze(0)
        xd = dict(b.data.x_dict)
        xd[et] = xm
        logits, _ = model(xd, b.data.edge_index_dict)
        loss = ((logits[task] - base_logit) ** 2).mean() + lam * g.mean()
        loss.backward()
        opt.step()
    head.eval()
    return head


def _depth_of(model, task: str) -> int:
    from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH
    return MARKOV_READOUT_DEPTH[task]


@torch.no_grad()
def m_learned_gate(model, bundle, targets, task, names, bvec, head: GateHead) -> np.ndarray:
    """Relevance = the gate value. A feature the gate keeps open is one the prediction needs."""
    et = TASK_ENTITY_TYPE[task]
    _, layers = model(bundle.data.x_dict, bundle.data.edge_index_dict)
    h = layers[_depth_of(model, task)][et]
    g = head(h)[torch.as_tensor(targets, dtype=torch.long)]
    return g.cpu().numpy()


def group_factors(data, entity_type: str, names: list[str]) -> list:
    """Method 4's factor set: feature BLOCKS plus each incident RELATION."""
    pos = {n: i for i, n in enumerate(names)}
    out = []
    for gname, cols in FEATURE_GROUPS.get(entity_type, {}).items():
        idx = [pos[c] for c in cols if c in pos]
        if idx:
            out.append(Factor(f"group:{gname}", "group", f"block[{gname}]", cols=idx))
    for et in data.edge_types:
        if et[1].startswith("rev_"):
            continue
        if entity_type in (et[0], et[2]):
            out.append(Factor(f"rel:{et[1]}", "relation", f"edges[{et[1]}]", relation=et))
    return out


@torch.no_grad()
def m_group_perturb(model, bundle, targets, task, factors, bvec) -> np.ndarray:
    """Block/relation-level perturbation, one forward pass per factor for the whole target set.

    Batched over targets exactly as `occlusion_attribution` batches its single-feature pass, so
    the two methods' costs and their contamination behaviour are the same.
    """
    et = TASK_ENTITY_TYPE[task]
    base = predict(model, bundle)[task][targets]
    idx = torch.as_tensor(targets, dtype=torch.long)
    phi = np.zeros((len(targets), len(factors)), dtype=float)
    for j, f in enumerate(factors):
        if f.kind == "group":
            x = bundle.data[et].x
            saved = x[idx][:, f.cols].clone()
            for c in f.cols:
                x[idx, c] = bvec[c]
            try:
                p = predict(model, _Shim(bundle.data))[task][targets]
            finally:
                for k, c in enumerate(f.cols):
                    x[idx, c] = saved[:, k]
        else:
            saved = []
            for et_ in [f.relation, _mirror_of(bundle.data, f.relation)]:
                if et_ is None or et_ not in bundle.data.edge_types:
                    continue
                ei = bundle.data[et_].edge_index
                row = (0 if et_[0] == et else (1 if et_[2] == et else None))
                if row is None:
                    continue
                keep = ~torch.isin(ei[row], idx)
                saved.append((et_, ei.clone()))
                bundle.data[et_].edge_index = ei[:, keep]
            try:
                p = predict(model, _Shim(bundle.data))[task][targets]
            finally:
                _restore(bundle.data, saved)
        phi[:, j] = base - p
    return phi


def coparent_factors(data, target: int, max_partners: int = 3) -> list:
    """Method 5's factor set for one Supplier: the entities it shares a component with.

    This is the only method of the five whose factors are **entities**, which is what "entity
    relevance" means in Layer 4's schema. Its coverage is bounded by the world: the generator
    gives roughly one supplier in six a co-parent at all, and a supplier with none has no
    operational co-parent lever, which is a real limit rather than a measurement failure.
    """
    et = ("Supplier", "SUPPLIES", "Component")
    if et not in data.edge_types:
        return []
    ei = data[et].edge_index
    mine = set(ei[1][ei[0] == target].tolist())
    if not mine:
        return []
    partners = {}
    for s, c in zip(ei[0].tolist(), ei[1].tolist()):
        if s != target and c in mine:
            partners[s] = partners.get(s, 0) + 1
    top = sorted(partners.items(), key=lambda kv: -kv[1])[:max_partners]
    return [Factor(f"coparent:{p}", "coparent", f"co-parent[{p}]", partner=p) for p, _ in top]


# --------------------------------------------------------------------------- verification

@torch.no_grad()
def response(model, bundle, target: int, factor: Factor, task: str, et: str,
             bvec: torch.Tensor, base_p: np.ndarray, base_l: np.ndarray) -> tuple:
    """`(delta_p, delta_logit)` from ablating one factor on one entity."""
    with factor.applied(bundle, target, et, bvec):
        logits, _ = model(bundle.data.x_dict, bundle.data.edge_index_dict)
        p = float(torch.sigmoid(logits[task][target]))
        lg = float(logits[task][target])
    return base_p[target] - p, base_l[target] - lg


def null_for(model, bundle, factor: Factor, task: str, et: str, bvec: torch.Tensor,
             pool: np.ndarray, base_p, base_l, rng, n_draws: int) -> tuple:
    """The specificity null: the SAME factor ablated on other entities from the SAME population.

    For an entity-valued factor (`coparent:<p>`) the identical partner does not exist on another
    supplier, so the null is the same *kind* of edit -- drop that supplier's largest co-parent
    link -- which is the matched question ("does removing a co-parent move this supplier more
    than it moves a comparable one") rather than a different one.
    """
    draws = rng.choice(pool, size=min(n_draws, len(pool)), replace=False)
    dp, dl = [], []
    for t in draws:
        f = factor
        if factor.kind == "coparent":
            cand = coparent_factors(bundle.data, int(t), max_partners=1)
            if not cand:
                continue
            f = cand[0]
        a, b = response(model, bundle, int(t), f, task, et, bvec, base_p, base_l)
        dp.append(abs(a)); dl.append(abs(b))
    if not dp:
        return None, None, 0
    return (float(np.percentile(dp, NULL_PCTL)), float(np.percentile(dl, NULL_PCTL)), len(dp))


# --------------------------------------------------------------------------- per-snapshot run

def run_snapshot(model, bundle, task: str, n_targets: int, n_null: int, rng_seed: int,
                 gate_head, train_pop: str) -> dict:
    et = TASK_ENTITY_TYPE[task]
    names = names_for(bundle.data, et)
    bvec = baseline_vector(bundle.data, et, "median")
    p = predict(model, bundle)[task]
    lg = predict_logit(model, bundle, task)
    if len(p) == 0:
        return {}
    sat = saturation_report(p, SAT_HI, SAT_LO)
    band_idx = np.where((p < SAT_HI) & (p > SAT_LO))[0]
    # **The high-confidence population is defined empirically, and measuring why was necessary.**
    # `reports/decision_support_build.md` §3.2a found 96.5% of `delay` predictions at p >= 0.999
    # and reported occlusion coverage of 3.4% as a consequence. On this repository's pipeline
    # that does not reproduce: Phase 0 measures the p >= 0.999 fraction at **0.0%** on all three
    # tasks. A fixed-threshold "saturated" population would therefore be empty here and the
    # prompt's question -- does the method stay informative where the model is most confident --
    # would go unanswered rather than answered negatively. The population is consequently the
    # **top decile of predicted probability**, which asks the same question on whatever
    # confidence range the pipeline actually produces; the p >= 0.999 count is still reported,
    # and it is zero.
    hi_cut = np.quantile(p, 0.9) if len(p) else 1.0
    hi_idx = np.where(p >= hi_cut)[0]
    sat["high_conf_cut"] = float(hi_cut)
    sat["high_conf_n"] = int(len(hi_idx))
    rng = np.random.default_rng(rng_seed)

    out = {"saturation": sat, "populations": {}}
    for pop, pool in (("band", band_idx), ("high_conf", hi_idx)):
        if len(pool) < 10:
            out["populations"][pop] = {"n_pool": int(len(pool)), "insufficient": True}
            continue
        targets = (select_targets(p, n_targets, "stratified", SAT_HI, rng_seed) if pop == "band"
                   else rng.choice(pool, size=min(n_targets, len(pool)), replace=False))
        targets = np.asarray(targets, dtype=int)
        if len(targets) == 0:
            out["populations"][pop] = {"n_pool": int(len(pool)), "insufficient": True}
            continue

        # ---- score every method on the same targets -------------------------------------
        feat_factors = [Factor(f"feat:{i}", "feature", names[i], cols=[i], scm_feature=names[i])
                        for i in range(len(names))]
        gfactors = group_factors(bundle.data, et, names)
        scores = {}
        scores["occlusion"] = (m_occlusion(model, bundle, targets, task, names, bvec),
                               feat_factors)
        scores["grad_input"] = (m_grad_input(model, bundle, targets, task, names, bvec),
                                feat_factors)
        if gate_head is not None:
            scores["learned_gate"] = (m_learned_gate(model, bundle, targets, task, names, bvec,
                                                     gate_head), feat_factors)
        scores["group_perturb"] = (m_group_perturb(model, bundle, targets, task, gfactors, bvec),
                                   gfactors)
        if et == "Supplier":
            cf_rows, cf_facs = [], []
            for t in targets:
                cand = coparent_factors(bundle.data, int(t))
                cf_facs.append(cand)
                if cand:
                    vals = [response(model, bundle, int(t), f, task, et, bvec, p, lg)[0]
                            for f in cand]
                    cf_rows.append(vals)
                else:
                    cf_rows.append([])
            scores["operational_cf"] = ("per_target", (cf_rows, cf_facs))

        # ---- cited factor per (method, target) ------------------------------------------
        cited = {}
        for meth, val in scores.items():
            if meth == "operational_cf":
                rows, facs = val[1]
                cited[meth] = [(facs[r][int(np.argmax(rows[r]))], float(max(rows[r])))
                               if rows[r] and max(rows[r]) > 0 else (None, 0.0)
                               for r in range(len(targets))]
            else:
                phi, facs = val
                cited[meth] = []
                for r in range(len(targets)):
                    j = int(np.argmax(phi[r]))
                    cited[meth].append((facs[j], float(phi[r, j])) if phi[r, j] > 0
                                       else (None, 0.0))

        # ---- one null per distinct CITED factor, shared across methods -------------------
        wanted = {}
        for meth, lst in cited.items():
            for f, _ in lst:
                if f is not None:
                    wanted.setdefault(f.key, f)
        nulls = {}
        for key, f in wanted.items():
            nulls[key] = null_for(model, bundle, f, task, et, bvec, pool, p, lg, rng, n_null)

        # ---- verify ---------------------------------------------------------------------
        per_method = {}
        for meth, lst in cited.items():
            rows = []
            for r, t in enumerate(targets):
                f, claimed = lst[r]
                if f is None:
                    rows.append({"target": int(t), "p": float(p[t]), "computable": False})
                    continue
                dp, dl = response(model, bundle, int(t), f, task, et, bvec, p, lg)
                np_, nl_, nn = nulls[f.key]
                rows.append({
                    "target": int(t), "p": float(p[t]), "logit": float(lg[t]),
                    "computable": True, "factor": f.label, "factor_key": f.key,
                    "factor_kind": f.kind, "claimed": claimed,
                    "observed_p": dp, "observed_logit": dl,
                    "null_p": np_, "null_logit": nl_, "n_null": nn,
                    "scm_tier": (scm_tier(task, f.scm_feature) if f.scm_feature
                                 else ("scm_parent" if f.kind == "coparent" else "scm_group")),
                    "direction_ok_p": dp > 0, "direction_ok_logit": dl > 0,
                    "magnitude_ok_p": (dp >= REL_TOL * claimed
                                       if f.kind in ("feature", "group") and
                                       meth in ("occlusion", "group_perturb") else True),
                    "beats_null_p": (np_ is not None and abs(dp) >= np_),
                    "beats_null_logit": (nl_ is not None and abs(dl) >= nl_),
                })
                rows[-1]["passed_p"] = bool(rows[-1]["direction_ok_p"]
                                            and rows[-1]["magnitude_ok_p"]
                                            and rows[-1]["beats_null_p"])
                rows[-1]["passed_logit"] = bool(rows[-1]["direction_ok_logit"]
                                                and rows[-1]["beats_null_logit"])
            per_method[meth] = rows
        out["populations"][pop] = {"n_pool": int(len(pool)), "n_targets": int(len(targets)),
                                   "methods": per_method}
    return out


def summarise_rows(rows: list) -> dict:
    n = len(rows)
    comp = [r for r in rows if r.get("computable")]
    d = {"n_targets": n, "n_computable": len(comp),
         "coverage": (len(comp) / n) if n else None}
    if not comp:
        return d
    d.update({
        "faithfulness_p": float(np.mean([r["passed_p"] for r in comp])),
        "faithfulness_logit": float(np.mean([r["passed_logit"] for r in comp])),
        "direction_rate_p": float(np.mean([r["direction_ok_p"] for r in comp])),
        "beats_null_rate_p": float(np.mean([r["beats_null_p"] for r in comp])),
        "beats_null_rate_logit": float(np.mean([r["beats_null_logit"] for r in comp])),
        "mean_abs_response_p": float(np.mean([abs(r["observed_p"]) for r in comp])),
        "mean_abs_response_logit": float(np.mean([abs(r["observed_logit"]) for r in comp])),
        "scm_tiers": _counts([r["scm_tier"] for r in comp]),
        "scm_agreement": float(np.mean([r["scm_tier"] in ("scm_parent", "scm_descendant")
                                        for r in comp])),
        "scm_parent_rate": float(np.mean([r["scm_tier"] == "scm_parent" for r in comp])),
        "top_factors": _counts([r["factor"] for r in comp]),
    })
    return d


def _counts(xs) -> dict:
    out: dict = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------- driver

def run_cell(csv_dir: str, variant: str, dseed: int, mseed: int, task: str, n_targets: int,
             n_null: int, n_snapshots: int, rng_seed: int) -> dict:
    model, meta = get_backbone(csv_dir, variant, dseed, mseed, device="cpu", verbose=False)
    tr, _va, te, _ids = load_world(csv_dir, "cpu")
    et = TASK_ENTITY_TYPE[task]
    bvec = baseline_vector(tr[0].data, et, "median")
    try:
        gate_head = train_gate(model, tr, task, bvec, seed=mseed)
    except Exception as exc:            # a gate that will not fit is reported, not hidden
        print(f"    learned_gate unavailable: {exc}", flush=True)
        gate_head = None

    picks = np.linspace(0, len(te) - 1, min(n_snapshots, len(te))).astype(int)
    snaps = [run_snapshot(model, te[i], task, n_targets, n_null, rng_seed + i,
                          gate_head, "band") for i in picks]

    agg = {"dseed": dseed, "mseed": mseed, "task": task, "auc": meta["auc"],
           "snapshots": [int(i) for i in picks],
           "saturation": {k: float(np.mean([s["saturation"][k] for s in snaps if s]))
                          for k in ("saturated_high_frac", "explainable_frac")},
           "populations": {}}
    for pop in ("band", "high_conf"):
        agg["populations"][pop] = {}
        for meth in METHODS:
            rows = [r for s in snaps if s for r in
                    s.get("populations", {}).get(pop, {}).get("methods", {}).get(meth, [])]
            if rows:
                agg["populations"][pop][meth] = summarise_rows(rows)
    return agg


def dual_floor(vals_by_cell: dict, dseeds, mseeds) -> dict:
    """`max(init-seed floor, dataset-seed floor)` for one scalar metric across the grid."""
    M = np.array([[vals_by_cell.get((d, m), np.nan) for m in mseeds] for d in dseeds],
                 dtype=float)
    if not np.isfinite(M).any():
        return {"available": False}
    per_world = [float(np.nanmax(r) - np.nanmin(r)) for r in M if np.isfinite(r).sum() > 1]
    wm = np.array([np.nanmean(r) for r in M], dtype=float)
    init_f = max(per_world) if per_world else float("nan")
    dset_f = (float(np.nanmax(wm) - np.nanmin(wm)) if np.isfinite(wm).sum() > 1
              else float("nan"))
    return {"available": True, "mean": float(np.nanmean(M)),
            "per_world_mean": [float(v) for v in wm],
            "init_seed_floor": init_f, "dataset_seed_floor": dset_f,
            "floor": (max([f for f in (init_f, dset_f) if f == f] or [float("nan")]))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--variant", default="0")
    ap.add_argument("--tasks", default="delay,impact")
    ap.add_argument("--dseeds", default="42,43,44,45,46")
    ap.add_argument("--mseeds", default="0,1,2")
    ap.add_argument("--n-targets", type=int, default=30)
    ap.add_argument("--n-null", type=int, default=30)
    ap.add_argument("--n-snapshots", type=int, default=3)
    ap.add_argument("--rng-seed", type=int, default=20260820)
    ap.add_argument("--out", default=os.path.join(REPO, "out", "layer3_v3", "gate1.json"))
    a = ap.parse_args()

    dseeds = [int(x) for x in a.dseeds.split(",") if x.strip()]
    mseeds = [int(x) for x in a.mseeds.split(",") if x.strip()]
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]

    cells = []
    for task in tasks:
        for d in dseeds:
            csv_dir = os.path.join(a.csv_root, f"v{a.variant}_seed{d}")
            for m in mseeds:
                c = run_cell(csv_dir, a.variant, d, m, task, a.n_targets, a.n_null,
                             a.n_snapshots, a.rng_seed)
                cells.append(c)
                b = c["populations"]["band"]
                print(f"  {task} d{d} m{m}  " + "  ".join(
                    f"{k[:4]}:cov={b[k]['coverage']:.2f}/f={b[k].get('faithfulness_p', 0):.3f}"
                    for k in b), flush=True)

    summary = {}
    for task in tasks:
        summary[task] = {}
        for pop in ("band", "high_conf"):
            summary[task][pop] = {}
            for meth in METHODS:
                cov, fap, fal, agr = {}, {}, {}, {}
                for c in cells:
                    if c["task"] != task:
                        continue
                    s = c["populations"].get(pop, {}).get(meth)
                    if not s:
                        continue
                    k = (c["dseed"], c["mseed"])
                    cov[k] = s.get("coverage")
                    fap[k] = s.get("faithfulness_p")
                    fal[k] = s.get("faithfulness_logit")
                    agr[k] = s.get("scm_agreement")
                if not cov:
                    continue
                fp = dual_floor({k: v for k, v in fap.items() if v is not None},
                                dseeds, mseeds)
                fl = dual_floor({k: v for k, v in fal.items() if v is not None},
                                dseeds, mseeds)
                entry = {
                    "coverage": dual_floor({k: v for k, v in cov.items() if v is not None},
                                           dseeds, mseeds),
                    "faithfulness_p": fp, "faithfulness_logit": fl,
                    "scm_agreement": dual_floor({k: v for k, v in agr.items()
                                                 if v is not None}, dseeds, mseeds),
                    "chance": CHANCE,
                }
                for space, f in (("p", fp), ("logit", fl)):
                    if f.get("available"):
                        entry[f"above_chance_{space}"] = f["mean"] - CHANCE
                        entry[f"clears_floor_{space}"] = bool(
                            (f["mean"] - CHANCE) > f["floor"])
                        entry[f"sign_consistent_{space}"] = bool(
                            all(v > CHANCE for v in f["per_world_mean"]))
                summary[task][pop][meth] = entry

    print("\n" + "=" * 138)
    print("GATE 1 — five relevance methods, one shared verification harness")
    print("=" * 138)
    hdr = (f"{'task':<8}{'population':<11}{'method':<16}{'coverage':>10}{'faith(p)':>10}"
           f"{'faith(lg)':>11}{'floor(p)':>10}{'>chance':>9}{'clears':>8}{'5/5':>6}"
           f"{'SCM agr':>9}{'SCM par':>9}")
    print(hdr); print("-" * len(hdr))
    for task in tasks:
        for pop in ("band", "high_conf"):
            for meth in METHODS:
                s = summary.get(task, {}).get(pop, {}).get(meth)
                if not s or not s["faithfulness_p"].get("available"):
                    continue
                cv = s["coverage"]["mean"]
                print(f"{task:<8}{pop:<11}{meth:<16}{cv:>10.3f}"
                      f"{s['faithfulness_p']['mean']:>10.3f}"
                      f"{s['faithfulness_logit']['mean']:>11.3f}"
                      f"{s['faithfulness_p']['floor']:>10.3f}"
                      f"{s.get('above_chance_p', float('nan')):>+9.3f}"
                      f"{('YES' if s.get('clears_floor_p') else 'no'):>8}"
                      f"{('yes' if s.get('sign_consistent_p') else 'NO'):>6}"
                      f"{s['scm_agreement']['mean']:>9.3f}"
                      f"{_par(cells, task, pop, meth):>9.3f}")
    print("-" * len(hdr))
    print(f"chance = {CHANCE:.3f} (a {NULL_PCTL}th-percentile null is exceeded "
          f"{CHANCE:.0%} of the time by construction)")

    blob = {"config": vars(a), "cells": cells, "summary": summary, "chance": CHANCE}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"\nwrote {a.out}")
    return 0


def _par(cells, task, pop, meth) -> float:
    v = [c["populations"][pop][meth]["scm_parent_rate"] for c in cells
         if c["task"] == task and meth in c["populations"].get(pop, {})
         and "scm_parent_rate" in c["populations"][pop][meth]]
    return statistics.fmean(v) if v else float("nan")


if __name__ == "__main__":
    sys.exit(main())
