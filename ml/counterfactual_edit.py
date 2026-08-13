#!/usr/bin/env python3
"""
Naive counterfactual: edit the graph, re-run the frozen forward pass, read the change.

This is Phase 1's model side. It builds **no new neural architecture**. It takes a trained
SHARE + Markov Blanket depth readout checkpoint, applies a structural edit to a snapshot's
`HeteroData`, and calls the *existing, unmodified* forward pass on the edited graph. The
question it exists to answer is whether an ordinary supervised predictor, handed an edited
input, produces a change in risk that agrees with the change the generator's own causal model
says would actually occur.

**Why no architecture change is needed.** SHARE's parameters are indexed by *node type* and
*relation type*, never by node identity or edge count: `lin_in` is per node type,
`rel_basis`/`rel_coeff` give `W_r` per relation, self-loops are per node type per layer, and
`att_msg`/`att_dst` are shared across every relation. Adding or deleting edges changes only
which messages are built and how the per-destination softmax normalises. The Markov readout is
an index-select and is likewise indifferent. So the edited graph runs through the identical
stack, which is exactly what makes this a fair test of the trained model rather than of a new
one.

**The three interventions**, matched one-to-one with
`ml/counterfactual_ground_truth.py`'s so that predicted and true deltas describe the same
event:

* `add_dual_source(primary, secondary)` -- add a `SUPPLIES` edge from `secondary` to every
  component that `primary` sources. In the generator this makes the two suppliers co-parents,
  so each picks up `COPARENT_COUPLING x` the other's own stress.
* `remove_supplier(s)` -- delete every edge incident to `s`, in both directions.
* `substitute_supplier(old, new)` -- re-point `old`'s outgoing `SUPPLIES` edges at `new`, and
  re-point shipments that shipped from `old` to `new`.

**Edge edits are mirrored.** The loader applies `ToUndirected()`, so every forward relation has
a reverse twin. An edit applied to only one direction would leave the graph internally
inconsistent in a way that is invisible until the numbers are wrong, so `_mirror_of` resolves
the twin and every edit touches both.

    python3 ml/counterfactual_edit.py --csv-dir db/csv_v1scale/v0_seed42 --seed 42 --smoke
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ml.ds_backbone import get_backbone, predict  # noqa: E402

SUPPLIES = ("Supplier", "SUPPLIES", "Component")
SHIPS_FROM = ("Shipment", "SHIPS_FROM", "Supplier")


def _mirror_of(data, et: tuple) -> tuple | None:
    """The `ToUndirected` twin of a relation, if present.

    PyG names the reverse of `(src, REL, dst)` as `(dst, 'rev_REL', src)`; when `src == dst`
    it may instead fold both directions into the one relation. Both cases are handled, and an
    absent twin returns None rather than raising -- some relations are genuinely one-sided
    after an edit removes the last edge.
    """
    src, rel, dst = et
    for cand in ((dst, f"rev_{rel}", src), (dst, rel, src)):
        if cand in data.edge_types and cand != et:
            return cand
    return None


def _drop_columns(data, et: tuple, mask_keep: np.ndarray) -> None:
    """Keep only the columns of `et`'s `edge_index` where `mask_keep` is True, and carry any
    edge attributes along with them."""
    store = data[et]
    keep = torch.as_tensor(mask_keep, dtype=torch.bool)
    store.edge_index = store.edge_index[:, keep]
    for attr in ("edge_attr", "edge_weight"):
        if hasattr(store, attr) and getattr(store, attr) is not None:
            v = getattr(store, attr)
            if v.size(0) == keep.size(0):
                setattr(store, attr, v[keep])


def _append_columns(data, et: tuple, src_idx: list[int], dst_idx: list[int]) -> None:
    """Append edges to `et`, padding edge attributes with their column means.

    Padding with the mean rather than zero matters: several of this graph's edge attributes are
    quantities (`quantity_required`, `stock_level`, days-to-ETA) where zero is not a neutral
    value but a specific and often extreme one.
    """
    if not src_idx:
        return
    store = data[et]
    new = torch.tensor([src_idx, dst_idx], dtype=store.edge_index.dtype,
                       device=store.edge_index.device)
    store.edge_index = torch.cat([store.edge_index, new], dim=1)
    for attr in ("edge_attr", "edge_weight"):
        if hasattr(store, attr) and getattr(store, attr) is not None:
            v = getattr(store, attr)
            if v.size(0) == store.edge_index.size(1) - len(src_idx):
                fill = v.mean(dim=0, keepdim=True) if v.numel() else v.new_zeros((1,) + v.shape[1:])
                setattr(store, attr, torch.cat([v, fill.expand(len(src_idx), *v.shape[1:])], 0))


def edit_graph(data, kind: str, spec: dict, sup_index: dict):
    """Return an edited **copy** of `data`. The input is never mutated.

    `sup_index` maps supplier id -> row index in the `Supplier` node store.
    """
    g = copy.deepcopy(data)

    if kind == "add_dual_source":
        a, b = sup_index.get(spec["primary"]), sup_index.get(spec["secondary"])
        if a is None or b is None:
            return None
        ei = g[SUPPLIES].edge_index
        comps = ei[1][ei[0] == a].tolist()          # components sourced by the primary
        if not comps:
            return None
        _append_columns(g, SUPPLIES, [b] * len(comps), comps)
        m = _mirror_of(g, SUPPLIES)
        if m is not None:
            _append_columns(g, m, comps, [b] * len(comps))

    elif kind == "remove_supplier":
        s = sup_index.get(spec["supplier"])
        if s is None:
            return None
        for et in list(g.edge_types):
            src_t, _, dst_t = et
            if "Supplier" not in (src_t, dst_t):
                continue
            ei = g[et].edge_index
            keep = np.ones(ei.size(1), dtype=bool)
            if src_t == "Supplier":
                keep &= (ei[0] != s).cpu().numpy()
            if dst_t == "Supplier":
                keep &= (ei[1] != s).cpu().numpy()
            _drop_columns(g, et, keep)

    elif kind == "substitute_supplier":
        old, new = sup_index.get(spec["old"]), sup_index.get(spec["new"])
        if old is None or new is None:
            return None
        for et in list(g.edge_types):
            src_t, _, dst_t = et
            if "Supplier" not in (src_t, dst_t):
                continue
            ei = g[et].edge_index.clone()
            if src_t == "Supplier":
                ei[0][ei[0] == old] = new
            if dst_t == "Supplier":
                ei[1][ei[1] == old] = new
            g[et].edge_index = ei

    else:
        raise ValueError(f"unknown intervention: {kind}")

    _validate(g)
    return g


def _validate(g) -> None:
    """Structural sanity after an edit. Cheap, and it catches the class of bug that would
    otherwise surface as a plausible-looking wrong number."""
    for et in list(g.edge_types):
        src_t, _, dst_t = et
        ei = g[et].edge_index
        if ei.numel() == 0:
            continue
        if int(ei[0].max()) >= g[src_t].num_nodes or int(ei[1].max()) >= g[dst_t].num_nodes:
            raise ValueError(f"dangling index in {et} after edit")
        if int(ei.min()) < 0:
            raise ValueError(f"negative index in {et} after edit")


class _Shim:
    """Minimal bundle-like wrapper, so `predict` can be called on a bare `HeteroData`."""

    def __init__(self, data):
        self.data = data


def naive_counterfactual(model, bundle, kind: str, spec: dict, sup_index: dict,
                         task: str = "impact",
                         baseline: np.ndarray | None = None) -> np.ndarray | None:
    """`delta[i] = p_after[i] - p_before[i]` for every entity of `task`'s type.

    Both halves come from the same frozen model and the same forward pass; only the graph
    differs. Returns None when the edit is not applicable to this snapshot.

    `baseline` may be supplied to skip recomputing the unedited prediction. It is identical
    for every intervention on the same snapshot, so passing it in halves the forward passes
    across a grid -- worth doing, since the grid is thousands of passes.
    """
    edited = edit_graph(bundle.data, kind, spec, sup_index)
    if edited is None:
        return None
    before = predict(model, bundle)[task] if baseline is None else baseline
    after = predict(model, _Shim(edited))[task]
    if after.shape != before.shape:
        return None                 # entity count changed; not comparable pointwise
    return after - before


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default=os.path.join(REPO, "db", "csv_v1scale", "v0_seed42"))
    ap.add_argument("--variant", default="0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    model, meta = get_backbone(a.csv_dir, a.variant, a.seed, 0, device="cpu")
    te = meta["test_bundles"]
    sup_ids = meta["supplier_ids"]
    sup_index = {s: i for i, s in enumerate(sup_ids)}
    b = te[0]
    print(f"test snapshots: {len(te)}  suppliers: {len(sup_ids)}")
    if a.smoke:
        s0, s1 = sup_ids[0], sup_ids[1]
        for kind, spec in (("add_dual_source", {"primary": s0, "secondary": s1}),
                           ("remove_supplier", {"supplier": s0}),
                           ("substitute_supplier", {"old": s0, "new": s1})):
            d = naive_counterfactual(model, b, kind, spec, sup_index)
            if d is None:
                print(f"  {kind:<22} not applicable")
            else:
                nz = int((np.abs(d) > 1e-9).sum())
                print(f"  {kind:<22} |delta| max={np.abs(d).max():.5f} "
                      f"mean={np.abs(d).mean():.6f} nonzero={nz}/{len(d)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
