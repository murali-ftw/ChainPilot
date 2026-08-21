#!/usr/bin/env python3
"""
Layer 3 redesign, PHASE 5 -- the hard Layer 3 -> Layer 4 interface, and the required ablations.

**The interface.** Each family's output is packaged as

    O_s = (Z_s, P_s [only if Phase 3 marked it Supported], C_s, status)

`P_s` -- the calibrated probability -- is present ONLY for a Supported family. That is the whole
point of the hard interface: a family whose calibration did not transfer cross-world has no
honest probability to hand downstream, so the field is absent rather than filled with an
uncalibrated number that a consumer would read as one. `assemble()` enforces this structurally
(the key is not created), not by convention.

**The wiring rule**, enforced by `wire_for_task()`:

    Supported cell      -> Z_s reaches the risk head
    Uncertain family    -> Z_s may reach the head ONLY alongside its confidence C_s
    Redundant cell      -> Z_s must NOT reach the head
    Unidentifiable      -> Z_s must NOT reach the head

**The ablation set**, run before any risk-head result is reported as an improvement:

    tabular_no_share   the task entity's raw observable features, no SHARE at all
    depth_h0..h4       a direct task head from each encoder depth H^k
    markov_readout     the depth this task's Markov readout actually uses
                       (MARKOV_READOUT_DEPTH: delay h^1, shortage h^3, impact h^4)
    state_all_wired    Markov depth + EVERY family's Z, regardless of its cell's class
    state_matrix_wired Markov depth + only the Supported cells' Z for this task

    markov_frozen_production   the trained backbone's own recorded test AUC

**One footing difference is flagged rather than smoothed over.** Every probe arm above is a small
head fit on `va` and scored on `te` over the frozen backbone. `markov_frozen_production` is the
end-to-end-trained model's own task head, fitted on `tr` for 100 epochs. Those are different
objects on different budgets, and the prompt requires both, so both are reported -- but the
like-for-like comparison for "does matrix-validated wiring help" is
`state_matrix_wired` vs `markov_readout`, not vs the production number.

**The metric is fixed in advance**: test ROC-AUC, pooled over 5 dataset seeds, and an arm is
credited over another only if the mean gain exceeds the reproduction floor
`max(init-seed spread, dataset-seed spread)` AND is sign-consistent across all five seeds. That
is the same bar every other result in this project clears or fails.

    python3 ml/layer3_interface.py --variant A --tasks delay,shortage,impact \
        --out out/layer3_redesign/phase5_ablation.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.ds_backbone import get_backbone, load_world              # noqa: E402
from ml.hypothesis_ranker import roc_auc                         # noqa: E402
from ml.latent_state_head import (                               # noqa: E402
    align, assert_backbone_frozen, binarise, depth_embeddings, latent_targets,
)
from ml.layer3_sufficiency import fit_probs                      # noqa: E402
from ml.models.depth import TASK_ENTITY_TYPE                     # noqa: E402
from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH   # noqa: E402
from ml.state_task_sufficiency import (                          # noqa: E402
    PRIMARY_AGG, _neighbour_suppliers, supplier_z_map,
)

DEPTHS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# the interface object
# ---------------------------------------------------------------------------

def assemble(state: str, status: str, z: np.ndarray, c: np.ndarray,
             p: np.ndarray | None) -> dict:
    """`O_s = (Z_s, P_s [Supported only], C_s, status)`.

    `P_s` is created only for a Supported family. A consumer that reaches for `O_s["P"]` on an
    Uncertain family gets a KeyError rather than a plausible-looking uncalibrated number, which
    is the failure mode this interface exists to make impossible.
    """
    o = {"state": state, "status": status, "Z": z, "C": c}
    if status == "Supported":
        if p is None:
            raise ValueError(f"{state} is marked Supported but no calibrated P was supplied")
        o["P"] = p
    return o


def wire_for_task(outputs: list[dict], matrix: dict, task: str,
                  validated: bool) -> tuple[list[np.ndarray], list[str]]:
    """Columns this task's risk head is allowed to consume, and their provenance.

    `validated=False` is the ablation arm that wires every family in regardless of its cell --
    the control that shows what the matrix is actually buying.
    """
    cols, why = [], []
    for o in outputs:
        cell = matrix.get(o["state"], {}).get(task, "Unidentifiable")
        if not validated:
            cols.append(o["Z"])
            why.append(f"{o['state']}:Z (unvalidated arm; cell={cell})")
            continue
        if cell != "Supported":
            continue                      # Redundant / Unsupported / Unidentifiable: excluded
        cols.append(o["Z"])
        why.append(f"{o['state']}:Z (cell=Supported)")
        if o["status"] != "Supported":
            # Uncertain family, Supported cell: Z may travel, but only with its uncertainty.
            cols.append(o["C"])
            why.append(f"{o['state']}:C (family status={o['status']}, Z never travels alone)")
    return cols, why


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

@torch.no_grad()
def entity_embeddings(model, bundles, entity: str) -> dict:
    """`{depth: X}` for an arbitrary node type, in bundle/node order. `depth_embeddings` in
    `ml/latent_state_head.py` is Supplier-only; this is the same computation for any entity,
    written here rather than by editing that module."""
    per = {d: [] for d in DEPTHS}
    for b in bundles:
        _logits, layers = model(b.data.x_dict, b.data.edge_index_dict)
        for d in DEPTHS:
            per[d].append(layers[d][entity].detach().cpu().numpy())
    return {d: np.concatenate(per[d], axis=0) for d in DEPTHS}


def task_matrix(model, bundles, task: str, zmaps: dict) -> dict:
    """Assemble every arm's feature block for one split, on ONE common row set.

    The row set is the task's labelled rows that also have a defined `Z` for every family in
    play -- identical across arms, because an arm scored on a different population is not
    comparable to the others.
    """
    ent = TASK_ENTITY_TYPE[task]
    emb = entity_embeddings(model, bundles, ent)
    Xraw, offs, off = [], [], 0
    for b in bundles:
        Xraw.append(b.data[ent].x.cpu().numpy())
        offs.append(off)
        off += b.data[ent].x.size(0)
    Xraw = np.concatenate(Xraw, axis=0)

    rows, Y, Zc = [], [], {s: [] for s in zmaps}
    for bi, b in enumerate(bundles):
        idx = b.labels[task][0].cpu().numpy()
        y = b.labels[task][1].cpu().numpy()
        t = b.t0.isoformat()
        sup_ids = b.data["Supplier"].node_id
        zsup = {s: np.array([zmaps[s].get((sid, t), np.nan) for sid in sup_ids])
                for s in zmaps}
        nbr = _neighbour_suppliers(b, task)
        for r, e in enumerate(idx):
            sups = nbr.get(int(e))
            if not sups:
                continue
            vals = {s: zsup[s][np.asarray(sups)] for s in zmaps}
            vals = {s: v[~np.isnan(v)] for s, v in vals.items()}
            if any(len(v) == 0 for v in vals.values()):
                continue
            rows.append(offs[bi] + int(e))
            Y.append(float(y[r]))
            for s in zmaps:
                Zc[s].append(float(vals[s].max() if PRIMARY_AGG == "max" else vals[s].mean()))
    if not rows:
        return {}
    rows = np.asarray(rows)
    return {"X": Xraw[rows], "H": {d: emb[d][rows] for d in DEPTHS},
            "Y": np.asarray(Y), "Z": {s: np.asarray(v) for s, v in Zc.items()},
            "n_labelled": sum(len(b.labels[task][0]) for b in bundles), "n_kept": len(rows)}


# ---------------------------------------------------------------------------
# one world
# ---------------------------------------------------------------------------

def run_seed(variant: str, dseed: int, config: str, tasks: list[str], families: dict,
             matrix: dict, statuses: dict, init_seeds: list[int], csv_root: str) -> dict:
    csv_dir = os.path.join(csv_root, f"v{variant}_seed{dseed}")
    model, meta = get_backbone(csv_dir, variant, dseed, mseed=0, device="cpu")
    assert_backbone_frozen(model)
    tr, va, te, _ = load_world(csv_dir, "cpu")
    t0s = [b.t0 for b in tr] + [b.t0 for b in va] + [b.t0 for b in te]
    targets = latent_targets(variant, dseed, config, t0s)

    prep = {}
    for state, depth in families.items():
        tgt = targets[state]
        Xtr, ytr = align(*depth_embeddings(model, tr)[depth], tgt)
        ytr_b, _, _ = binarise(ytr, ytr, tgt["kind"])
        prep[state] = (tgt, depth, Xtr, ytr_b)

    out = {"markov_frozen_production": {t: meta["auc"].get(t) for t in tasks}}
    for task in tasks:
        arms: dict = {}
        for s in init_seeds:
            zva = {st: supplier_z_map(model, va, prep[st][0], prep[st][1],
                                      prep[st][2], prep[st][3], s) for st in prep}
            zte = {st: supplier_z_map(model, te, prep[st][0], prep[st][1],
                                      prep[st][2], prep[st][3], s) for st in prep}
            Mva, Mte = task_matrix(model, va, task, zva), task_matrix(model, te, task, zte)
            if not Mva or not Mte or Mva["Y"].sum() == 0 or not 0 < Mte["Y"].sum() < len(Mte["Y"]):
                continue

            # Confidence per family, on the same rows: |p - 0.5| on the propagated Z. The
            # learned C_s head (Phase 3) does not beat this control, so the deployable
            # interface carries the control rather than a head that costs parameters and
            # buys nothing -- Phase 3's own measurement, applied.
            outs_va = [assemble(st, statuses.get(st, "Uncertain"),
                                Mva["Z"][st], np.abs(Mva["Z"][st] - 0.5), None) for st in prep]
            outs_te = [assemble(st, statuses.get(st, "Uncertain"),
                                Mte["Z"][st], np.abs(Mte["Z"][st] - 0.5), None) for st in prep]

            md = MARKOV_READOUT_DEPTH[task]
            blocks = {"tabular_no_share": (Mva["X"], Mte["X"])}
            for d in DEPTHS:
                blocks[f"depth_h{d}"] = (Mva["H"][d], Mte["H"][d])
            blocks["markov_readout"] = (Mva["H"][md], Mte["H"][md])
            for tag, validated in (("state_all_wired", False), ("state_matrix_wired", True)):
                cva, why = wire_for_task(outs_va, matrix, task, validated)
                cte, _ = wire_for_task(outs_te, matrix, task, validated)
                blocks[tag] = (
                    np.concatenate([Mva["H"][md]] + [c.reshape(-1, 1) for c in cva], axis=1),
                    np.concatenate([Mte["H"][md]] + [c.reshape(-1, 1) for c in cte], axis=1))
                arms.setdefault(f"_{tag}_columns", why)

            for tag, (Fva, Fte) in blocks.items():
                pr = fit_probs(Fva, Mva["Y"], [Fte], s, model=model)
                if pr is not None:
                    arms.setdefault(tag, []).append(roc_auc(pr[0], Mte["Y"].astype(bool)))
            arms.setdefault("_coverage", []).append(Mte["n_kept"] / Mte["n_labelled"])
        out[task] = arms
        base = arms.get("markov_readout", [])
        mw = arms.get("state_matrix_wired", [])
        if base and mw:
            print(f"  {task:<9} markov_readout {statistics.fmean(base):.4f}   "
                  f"matrix_wired {statistics.fmean(mw):.4f}   "
                  f"all_wired {statistics.fmean(arms.get('state_all_wired', [np.nan])):.4f}   "
                  f"production {out['markov_frozen_production'][task]:.4f}", flush=True)
    return out


def summarise(res: dict, tasks: list[str]) -> dict:
    """Pooled AUC per arm per task, with floors, and every credited comparison."""
    summary = {}
    dseeds = res["dataset_seeds"]
    for task in tasks:
        tags = sorted({t for d in dseeds for t in res["per_seed"][str(d)].get(task, {})
                       if not t.startswith("_")})
        entry = {}
        for tag in tags:
            per_seed, spreads = [], []
            for d in dseeds:
                v = res["per_seed"][str(d)].get(task, {}).get(tag) or []
                if v:
                    per_seed.append(statistics.fmean(v))
                    spreads.append(max(v) - min(v))
            if not per_seed:
                continue
            init_floor = max(spreads)
            dfloor = (max(per_seed) - min(per_seed)) if len(per_seed) > 1 else float("nan")
            entry[tag] = {"auc_mean": statistics.fmean(per_seed),
                          "auc_per_dataset_seed": per_seed,
                          "init_seed_floor": init_floor, "dataset_seed_floor": dfloor,
                          "reproduction_floor": max(f for f in (init_floor, dfloor) if f == f)}
        prod = [res["per_seed"][str(d)]["markov_frozen_production"][task] for d in dseeds
                if res["per_seed"][str(d)]["markov_frozen_production"].get(task) is not None]
        if prod:
            entry["markov_frozen_production"] = {
                "auc_mean": statistics.fmean(prod), "auc_per_dataset_seed": prod,
                "init_seed_floor": float("nan"),
                "dataset_seed_floor": max(prod) - min(prod),
                "reproduction_floor": max(prod) - min(prod),
                "note": "end-to-end trained head, different footing from the probe arms"}

        comps = {}
        for a, b in (("state_matrix_wired", "markov_readout"),
                     ("state_all_wired", "markov_readout"),
                     ("state_matrix_wired", "state_all_wired"),
                     ("state_matrix_wired", "markov_frozen_production"),
                     ("markov_readout", "tabular_no_share")):
            if a not in entry or b not in entry:
                continue
            pa, pb = entry[a]["auc_per_dataset_seed"], entry[b]["auc_per_dataset_seed"]
            if len(pa) != len(pb):
                continue
            per = [x - y for x, y in zip(pa, pb)]
            gain = statistics.fmean(per)
            floor = max(entry[a]["reproduction_floor"], entry[b]["reproduction_floor"])
            comps[f"{a}_vs_{b}"] = {
                "gain": gain, "per_dataset_seed": per, "floor": floor,
                "clears_floor": gain > floor,
                "sign_consistent": all(v > 0 for v in per) or all(v < 0 for v in per),
                "credited": bool(gain > floor and all(v > 0 for v in per))}
        entry["_comparisons"] = comps
        entry["_coverage"] = statistics.fmean(
            [v for d in dseeds for v in res["per_seed"][str(d)].get(task, {}).get("_coverage", [])]
            or [float("nan")])
        summary[task] = entry
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="A")
    ap.add_argument("--states", default="supply_stress")
    ap.add_argument("--tasks", default="delay,shortage,impact")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--init-seeds", default="0,1,2,3,4")
    ap.add_argument("--config", default="v1")
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--phase3", default=os.path.join(REPO, "out", "layer3_redesign",
                                                     "phase3_status.json"))
    ap.add_argument("--phase4", default=os.path.join(REPO, "out", "layer3_redesign",
                                                     "phase4_matrix.json"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dseeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    iseeds = [int(s) for s in a.init_seeds.split(",") if s.strip()]
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    states = [s.strip() for s in a.states.split(",") if s.strip()]

    p1 = json.load(open(os.path.join(REPO, "out", "phase1", f"heads_{a.variant}.json")))
    families = {s: int(p1["summary"][s]["best_depth"]) for s in states}

    st3 = json.load(open(a.phase3))
    statuses = {s: st3["arms"][f"{a.variant}:{s}"]["status"] for s in states
                if f"{a.variant}:{s}" in st3["arms"]}
    st4 = json.load(open(a.phase4))
    matrix: dict = {}
    for s in states:
        key = f"{a.variant}:{s}"
        matrix[s] = {}
        if key in st4.get("cited_impact", {}):
            from ml.state_task_sufficiency import classify_cited
            matrix[s]["impact"] = classify_cited(st4["cited_impact"][key])
        for task, cell in st4.get("summary", {}).get(key, {}).items():
            matrix[s][task] = cell["classification"]

    print(f"Phase 3 statuses: {statuses}")
    print(f"Phase 4 matrix:   {matrix}")
    n_supported = sum(1 for s in matrix for t in matrix[s] if matrix[s][t] == "Supported")
    print(f"Supported cells wired into the risk heads: {n_supported}")

    res = {"variant": a.variant, "dataset_seeds": dseeds, "init_seeds": iseeds,
           "tasks": tasks, "families": families, "statuses": statuses, "matrix": matrix,
           "per_seed": {}}
    for d in dseeds:
        print(f"\n=== variant {a.variant} seed {d} ===", flush=True)
        t = time.time()
        res["per_seed"][str(d)] = run_seed(a.variant, d, a.config, tasks, families,
                                           matrix, statuses, iseeds, a.csv_root)
        print(f"  ({time.time() - t:.0f}s)", flush=True)
    res["summary"] = summarise(res, tasks)

    print("\n" + "=" * 116)
    print(f"PHASE 5 — ablation set, variant {a.variant}, metric fixed in advance: test ROC-AUC")
    print("=" * 116)
    for task in tasks:
        e = res["summary"][task]
        print(f"\n--- {task}  (Markov readout depth h^{MARKOV_READOUT_DEPTH[task]}, "
              f"Z coverage {e['_coverage']:.1%}) ---")
        print(f"{'arm':<28}{'AUC':>9}{'init fl':>9}{'dset fl':>9}{'floor':>9}")
        for tag in ("tabular_no_share", *[f"depth_h{d}" for d in DEPTHS], "markov_readout",
                    "state_all_wired", "state_matrix_wired", "markov_frozen_production"):
            c = e.get(tag)
            if not c:
                continue
            print(f"{tag:<28}{c['auc_mean']:>9.4f}{c['init_seed_floor']:>9.4f}"
                  f"{c['dataset_seed_floor']:>9.4f}{c['reproduction_floor']:>9.4f}")
        print(f"{'comparison':<44}{'gain':>9}{'floor':>9}{'sign':>6}{'CREDITED':>10}")
        for k, c in e["_comparisons"].items():
            print(f"{k:<44}{c['gain']:>+9.4f}{c['floor']:>9.4f}"
                  f"{('yes' if c['sign_consistent'] else 'NO'):>6}"
                  f"{('YES' if c['credited'] else 'no'):>10}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
