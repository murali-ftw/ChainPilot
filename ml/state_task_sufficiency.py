#!/usr/bin/env python3
"""
Layer 3 redesign, PHASE 4 -- the state-to-task sufficiency matrix.

For every (family, task) pair among the Supported and Uncertain families x {delay, shortage,
impact}, fit and compare

    P(Y_t | X)      the task's own observable features alone
    P(Y_t | Z_s)    the estimated latent state alone
    P(Y_t | X, Z_s) both

and compute the marginal contribution `delta_{s,t} = P(Y|X,Z) - P(Y|X)` against a reproduction
floor measured for that specific task.

**This grid does not start from zero.** `reports/layer3_uncertainty_aware.md` STEP 1 already
measured every family x `impact` cell, and those numbers are entered here as citations rather
than re-fitted -- marked `cited: true` in the output and in the report. Only the delay and
shortage columns require a fresh fit.

---

**Propagating a per-supplier state to a non-Supplier task.** `impact` is the only benchmark task
whose label sits on the Supplier entity, which is exactly why STEP 1 measured that column and no
other. `delay` is labelled on Shipment and `shortage` on Product, so a per-supplier `Z_s` has to
be carried along the graph's own edges to reach them. The mapping is the generator's own
sourcing path, not a learned projection:

    delay      Shipment --SHIPS_FROM--> Supplier                       (>=1 supplier per shipment)
    shortage   Product <--USED_IN-- Component <--SUPPLIES-- Supplier   (many suppliers per product)

**The aggregator is pre-registered as `max`, before any cell was fitted**, and the reason is the
generator's: `sourcing_stress()` (`db/generate_dataset.py`) reduces a product's component sources
with `max` on its AND branch -- worst-case sourcing is the generator's own semantics for how
supplier state reaches a product. Choosing the aggregator after seeing which one scored better
would be the goalpost-move this project's reporting standard exists to prevent. `mean` is
computed alongside and reported as a secondary, never gated on.

**Rows without a defined `Z`.** A shipment dispatched from a factory rather than a supplier has
no upstream supplier and therefore no `Z`. All three arms are scored on the SAME row set -- the
rows where `Z` is defined -- because scoring `P(Y|X)` on a larger population than `P(Y|X,Z)`
would make the delta a comparison between two different tasks. Coverage is reported per cell.

**Classification**, exactly per the architecture's rule:

    Supported      delta positive, reproducible, clears the task floor
    Redundant      Z alone predicts Y, but delta does not clear the floor
    Unsupported    neither shows signal
    Unidentifiable the family never reached this phase

    python3 ml/state_task_sufficiency.py --arms A:supply_stress,E:supply_stress,E:recovery_capability \
        --tasks delay,shortage --out out/layer3_redesign/phase4_matrix.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

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

ARMS = ("Z", "X", "XZ")
AGGREGATORS = ("max", "mean")
PRIMARY_AGG = "max"          # pre-registered; see module docstring

# `reports/layer3_uncertainty_aware.md` STEP 1, verbatim. Entered as citations, never re-fitted.
CITED_IMPACT = {
    "A:supply_stress": {
        "Z": 0.7075, "X": 0.7832, "XZ": 0.7914,
        "Z_floor": 0.1804, "XZ_floor": 0.0485, "delta": 0.0082, "delta_floor": 0.0485,
        "source": "layer3_uncertainty_aware.md STEP 1 (Variant A)"},
    "E:supply_stress": {
        "Z": 0.7669, "X": 0.8124, "XZ": 0.8209,
        "Z_floor": 0.1179, "XZ_floor": 0.0676, "delta": 0.0084, "delta_floor": 0.0676,
        "source": "layer3_uncertainty_aware.md STEP 1 (Variant E)"},
    "E:recovery_capability": {
        "Z": 0.5711, "X": 0.8124, "XZ": 0.8090,
        "Z_floor": 0.1391, "XZ_floor": 0.0615, "delta": -0.0034, "delta_floor": 0.0615,
        "source": "layer3_uncertainty_aware.md STEP 1 (Variant E)"},
}


# ---------------------------------------------------------------------------
# propagating Z from Supplier to the task entity
# ---------------------------------------------------------------------------

def supplier_z_map(model, bundles, tgt, depth, Xtr, ytr_b, init_seed) -> dict:
    """`{(sup_id, t0_iso): z}` from one Model A head fit, for every aligned supplier row."""
    H, ids, times = depth_embeddings(model, bundles)[depth]
    keep = [i for i, (sid, t) in enumerate(zip(ids, times))
            if (tgt["values"].get(t) is not None and sid in tgt["values"][t])]
    if not keep:
        return {}
    got = fit_probs(Xtr, ytr_b, [H[np.asarray(keep)]], init_seed, model=model)
    if got is None:
        return {}
    z = got[0]
    return {(ids[i], times[i]): float(z[j]) for j, i in enumerate(keep)}


def _neighbour_suppliers(b, task: str) -> dict:
    """`{task_entity_index: [supplier_index, ...]}` along the generator's sourcing path."""
    data = b.data
    if task == "impact":
        n = data["Supplier"].x.size(0)
        return {i: [i] for i in range(n)}

    if task == "delay":
        ei = data[("Shipment", "SHIPS_FROM", "Supplier")].edge_index.cpu().numpy()
        out: dict = {}
        for s, v in zip(ei[0], ei[1]):
            out.setdefault(int(s), []).append(int(v))
        return out

    # shortage: Supplier -SUPPLIES-> Component -USED_IN-> Product
    sc = data[("Supplier", "SUPPLIES", "Component")].edge_index.cpu().numpy()
    cp = data[("Component", "USED_IN", "Product")].edge_index.cpu().numpy()
    comp_sup: dict = {}
    for sup, comp in zip(sc[0], sc[1]):
        comp_sup.setdefault(int(comp), []).append(int(sup))
    out = {}
    for comp, prod in zip(cp[0], cp[1]):
        sups = comp_sup.get(int(comp))
        if sups:
            out.setdefault(int(prod), []).extend(sups)
    return out


def task_rows(model, bundles, task: str, zmap: dict) -> dict:
    """`{agg: (X, Z, Y)}` for one split -- the task's labelled rows, restricted to those with a
    defined Z, with the raw task-entity features and the aggregated supplier state."""
    ent = TASK_ENTITY_TYPE[task]
    Xs, Zs, Ys = {a: [] for a in AGGREGATORS}, {a: [] for a in AGGREGATORS}, []
    n_lab = n_kept = 0
    for b in bundles:
        idx, y = b.labels[task]
        idx = idx.cpu().numpy()
        y = y.cpu().numpy()
        n_lab += len(idx)
        t = b.t0.isoformat()
        sup_ids = b.data["Supplier"].node_id
        zsup = np.array([zmap.get((s, t), np.nan) for s in sup_ids])
        nbr = _neighbour_suppliers(b, task)
        Xent = b.data[ent].x.cpu().numpy()

        keep, zvals = [], {a: [] for a in AGGREGATORS}
        for r, (e, yy) in enumerate(zip(idx, y)):
            sups = nbr.get(int(e))
            if not sups:
                continue
            vals = zsup[np.asarray(sups)]
            vals = vals[~np.isnan(vals)]
            if not len(vals):
                continue
            keep.append(r)
            zvals["max"].append(float(vals.max()))
            zvals["mean"].append(float(vals.mean()))
        if not keep:
            continue
        n_kept += len(keep)
        kr = np.asarray(keep)
        for a in AGGREGATORS:
            Xs[a].append(Xent[idx[kr]])
            Zs[a].append(np.asarray(zvals[a]))
        Ys.append(y[kr])
    if not Ys:
        return {}
    Y = np.concatenate(Ys)
    return {a: (np.concatenate(Xs[a]), np.concatenate(Zs[a]), Y) for a in AGGREGATORS} | {
        "_coverage": {"n_labelled": n_lab, "n_with_z": n_kept,
                      "coverage": n_kept / n_lab if n_lab else 0.0}}


def stack(z, X, arm):
    if arm == "Z":
        return z.reshape(-1, 1)
    if arm == "X":
        return X
    return np.concatenate([X, z.reshape(-1, 1)], axis=1)


# ---------------------------------------------------------------------------
# one (variant, state, seed) -> every requested task
# ---------------------------------------------------------------------------

def run_seed(variant: str, state: str, dseed: int, config: str, tasks: list[str],
             init_seeds: list[int], csv_root: str, depth: int) -> dict:
    csv_dir = os.path.join(csv_root, f"v{variant}_seed{dseed}")
    model, _ = get_backbone(csv_dir, variant, dseed, mseed=0, device="cpu")
    assert_backbone_frozen(model)
    tr, va, te, _ = load_world(csv_dir, "cpu")
    tgt = latent_targets(variant, dseed, config,
                         [b.t0 for b in tr] + [b.t0 for b in va] + [b.t0 for b in te])[state]

    # The Z head is fitted ONCE per init seed on `tr` and reused across every task -- the state
    # does not depend on which downstream label it is later asked about.
    Xtr, ytr = align(*depth_embeddings(model, tr)[depth], tgt)
    ytr_b, _, _ = binarise(ytr, ytr, tgt["kind"])

    out: dict = {}
    for task in tasks:
        per_arm = {a: {arm: [] for arm in ARMS} for a in AGGREGATORS}
        cov = None
        for s in init_seeds:
            zmap_va = supplier_z_map(model, va, tgt, depth, Xtr, ytr_b, s)
            zmap_te = supplier_z_map(model, te, tgt, depth, Xtr, ytr_b, s)
            rv, rt = task_rows(model, va, task, zmap_va), task_rows(model, te, task, zmap_te)
            if not rv or not rt:
                continue
            cov = {"va": rv["_coverage"], "te": rt["_coverage"]}
            for a in AGGREGATORS:
                Xv, Zv, Yv = rv[a]
                Xt, Zt, Yt = rt[a]
                if Yv.sum() == 0 or Yt.sum() == 0 or Yt.sum() == len(Yt):
                    continue
                for arm in ARMS:
                    pr = fit_probs(stack(Zv, Xv, arm), Yv,
                                   [stack(Zt, Xt, arm)], s, model=model)
                    per_arm[a][arm].append(
                        roc_auc(pr[0], Yt.astype(bool)) if pr is not None else None)
        cell = {"coverage": cov}
        for a in AGGREGATORS:
            e = {}
            for arm in ARMS:
                v = [x for x in per_arm[a][arm] if x is not None]
                e[arm] = {"auc_mean": statistics.fmean(v) if v else None, "auc_all": v,
                          "init_seed_spread": (max(v) - min(v)) if len(v) > 1 else None}
            cell[a] = e
        out[task] = cell
        pa = cell[PRIMARY_AGG]
        if pa["X"]["auc_mean"] is not None:
            print(f"  {task:<9} cov {cov['te']['coverage']:.1%}  "
                  f"P(Y|Z) {pa['Z']['auc_mean']:.4f}  P(Y|X) {pa['X']['auc_mean']:.4f}  "
                  f"P(Y|X,Z) {pa['XZ']['auc_mean']:.4f}  "
                  f"delta {pa['XZ']['auc_mean'] - pa['X']['auc_mean']:+.4f}", flush=True)
    return out


def summarise(res: dict, tasks: list[str]) -> dict:
    """Per (arm, task): pooled AUCs, the task-specific reproduction floor, and the cell class."""
    summary = {}
    for key, per_seed in res["per_arm_seed"].items():
        summary[key] = {}
        for task in tasks:
            cells = [per_seed[str(d)][task] for d in res["dataset_seeds"]
                     if task in per_seed.get(str(d), {})]
            if not cells:
                continue
            entry = {"n_dataset_seeds": len(cells),
                     "coverage_te": statistics.fmean(
                         [c["coverage"]["te"]["coverage"] for c in cells if c.get("coverage")])}
            for a in AGGREGATORS:
                agg = {}
                for arm in ARMS:
                    vals = [c[a][arm]["auc_mean"] for c in cells
                            if c[a][arm]["auc_mean"] is not None]
                    spr = [c[a][arm]["init_seed_spread"] for c in cells
                           if c[a][arm]["init_seed_spread"] is not None]
                    if not vals:
                        continue
                    init_floor = max(spr) if spr else float("nan")
                    dfloor = (max(vals) - min(vals)) if len(vals) > 1 else float("nan")
                    floor = max(f for f in (init_floor, dfloor) if f == f)
                    mean = statistics.fmean(vals)
                    agg[arm] = {"auc_mean": mean, "auc_per_dataset_seed": vals,
                                "init_seed_floor": init_floor, "dataset_seed_floor": dfloor,
                                "reproduction_floor": floor,
                                "above_chance_by": mean - 0.5,
                                "clears_floor": (mean - 0.5) > floor,
                                "sign_consistent": (all(v > 0.5 for v in vals)
                                                    or all(v < 0.5 for v in vals))}
                if "X" in agg and "XZ" in agg:
                    # The delta's OWN floor, measured on the paired per-seed deltas rather than
                    # inherited from either arm -- the quantity being gated is the difference.
                    d_per_seed = [xz - x for xz, x in zip(agg["XZ"]["auc_per_dataset_seed"],
                                                          agg["X"]["auc_per_dataset_seed"])]
                    delta = statistics.fmean(d_per_seed)
                    d_floor = max(agg["XZ"]["reproduction_floor"],
                                  agg["X"]["reproduction_floor"])
                    agg["delta"] = {
                        "value": delta, "per_dataset_seed": d_per_seed,
                        "floor": d_floor, "clears_floor": delta > d_floor,
                        "sign_consistent": (all(v > 0 for v in d_per_seed)
                                            or all(v < 0 for v in d_per_seed))}
                entry[a] = agg
            entry["classification"] = classify(entry.get(PRIMARY_AGG, {}))
            summary[key][task] = entry
    return summary


def classify(agg: dict) -> str:
    """The architecture's rule, applied literally and with nothing else consulted."""
    if not agg or "delta" not in agg or "Z" not in agg:
        return "Unsupported"
    d = agg["delta"]
    if d["clears_floor"] and d["value"] > 0 and d["sign_consistent"]:
        return "Supported"
    z_predicts = bool(agg["Z"]["clears_floor"] and agg["Z"]["sign_consistent"])
    return "Redundant" if z_predicts else "Unsupported"


def classify_cited(c: dict) -> str:
    z_predicts = (c["Z"] - 0.5) > c["Z_floor"]
    if c["delta"] > c["delta_floor"]:
        return "Supported"
    return "Redundant" if z_predicts else "Unsupported"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="A:supply_stress,E:supply_stress,E:recovery_capability")
    ap.add_argument("--tasks", default="delay,shortage")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--init-seeds", default="0,1,2,3,4")
    ap.add_argument("--config", default="v1")
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dseeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    iseeds = [int(s) for s in a.init_seeds.split(",") if s.strip()]
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    arms = [tuple(x.split(":")) for x in a.arms.split(",") if x.strip()]

    res = {"dataset_seeds": dseeds, "init_seeds": iseeds, "tasks": tasks,
           "primary_aggregator": PRIMARY_AGG, "per_arm_seed": {},
           "cited_impact": CITED_IMPACT}
    for variant, state in arms:
        key = f"{variant}:{state}"
        depth = int(json.load(open(os.path.join(
            REPO, "out", "phase1", f"heads_{variant}.json")))["summary"][state]["best_depth"])
        res["per_arm_seed"][key] = {}
        for d in dseeds:
            print(f"\n=== {key} seed {d} (Z from h^{depth}) ===", flush=True)
            t = time.time()
            res["per_arm_seed"][key][str(d)] = run_seed(
                variant, state, d, a.config, tasks, iseeds, a.csv_root, depth)
            print(f"  ({time.time() - t:.0f}s)", flush=True)

    res["summary"] = summarise(res, tasks)

    print("\n" + "=" * 122)
    print("PHASE 4 — state-to-task sufficiency matrix "
          f"(aggregator: {PRIMARY_AGG}, pre-registered)")
    print("=" * 122)
    hdr = (f"{'arm':<26}{'task':<10}{'cov':>7}{'P(Y|Z)':>9}{'P(Y|X)':>9}{'P(Y|X,Z)':>10}"
           f"{'delta':>9}{'floor':>9}{'sign':>6}{'CLASS':>14}{'cited':>7}")
    print(hdr); print("-" * len(hdr))
    for key in res["per_arm_seed"]:
        c = CITED_IMPACT.get(key)
        if c:
            print(f"{key:<26}{'impact':<10}{'--':>7}{c['Z']:>9.4f}{c['X']:>9.4f}"
                  f"{c['XZ']:>10.4f}{c['delta']:>+9.4f}{c['delta_floor']:>9.4f}{'--':>6}"
                  f"{classify_cited(c):>14}{'YES':>7}")
        for task in tasks:
            e = res["summary"][key].get(task)
            if not e:
                continue
            g = e[PRIMARY_AGG]
            print(f"{key:<26}{task:<10}{e['coverage_te']:>6.1%} {g['Z']['auc_mean']:>9.4f}"
                  f"{g['X']['auc_mean']:>9.4f}{g['XZ']['auc_mean']:>10.4f}"
                  f"{g['delta']['value']:>+9.4f}{g['delta']['floor']:>9.4f}"
                  f"{('yes' if g['delta']['sign_consistent'] else 'NO'):>6}"
                  f"{e['classification']:>14}{'no':>7}")
    print("-" * len(hdr))

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
