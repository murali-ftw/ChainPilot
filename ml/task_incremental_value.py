#!/usr/bin/env python3
"""
Layer 3 v2, PHASE 2 -- residual-information / incremental-value evaluation, both seed axes.

For each task with a trained encoder:

    baseline: P(Y_t | X, H)        vs        layer3: P(Y_t | X, H, Z_t)
    delta_t = Performance(layer3) - Performance(baseline)

This is the operational test for `I(Y_t; Z_t | X, H) > 0`. Mutual information is **not**
estimated directly -- a downstream performance comparison is far less noisy than a direct MI
estimate in this data regime.

**Read `ml/task_latent_encoder.py`'s docstring before reading any number here.** `Z_t = f_t(X,H)`
is a deterministic function of the conditioning set, so `I(Y;Z|X,H) = 0` identically in the
population limit. What `delta_t` can therefore show is a representation/optimisation effect at
finite sample -- whether an explicit compressed feature is easier for a small head to use than
the same information implicit in `(X,H)` -- never the presence of information the Layer-2
pathway lacks.

**Four arms, and two of them are controls without which `delta_t` is uninterpretable:**

    baseline   [X ; H^k]                      the bar
    layer3     [X ; H^k ; Z_t]                the arm
    noise      [X ; H^k ; N(0,1)^{Z_DIM}]     WIDTH-MATCHED CONTROL -- identical input width and
                                              first-layer parameter count as `layer3`, carrying
                                              no information at all. Any `delta_t` that the noise
                                              arm also produces is capacity, not `Z_t`.
    z_only     [Z_t]                          diagnostic: did the encoder learn anything at all

**Both seed axes, and the pass requires both independently.** The grid is 5 dataset seeds x 5
init seeds. `floor_t = max(init_seed_floor_t, dataset_seed_floor_t)`, and a task passes only if
`delta_t` is positive, clears the init-seed floor, clears the dataset-seed floor, and is
sign-consistent across all five dataset seeds -- not merely positive on the averaged number,
which can hide an axis that fails on its own.

**The second grid axis is the head init seed, not the backbone seed**, carried verbatim from
`reports/layer3_uncertainty_aware.md` §3 and forced by the same constraint: only `m0` backbone
checkpoints exist in `out/ds_ckpt/`, and building `m1..m4` would mean training SHARE, which the
ground rules forbid. `ml/uncertainty_ensemble.py::metric_decomposition` is imported **unmodified**
and applied to this grid, so the two-way variance decomposition is computed by the same estimator
that produced the project's existing 68-99% figures.

**Expected result, stated up front so it is not mistaken for a bug.** The closest existing
analogue -- Supply Stress's and Recovery Capability's marginal contribution to `impact` --
measured +0.0082 and +0.0084 against floors of 0.0485 and 0.0676, both well inside. A
below-floor result here is a real, useful, negative finding, not evidence the pipeline is broken.

    python3 ml/task_incremental_value.py --variant A --tasks delay,shortage,impact \
        --seeds 42,43,44,45,46 --init-seeds 0,1,2,3,4 --out out/layer3_v2/phase2_A.json
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

from ml.ds_backbone import get_backbone, load_world                  # noqa: E402
from ml.hypothesis_ranker import roc_auc                             # noqa: E402
from ml.latent_state_head import assert_backbone_frozen              # noqa: E402
from ml.layer3_sufficiency import fit_probs                          # noqa: E402
from ml.models.depth import TASK_ENTITY_TYPE                         # noqa: E402
from ml.task_latent_encoder import TASK_DEPTH, Z_DIM, train_encoder  # noqa: E402
from ml.uncertainty_ensemble import metric_decomposition             # noqa: E402

ARMS = ("baseline", "layer3", "noise", "z_only")


@torch.no_grad()
def task_rows(model, bundles, task: str) -> dict:
    """`{"X":…, "H":…, "Y":…, "ent":…}` -- the task's labelled rows on one split.

    `X` is the entity's own raw input feature vector, `H` its frozen SHARE representation at
    `TASK_DEPTH[task]`. Label rows are taken exactly as `ml/evaluate.py` reads them, so a row
    here is a row there. For `shortage` several (product, warehouse) label rows share one
    Product embedding -- the documented coarsening in `ml/models/heads.py`, reproduced rather
    than silently deduplicated.
    """
    ent = TASK_ENTITY_TYPE[task]
    depth = TASK_DEPTH[task]
    X, H, Y, E = [], [], [], []
    for b in bundles:
        _logits, layers = model(b.data.x_dict, b.data.edge_index_dict)
        h = layers[depth][ent].detach().cpu().numpy()
        x = b.data[ent].x.cpu().numpy()
        idx = b.labels[task][0].cpu().numpy()
        y = b.labels[task][1].cpu().numpy()
        node_ids = b.data[ent].node_id
        X.append(x[idx]); H.append(h[idx]); Y.append(y)
        E.extend([node_ids[int(i)] for i in idx])
    if not Y:
        return {}
    return {"X": np.concatenate(X), "H": np.concatenate(H), "Y": np.concatenate(Y), "ent": E}


def stack(arm: str, X, H, Z, rng):
    if arm == "baseline":
        return np.concatenate([X, H], axis=1)
    if arm == "layer3":
        return np.concatenate([X, H, Z], axis=1)
    if arm == "noise":
        return np.concatenate([X, H, rng.randn(len(X), Z.shape[1])], axis=1)
    return Z


def run_seed(variant: str, dseed: int, tasks: list[str], init_seeds: list[int],
             csv_root: str) -> dict:
    csv_dir = os.path.join(csv_root, f"v{variant}_seed{dseed}")
    model, _meta = get_backbone(csv_dir, variant, dseed, mseed=0, device="cpu")
    assert_backbone_frozen(model)
    tr, va, te, _ = load_world(csv_dir, "cpu")

    out = {}
    for task in tasks:
        R = {k: task_rows(model, b, task) for k, b in (("tr", tr), ("va", va), ("te", te))}
        if not all(R.values()) or R["te"]["Y"].sum() in (0, len(R["te"]["Y"])):
            print(f"  {task}: no usable rows", flush=True)
            continue
        Ftr = np.concatenate([R["tr"]["X"], R["tr"]["H"]], axis=1)
        Fva = np.concatenate([R["va"]["X"], R["va"]["H"]], axis=1)
        Fte = np.concatenate([R["te"]["X"], R["te"]["H"]], axis=1)

        per_arm = {a: [] for a in ARMS}
        for s in init_seeds:
            # f_t is fitted on `tr`; Z on va/te is therefore out of sample for it, and the
            # downstream head is fitted on `va` and scored on `te`, so no arm is scored on its
            # own training data at either level.
            Zs = train_encoder(Ftr, R["tr"]["Y"], [Fva, Fte], s, model=model)
            if Zs is None:
                continue
            Zva, Zte = Zs
            rng = np.random.RandomState(1000 + s)
            for arm in ARMS:
                A = stack(arm, R["va"]["X"], R["va"]["H"], Zva, rng)
                B = stack(arm, R["te"]["X"], R["te"]["H"], Zte,
                          np.random.RandomState(1000 + s))
                pr = fit_probs(A, R["va"]["Y"], [B], s, model=model)
                per_arm[arm].append(roc_auc(pr[0], R["te"]["Y"].astype(bool))
                                    if pr is not None else None)

        cell = {"n_tr": int(len(R["tr"]["Y"])), "n_va": int(len(R["va"]["Y"])),
                "n_te": int(len(R["te"]["Y"])), "pos_te": int(R["te"]["Y"].sum()),
                "depth": TASK_DEPTH[task], "x_dim": int(R["te"]["X"].shape[1]),
                "h_dim": int(R["te"]["H"].shape[1]), "z_dim": Z_DIM}
        for arm in ARMS:
            v = [x for x in per_arm[arm] if x is not None]
            cell[arm] = {"auc_mean": statistics.fmean(v) if v else None, "auc_all": v,
                         "init_seed_spread": (max(v) - min(v)) if len(v) > 1 else None}
        if cell["baseline"]["auc_mean"] is not None and cell["layer3"]["auc_mean"] is not None:
            cell["delta"] = cell["layer3"]["auc_mean"] - cell["baseline"]["auc_mean"]
            cell["delta_noise"] = cell["noise"]["auc_mean"] - cell["baseline"]["auc_mean"]
            # Paired per-init-seed deltas: the same encoder init and the same head init on both
            # sides, so this is a matched comparison rather than two independent means.
            cell["delta_paired"] = [l - b for l, b in
                                    zip(cell["layer3"]["auc_all"], cell["baseline"]["auc_all"])]
            cell["delta_noise_paired"] = [n - b for n, b in
                                          zip(cell["noise"]["auc_all"],
                                              cell["baseline"]["auc_all"])]
            print(f"  {task:<9} h^{cell['depth']}  base {cell['baseline']['auc_mean']:.4f}  "
                  f"L3 {cell['layer3']['auc_mean']:.4f}  noise {cell['noise']['auc_mean']:.4f}  "
                  f"z_only {cell['z_only']['auc_mean']:.4f}  "
                  f"delta {cell['delta']:+.4f} (noise {cell['delta_noise']:+.4f})", flush=True)
        out[task] = cell
    return out


def summarise(res: dict) -> dict:
    summary = {}
    dseeds = res["dataset_seeds"]
    for task in res["tasks"]:
        cells = [res["per_seed"][str(d)][task] for d in dseeds
                 if task in res["per_seed"].get(str(d), {})]
        if not cells:
            continue
        entry = {"n_dataset_seeds": len(cells), "depth": cells[0]["depth"],
                 "n_te_total": sum(c["n_te"] for c in cells),
                 "pos_te_total": sum(c["pos_te"] for c in cells)}
        for arm in ARMS:
            vals = [c[arm]["auc_mean"] for c in cells if c[arm]["auc_mean"] is not None]
            spr = [c[arm]["init_seed_spread"] for c in cells
                   if c[arm]["init_seed_spread"] is not None]
            if not vals:
                continue
            entry[arm] = {"auc_mean": statistics.fmean(vals), "auc_per_dataset_seed": vals,
                          "init_seed_floor": max(spr) if spr else float("nan"),
                          "dataset_seed_floor": ((max(vals) - min(vals)) if len(vals) > 1
                                                 else float("nan"))}
        for tag, key in (("delta", "delta_paired"), ("delta_noise", "delta_noise_paired")):
            per_seed = [statistics.fmean(c[key]) for c in cells if c.get(key)]
            if not per_seed:
                continue
            # init-seed floor for the DELTA: how far the paired delta moves across init seeds
            # within a world, maximised over worlds. dataset-seed floor: how far the per-world
            # mean delta moves across worlds.
            init_floor = max(max(c[key]) - min(c[key]) for c in cells if c.get(key))
            dset_floor = (max(per_seed) - min(per_seed)) if len(per_seed) > 1 else float("nan")
            mean = statistics.fmean(per_seed)
            entry[tag] = {
                "value": mean, "per_dataset_seed": per_seed,
                "init_seed_floor": init_floor, "dataset_seed_floor": dset_floor,
                "floor": max(f for f in (init_floor, dset_floor) if f == f),
                "clears_init_axis": mean > init_floor,
                "clears_dataset_axis": mean > dset_floor,
                "sign_consistent": all(v > 0 for v in per_seed) or all(v < 0 for v in per_seed),
            }
        d = entry.get("delta")
        if d:
            # THE GATE: positive, clears BOTH axes independently, sign-consistent 5/5.
            entry["gate_pass"] = bool(d["value"] > 0 and d["clears_init_axis"]
                                      and d["clears_dataset_axis"]
                                      and all(v > 0 for v in d["per_dataset_seed"]))
            n = entry.get("delta_noise")
            if n:
                entry["delta_minus_noise"] = d["value"] - n["value"]
                entry["beats_width_control"] = bool(
                    (d["value"] - n["value"]) > max(d["floor"], n["floor"]))
        summary[task] = entry

    # Two-way variance decomposition, by ml/uncertainty_ensemble.py's own estimator, unmodified.
    aucs = {}
    for d in dseeds:
        for mi, _ in enumerate(res["init_seeds"]):
            cell = {}
            for task in res["tasks"]:
                c = res["per_seed"].get(str(d), {}).get(task)
                if c and c.get("layer3", {}).get("auc_all"):
                    v = c["layer3"]["auc_all"]
                    cell[task] = v[mi] if mi < len(v) else float("nan")
            aucs[(d, mi)] = cell
    try:
        summary["_variance_decomposition_layer3"] = metric_decomposition(
            aucs, dseeds, list(range(len(res["init_seeds"]))))
    except Exception as exc:                      # noqa: BLE001
        summary["_variance_decomposition_layer3"] = {"unavailable": str(exc)}
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="A")
    ap.add_argument("--tasks", default="delay,shortage,impact")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--init-seeds", default="0,1,2,3,4")
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dseeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    iseeds = [int(s) for s in a.init_seeds.split(",") if s.strip()]
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]

    res = {"variant": a.variant, "dataset_seeds": dseeds, "init_seeds": iseeds,
           "tasks": tasks, "task_depth": {t: TASK_DEPTH[t] for t in tasks},
           "z_dim": Z_DIM, "per_seed": {}}
    for d in dseeds:
        print(f"\n=== variant {a.variant} seed {d} ===", flush=True)
        t = time.time()
        res["per_seed"][str(d)] = run_seed(a.variant, d, tasks, iseeds, a.csv_root)
        print(f"  ({time.time() - t:.0f}s)", flush=True)
    res["summary"] = summarise(res)

    print("\n" + "=" * 128)
    print(f"PHASE 2 — incremental value of Z_t over (X, H), variant {a.variant}")
    print("=" * 128)
    hdr = (f"{'task':<10}{'depth':>6}{'base':>9}{'layer3':>9}{'noise':>9}{'z_only':>9}"
           f"{'delta':>9}{'init fl':>9}{'dset fl':>9}{'sign':>6}{'vs noise':>10}{'GATE':>7}")
    print(hdr); print("-" * len(hdr))
    for task in tasks:
        s = res["summary"].get(task)
        if not s or "delta" not in s:
            continue
        d = s["delta"]
        print(f"{task:<10}{'h^'+str(s['depth']):>6}{s['baseline']['auc_mean']:>9.4f}"
              f"{s['layer3']['auc_mean']:>9.4f}{s['noise']['auc_mean']:>9.4f}"
              f"{s['z_only']['auc_mean']:>9.4f}{d['value']:>+9.4f}"
              f"{d['init_seed_floor']:>9.4f}{d['dataset_seed_floor']:>9.4f}"
              f"{('yes' if d['sign_consistent'] else 'NO'):>6}"
              f"{s.get('delta_minus_noise', float('nan')):>+10.4f}"
              f"{('PASS' if s['gate_pass'] else 'STOP'):>7}")
    print("-" * len(hdr))
    dec = res["summary"].get("_variance_decomposition_layer3", {})
    if dec and "unavailable" not in dec:
        print(f"\n{'task':<10}{'std_init':>10}{'std_dataset':>13}{'ratio':>8}{'dataset share':>15}"
              "   (ml/uncertainty_ensemble.py::metric_decomposition, unmodified)")
        for task in tasks:
            c = dec.get(task)
            if not c:
                continue
            r = c["std_ratio_dataset_over_model"]
            share = c["dataset_share"]
            print(f"{task:<10}{c['std_model']:>10.4f}{c['std_dataset']:>13.4f}"
                  f"{(f'{r:.1f}x' if r else 'n/a'):>8}"
                  f"{(f'{share:.1%}' if share is not None else 'n/a'):>15}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
