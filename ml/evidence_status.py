#!/usr/bin/env python3
"""
Layer 3 redesign, PHASE 3 -- split the state estimate from its confidence, and assign each
family a three-way epistemic status.

Two things happen here, and only the second is a translation layer.

**(1) The calibration/confidence ladder is completed for Recovery Capability.** The original
pipeline (`reports/layer3_uncertainty_aware.md`) ran Steps 3-5 for Supply Stress only: Recovery
Capability was excluded at STEP 1 by that ladder's ordering, on a downstream-sufficiency test
against `impact`, before its calibration was ever measured. The redesign decouples those --
sufficiency is Phase 4's question, not a gate on Phase 3 -- so Recovery Capability's cross-world
calibration and per-entity confidence are measured here for the first time. `step3`,
`step3_diagnostics` and `step4` are imported from `ml/layer3_uncertainty.py` **unmodified**, so
Supply Stress's arms reproduce the published figures and Recovery Capability's are produced by
the identical instrument.

**(2) The confidence output is built as a SEPARATE HEAD, which is new.** The original Step 4
signal was `-var` across head init seeds, and it was measured and rejected: the most-confident
bin was the *least* discriminative on both variants. That closes the ensemble-variance route,
not the learned-head route, and the redesign asks specifically for

    C_s = g_s(H, X)

-- a head that predicts, from the same frozen representation and observables the state head
reads, whether the state head is right about this entity. It is fit on `va` (where the state
head's predictions are out of sample for it) and scored on `te`, the same three-way discipline
`ml/layer3_sufficiency.py` uses, because fitting it on `tr` would feed it in-sample correctness
whose error distribution is not the one it meets at test time.

**Two controls, without which a confidence AUC means nothing.**

    margin   -- |p - 0.5|, the state head's own decision margin. Free, needs no head, and is
                what any learned confidence signal must beat to have earned its parameters.
    ens_var  -- `-var` across the five head init seeds, i.e. exactly Step 4's rejected signal,
                recomputed here so the new head is compared against the old one on one table.

**Status assignment**, per the redesign brief:

    Supported      -- the calibration transfer floor is cleared CROSS-WORLD, AND the confidence
                      signal is monotone (least-confident-first bins show worse accuracy than
                      most-confident-first bins), per world
    Uncertain      -- some signal exists (identifiability passed) but calibration or confidence
                      does not clear its floor cross-world
    Unidentifiable -- the family never reached this phase (closed at Phase 0 or Phase 1)

Nothing is fitted within-world and evaluated within-world anywhere below. That is leakage, it
produced a false-positive-looking result once already in this project's history, and Step 3's
own protocol forbids it.

    python3 ml/evidence_status.py --arms A:supply_stress,E:supply_stress,E:recovery_capability \
        --out out/layer3_redesign/phase3_status.json
"""
from __future__ import annotations

import argparse
import itertools
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

from ml.calibration_grid_cache import load_grid                              # noqa: E402
from ml.ds_backbone import get_backbone, load_world                          # noqa: E402
from ml.hypothesis_ranker import ece, isotonic_apply, isotonic_fit, roc_auc  # noqa: E402
from ml.latent_state_head import (                                           # noqa: E402
    assert_backbone_frozen, binarise, depth_embeddings, latent_targets,
)
from ml.layer3_sufficiency import fit_probs, observed_rows                   # noqa: E402
from ml.layer3_uncertainty import (                                          # noqa: E402
    brier, confidence_bins, step3, step3_diagnostics, step4,
)

DSEEDS = [42, 43, 44, 45, 46]
INIT_SEEDS = [0, 1, 2, 3, 4]
N_BINS = 10
CACHE = os.path.join(REPO, "out", "layer3_redesign")

# Families closed before this phase, with the disposition that closed them. Carried so the
# status table is complete rather than only listing what survived.
CLOSED_BEFORE_PHASE_3 = {
    "supplier_reliability": ("Unidentifiable", "Phase 1 (cited): layer3_uncertainty_aware.md "
                             "STEP 2, Case 3 / classification C"),
    "logistics": ("Unidentifiable", "Phase 0: not-simulated as an independent per-entity state "
                  "/ already-a-feature"),
    "inventory": ("Unidentifiable", "Phase 0: already-a-feature"),
    "production": ("Unidentifiable", "Phase 0: not-simulated at any observed t0 / too-sparse"),
}


# ---------------------------------------------------------------------------
# features for the confidence head
# ---------------------------------------------------------------------------

def align_keep(ids, times, target: dict) -> np.ndarray:
    """The row indices `ml/latent_state_head.py::align` keeps, without modifying it.

    `align` returns the filtered matrix but not the mask, and the confidence head needs to
    line the raw observables X up with the SAME rows the state head was scored on. Reproducing
    the mask here (rather than editing the inherited module) keeps `latent_state_head.py`
    byte-identical, which every prior report's integrity check depends on.
    """
    keep = []
    for i, (sid, t) in enumerate(zip(ids, times)):
        row = target["values"].get(t)
        if row is not None and sid in row:
            keep.append(i)
    return np.asarray(keep, dtype=int)


def conf_inputs(variant: str, state: str, depth: int, dseeds: list[int], config: str,
                csv_root: str, rebuild: bool = False) -> dict:
    """`{dseed: {H_va, X_va, H_te, X_te}}` -- the confidence head's inputs, cached.

    `H` is the frozen backbone's representation at the state's own best depth (the same
    tensor the state head consumes); `X` is the raw Supplier input feature vector. Row order
    is `align`'s, so these line up index-for-index with the prediction grid.
    """
    path = os.path.join(CACHE, f"confinputs_{variant}_{state}.npz")
    if os.path.exists(path) and not rebuild:
        z = np.load(path, allow_pickle=False)
        print(f"  [cached] {os.path.basename(path)}", flush=True)
        return {int(d): {k: z[f"{d}_{k}"] for k in ("H_va", "X_va", "H_te", "X_te")}
                for d in dseeds}

    out = {}
    for d in dseeds:
        csv_dir = os.path.join(csv_root, f"v{variant}_seed{d}")
        model, _ = get_backbone(csv_dir, variant, d, mseed=0, device="cpu")
        assert_backbone_frozen(model)
        tr, va, te, _ = load_world(csv_dir, "cpu")
        tgt = latent_targets(variant, d, config,
                             [b.t0 for b in tr] + [b.t0 for b in va] + [b.t0 for b in te])[state]
        cell = {}
        for tag, bundles in (("va", va), ("te", te)):
            H, ids, times = depth_embeddings(model, bundles)[depth]
            keep = align_keep(ids, times, tgt)
            Xo, ids_o, times_o = observed_rows(bundles)
            assert ids_o == ids and times_o == times, "observable/embedding row order diverged"
            cell[f"H_{tag}"] = H[keep]
            cell[f"X_{tag}"] = Xo[keep]
        out[d] = cell
        print(f"  seed {d}: H {cell['H_te'].shape}  X {cell['X_te'].shape}", flush=True)

    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **{f"{d}_{k}": v for d, c in out.items() for k, v in c.items()})
    print(f"  cached -> {path}", flush=True)
    return out


# ---------------------------------------------------------------------------
# the confidence head, and its two controls
# ---------------------------------------------------------------------------

def confidence_signals(grid: dict, ci: dict, dseeds: list[int],
                       init_seeds: list[int]) -> dict:
    """`{dseed: {arm: {"c_te": [M,N], "correct_te": [M,N]}}}` for the three arms.

    One confidence head per (world, init seed), fit on `va` against the state head's
    out-of-sample correctness there and scored on `te`. `fit_probs` is Model A's own head and
    optimiser, unmodified -- the confidence head differs from the state head only in what it
    is asked to predict.
    """
    out = {}
    for d in dseeds:
        g, c = grid[d], ci[d]
        Fva = np.concatenate([c["H_va"], c["X_va"]], axis=1)
        Fte = np.concatenate([c["H_te"], c["X_te"]], axis=1)
        y_va, y_te = g["y_va"].astype(bool), g["y_te"].astype(bool)

        hx, margin, corr = [], [], []
        for m in range(len(init_seeds)):
            p_va, p_te = g["p_va"][m], g["p_te"][m]
            ok_va = ((p_va > 0.5) == y_va).astype(float)
            ok_te = ((p_te > 0.5) == y_te).astype(float)
            pr = fit_probs(Fva, ok_va, [Fte], init_seeds[m])
            hx.append(pr[0] if pr is not None else np.full(len(ok_te), np.nan))
            margin.append(np.abs(p_te - 0.5))
            corr.append(ok_te)
        # Step 4's rejected signal, recomputed so old and new sit on one table. Higher =
        # more confident, so the sign is flipped from `var`.
        var = grid[d]["p_te"][list(range(len(init_seeds)))].var(axis=0, ddof=1)
        out[d] = {
            "hx": {"c_te": np.stack(hx), "correct_te": np.stack(corr)},
            "margin": {"c_te": np.stack(margin), "correct_te": np.stack(corr)},
            "ens_var": {"c_te": np.stack([-var] * len(init_seeds)),
                        "correct_te": np.stack(corr)},
        }
        print(f"  seed {d}: state-head accuracy {np.stack(corr).mean():.4f}   "
              f"C_s(H,X) AUC vs correctness "
              f"{statistics.fmean([roc_auc(hx[m], corr[m].astype(bool)) for m in range(len(hx))]):.4f}",
              flush=True)
    return out


def discrimination(sig: dict, dseeds: list[int], arms=("hx", "margin", "ens_var")) -> dict:
    """Does the confidence signal order the state head's errors? AUC against correctness,
    with the project's standard `max(init-seed spread, dataset-seed spread)` floor."""
    out = {}
    for arm in arms:
        per_seed, spreads = [], []
        for d in dseeds:
            C, K = sig[d][arm]["c_te"], sig[d][arm]["correct_te"]
            a = [roc_auc(C[m], K[m].astype(bool)) for m in range(C.shape[0])
                 if 0 < K[m].sum() < len(K[m])]
            if not a:
                continue
            per_seed.append(statistics.fmean(a))
            spreads.append(max(a) - min(a))
        if not per_seed:
            continue
        init_floor = max(spreads)
        dseed_floor = (max(per_seed) - min(per_seed)) if len(per_seed) > 1 else float("nan")
        floor = max(f for f in (init_floor, dseed_floor) if f == f)
        mean = statistics.fmean(per_seed)
        out[arm] = {
            "auc_mean": mean, "auc_per_dataset_seed": per_seed,
            "init_seed_floor": init_floor, "dataset_seed_floor": dseed_floor,
            "reproduction_floor": floor, "above_chance_by": mean - 0.5,
            "clears_floor": (mean - 0.5) > floor,
            "sign_consistent": all(v > 0.5 for v in per_seed) or all(v < 0.5 for v in per_seed),
        }
    if "hx" in out and "margin" in out:
        d = out["hx"]["auc_mean"] - out["margin"]["auc_mean"]
        out["hx_minus_margin"] = {
            "delta": d,
            "floor": max(out["hx"]["reproduction_floor"], out["margin"]["reproduction_floor"]),
            "beats_margin_control": d > max(out["hx"]["reproduction_floor"],
                                            out["margin"]["reproduction_floor"]),
        }
    return out


def calibration_transfer(sig: dict, dseeds: list[int], arm: str = "hx") -> dict:
    """Step 3's protocol, applied to the confidence head's own probabilities.

    Fit the isotonic map on one world, evaluate on a DIFFERENT world, rotate over all ordered
    pairs. Both of Step 3's floors are reproduced: the fit floor (refit on identical inputs --
    deterministic, expected exactly 0) and the member-composition floor, which is the operative
    one and the only one gated on.
    """
    members = list(range(sig[dseeds[0]][arm]["c_te"].shape[0]))

    def ens(d, mem):
        C = sig[d][arm]["c_te"][list(mem)]
        return C.mean(axis=0), sig[d][arm]["correct_te"][0]

    folds = []
    for calib_d, test_d in itertools.permutations(dseeds, 2):
        c_c, k_c = ens(calib_d, members)
        c_t, k_t = ens(test_d, members)
        cal = isotonic_apply(isotonic_fit(c_c, k_c.astype(float)), c_t)
        e_raw, _ = ece(c_t, k_t, N_BINS)
        e_cal, _ = ece(cal, k_t, N_BINS)
        folds.append({"calib_world": calib_d, "test_world": test_d, "n": int(len(k_t)),
                      "ece_raw": e_raw, "ece_cal": e_cal,
                      "brier_raw": brier(c_t, k_t), "brier_cal": brier(cal, k_t),
                      "improvement": e_raw - e_cal})

    c0, t0 = dseeds[0], dseeds[1]
    fit_vals = []
    for _ in range(8):
        c_c, k_c = ens(c0, members)
        c_t, k_t = ens(t0, members)
        fit_vals.append(ece(isotonic_apply(isotonic_fit(c_c, k_c.astype(float)), c_t),
                            k_t, N_BINS)[0])
    mem_vals = []
    for combo in itertools.combinations(members, 3):
        c_c, k_c = ens(c0, combo)
        c_t, k_t = ens(t0, combo)
        mem_vals.append(ece(isotonic_apply(isotonic_fit(c_c, k_c.astype(float)), c_t),
                            k_t, N_BINS)[0])
    member_floor = float(np.max([abs(a - b) for a, b in itertools.combinations(mem_vals, 2)]))
    impr = statistics.fmean([f["improvement"] for f in folds])
    return {
        "arm": arm, "n_folds": len(folds), "folds": folds,
        "ece_raw": statistics.fmean([f["ece_raw"] for f in folds]),
        "ece_cal": statistics.fmean([f["ece_cal"] for f in folds]),
        "ece_improvement": impr,
        "folds_improved": int(sum(1 for f in folds if f["improvement"] > 0)),
        "fit_floor_max_abs_dev": float(np.max(
            [abs(a - b) for a, b in itertools.combinations(fit_vals, 2)])),
        "member_floor_max_abs_dev": member_floor,
        "clears_member_floor": bool(impr > member_floor),
        "gate_pass": bool(impr > member_floor and all(f["improvement"] > 0 for f in folds)),
    }


def monotonicity(sig: dict, grid: dict, dseeds: list[int], arm: str = "hx") -> dict:
    """Step 4's protocol on the new signal: bin least-confident-first and ask whether the
    state head's error falls across bins. Floor from 3-of-5 member subsets, as Step 4 used."""
    members = list(range(sig[dseeds[0]][arm]["c_te"].shape[0]))

    def bins_for(mem):
        P, C, Y = [], [], []
        for d in dseeds:
            P.append(grid[d]["p_te"][list(mem)].mean(axis=0))
            C.append(sig[d][arm]["c_te"][list(mem)].mean(axis=0))
            Y.append(grid[d]["y_te"])
        # `confidence_bins` orders by DESCENDING its second argument, i.e. least-confident
        # first when handed a variance. Passing -C reuses it unmodified for a confidence.
        return confidence_bins(np.concatenate(P), -np.concatenate(C), np.concatenate(Y))

    per_world = {}
    for d in dseeds:
        p = grid[d]["p_te"][members].mean(axis=0)
        c = sig[d][arm]["c_te"][members].mean(axis=0)
        b = confidence_bins(p, -c, grid[d]["y_te"])
        br = [x["brier"] for x in b]
        per_world[d] = {"bins": b, "monotone_brier": all(br[i] >= br[i + 1]
                                                         for i in range(len(br) - 1)),
                        "brier_lowconf_minus_highconf": br[0] - br[-1]}

    pooled = bins_for(members)
    pb = [x["brier"] for x in pooled]
    sub = np.array([[x["brier"] for x in bins_for(c)]
                    for c in itertools.combinations(members, 3)])
    floor = float(np.max(sub.max(axis=0) - sub.min(axis=0)))
    spread = pb[0] - pb[-1]
    return {
        "arm": arm, "pooled_bins": pooled,
        "pooled_monotone_brier": all(pb[i] >= pb[i + 1] for i in range(len(pb) - 1)),
        "pooled_brier_lowconf_minus_highconf": spread,
        "worlds_monotone": int(sum(1 for v in per_world.values() if v["monotone_brier"])),
        "n_worlds": len(dseeds), "per_world": per_world,
        "member_floor_on_bin_brier": floor,
        "spread_exceeds_floor": bool(spread > floor),
        "gate_pass": bool(all(pb[i] >= pb[i + 1] for i in range(len(pb) - 1))
                          and spread > floor),
    }


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def assign_status(cell: dict) -> dict:
    """Supported / Uncertain / Unidentifiable, derived from the gates rather than chosen."""
    cal_ok = bool(cell["state_calibration"]["clears_member_floor"])
    conf_ok = bool(cell["confidence_monotonicity"]["gate_pass"])
    conf_cal_ok = bool(cell["confidence_calibration"]["clears_member_floor"])
    supported = cal_ok and conf_ok and conf_cal_ok
    reasons = []
    if not cal_ok:
        s3 = cell["state_calibration"]
        reasons.append(f"state calibration does not transfer cross-world: ECE improvement "
                       f"{s3['ece_improvement']:+.4f} against a member floor of "
                       f"{s3['member_floor_max_abs_dev']:.4f}, {s3['folds_improved']}/"
                       f"{s3['n_folds']} folds improved")
    if not conf_cal_ok:
        cc = cell["confidence_calibration"]
        reasons.append(f"confidence-head calibration does not transfer cross-world: "
                       f"{cc['ece_improvement']:+.4f} against a member floor of "
                       f"{cc['member_floor_max_abs_dev']:.4f}, {cc['folds_improved']}/"
                       f"{cc['n_folds']} folds improved")
    if not conf_ok:
        m = cell["confidence_monotonicity"]
        reasons.append(f"confidence signal is not monotone: {m['worlds_monotone']}/"
                       f"{m['n_worlds']} worlds monotone, pooled least-minus-most-confident "
                       f"Brier {m['pooled_brier_lowconf_minus_highconf']:+.4f} against a "
                       f"member floor of {m['member_floor_on_bin_brier']:.4f}")
    return {"status": "Supported" if supported else "Uncertain",
            "state_calibration_clears": cal_ok,
            "confidence_calibration_clears": conf_cal_ok,
            "confidence_monotone": conf_ok,
            "reasons": reasons}


def run_arm(variant: str, state: str, dseeds: list[int], init_seeds: list[int],
            config: str, csv_root: str) -> dict:
    print(f"\n=== variant {variant} / {state} ===", flush=True)
    t = time.time()
    grid, depth = load_grid(variant, state, dseeds=dseeds, init_seeds=init_seeds,
                            csv_root=csv_root, config=config)
    print(f"  state head reads h^{depth} (Phase 1's recorded best depth, not re-selected)",
          flush=True)

    print("  -- state-estimate calibration (ml/layer3_uncertainty.py, unmodified) --",
          flush=True)
    s3 = step3(grid, dseeds, init_seeds)
    s3d = step3_diagnostics(grid, dseeds, init_seeds, s3)
    s4 = step4(grid, dseeds, init_seeds)

    print("  -- confidence head C_s = g_s(H, X) --", flush=True)
    ci = conf_inputs(variant, state, depth, dseeds, config, csv_root)
    sig = confidence_signals(grid, ci, dseeds, init_seeds)
    disc = discrimination(sig, dseeds)
    ccal = calibration_transfer(sig, dseeds, "hx")
    mono = monotonicity(sig, grid, dseeds, "hx")
    mono_margin = monotonicity(sig, grid, dseeds, "margin")

    cell = {"variant": variant, "state": state, "depth": depth,
            "state_calibration": s3, "state_calibration_diagnostics": s3d,
            "state_confidence_ensemble_variance": s4,
            "confidence_discrimination": disc,
            "confidence_calibration": ccal,
            "confidence_monotonicity": mono,
            "confidence_monotonicity_margin_control": mono_margin,
            "seconds": time.time() - t}
    cell.update(assign_status(cell))
    print(f"  STATUS: {cell['status']}   ({cell['seconds']:.0f}s)", flush=True)
    return cell


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="A:supply_stress,E:supply_stress,E:recovery_capability")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--init-seeds", default="0,1,2,3,4")
    ap.add_argument("--config", default="v1")
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dseeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    iseeds = [int(s) for s in a.init_seeds.split(",") if s.strip()]
    arms = [tuple(x.split(":")) for x in a.arms.split(",") if x.strip()]

    res = {"dataset_seeds": dseeds, "init_seeds": iseeds, "config": a.config,
           "arms": {}, "closed_before_phase_3": CLOSED_BEFORE_PHASE_3}
    for variant, state in arms:
        res["arms"][f"{variant}:{state}"] = run_arm(variant, state, dseeds, iseeds,
                                                    a.config, a.csv_root)

    print("\n" + "=" * 124)
    print("PHASE 3 — evidence status per family")
    print("=" * 124)
    hdr = (f"{'arm':<30}{'state cal':>11}{'conf cal':>10}{'conf mono':>11}"
           f"{'C_s AUC':>10}{'margin':>9}{'floor':>9}{'beats?':>8}{'STATUS':>13}")
    print(hdr); print("-" * len(hdr))
    for k, c in res["arms"].items():
        d = c["confidence_discrimination"]
        hx, mg = d.get("hx", {}), d.get("margin", {})
        cmp_ = d.get("hx_minus_margin", {})
        print(f"{k:<30}"
              f"{('clears' if c['state_calibration_clears'] else 'STOP'):>11}"
              f"{('clears' if c['confidence_calibration_clears'] else 'STOP'):>10}"
              f"{('yes' if c['confidence_monotone'] else 'NO'):>11}"
              f"{hx.get('auc_mean', float('nan')):>10.4f}"
              f"{mg.get('auc_mean', float('nan')):>9.4f}"
              f"{cmp_.get('floor', float('nan')):>9.4f}"
              f"{('YES' if cmp_.get('beats_margin_control') else 'no'):>8}"
              f"{c['status']:>13}")
    print("-" * len(hdr))
    for k, c in res["arms"].items():
        print(f"\n{k} -> {c['status']}")
        for r in c["reasons"]:
            print(f"  - {r}")
    print("\nclosed before this phase (status Unidentifiable):")
    for fam, (st, why) in CLOSED_BEFORE_PHASE_3.items():
        print(f"  {fam:<24}{st:<16}{why}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
