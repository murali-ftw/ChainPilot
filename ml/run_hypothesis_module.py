#!/usr/bin/env python3
"""
§10 — Behavioural Hypothesis Generation. Detect, label, rank, calibrate, and show a human.

This is the runner for `reports/layer3_testing.md` §10. It does **not** retry hidden-cause
discovery: §9.8 established that at no depth of SHARE's stack -- including the raw input
features -- does hidden-parent co-membership separate from a size-matched random regrouping,
and nothing here reopens that. It builds the weaker, checkable claim instead: given an
observed co-degradation between two suppliers, emit a **ranked, calibrated** distribution
over candidate explanations, with an explicit "unknown", and leave the call to a reviewer.

**The prediction, recorded here in code before any number exists**, so that a specific result
is not later misread as a bug. `regional_logistics` maps onto the base world's `H_PORT` /
`H_TRUCK` / `H_CUSTOMS` pools, two of which are *defined* by `country` and the third by a
`sea` flag closely proxied by lead time -- all observable, so it should rank and calibrate
well. `shared_upstream` is exactly the object §9.8 found has zero encodable signal anywhere.
A correctly calibrated module should therefore push `shared_upstream` toward its base rate or
toward `unknown` for most detected patterns. **That outcome is the module working, not
failing.**

Pipeline, in the order it runs:

1. **Detect** (`ml/hypothesis_features.py::detect_patterns`) -- reuse §9.7's `on_time_rate_90d`
   correlation, the exact column §2.2's generator-side check validated. No new instrument.
2. **Label** (`ml/hypothesis_labels.py`) -- multi-label ground truth from generator mechanisms,
   read privileged, never a feature.
3. **Report composition first.** How imbalanced the ranking task is decides how it has to be
   trained, so the table is printed before a model exists.
4. **Rank** (`ml/hypothesis_ranker.py`) -- leave-one-dataset-seed-out CV, with a second seed
   held out inside each fold for the isotonic calibration fit.
5. **Floor.** The whole CV is repeated `--replicates` times identically; per-class AUC and ECE
   movement across replicates is the floor every §10 claim has to clear, re-measured on this
   session's device per §9.10 rather than carried over.
6. **Reviewer output** -- a sample of instances as a reviewer would see them, `unknown` wins
   included.

    python3 ml/run_hypothesis_module.py --variants B,D --seeds 42,43,44,45,46
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.hypothesis_features import (  # noqa: E402
    FUTURE, PairContext, assert_no_privileged_features, build_features, detect_patterns,
    usable_universe)
from ml.hypothesis_labels import (  # noqa: E402
    CLASSES, composition, label_matrix, load_mechanism_state, pair_subtypes, truth_ceiling)
from ml.hypothesis_ranker import (  # noqa: E402
    evaluate, isotonic_apply, isotonic_fit, predict, train_head)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_one(csv_dir: str, state_path: str, hidden_ref: str, top_k: int,
              min_corr: float, as_of: str, ablate_coparent: bool,
              candidate_set: str = "detected") -> dict:
    """Detect, featurise and label one variant-seed."""
    state = load_mechanism_state(state_path, csv_dir, cross_check_dir=hidden_ref)
    ctx = PairContext(csv_dir, as_of=as_of)
    pairs = (usable_universe(ctx) if candidate_set == "usable_universe"
             else detect_patterns(ctx, top_k=top_k, min_corr=min_corr))
    X, names = build_features(ctx, pairs)
    assert_no_privileged_features(names)

    # The `shared_sourcing` class is a positive control only if the observable rebuild of the
    # co-parent graph really is the privileged one. Asserted, not claimed: `component_suppliers`
    # plus `components.supplier_id` must reproduce the generator's internal `coparents` edge
    # for edge. (This also confirms Mechanism C is inactive -- a live rewire would deactivate
    # edges mid-timeline and the static rebuild would drift.)
    idx = {s: i for i, s in enumerate(state["supplier_ids"])}
    priv_edges = {(min(idx[a], idx[b]), max(idx[a], idx[b]))
                  for a, ds in state["coparents"].items() if a in idx
                  for b in ds if b in idx}
    if priv_edges != ctx.coparent_edges:
        raise ValueError(
            f"{csv_dir}: observable co-parent rebuild != privileged coparents "
            f"({len(ctx.coparent_edges)} vs {len(priv_edges)} edges); "
            f"`shared_sourcing` cannot be used as a positive control")

    if ablate_coparent:
        keep = [i for i, n in enumerate(names) if n != "is_coparent"]
        X, names = X[:, keep], [names[i] for i in keep]

    y = label_matrix(state, pairs)
    subs = pair_subtypes(state, pairs)
    return {
        "variant": state["variant"], "seed": state["seed"], "pairs": pairs, "X": X,
        "y": y, "names": names, "subtypes": subs, "sup_ids": state["supplier_ids"],
        "corr": ctx.s90[pairs[:, 0], pairs[:, 1]],
        "corr_cohort": ctx.s90c[pairs[:, 0], pairs[:, 1]],
        "n_usable": int(ctx.usable.sum()), "n_any_history": int((ctx.n_obs > 0).sum()),
        "n_suppliers": ctx.n, "n_snapshots": int(ctx.resid.shape[1]),
        "composition": composition(y, subs),
        "ceiling": truth_ceiling(state, ctx.usable),
        "generator_checks_failed": state.get("generator_checks_failed", []),
        "hp_alpha": state.get("hp_alpha"),
    }


def _subsample_negatives(X: np.ndarray, Y: np.ndarray, cap: int,
                         rng_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep every mechanism-positive row and at most `cap` of the rest. **Training only.**

    On the `usable_universe` candidate set a fold's training half runs to ~750,000 pairs, of
    which >90% carry no mechanism at all, and full-batch fitting on all of them costs minutes
    per fold for no gain in what the fit can learn. Capping the negatives is the standard
    remedy and it shifts the base rate, which would normally wreck calibration -- except that
    the isotonic curve is fit on the **complete, uncapped** calibration seed and evaluated on
    the **complete, uncapped** test seed, so the base rate the reported confidences are
    mapped onto is the true one. The cap changes what the ranker sees; it does not change
    what the calibration is measured against.
    """
    if cap <= 0:
        return X, Y
    pos = Y[:, :3].any(axis=1)
    neg = np.where(~pos)[0]
    if len(neg) <= cap:
        return X, Y
    keep = np.random.default_rng(rng_seed).choice(neg, size=cap, replace=False)
    sel = np.sort(np.concatenate([np.where(pos)[0], keep]))
    return X[sel], Y[sel]


def cross_validate(data: list[dict], seeds: list[int], model_seed: int, epochs: int,
                   device: str, max_pos_weight: float, train_neg_cap: int = 0) -> dict:
    """Leave-one-dataset-seed-out CV with an inner calibration seed.

    Folds are whole dataset seeds because §9.8's finding is that co-membership is
    distinguishable only by supplier *identity*: a random pair-level split would leave the
    same suppliers on both sides and let the head memorise them, which is what §9.4 caught
    Stage 1 doing. Nothing about a test seed -- its suppliers, its groups, its pools -- has
    been seen during training or calibration.
    """
    by_seed = {d["seed"]: d for d in data}
    folds, pooled_p, pooled_c, pooled_y, pooled_meta = [], [], [], [], []
    for i, test_seed in enumerate(seeds):
        calib_seed = seeds[(i + 1) % len(seeds)]
        train_seeds = [s for s in seeds if s not in (test_seed, calib_seed)]
        Xtr = np.concatenate([by_seed[s]["X"] for s in train_seeds])
        Ytr = np.concatenate([by_seed[s]["y"] for s in train_seeds])
        n_train_full = len(Xtr)
        Xtr, Ytr = _subsample_negatives(Xtr, Ytr, train_neg_cap,
                                        rng_seed=1000 * model_seed + test_seed)
        Xca, Yca = by_seed[calib_seed]["X"], by_seed[calib_seed]["y"]
        Xte, Yte = by_seed[test_seed]["X"], by_seed[test_seed]["y"]

        fit = train_head(Xtr, Ytr, Xca, Yca, epochs=epochs, seed=model_seed,
                         device=device, max_pos_weight=max_pos_weight)
        p_ca = predict(fit, Xca, device=device)
        p_te = predict(fit, Xte, device=device)
        curves = [isotonic_fit(p_ca[:, c], Yca[:, c]) for c in range(len(CLASSES))]
        p_cal = np.stack([isotonic_apply(curves[c], p_te[:, c])
                          for c in range(len(CLASSES))], axis=1)

        folds.append({
            "test_seed": test_seed, "calib_seed": calib_seed, "train_seeds": train_seeds,
            "n_train": int(len(Xtr)), "n_train_before_cap": int(n_train_full),
            "n_calib": int(len(Xca)), "n_test": int(len(Xte)),
            "pos_weight": fit["pos_weight"], "best_epoch": fit["best_epoch"],
            "params": fit["params"], "per_class": evaluate(p_te, p_cal, Yte),
        })
        pooled_p.append(p_te)
        pooled_c.append(p_cal)
        pooled_y.append(Yte)
        pooled_meta.append(test_seed)

    P, C, Y = np.concatenate(pooled_p), np.concatenate(pooled_c), np.concatenate(pooled_y)
    return {"folds": folds, "pooled": evaluate(P, C, Y),
            "pooled_arrays": (P, C, Y, pooled_meta)}


def subtype_breakdown(data: list[dict], cv: dict) -> dict:
    """`shared_upstream` AUC split by Type A vs Type B co-membership.

    The headline class merges them, but they are not the same object. On **Variant B**
    `HP_ALPHA` is 0, so a Type A group has no downstream effect at all and is behaviourally
    identical to the Type C decoy; only on Variant D (`HP_ALPHA = 0.35`) does Type A couple.
    Type C is scored too and must stay at chance -- it is the decoy control, and a module
    that ranked it above chance would be reading structure that is not there.
    """
    P, _, Y, seeds = cv["pooled_arrays"]
    by_seed = {d["seed"]: d for d in data}
    offs, out = {}, {}
    pos = 0
    for s in seeds:
        n = len(by_seed[s]["y"])
        offs[s] = (pos, pos + n)
        pos += n
    from ml.hypothesis_ranker import roc_auc
    score = P[:, CLASSES.index("shared_upstream")]
    neg = ~Y[:, CLASSES.index("shared_upstream")].astype(bool)
    masks = {}
    for name in ("type_a_pair", "type_b_pair", "type_c_pair"):
        mask = np.zeros(len(Y), dtype=bool)
        for s in seeds:
            lo, hi = offs[s]
            mask[lo:hi] = by_seed[s]["subtypes"][name]
        masks[name] = mask
        sel = mask | neg
        # Per fold as well as pooled. With 43-51 pooled positives a fold holds only 8-10, so
        # a per-fold AUC is very noisy -- but sign consistency across folds is the standard
        # this project has applied to every other effect (§3.2 identified a mean carried by
        # one or two seeds as exactly the shape of noise), and it cannot be applied to a
        # number that was only ever computed pooled.
        per_fold = []
        for s in seeds:
            lo, hi = offs[s]
            fm, fn = mask[lo:hi], neg[lo:hi]
            fsel = fm | fn
            per_fold.append(roc_auc(score[lo:hi][fsel], fm[fsel]))
        above = [a for a in per_fold if a is not None and a > 0.5]
        out[name] = {"n_positive": int(mask.sum()),
                     "auc_vs_negatives": roc_auc(score[sel], mask[sel]),
                     "auc_per_fold": per_fold,
                     "folds_above_chance": f"{len(above)}/{sum(a is not None for a in per_fold)}"}

    # The decoy-controlled contrast §9.7.3 used: a real type must beat the inert Type C, not
    # merely beat chance. Type C exists precisely so that "the model found something" and
    # "the model found the thing that is there" can be told apart.
    for real in ("type_a_pair", "type_b_pair"):
        a, c = out[real]["auc_vs_negatives"], out["type_c_pair"]["auc_vs_negatives"]
        out[f"{real}_minus_decoy"] = (None if a is None or c is None else float(a - c))
    return out


def reviewer_sample(data: list[dict], cv: dict, n: int, rng_seed: int = 20260813) -> list[dict]:
    """What a supply-chain reviewer would actually be handed, for a sample of instances.

    Sampling is **stratified to include the cases that matter**, not uniform: a uniform draw
    from a set this imbalanced would return `unknown` on essentially every row and would
    show nothing about the module's behaviour on a true `shared_upstream` pair. Rows where
    `unknown` wins are deliberately kept rather than filtered -- that is the output the
    module exists to be able to produce.
    """
    P, C, Y, seeds = cv["pooled_arrays"]
    by_seed = {d["seed"]: d for d in data}
    rows, offs, pos = [], {}, 0
    for s in seeds:
        offs[s] = pos
        pos += len(by_seed[s]["y"])
    for s in seeds:
        d = by_seed[s]
        base = offs[s]
        for j in range(len(d["y"])):
            rows.append((base + j, s, j))

    rng = np.random.default_rng(rng_seed)
    idx_up = [r for r in rows if Y[r[0], 0]]
    idx_reg = [r for r in rows if Y[r[0], 1] and not Y[r[0], 0]]
    idx_src = [r for r in rows if Y[r[0], 2] and not Y[r[0], 0]]
    idx_unk = [r for r in rows if Y[r[0], 3]]
    picked = []
    for bucket, want in ((idx_up, n // 4), (idx_reg, n // 4),
                         (idx_src, n // 4), (idx_unk, n - 3 * (n // 4))):
        if bucket:
            take = rng.choice(len(bucket), size=min(want, len(bucket)), replace=False)
            picked.extend(bucket[t] for t in take)

    out = []
    for gi, seed, j in picked:
        d = by_seed[seed]
        a, b = int(d["pairs"][j, 0]), int(d["pairs"][j, 1])
        ranked = sorted(
            ({"hypothesis": cls, "confidence": float(C[gi, k])}
             for k, cls in enumerate(CLASSES)),
            key=lambda r: -r["confidence"])
        out.append({
            "variant": d["variant"], "seed": seed,
            "supplier_a": d["sup_ids"][a], "supplier_b": d["sup_ids"][b],
            "pattern": {"corr_on_time_90d": float(d["corr"][j]),
                        "corr_cohort_residualised": float(d["corr_cohort"][j]),
                        "joint_dip_count": float(
                            d["X"][j, d["names"].index("joint_dip_count")]),
                        "n_overlapping_snapshots": float(
                            d["X"][j, d["names"].index("n_overlap")])},
            "ranked_hypotheses": ranked,
            "truth": [c for k, c in enumerate(CLASSES) if Y[gi, k]],
        })
    return out


def subtype_floor(data: list[dict], reps: list[dict]) -> dict:
    """Floor for the Type A / Type B / decoy AUCs, which `floor_from_replicates` misses.

    Necessary because those three numbers rest on 12-35 positives each, and the whole
    reading of §10.3 turns on whether Type B's elevation is larger than what the fit moves
    by on its own. A subtype AUC quoted without this would be exactly the kind of
    seed-carried effect §3.2 identified as noise.
    """
    out: dict = {}
    per_rep = [subtype_breakdown(data, r) for r in reps]
    for name in ("type_a_pair", "type_b_pair", "type_c_pair",
                 "type_a_pair_minus_decoy", "type_b_pair_minus_decoy"):
        vals = [(r[name] if name.endswith("_minus_decoy") else r[name]["auc_vs_negatives"])
                for r in per_rep]
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            continue
        d = [abs(vals[i] - vals[j]) for i in range(len(vals)) for j in range(i + 1, len(vals))]
        out[name] = {"n_replicates": len(vals), "mean": float(np.mean(vals)),
                     "mean_abs_dev": float(np.mean(d)), "max_abs_dev": float(np.max(d)),
                     "values": [float(v) for v in vals]}
    return out


def floor_from_replicates(reps: list[dict]) -> dict:
    """Per-class AUC and ECE movement across identically-configured repeats of the whole CV.

    Reported as mean-abs and **max-abs** pairwise deviation, and the gate below uses max-abs
    -- the same conservative choice §4 and §9.2 made. If this comes back at exactly zero the
    fit is bit-deterministic on this device, which is a fact to state rather than a floor to
    celebrate: it means run-to-run numerical noise is not the operative uncertainty, and the
    model-seed spread reported alongside it is.
    """
    out = {}
    for cls in CLASSES:
        for metric in ("auc", "ece_calibrated", "ece_raw"):
            vals = [r["pooled"][cls][metric] for r in reps
                    if r["pooled"][cls][metric] is not None]
            if len(vals) < 2:
                continue
            d = [abs(vals[i] - vals[j]) for i in range(len(vals))
                 for j in range(i + 1, len(vals))]
            out.setdefault(cls, {})[metric] = {
                "n_replicates": len(vals), "mean": float(np.mean(vals)),
                "mean_abs_dev": float(np.mean(d)), "max_abs_dev": float(np.max(d)),
                "values": [float(v) for v in vals],
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default=os.path.join(REPO, "db", "csv_mid"))
    ap.add_argument("--state-dir", default=os.path.join(REPO, "out", "hidden_mid_ext"))
    ap.add_argument("--hidden-ref", default=os.path.join(REPO, "out", "hidden_mid"),
                    help="§9's HP_GROUPS, cross-checked against so §10's ground truth is "
                         "provably the same object §2 and §9 scored")
    ap.add_argument("--variants", default="B,D")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--candidate-set", default="detected",
                    choices=("detected", "usable_universe"),
                    help="'detected' is the module as it would deploy; 'usable_universe' is "
                         "the maximum-power version used to state the shared_upstream null "
                         "on more than a handful of positives")
    ap.add_argument("--train-neg-cap", type=int, default=0,
                    help="cap on mechanism-negative rows in the TRAINING split only; "
                         "calibration and test splits are always complete")
    ap.add_argument("--top-k", type=int, default=64,
                    help="pool width, matching the top_k=64 Transformer 2 attends over "
                         "throughout sections 2 and 9")
    ap.add_argument("--min-corr", type=float, default=0.0)
    ap.add_argument("--respect-t0", action="store_true",
                    help="restrict detection to pre-t0 snapshots, i.e. §9.7's forecasting "
                         "setting rather than §10's retrospective one")
    ap.add_argument("--test-frac", type=float, default=0.6)
    ap.add_argument("--ablate-coparent", action="store_true",
                    help="withhold the observable co-parent edge, turning the "
                         "`shared_sourcing` positive control into a behavioural test")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--max-pos-weight", type=float, default=50.0)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--replicates", type=int, default=10,
                    help="identically-configured repeats of the full CV: the reproduction "
                         "floor")
    ap.add_argument("--seed-replicates", type=int, default=10,
                    help="repeats with a different model-init seed each time")
    ap.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    ap.add_argument("--sample-size", type=int, default=48)
    ap.add_argument("--out", default=os.path.join(REPO, "out", "hypothesis_module.json"))
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    blob: dict = {"config": vars(args), "variants": {}}

    for variant in variants:
        t0 = time.time()
        data = []
        for seed in seeds:
            d = os.path.join(args.csv_dir, f"v{variant}_seed{seed}")
            as_of = FUTURE
            if args.respect_t0:
                import pandas as pd
                dates = sorted(pd.read_csv(f"{d}/supplier_temporal_features.csv.gz",
                                           usecols=["as_of_date"])["as_of_date"].unique())
                as_of = dates[int(round(args.test_frac * len(dates)))]
            data.append(build_one(d, os.path.join(args.state_dir, f"{variant}_{seed}.json"),
                                  args.hidden_ref, args.top_k, args.min_corr, as_of,
                                  args.ablate_coparent, args.candidate_set))
            c, ce = data[-1]["composition"], data[-1]["ceiling"]
            print(f"  v{variant} s{seed}: {c['n_pairs']:,} instances from "
                  f"{data[-1]['n_usable']} usable suppliers  " +
                  "  ".join(f"{k}={v:,}" for k, v in c["positives"].items()) +
                  f"  [ceiling: {ce['shared_upstream_pairs_both_usable']}"
                  f"/{ce['shared_upstream_pairs_total']} A/B pairs evaluable]", flush=True)
        print(f"  [detect+featurise v{variant}: {time.time() - t0:.0f}s]", flush=True)

        t1 = time.time()
        reps = [cross_validate(data, seeds, args.model_seed, args.epochs, args.device,
                               args.max_pos_weight, args.train_neg_cap)
                for _ in range(max(1, args.replicates))]
        seed_reps = [cross_validate(data, seeds, args.model_seed + 1 + k, args.epochs,
                                    args.device, args.max_pos_weight, args.train_neg_cap)
                     for k in range(max(0, args.seed_replicates))]
        cv = reps[0]
        print(f"  [rank v{variant}: {time.time() - t1:.0f}s for "
              f"{len(reps) + len(seed_reps)} x {len(seeds)}-fold CV]", flush=True)

        blob["variants"][variant] = {
            "feature_names": data[0]["names"],
            "hp_alpha": data[0]["hp_alpha"],
            "per_seed": [{k: v for k, v in d.items()
                          if k in ("seed", "composition", "ceiling", "n_usable",
                                   "n_any_history", "n_suppliers", "n_snapshots",
                                   "generator_checks_failed")}
                         for d in data],
            "cv": {"folds": cv["folds"], "pooled": cv["pooled"]},
            "subtypes": subtype_breakdown(data, cv),
            "floor_identical": floor_from_replicates(reps),
            "floor_model_seed": floor_from_replicates(seed_reps) if seed_reps else {},
            "floor_subtype_model_seed": subtype_floor(data, seed_reps) if seed_reps else {},
            "reviewer_sample": reviewer_sample(data, cv, args.sample_size),
        }
        for cls in CLASSES:
            r = cv["pooled"][cls]
            auc = "n/a" if r["auc"] is None else f"{r['auc']:.4f}"
            print(f"    {cls:<20} base={r['base_rate']:.4f} AUC={auc} "
                  f"ECE_raw={r['ece_raw']:.4f} ECE_cal={r['ece_calibrated']:.4f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
