#!/usr/bin/env python3
"""
Phase 2 — validate the extracted SCM, then drive it with true vs. estimated latent state.

Three checks, in order of how much they prove:

  1. EQUATION FIDELITY. `ml/scm.py` re-implements the generator's equations standalone. Compare
     its `own_stress`/`stress` against the generator's OWN functions on every visible supplier at
     every snapshot. A wrapper would agree trivially; this does not wrap, so agreement is
     evidence. Reported as max absolute error -- exact means 0.0, not "small".

  2. OUTCOME FIDELITY. Equation agreement only shows the transcription is faithful. This check
     runs the SCM forward to `p_delay` on real shipments and compares against the generator's own
     realised outcomes read from the EMITTED data (`shipments.csv`: was the shipment late?).
     Reported as calibration by predicted-probability bucket plus a Brier score, since the
     generator's outcome is a Bernoulli draw and cannot be reproduced exactly by anything.

  3. TRUE vs ESTIMATED STATE. The SCM cannot consume privileged signals at inference. For each
     state Phase 1 cleared, drive `p_delay` with (a) the true generator state -- an upper bound on
     what correct equations can do -- and (b) a Layer 3 head's estimate. The gap is the first
     measurement of what Layer 3's estimation error costs Layer 4.

     Estimated stress is a probability from a head trained on a MEDIAN-SPLIT target, so it is not
     on the stress scale. It is mapped back by rank/quantile matching against the true stress
     distribution -- monotone, so it preserves exactly the ordering information the head carries
     and adds no new information. That mapping is itself an approximation and is reported as such.

The validity caveat for the whole exercise lives in `ml/scm.py`'s module docstring.

    python3 ml/run_scm_validation.py --variant E --seeds 42,43,44,45,46 --config v1
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.confirm_latent_states import namespace                       # noqa: E402
from ml.ds_backbone import get_backbone, load_world                  # noqa: E402
from ml.hypothesis_ranker import roc_auc                             # noqa: E402
from ml.latent_state_head import (DEPTHS, align, assert_backbone_frozen,  # noqa: E402
                                  binarise, depth_embeddings, latent_targets,
                                  train_head, LatentStateHead)
from ml.scm import SupplyChainSCM, from_namespace                    # noqa: E402

SEA = None   # resolved per world from the generator namespace


def equation_fidelity(scm: SupplyChainSCM, ns: dict, probes) -> dict:
    """Check 1 — standalone SCM vs the generator's own functions."""
    gen_stress, gen_own = ns["stress"], ns["own_stress"]
    sup_by_id, visible = ns["sup_by_id"], ns["VISIBLE_SUP"]
    sids = sorted(visible)

    max_own = max_tot = 0.0
    n = 0
    for t in probes:
        tt = t.to_pydatetime() if hasattr(t, "to_pydatetime") else t
        for sid in sids:
            br = sup_by_id[sid]["base_rel"]
            max_own = max(max_own, abs(scm.own_stress(sid, tt) - gen_own(sid, br, tt)))
            max_tot = max(max_tot, abs(scm.stress(sid, tt) - gen_stress(sid, br, tt)))
            n += 1
    return {"n_comparisons": n, "max_abs_err_own_stress": max_own,
            "max_abs_err_stress": max_tot, "exact": (max_own == 0.0 and max_tot == 0.0)}


def outcome_fidelity(scm: SupplyChainSCM, ns: dict) -> dict:
    """Check 2 — SCM p_delay vs the generator's realised late/on-time outcomes."""
    sea = set(ns["SEA"])
    resilience = ns["RESILIENCE"] or None
    fac_outage = ns["FACTORY_OUTAGE"]
    sup_by_id = ns["sup_by_id"]
    T_END = ns["T_END"]

    # The delay EVENT is the generator's `delayed` status transition (:1096-1098) -- the same
    # ground truth training_labels is built from (:1355). It is NOT `delivered_at > eta`:
    # `actual = J(eta - timedelta(hours=random.randint(0, 30)))` on the on-time branch, and
    # J() adds 0..2700s of jitter (:207-212), so whenever that hours draw is 0 -- about 1 in 31
    # on-time shipments -- the delivery timestamp lands after eta with no delay having occurred.
    # Using the timestamp comparison inflates the observed rate by ~0.03 and would have been
    # misread as SCM under-prediction.
    delayed_ids = {sid for (sid, stat, _prev, _at, _rec) in ns["transitions"] if stat == "delayed"}

    p, y = [], []
    for sh in ns["shipments"]:
        sid, disp, eta = sh["supplier_id"], sh["dispatched_at"], sh["eta"]
        if not sid or sid not in sup_by_id or not disp:
            continue
        # a delayed shipment only receives the transition when eta lands inside the timeline
        # (:1096), so shipments past that boundary are censored, not on-time.
        if eta > T_END:
            continue
        st = scm.stress(sid, disp)
        hit = bool(fac_outage and sh.get("factory_id") == fac_outage[0]
                   and fac_outage[1] <= disp <= fac_outage[2])
        p.append(scm.p_delay(st, sup_id=sid, resilience=resilience,
                             sea_carrier=sh["carrier"] in sea, dispatch=disp,
                             factory_outage_hit=hit))
        y.append(1.0 if sh["id"] in delayed_ids else 0.0)

    p_arr, y_arr = np.asarray(p), np.asarray(y)
    edges = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.01]
    buckets = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_arr >= lo) & (p_arr < hi)
        if m.sum() >= 30:
            buckets.append({"range": f"[{lo:.2f},{hi:.2f})", "n": int(m.sum()),
                            "predicted": float(p_arr[m].mean()),
                            "observed": float(y_arr[m].mean())})
    return {"n_shipments": int(len(y_arr)),
            "brier": float(np.mean((p_arr - y_arr) ** 2)),
            "mean_predicted": float(p_arr.mean()), "observed_rate": float(y_arr.mean()),
            "auc_p_delay_vs_actual_late": roc_auc(p_arr, y_arr.astype(bool)),
            "max_abs_calibration_gap": max((abs(b["predicted"] - b["observed"])
                                            for b in buckets), default=None),
            "buckets": buckets}


def quantile_map(pred: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map head probabilities onto the true-state scale, rank-preserving.

    Monotone by construction, so it carries exactly the ordering the head learned and injects
    no new information -- it only puts the estimate on the units `p_delay` expects."""
    order = np.argsort(pred, kind="stable")
    ranks = np.empty(len(pred), dtype=float)
    ranks[order] = np.arange(len(pred))
    q = ranks / max(1, len(pred) - 1)
    return np.quantile(reference, q)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="E")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--config", default="v1")
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--depth", type=int, default=1, help="Layer 3 readout depth for estimates")
    ap.add_argument("--skip-heads", action="store_true",
                    help="checks 1-2 only. Lets the equation/outcome fidelity checks run on a "
                         "variant whose backbone is not cached -- notably Variant K, which is "
                         "the only variant that activates EVERY term in the extracted equations "
                         "(co-parent coupling, HP_ALPHA, upstream chains, F attenuation).")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results = {"variant": args.variant, "config": args.config, "depth": args.depth,
               "per_seed": {}}

    for dseed in seeds:
        csv_dir = os.path.join(args.csv_root, f"v{args.variant}_seed{dseed}")
        print(f"\n=== variant {args.variant} seed {dseed} ===", flush=True)
        ns = namespace(args.variant, dseed, args.config)
        params, structure = from_namespace(ns)
        scm = SupplyChainSCM(params, structure)

        probes = list(ns["T0S"])
        eq = equation_fidelity(scm, ns, probes)
        print(f"  [1] equation fidelity : {eq['n_comparisons']:,} comparisons  "
              f"max|err| own_stress={eq['max_abs_err_own_stress']:.3e}  "
              f"stress={eq['max_abs_err_stress']:.3e}  "
              f"{'EXACT' if eq['exact'] else 'INEXACT'}", flush=True)

        oc = outcome_fidelity(scm, ns)
        print(f"  [2] outcome fidelity  : {oc['n_shipments']:,} shipments  "
              f"Brier={oc['brier']:.4f}  predicted={oc['mean_predicted']:.4f} "
              f"observed={oc['observed_rate']:.4f}  "
              f"max calib gap={oc['max_abs_calibration_gap']:.4f}  "
              f"AUC(p_delay vs late)={oc['auc_p_delay_vs_actual_late']:.4f}", flush=True)

        # record which equation terms this variant actually exercised, so a clean fidelity
        # result cannot be read as covering terms that were structurally inert.
        eq["terms_active"] = {
            "own_stress_events": bool(structure.events),
            "hp_b_events": bool(structure.hp_b_events),
            "shock_events": bool(structure.shock_events),
            "idio": any(structure.idio.values()),
            "coparent_coupling": bool(structure.coparents) and bool(params.coparent_coupling),
            "hp_alpha_type_a": bool(params.hp_alpha) and bool(structure.hp_groups),
            "upstream_chain": bool(structure.sup_chain),
            "f_attenuation": structure.f_on,
            "resilience_absorption": bool(ns["RESILIENCE"]),
        }
        print(f"      terms active: "
              f"{sorted(k for k, v in eq['terms_active'].items() if v)}", flush=True)

        if args.skip_heads:
            results["per_seed"][str(dseed)] = {"equation_fidelity": eq, "outcome_fidelity": oc,
                                               "true_vs_estimated": {}}
            continue

        # ---- check 3: true vs estimated state ------------------------------------
        model, _meta = get_backbone(csv_dir, args.variant, dseed, mseed=0, device="cpu",
                                    verbose=False)
        assert_backbone_frozen(model)
        tr, va, te, _ = load_world(csv_dir, "cpu")
        targets = latent_targets(args.variant, dseed, args.config,
                                 [b.t0 for b in tr] + [b.t0 for b in va] + [b.t0 for b in te])
        emb_tr, emb_te = depth_embeddings(model, tr), depth_embeddings(model, te)

        cmp_out = {}
        for state in ("supply_stress", "recovery_capability", "mitigation_level"):
            tgt = targets.get(state)
            if tgt is None:
                continue
            Xtr, ytr = align(*emb_tr[args.depth], tgt)
            Xte, yte = align(*emb_te[args.depth], tgt)
            if Xtr is None or Xte is None:
                continue
            ytr_b, yte_b, _thr = binarise(ytr, yte, tgt["kind"])

            # train the Layer 3 head, then read its continuous score on the test split
            torch.manual_seed(0); np.random.seed(0)
            mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
            xtr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
            xte = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
            ttr = torch.tensor(ytr_b, dtype=torch.float32)
            head = LatentStateHead(xtr.shape[1])
            opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
            assert_backbone_frozen(model, head=head, optimizer=opt)
            pos = float(ttr.sum())
            pw = torch.tensor(max(1.0, (len(ttr) - pos) / max(1.0, pos)), dtype=torch.float32)
            lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
            head.train()
            for _ in range(120):
                opt.zero_grad(); lossf(head(xtr), ttr).backward(); opt.step()
            head.eval()
            with torch.no_grad():
                est = torch.sigmoid(head(xte)).numpy()
            assert_backbone_frozen(model, head=head, optimizer=opt)

            if state == "mitigation_level":
                # Mitigation does not enter p_delay at all. Its causal role is the
                # replenishment trigger (:1258-1259) and the order-quantity multiplier
                # (:1269), so it must be evaluated through THAT path -- scoring it through
                # p_delay would measure a channel the generator does not have.
                true_state = yte
                est_state = quantile_map(est, yte)
                p_true = np.array([scm.replenish_trigger(v) for v in true_state])
                p_est = np.array([scm.replenish_trigger(v) for v in est_state])
            elif state == "supply_stress":
                true_state = yte                              # true stress on the test rows
                est_state = quantile_map(est, yte)
                p_true = np.array([scm.p_delay(v) for v in true_state])
                p_est = np.array([scm.p_delay(v) for v in est_state])
            else:
                # resilience enters through absorption(); hold stress at its true value so the
                # comparison isolates THIS state's estimation error.
                st_true = np.asarray(targets["supply_stress"]["values"] and
                                     align(*emb_te[args.depth], targets["supply_stress"])[1])
                true_r = yte
                est_r = quantile_map(est, yte)
                n = min(len(st_true), len(true_r))
                p_true = np.array([min(0.80, 0.025 + 0.38 * st_true[i]
                                       * max(0.0, 1 - params.resilience_lambda * true_r[i]))
                                   for i in range(n)])
                p_est = np.array([min(0.80, 0.025 + 0.38 * st_true[i]
                                      * max(0.0, 1 - params.resilience_lambda * est_r[i]))
                                  for i in range(n)])

            corr = float(np.corrcoef(p_true, p_est)[0, 1])
            cmp_out[state] = {
                "head_auc_on_binarised_target": roc_auc(est, yte_b.astype(bool)),
                "mean_p_delay_true": float(p_true.mean()),
                "mean_p_delay_estimated": float(p_est.mean()),
                "mean_abs_p_delay_error": float(np.mean(np.abs(p_true - p_est))),
                "max_abs_p_delay_error": float(np.max(np.abs(p_true - p_est))),
                "pearson_r_true_vs_estimated": corr,
                "n_rows": int(len(p_true)),
            }
            c = cmp_out[state]
            print(f"  [3] {state:<21}: head AUC={c['head_auc_on_binarised_target']:.4f}  "
                  f"p_delay true={c['mean_p_delay_true']:.4f} est={c['mean_p_delay_estimated']:.4f}  "
                  f"MAE={c['mean_abs_p_delay_error']:.4f}  r={c['pearson_r_true_vs_estimated']:.4f}",
                  flush=True)

        results["per_seed"][str(dseed)] = {"equation_fidelity": eq, "outcome_fidelity": oc,
                                           "true_vs_estimated": cmp_out}

    # ---- aggregate ---------------------------------------------------------------
    per = results["per_seed"].values()
    agg = {
        "equation_exact_all_seeds": all(s["equation_fidelity"]["exact"] for s in per),
        "max_abs_err_stress_worst": max(s["equation_fidelity"]["max_abs_err_stress"] for s in per),
        "brier_mean": statistics.fmean(s["outcome_fidelity"]["brier"] for s in per),
        "mean_predicted": statistics.fmean(s["outcome_fidelity"]["mean_predicted"] for s in per),
        "observed_rate": statistics.fmean(s["outcome_fidelity"]["observed_rate"] for s in per),
        "max_calibration_gap_worst": max(s["outcome_fidelity"]["max_abs_calibration_gap"]
                                         for s in per),
        "auc_p_delay_vs_late_mean": statistics.fmean(
            s["outcome_fidelity"]["auc_p_delay_vs_actual_late"] for s in per),
    }
    for state in ("supply_stress", "recovery_capability", "mitigation_level"):
        vals = [s["true_vs_estimated"][state] for s in per if state in s["true_vs_estimated"]]
        if vals:
            agg[state] = {
                "mean_abs_p_delay_error": statistics.fmean(v["mean_abs_p_delay_error"] for v in vals),
                "max_abs_p_delay_error": max(v["max_abs_p_delay_error"] for v in vals),
                "pearson_r": statistics.fmean(v["pearson_r_true_vs_estimated"] for v in vals),
                "head_auc": statistics.fmean(v["head_auc_on_binarised_target"] for v in vals),
            }
    results["summary"] = agg

    print("\n" + "=" * 92)
    print(f"PHASE 2 SCM VALIDATION — variant {args.variant}, config {args.config}, "
          f"{len(seeds)} seeds, Layer 3 depth h^{args.depth}")
    print("=" * 92)
    print(f"  [1] equations exact on every seed      : {agg['equation_exact_all_seeds']}  "
          f"(worst max|err| = {agg['max_abs_err_stress_worst']:.3e})")
    print(f"  [2] p_delay vs realised outcomes       : Brier {agg['brier_mean']:.4f}, "
          f"predicted {agg['mean_predicted']:.4f} vs observed {agg['observed_rate']:.4f}, "
          f"worst bucket gap {agg['max_calibration_gap_worst']:.4f}")
    print(f"      AUC(p_delay vs actual late)        : {agg['auc_p_delay_vs_late_mean']:.4f}")
    OUTPUT = {"supply_stress": "p_delay", "recovery_capability": "p_delay",
              "mitigation_level": "replenish_trigger"}
    for state in ("supply_stress", "recovery_capability", "mitigation_level"):
        if state in agg:
            a = agg[state]
            print(f"  [3] {state:<21}: head AUC {a['head_auc']:.4f}  ->  "
                  f"{OUTPUT[state]} MAE {a['mean_abs_p_delay_error']:.4f} "
                  f"(max {a['max_abs_p_delay_error']:.4f}), r={a['pearson_r']:.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True, default=str)
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
