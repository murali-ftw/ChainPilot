#!/usr/bin/env python3
"""
Layer 3 v2, PHASE 5 -- evidence status, and the three-field output spec per task.

Every task that reaches this phase gets exactly one terminal evidence-status label. This is the
second of the three output fields; it is a categorical summary of **which gates the task
cleared**, kept strictly separate from both the probability (Phase 3) and the confidence band
(Phase 4).

| Status | Condition |
|---|---|
| **Supported**      | Passed Phase 0, cleared `floor_t` on both seed axes in Phase 2, calibration transferred in Phase 3 |
| **Uncertain**      | Passed Phase 0 and Phase 2, but calibration did not clearly transfer (mixed folds, not a majority failure) |
| **Unidentifiable** | Closed at Phase 0, or failed Phase 2's floor outright |

**Two things about that table are worth stating rather than leaving for the reader to trip over.**

1. *"Unidentifiable" is doing double duty.* It covers both "closed at Phase 0" (genuinely no
   measurable observable footprint) and "failed Phase 2's floor" (a real footprint, but this
   encoder did not convert it into downstream value). Those are different findings with different
   follow-ups, so `phase0_outcome` and `phase2_outcome` are carried alongside the label and the
   report distinguishes them in prose. The label follows the specified table; the detail is not
   thrown away.

2. *Calibration transfer is operationalised explicitly*, because "did not clearly transfer" needs
   a rule that was fixed before the numbers were read:

       transferred      > 50% of ordered folds improve ECE **and** the mean improvement clears
                        the member floor
       mixed            >= 50% of folds improve but the improvement does not clear its floor
       majority failure < 50% of folds improve                              -> STOP-G

**Layer 4 wiring**, per the build's own scope rules: a `Supported` task's `Z_t` may be consumed by
its risk head; an `Uncertain` task's output may be surfaced for observability but **not** silently
fed into a risk head; an `Unidentifiable` task falls back to the existing Layer-2-only
(SHARE + Markov) path unchanged.

    python3 ml/task_evidence_status.py --variant A --out out/layer3_v2/phase5_status_A.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

TASKS = ("delay", "shortage", "impact")
WIRING = {
    "Supported": "Z_t may be consumed by this task's Layer-4 risk head",
    "Uncertain": ("surface for observability only — must NOT be silently fed into a risk head"),
    "Unidentifiable": ("fall back to the existing Layer-2-only (SHARE + Markov) path, unchanged"),
}


def calibration_verdict(folds: list, member_floor: float) -> dict:
    """`transferred` / `mixed` / `majority_failure`, by the rule fixed in the docstring."""
    if not folds:
        return {"verdict": "not_run", "n_folds": 0}
    impr = [f["ece_ens_raw"] - f["ece_ens_cal"] for f in folds]
    n_up = sum(1 for v in impr if v > 0)
    frac = n_up / len(impr)
    mean = statistics.fmean(impr)
    clears = mean > member_floor if member_floor == member_floor else False
    if frac > 0.5 and clears:
        v = "transferred"
    elif frac >= 0.5:
        v = "mixed"
    else:
        v = "majority_failure"
    return {"verdict": v, "n_folds": len(impr), "folds_improved": n_up,
            "fraction_improved": frac, "mean_ece_improvement": mean,
            "member_floor": member_floor, "clears_member_floor": bool(clears),
            "ece_raw": statistics.fmean([f["ece_ens_raw"] for f in folds]),
            "ece_cal": statistics.fmean([f["ece_ens_cal"] for f in folds]),
            "stop_code": "STOP-G" if v == "majority_failure" else None}


def assign(task: str, p0: dict, p2: dict, p3: dict, p4: dict) -> dict:
    """One terminal label per task, derived from the gates rather than chosen."""
    # -- Phase 0 -------------------------------------------------------
    tiers = p0.get("summary", {}).get(task, {})
    if not tiers:
        phase0 = {"passed": False, "tiers_run": [],
                  "reason": p0.get("_closed_reason", {}).get(task, "gate not runnable")}
    else:
        ident = {t: bool(v["identifiable"]) for t, v in tiers.items()}
        phase0 = {"passed": any(ident.values()), "tiers_run": sorted(tiers),
                  "identifiable_by_tier": ident,
                  "residual_auc_by_tier": {t: v["residual_auc_mean"] for t, v in tiers.items()},
                  "null_auc_by_tier": {t: v["null_auc_mean"] for t, v in tiers.items()},
                  "floor_by_tier": {t: v["reproduction_floor"] for t, v in tiers.items()}}

    # -- Phase 2 -------------------------------------------------------
    s2 = p2.get("summary", {}).get(task, {})
    d = s2.get("delta")
    phase2 = {"passed": bool(s2.get("gate_pass")),
              "delta": d.get("value") if d else None,
              "init_seed_floor": d.get("init_seed_floor") if d else None,
              "dataset_seed_floor": d.get("dataset_seed_floor") if d else None,
              "clears_init_axis": d.get("clears_init_axis") if d else None,
              "clears_dataset_axis": d.get("clears_dataset_axis") if d else None,
              "sign_consistent": d.get("sign_consistent") if d else None,
              "baseline_auc": s2.get("baseline", {}).get("auc_mean"),
              "layer3_auc": s2.get("layer3", {}).get("auc_mean"),
              "noise_control_auc": s2.get("noise", {}).get("auc_mean"),
              "delta_minus_noise": s2.get("delta_minus_noise"),
              "beats_width_control": s2.get("beats_width_control")}

    # -- Phase 3 -------------------------------------------------------
    folds = [f for f in p3.get("folds", []) if f["task"] == task]
    mf = p3.get("floors", {}).get(task, {}).get("member_floor", {}).get("max_abs_dev", float("nan"))
    phase3 = calibration_verdict(folds, mf)

    # -- Phase 4 -------------------------------------------------------
    c = p4.get("per_task", {}).get(task, {})
    phase4 = {"accepted": bool(c.get("accepted")), "reasons": c.get("reasons", []),
              "worlds_monotone": c.get("worlds_monotone"), "n_worlds": c.get("n_worlds"),
              "cross_world_folds_monotone": c.get("cross_world_folds_monotone"),
              "n_folds": c.get("n_folds"),
              "bands": c.get("bands")}

    # -- the label -----------------------------------------------------
    if not phase0["passed"] or not phase2["passed"]:
        status = "Unidentifiable"
    elif phase3["verdict"] == "transferred":
        status = "Supported"
    elif phase3["verdict"] == "mixed":
        status = "Uncertain"
    else:
        status = "Uncertain"          # majority failure: STOP-G, still past Phases 0 and 2

    why = []
    if not phase0["passed"]:
        # Two very different Phase-0 closures share this branch, and conflating them would
        # misreport the finding: a gate that RAN and measured no above-null footprint, versus a
        # gate that could not be run at all on this generator.
        why.append(phase0.get("reason")
                   or "closed at Phase 0 — no reproducible above-null residual footprint")
    elif not phase2["passed"]:
        why.append(
            f"passed Phase 0 but failed Phase 2's floor: delta {phase2['delta']:+.4f} against "
            f"init-seed floor {phase2['init_seed_floor']:.4f} / dataset-seed floor "
            f"{phase2['dataset_seed_floor']:.4f}")
    elif phase3["verdict"] != "transferred":
        why.append(f"calibration {phase3['verdict']} "
                   f"({phase3['folds_improved']}/{phase3['n_folds']} folds improved)")

    return {"task": task, "status": status, "why": why,
            "phase0": phase0, "phase2": phase2, "phase3": phase3, "phase4": phase4,
            "output_spec": {
                # Phase 3 calibrated the P(Y|X,H,Z) head. Where Phase 2 found delta ~ 0 that
                # head is empirically the P(Y|X,H) baseline, so the calibrated probability is a
                # property of the LAYER-2 pathway and must not be presented as something Z_t
                # delivered.
                "prediction": (
                    ("calibrated P(Y_t=1) from Phase 3 — attributable to the Layer-2 (X,H) "
                     "pathway, not to Z_t, since Phase 2 found delta inside its floor"
                     if not phase2["passed"] else "calibrated P(Y_t=1) from Phase 3")
                    if phase3["verdict"] == "transferred"
                    else "UNCALIBRATED P(Y_t=1) — the cross-world map did not transfer, so no "
                         "calibrated probability is offered"),
                "evidence_status": status,
                "confidence": (f"bands {phase4['bands']}" if phase4["accepted"]
                               else "ABSENT — signal failed its own validation and is not "
                                    "backfilled"),
            },
            "layer4_wiring": WIRING[status]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="A")
    ap.add_argument("--tasks", default="delay,shortage,impact")
    ap.add_argument("--root", default=os.path.join(REPO, "out", "layer3_v2"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]

    def load(name, default=None):
        p = os.path.join(a.root, name)
        return json.load(open(p)) if os.path.exists(p) else (default or {})

    p0 = load(f"phase0_gate_{a.variant}.json")
    p2 = load(f"phase2_{a.variant}.json")
    p3 = load(f"phase3_calibration_{a.variant}.json")
    p4 = load(f"phase4_confidence_{a.variant}.json")
    # `shortage` never enters the Phase 0 summary: its gate is not computable under CRN.
    p0.setdefault("_closed_reason", {})["shortage"] = (
        "gate not computable under the CRN protocol — the weekly inventory walk CREATES "
        "shipments as a function of the intervened quantity, so the entity set is not pinnable "
        "(measured: see phase0 shortage_feasibility)")

    res = {"variant": a.variant, "per_task": {t: assign(t, p0, p2, p3, p4) for t in tasks},
           "shortage_feasibility": p0.get("shortage_feasibility")}

    print("\n" + "=" * 122)
    print(f"PHASE 5 — evidence status, variant {a.variant}")
    print("=" * 122)
    hdr = (f"{'task':<10}{'Phase 0':>10}{'Phase 2':>10}{'Phase 3':>18}{'Phase 4':>11}"
           f"{'STATUS':>17}")
    print(hdr); print("-" * len(hdr))
    for t in tasks:
        c = res["per_task"][t]
        print(f"{t:<10}{('pass' if c['phase0']['passed'] else 'CLOSED'):>10}"
              f"{('pass' if c['phase2']['passed'] else 'STOP'):>10}"
              f"{c['phase3']['verdict']:>18}"
              f"{('accepted' if c['phase4']['accepted'] else 'REJECTED'):>11}"
              f"{c['status']:>17}")
    print("-" * len(hdr))
    for t in tasks:
        c = res["per_task"][t]
        print(f"\n{t} -> {c['status']}")
        for w in c["why"]:
            print(f"  - {w}")
        print(f"  Prediction     : {c['output_spec']['prediction']}")
        print(f"  Evidence status: {c['output_spec']['evidence_status']}")
        print(f"  Confidence     : {c['output_spec']['confidence']}")
        print(f"  Layer 4 wiring : {c['layer4_wiring']}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
