"""Phase 24 Stage 3 -- metrics pack v2 and progress v2: the Phase 23AC pack with its two PENDING PHASE 23B rows filled.
New file; the Phase 23AC files are read, never edited.

  * Predict-the-rescue: the rescue models' stored predictions are NOT on this machine (ml/artifacts/phase17 and
    ml/artifacts/phase23b are absent), so B1a published, B1b published (2 of 5 seeds) and B1a clean are QUOTED from
    reports/part2/phase-17.md and phase-23b.md (+ phase23b/stage2_rescue_clean.md).
  * Shortage simulation: RECOMPUTED from Stage 2's regenerated rows (ml/artifacts/phase24/sim/{leaky,clean}) with Phase 23AC's
    own table code (phase23ac_t4_metrics: class_metrics, band_flat, Phase 15 summary, md_use_case), imported unchanged. Two views:
      SIM-RESCUE  the simulation as a rescue detector (label: a transfer-in recorded that week), Phase 17's operating point
      SIM-UC5A    the simulation as a below-safety-stock detector (observed reference), plus Q1 (pre-rescue ratio) and Q2 (missed)
    INCUMBENT = BEST PUBLISHED = the simulation on its published (leaky) heads; CLEAN = on Phase 22's clean heads.
Writes reports/part2/phase24/metrics_pack_v2.md / .csv, progress_v2.md and charts/data/rescue_published_vs_clean.csv.

    python ml/eval/phase24_metrics_v2.py
"""
from __future__ import annotations
import os, sys, json, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "sim"), os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "data")]
import numpy as np, pandas as pd
import phase12_common as C
import phase23ac_t4_metrics as T4

SEEDS = (7, 17, 27, 37, 47)
ART24 = os.path.join(C.ART, "phase24")
SIM24 = os.path.join(ART24, "sim")
OUT = os.path.join(C.REPO, "reports", "part2", "phase24")
# the Phase 23AC pack and progress are read from the main checkout's tracked copies at the base commit (identical here)
P23AC = os.path.join(C.REPO, "reports", "part2", "phase23ac", "t4")
P23AC_CSV = os.environ.get("P23AC_CSV", os.path.join(P23AC, "metrics_pack.csv"))    # gitignored: read from where it was written
Q17 = "reports/part2/phase-17.md"
Q23B = "reports/part2/phase-23b.md"
Q23B2 = "reports/part2/phase23b/stage2_rescue_clean.md"


def q(v, lo=None, hi=None, n=1, iv="as quoted"):
    return dict(v=v, lo=lo, hi=hi, n=n, interval=iv)


def quoted_rescue_arms():
    """QUOTED rows for predict-the-rescue (its predictions are on the other machine)."""
    A = []
    pub = {"op.precision": q(0.823, 0.822, 0.824, 5, "5-seed mean [min, max]"), "op.recall": q(0.246), "base_rate": q(0.4535),
           "op.lift": q(1.81), "cov.1%.precision": q(0.938), "cov.5%.precision": q(0.883), "cov.10%.precision": q(0.842),
           "cov.20%.precision": q(0.788), "pr_auc": q(0.729, 0.7285, 0.7289, 5, "5-seed mean [min, max]"), "roc_auc": q(0.772),
           "ens.op.precision": q(0.825, 0.791, 0.854, 1, "5-seed ensemble, snapshot-block 95%"),
           "spread.op": dict(s="0.738 / 0.813 / 0.884"), "phase15.cls": dict(s="ALERT"),
           "phase15.reachable": dict(s="PARTIAL @ 0.85, lift 1.84")}
    A.append(T4.arm("b1a_published", "RESCUE", "INCUMBENT", "headline", "B1a LightGBM (Phase 17), 5 seeds", "quoted", True,
                    [Q17, Q23B2], pub))
    b1b = {"op.precision": q(0.859, 0.858, 0.859, 2, "2-seed mean [min, max] (seeds 7, 17 only)"), "op.recall": q(0.238),
           "base_rate": q(0.4535), "cov.1%.precision": q(0.947), "cov.5%.precision": q(0.909), "cov.10%.precision": q(0.874),
           "pr_auc": q(0.757, 0.7567, 0.7577, 2, "2-seed mean [min, max]"), "roc_auc": q(0.797),
           "phase15.cls": dict(s="not classified (2 of 5 seeds; Phase 17 STOPPED at the epoch cap)")}
    A.append(T4.arm("b1b_published", "RESCUE", "BEST PUBLISHED", "headline", "B1b neural (Phase 17), 2 of 5 seeds, capped (floor)",
                    "quoted", True, [Q17], b1b))
    cln = {"op.precision": q(0.733, 0.7325, 0.7336, 5, "5-seed mean [min, max]"), "op.recall": q(0.254), "base_rate": q(0.4535),
           "op.lift": q(1.62), "cov.1%.precision": q(0.881), "cov.5%.precision": q(0.812), "cov.10%.precision": q(0.772),
           "cov.20%.precision": q(0.711), "ens.op.precision": q(0.733, 0.697, 0.773, 1, "5-seed ensemble, snapshot-block 95%"),
           "spread.op": dict(s="0.653 / 0.728 / 0.840"), "phase15.cls": dict(s="ALERT"),
           "phase15.reachable": dict(s="PARTIAL @ 0.80, lift 1.76")}
    A.append(T4.arm("b1a_clean", "RESCUE", "CLEAN", "headline", "B1a LightGBM on v8clean (Phase 23B), 5 seeds", "quoted", False,
                    [Q23B, Q23B2], cln))
    return A


def snap_dates(arm):
    log = json.load(open(os.path.join(SIM24, arm, "run_log.json")))
    d = [g["t0"] for g in log["gate"] if g["seed"] == 7]
    assert len(d) == 9
    return np.array([np.datetime64(x) for x in d])


def sim_pairs(arm, label):
    import phase24_sim_clean as S2
    Z = S2.load_arm(arm)                                    # G4: all five seeds
    dates = snap_dates(arm)
    return [(Z[("val", s)]["p"], Z[("val", s)][label].astype(int), Z[("test", s)]["p"], Z[("test", s)][label].astype(int),
             dates[Z[("test", s)]["snap"].astype(int)]) for s in SEEDS], Z


def sim_arms():
    import phase24_sim_clean as S2
    st2 = json.load(open(os.path.join(ART24, "stage2_sim.json")))
    A = []
    for arm, cols, leaky in (("leaky", ("INCUMBENT", "BEST PUBLISHED"), True), ("clean", ("CLEAN",), False)):
        src = [f"ml/artifacts/phase24/sim/{arm}/test_s{{s}}.npz", "ml/artifacts/phase24/stage2_sim.json"]
        lab = "simulation on its published heads (leaky)" if leaky else "simulation on Phase 22's clean heads"
        # rescue view: Phase 17's operating point per seed
        pr, Z = sim_pairs(arm, "acted")
        q3 = st2["arms"][arm]["Q3"]["per_seed"]
        extra = [dict(**{"op.precision": x["test_precision"], "op.recall": x["test_recall"], "op.share_flagged": x["test_coverage"],
                         "op.lift": x["test_precision"] / Z[("test", s)]["acted"].mean()}) for x, s in zip(q3, SEEDS)]
        # uc5a view + Q1 / Q2 per seed
        pu, _ = sim_pairs(arm, "obs")
        q1 = st2["arms"][arm]["Q1"]["per_seed"]; q2 = st2["arms"][arm]["Q2"]["per_seed"]; pk = st2["arms"][arm]["Q4"]["per_seed_peak"]
        extra_u = [{"q1.ratio": q1[str(s)], "q2.missed_share": m["missed_share"], "q2.rescued": m["rescued_weeks"], "q2.flagged": m["flagged"],
                    "q4.peak_precision": p} for s, m, p in zip(SEEDS, q2, pk)]
        for c in cols:
            A.append(T4.seed_class_arm(f"sim_rescue_{arm}", "SIM-RESCUE", c, "headline", lab, pr, leaky, src, extra))
            A.append(T4.seed_class_arm(f"sim_uc5a_{arm}", "SIM-UC5A", c, "headline", lab, pu, leaky, src, extra_u))
    return A


def register_display():
    op_rows = [("op.precision", "**precision at the Phase 17 operating point** (val recall ≥ 0.20)"), ("op.recall", "test recall there"),
               ("op.lift", "lift there"), ("op.share_flagged", "share flagged there"),
               ("ens.op.precision", "5-seed ensemble at its operating point (block 95%)"),
               ("spread.op", "per-snapshot precision at the operating point: min / median / max")]
    T4.DISPLAY["RESCUE"] = op_rows[:3] + [("base_rate", "base rate (a transfer-in that week)")] + \
        [(f"cov.{c}.precision", f"precision @ {c} coverage") for c in ("1%", "5%", "10%", "20%")] + \
        [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC")] + op_rows[4:] + [("phase15.cls", "**Phase 15 class**"), ("phase15.reachable", "REACHABLE")]
    T4.DISPLAY["SIM-RESCUE"] = op_rows[:4] + T4.class_display()
    T4.DISPLAY["SIM-UC5A"] = [("q1.ratio", "**Q1 pre-rescue match ratio** (sim below-SS ÷ PRIVILEGED pre-rescue reference)"),
                              ("q2.missed_share", "**Q2 share of rescued weeks the simulation missed**"),
                              ("q2.rescued", "rescued part-plant-weeks"), ("q2.flagged", "of which flagged at the val max-F1 τ"),
                              ("q4.peak_precision", "**Q4 peak test precision** (any coverage ≥ 50 alerts; the 'ceiling')")] + T4.class_display()
    T4.UC_TITLE.update({"RESCUE": "Predict-the-rescue (base 0.4535): QUOTED, predictions on the other machine",
                        "SIM-RESCUE": "Shortage simulation as a rescue detector (the 'simulation proxy'; base 0.4535)",
                        "SIM-UC5A": "Shortage simulation as a below-safety-stock detector, and its pre-rescue match (base 0.080)"})
    T4.HEADLINE.update({"RESCUE": ("op.precision", True), "SIM-RESCUE": ("op.precision", True), "SIM-UC5A": ("q1.ratio", False)})


def rewrite_pack(text, new_sections, stamp):
    head = ["# Phase 24: metrics pack v2 (Phase 23AC T4.2 with the two PENDING PHASE 23B rows filled)", "",
            "**What changed from `reports/part2/phase23ac/t4/metrics_pack.md` (not edited):**",
            "- the *Predict-the-rescue* row: clean B1a 0.733 QUOTED from `reports/part2/phase-23b.md` (B1a published 0.823 and B1b",
            "  published 0.859 (2 of 5 seeds) beside it, LEAKY); new section **Predict-the-rescue** below;",
            "- the *Shortage simulation* row: RESTATED on clean heads by Phase 24 Stage 2; new sections **Shortage simulation as a rescue",
            "  detector** and **… as a below-safety-stock detector** below (RECOMPUTED, same table layout);",
            f"- everything else is the Phase 23AC pack verbatim. Code: `ml/eval/phase24_metrics_v2.py` at `{stamp['code_commit']}`, "
            f"run {stamp['run_ts']}.", ""]
    lines = text.splitlines()
    assert lines[0].startswith("# Phase 23AC T4.2")
    body = "\n".join(lines[1:])
    old_r = [l for l in lines if l.startswith("| Predict-the-rescue |")]
    old_s = [l for l in lines if l.startswith("| Shortage simulation |")]
    assert len(old_r) == 1 and len(old_s) == 1, "the two PENDING rows were not found exactly once"
    st2 = json.load(open(os.path.join(ART24, "stage2_sim.json")))
    b = st2["block"]
    new_r = ("| Predict-the-rescue | clean B1a **0.733** [0.7325, 0.7336] precision at recall 0.254, base 0.4535 (block [0.697, 0.773]); "
             "published 0.823 LEAKY (superseded, Phase 22) | **RESTATED (Phase 23B): still GO, ALERT (PARTIAL @ 0.80), marginal under "
             "block resampling** | clean | `reports/part2/phase-23b.md` §1 (535361d), QUOTED |")
    new_s = (f"| Shortage simulation | clean: pre-rescue ratio **{b['q1']['clean']:.3f}x** [{b['q1']['clean_ci'][0]:.3f}, {b['q1']['clean_ci'][1]:.3f}]; "
             f"rescue-detection precision **{b['q3_precision']['clean']:.3f}** at recall {b['q3_recall']['clean']:.3f}; "
             f"rescued weeks missed **{b['q2_missed']['clean']:.1%}**; below-SS ceiling {st2['arms']['clean']['Q4']['max_test_precision']:.3f} | "
             "**RESTATED (Phase 24 Stage 2)** | clean | `reports/part2/phase24/stage2_simulation.md`, RECOMPUTED |")
    body = body.replace(old_r[0], new_r).replace(old_s[0], new_s)
    return "\n".join(head) + body.rstrip() + "\n\n" + "\n".join(new_sections) + "\n"


def rewrite_progress(text, stamp):
    st2 = json.load(open(os.path.join(ART24, "stage2_sim.json"))); b = st2["block"]
    lines = text.splitlines()
    out = ["# Phase 24: progress v2 (Phase 23AC progress.md with the two PENDING PHASE 23B rows filled)", "",
           f"Changed rows only: *Predict-the-rescue* and *Shortage simulation* (and the matching summary line). Everything else is "
           f"`reports/part2/phase23ac/t4/progress.md` verbatim. Code `ml/eval/phase24_metrics_v2.py` at `{stamp['code_commit']}`.", ""]
    rep = {"- **Predict-the-rescue:**": ("- **Predict-the-rescue:** RESTATED by Phase 23B. Clean B1a **0.733** [0.7325, 0.7336] precision at recall "
                                         "0.254 against a 0.4535 base (lift 1.62), still GO, ALERT (PARTIAL @ 0.80) but marginal under block "
                                         "resampling ([0.697, 0.773]); published 0.823 was LEAKY (superseded, Phase 22) (QUOTED; "
                                         "`reports/part2/phase-23b.md` §1)."),
           "- **Shortage simulation:**": (f"- **Shortage simulation:** RESTATED by Phase 24 on clean heads. Pre-rescue ratio "
                                          f"**{b['q1']['clean']:.3f}x** (was {b['q1']['leaky']:.3f}x); rescue-detection precision "
                                          f"**{b['q3_precision']['clean']:.3f}** (was {b['q3_precision']['leaky']:.3f}); rescued weeks missed "
                                          f"**{b['q2_missed']['clean']:.1%}** (was {b['q2_missed']['leaky']:.1%}) (RECOMPUTED; "
                                          "`reports/part2/phase24/stage2_simulation.md`)."),
           "9. Predict-the-rescue and the shortage simulation: PENDING PHASE 23B.": (
               "9. Predict-the-rescue (clean 0.733, still GO, marginal) and the shortage simulation (restated on clean heads, Phase 24) are no "
               "longer pending. MILP, allocation and transfer are blocked on data.")}
    n = 0
    for l in lines[1:]:
        hit = [k for k in rep if l.startswith(k)]
        if hit:
            out.append(rep[hit[0]]); n += 1
        else:
            out.append(l)
    assert n == 3, f"expected 3 rows to replace in progress.md, replaced {n}"
    return "\n".join(out) + "\n"


def chart_rows():
    st2 = json.load(open(os.path.join(ART24, "stage2_sim.json"))); b = st2["block"]
    rows = [dict(arm="Simulation proxy\n(leaky heads)", value=b["q3_precision"]["leaky"], lo=b["q3_precision"]["leaky_ci"][0],
                 hi=b["q3_precision"]["leaky_ci"][1], leaky=True, status="RECOMPUTED", source="phase24/stage2_sim.json"),
            dict(arm="Simulation proxy\n(clean heads)", value=b["q3_precision"]["clean"], lo=b["q3_precision"]["clean_ci"][0],
                 hi=b["q3_precision"]["clean_ci"][1], leaky=False, status="RECOMPUTED", source="phase24/stage2_sim.json"),
            dict(arm="Own history\n(52-wk rate)", value=0.631, lo=0.583, hi=0.675, leaky=False, status="QUOTED", source=Q23B2),
            dict(arm="B1a published\n(leaky)", value=0.825, lo=0.791, hi=0.854, leaky=True, status="QUOTED", source=Q23B2),
            dict(arm="B1a clean", value=0.733, lo=0.697, hi=0.773, leaky=False, status="QUOTED", source=Q23B2)]
    return pd.DataFrame(rows).assign(base_rate=0.4535)


def main():
    st = C.require_clean()
    T4.register_worlds()
    register_display()
    arms = quoted_rescue_arms() + sim_arms()
    T = T4.populate(arms)
    sec = []
    for uc in ("RESCUE", "SIM-RESCUE", "SIM-UC5A"):
        sec += T4.md_use_case(arms, uc)
    sec += ["Interval kinds: simulation arms are 5 fill seeds, mean [min, max]. The paired snapshot-block intervals of the clean − leaky",
            "differences are in `reports/part2/phase24/stage2_simulation.md`. The ensemble rows for predict-the-rescue are QUOTED.", ""]
    os.makedirs(os.path.join(OUT, "charts", "data"), exist_ok=True)
    pack = rewrite_pack(open(os.path.join(P23AC, "metrics_pack.md")).read(), sec, st)
    open(os.path.join(OUT, "metrics_pack_v2.md"), "w").write(pack)
    old = pd.read_csv(P23AC_CSV) if os.path.exists(P23AC_CSV) else None
    new = T.frame()
    if old is not None:
        old = old[~old["use_case"].isin(["RESCUE", "SIM-RESCUE", "SIM-UC5A"])]
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df.to_csv(os.path.join(OUT, "metrics_pack_v2.csv"), index=False)
    open(os.path.join(OUT, "progress_v2.md"), "w").write(rewrite_progress(open(os.path.join(P23AC, "progress.md")).read(), st))
    chart_rows().to_csv(os.path.join(OUT, "charts", "data", "rescue_published_vs_clean.csv"), index=False)
    C.dump(dict(stamp=st, p23ac_csv=P23AC_CSV, p23ac_csv_rows=None if old is None else int(len(old)), new_rows=int(len(new)),
                written=["metrics_pack_v2.md", "metrics_pack_v2.csv", "progress_v2.md", "charts/data/rescue_published_vs_clean.csv"]),
           "phase24/metrics_v2.json")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
