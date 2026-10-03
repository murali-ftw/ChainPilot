"""Phase 19 Stage 1 scorer -- Gate v2 (reports/part2/phase-19-preregistration.md). Torch process, no LightGBM.

Metric functions are Phase 18's (ml/eval/phase18_score.py), unchanged. Exact per-seed filenames, no globs.

Reuse checks before any verdict:
  * BASE + fwd_load is Phase 18's stored arm: re-scored here and asserted EQUAL to ml/artifacts/phase18/scores.json.
  * BASE + fwd_season carries exactly the values Phase 18's diagnostic fwd_load_netmean read: its predictions are
    asserted equal to that arm's (same seed, same fold).

Gate v2:
  fwd_season   PASS  if fwd_season > BASE on >= 1 primary metric, worse on none, and fwd_season_sperm > BASE on none
  fwd_load     PASS  (supplier-specific) if fwd_load > fwd_season on >= 1, worse than it on none, and fwd_load_sperm > BASE on none
  cadence      PASS  if cadence > BASE on >= 1, worse on none, and cadence_shuf > BASE on none
  capacity: >= 3 of 6 points better (and none worse). "better / worse" = DISJOINT 5-seed bands.

  python ml/eval/phase19_score.py        # -> ml/artifacts/phase19/gate_v2.json
"""
from __future__ import annotations
import os, json
import phase12_common as C
import numpy as np
import phase18_score as S18

PR19 = os.path.join(C.ART, "phase19", "preds")
SEEDS = C.V8_SEEDS


def fmt19(t, arm):
    return os.path.join(PR19, f"v8_{t}_p19_{arm}_s{{s}}_{{f}}.npz")


def fmt18(t, arm):
    return os.path.join(S18.PR18, f"v8_{t}_p18_{arm}_s{{s}}_{{f}}.npz")


def bands5(task, loader):
    """Phase 18's per-seed metrics, but a band exists only when ALL FIVE seeds have the metric (deviation 171).

    Phase 18's band() silently dropped seeds whose recall-at-p was UNREACHABLE on validation, so a 1-seed value could be
    compared as if it were a 5-seed band. Phase 15 (the metric's definition) calls a point UNREACHABLE unless every seed
    reaches the bar; the pre-registration says bands are over the 5 seeds. Such a point is now None -> 'n/a'."""
    per = [S18.metrics_for(task, loader(s), True) for s in SEEDS]
    out, unreach = {}, {}
    for k in per[0]:
        v = [p[k] for p in per]
        n_ok = sum(x is not None and np.isfinite(x) for x in v)
        out[k] = S18.band(v) if n_ok == len(SEEDS) else None
        if n_ok < len(SEEDS):
            unreach[k] = f"UNREACHABLE on validation for {len(SEEDS) - n_ok} of {len(SEEDS)} seeds"
    return out, per, unreach


def cmp_all(task, a, b):
    return {k: S18.compare(a[k], b[k], h) for k, h in S18.DIRECTION[task].items()}


def gate_v2(task, fam_vs, ctrl_vs_base):
    """fam_vs: comparison of the family arm against its reference; ctrl_vs_base: the control arm against BASE."""
    better = [k for k, v in fam_vs.items() if v == "better"]; worse = [k for k, v in fam_vs.items() if v == "worse"]
    ctrl_better = [k for k, v in ctrl_vs_base.items() if v == "better"]
    need = 3 if task == "capacity" else 1
    ok = len(better) >= need and not worse and not ctrl_better
    if ok:
        why = "better on " + ", ".join(better) + "; control not better than BASE anywhere"
    elif worse:
        why = "worse on " + ", ".join(worse)
    elif ctrl_better and len(better) >= need:
        why = "the CONTROL is also better than BASE on " + ", ".join(ctrl_better)
    elif better:
        why = f"only {len(better)} of 6 points better (3 needed)" if task == "capacity" else "?"
    else:
        why = "no primary metric disjointly better (bands overlap)"
    if ok is False and ctrl_better and not worse and len(better) < need:
        why += "; control also better than BASE on " + ", ".join(ctrl_better)
    return dict(verdict="PASS" if ok else "FAIL", why=why, family_vs_reference=fam_vs, control_vs_base=ctrl_vs_base)


def main():
    st = C.require_clean()
    p18 = json.load(open(os.path.join(C.ART, "phase18", "scores.json")))
    out = dict(stamp=st, arms={}, gate_v2={}, phase18_gate={k: v["verdict"] for k, v in p18["gate"].items()}, reuse_checks={})
    for task, t in S18.TASK.items():
        L = {"base": lambda s: S18.load_flat(os.path.join(S18.PR7, f"v8_{t}_{S18.STORED_LGBM[task]}_s{{s}}_{{f}}.npz"), s),
             "neural_incumbent": lambda s: S18.load_neural(task, s),
             "fwd_load": lambda s: S18.load_flat(fmt18(t, "fwd_load"), s)}
        for arm in ("fwd_season", "fwd_season_sperm", "fwd_load_sperm", "cadence", "cadence_shuf"):
            L[arm] = lambda s, arm=arm: S18.load_flat(fmt19(t, arm), s)
        A, UN = {}, {}
        for arm, ld in L.items():
            b, p, u = bands5(task, ld); A[arm] = (b, p); UN[arm] = u
        out.setdefault("unreachable", {})[task] = {a: u for a, u in UN.items() if u}
        # the as-coded reading (Phase 18's band, NaN seeds dropped), kept for the record (deviation 171)
        legacy = {a: S18.arm_bands(task, ld, with_extra=True)[0] for a, ld in L.items()}
        # ---- reuse checks
        stored = p18["arms"][task]["fwd_load"]["bands"]
        for k, v in legacy["fwd_load"].items():
            if v is not None and stored.get(k) is not None:
                assert np.allclose(v, stored[k], rtol=0, atol=1e-12), f"{task} fwd_load {k}: re-score {v} != Phase 18 {stored[k]}"
        eq = []
        for s in SEEDS:
            for f in ("val", "test"):
                a = np.load(fmt19(t, "fwd_season").format(s=s, f=f)); b = np.load(fmt18(t, "fwd_load_netmean").format(s=s, f=f))
                assert np.array_equal(a["Y"], b["Y"]), "rows differ"
                eq.append(float(np.abs(np.asarray(a["P"], float) - np.asarray(b["P"], float)).max()))
        out["reuse_checks"][task] = dict(fwd_load_rescored_equals_phase18=True,
                                         fwd_season_vs_phase18_netmean_max_abs_diff=max(eq))
        B = {a: b for a, (b, _) in A.items()}
        out["arms"][task] = {a: dict(bands=b, per_seed=p) for a, (b, p) in A.items()}
        out["gate_v2"][f"{task}|fwd_season"] = gate_v2(task, cmp_all(task, B["fwd_season"], B["base"]), cmp_all(task, B["fwd_season_sperm"], B["base"]))
        out["gate_v2"][f"{task}|fwd_load_supplier_specific"] = gate_v2(task, cmp_all(task, B["fwd_load"], B["fwd_season"]),
                                                                       cmp_all(task, B["fwd_load_sperm"], B["base"]))
        out["gate_v2"][f"{task}|cadence"] = gate_v2(task, cmp_all(task, B["cadence"], B["base"]), cmp_all(task, B["cadence_shuf"], B["base"]))
        Lg = legacy
        out.setdefault("gate_v2_as_coded_nan_seeds_dropped", {})[f"{task}|fwd_season"] = gate_v2(task, cmp_all(task, Lg["fwd_season"], Lg["base"]), cmp_all(task, Lg["fwd_season_sperm"], Lg["base"]))["verdict"]
        out["gate_v2_as_coded_nan_seeds_dropped"][f"{task}|fwd_load_supplier_specific"] = gate_v2(task, cmp_all(task, Lg["fwd_load"], Lg["fwd_season"]), cmp_all(task, Lg["fwd_load_sperm"], Lg["base"]))["verdict"]
        out["gate_v2_as_coded_nan_seeds_dropped"][f"{task}|cadence"] = gate_v2(task, cmp_all(task, Lg["cadence"], Lg["base"]), cmp_all(task, Lg["cadence_shuf"], Lg["base"]))["verdict"]
        out.setdefault("context", {})[task] = {"fwd_load vs base": cmp_all(task, B["fwd_load"], B["base"]),
                                               "fwd_load vs neural": cmp_all(task, B["fwd_load"], B["neural_incumbent"]),
                                               "fwd_season vs neural": cmp_all(task, B["fwd_season"], B["neural_incumbent"])}
    C.dump(out, "phase19/gate_v2.json")
    print(json.dumps(out["reuse_checks"], indent=1))
    for task in S18.TASK:
        print(f"== {task}")
        for a, d in out["arms"][task].items():
            print(f"  {a:18s}", {k: [round(x, 4) for x in v] for k, v in d["bands"].items()
                                  if v and k in {**S18.DIRECTION[task], **S18.EXTRA[task]}})
    print(json.dumps(out["unreachable"], indent=1))
    for k, v in out["gate_v2"].items():
        print(f"{k:42s} {v['verdict']:4s} (as coded: {out['gate_v2_as_coded_nan_seeds_dropped'][k]})  -- {v['why']}")


if __name__ == "__main__":
    main()
