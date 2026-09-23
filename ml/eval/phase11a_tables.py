"""Phase 11A — the comparison tables: heads vs baselines, and the three graph arms.

Two sources, joined here and nowhere else:
  ml/artifacts/phase7_scores.json      baselines, scored under the HEAD'S OWN recalibration
                                       protocol (loop.fit_recalibration, fitted on validation)
  <scratch>/phase1_v8_scores.json      the heads, from ml/eval/phase1_v8_score.py

DISJOINTNESS IS REQUIRED FOR ANY CLAIM OF DIFFERENCE. Two arms are called different only when
their [min, max] seed bands do not overlap. Anything else is reported as NOT DISTINGUISHABLE,
which is a finding, not a failure to find one.

LightGBM is deterministic at a fixed seed, so its five fits differ only through the seed passed
to the sampler. A near-zero spread there is the expected result and is reported as such.

ARRIVAL: `promise_only` is carried under PRIVILEGED__ and is NEVER a ranking comparison. The
retirement in reports/part2/phase-0-1-v8.md S3.1 stands -- the promise date belongs to a line
raised ~7 weeks after the forecast instant.
"""
from __future__ import annotations
import json, sys, os, re
import numpy as np

HIGHER = {"cindex", "roc_auc_late", "roc_auc", "pr_auc", "coverage80", "rel_one"}


def band(vals):
    v = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    if not v:
        return None
    return dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)),
                sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, n=len(v))


def disjoint(a, b):
    """True when the two bands do not overlap at all."""
    if not a or not b:
        return None
    return bool(a["max"] < b["min"] or b["max"] < a["min"])


def verdict(a, b, metric, name_a="A", name_b="B"):
    """-> (winner, disjoint?) with direction taken from the metric."""
    if not a or not b:
        return ("insufficient", None)
    d = disjoint(a, b)
    hi = metric in HIGHER
    better = name_a if ((a["mean"] > b["mean"]) == hi) else name_b
    return (better if d else "NOT DISTINGUISHABLE", d)


def group(scores, prefix, metric):
    """Collect a metric across the seed variants of one baseline family."""
    out, names = [], []
    for k, v in scores.items():
        if not k.startswith(prefix):
            continue
        m = v.get(metric)
        if isinstance(m, (list, tuple)):
            m = m[0]
        if isinstance(m, (int, float)):
            out.append(float(m)); names.append(k)
    return band(out), names


def fam(scores, world, task, base):
    """All entries for one baseline family, e.g. v8|fill_rate|b5flat22_s*."""
    pat = re.compile(rf"^{re.escape(world)}\|{re.escape(task)}\|{re.escape(base)}(_s\d+)?$")
    return {k: v for k, v in scores.items() if pat.match(k)}


def famband(scores, world, task, base, metric):
    sub = fam(scores, world, task, base)
    vals = []
    for v in sub.values():
        m = v.get(metric)
        if isinstance(m, (list, tuple)):
            m = m[0]
        if isinstance(m, (int, float)):
            vals.append(float(m))
    return band(vals), len(sub)


def main():
    scratch = sys.argv[1] if len(sys.argv) > 1 else "."
    B = json.load(open("ml/artifacts/phase7_scores.json"))["scores"]
    H = json.load(open(os.path.join(scratch, "phase1_v8_scores.json")))
    heads = H["bands"]
    R = {"stage2": {}, "notes": []}

    SPEC = {
        "arrival_week": dict(
            head="arrival_week|lite|h4|real",
            metrics=["cindex", "roc_auc_late"],
            baselines=["b5flat_reg", "lgbm_id_reg", "naive_channel_median",
                       "naive_global_median", "naive_lane_median"],
            privileged=["promise_only"]),
        "fill_rate": dict(
            head="fill_rate|none|h0|h0",
            metrics=["crps_exact", "ece22"],
            head_metric={"crps_exact": "crps_exact_recal", "ece22": "ece22_recal"},
            baselines=["recal_b5flat22", "recal_lgbm22_id", "b5flat22", "lgbm22_id",
                       "b2_rolling52_cdf", "naive_channel_cdf", "naive_global_cdf"],
            privileged=[]),
        "capacity_strain": dict(
            head="capacity_strain|mp|h4|real",
            metrics=["pinball_mean", "coverage80"],
            baselines=["b5flat_q", "naive_supplier", "naive_channel", "naive_global"],
            privileged=[]),
        "shortage_qty": dict(
            head="shortage_qty|mp|h1|real",
            metrics=["roc_auc", "pr_auc"],
            baselines=["b5flat_bin", "naive_part_plant_rate", "naive_global_rate"],
            privileged=[]),
    }

    for task, sp in SPEC.items():
        hb = heads.get(sp["head"])
        if hb is None:
            R["notes"].append(f"{task}: head band {sp['head']} not found; scored cells only")
            continue
        rows = []
        for metric in sp["metrics"]:
            hm = sp.get("head_metric", {}).get(metric, metric)
            head_band = hb.get(hm)
            entry = dict(metric=metric, head_metric=hm, head=head_band, baselines={}, privileged={})
            for bname in sp["baselines"]:
                bb, n = famband(B, "v8", task, bname, metric)
                if bb is None:
                    continue
                w, d = verdict(head_band, bb, metric, "HEAD", bname)
                entry["baselines"][bname] = dict(band=bb, n_fits=n, winner=w, disjoint=d)
            for pname in sp["privileged"]:
                pb, n = famband(B, "v8", task, pname, metric)
                if pb is not None:
                    entry["privileged"]["PRIVILEGED__" + pname] = dict(
                        band=pb, n_fits=n,
                        note="NOT a fair comparison: reads a line raised ~7 weeks after t0")
            rows.append(entry)
        R["stage2"][task] = rows

    json.dump(R, open(os.path.join(scratch, "phase11a_stage2.json"), "w"), indent=1, default=str)

    # ------------------------------------------------------------------ print
    for task, rows in R["stage2"].items():
        print(f"\n{'=' * 92}\n{task}\n{'=' * 92}")
        for e in rows:
            h = e["head"]
            arrow = "higher better" if e["metric"] in HIGHER else "lower better"
            print(f"\n  {e['metric']}  ({arrow})   HEAD[{e['head_metric']}] = "
                  f"{h['mean']:.5f} [{h['min']:.5f}, {h['max']:.5f}] n={h.get('n', h.get('n_seeds','?'))}")
            for bn, bv in sorted(e["baselines"].items(),
                                 key=lambda kv: -kv[1]["band"]["mean"] if e["metric"] in HIGHER
                                 else kv[1]["band"]["mean"]):
                b = bv["band"]
                mark = {"HEAD": "head wins", "NOT DISTINGUISHABLE": "NOT DISTINGUISHABLE"}.get(
                    bv["winner"], f"** {bv['winner']} WINS **")
                print(f"      {bn:24s} {b['mean']:.5f} [{b['min']:.5f}, {b['max']:.5f}] "
                      f"n={b['n']:<2} sd={b['sd']:.5f}   {mark}")
            for pn, pv in e["privileged"].items():
                b = pv["band"]
                print(f"      {pn:24s} {b['mean']:.5f} [{b['min']:.5f}, {b['max']:.5f}]   "
                      f"(privileged, not a comparison)")
    print(f"\njson -> {os.path.join(scratch, 'phase11a_stage2.json')}")


if __name__ == "__main__":
    main()
