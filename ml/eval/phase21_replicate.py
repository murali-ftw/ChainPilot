"""Phase 21 Stage 7 -- replication verdicts on the second world (pre-registration D14). Reads the two worlds' score files;
no fitting, no torch-side scoring of its own beyond what phase21_score.py stored. Each world against ITS OWN BASE band.

Per arm and primary metric: gain (mean arm - mean BASE, better = positive), whether the world's bands are disjoint, and
  REPLICATES     same sign as v8-1001 and disjoint against world 2's own BASE
  DOES NOT       disjoint in the OPPOSITE direction on world 2
  UNDETERMINED   world 2's bands overlap
and the ratio of gains (world 2 / v8-1001). Deterministic standalone rules: their block-bootstrap difference vs the BASE
seed ensemble on each world.

  python ml/eval/phase21_replicate.py      -> ml/artifacts/phase21/stage7_replication.json
"""
from __future__ import annotations
import os, json
import phase12_common as C
import numpy as np

DIR = {"arrival": {"lateness_auc": True, "a3_median_abs_err_days": False},
       "fill": {"crps_exact": False, "p_full_auc": True, "uc2b_precision_at_5pct": True}}


def cmp(a, b, hi):
    if a is None or b is None:
        return "n/a"
    if hi:
        return "better" if a[0] > b[2] else "worse" if a[2] < b[0] else "undetermined"
    return "better" if a[2] < b[0] else "worse" if a[0] > b[2] else "undetermined"


def main():
    st = C.require_clean()
    S = {w: json.load(open(os.path.join(C.ART, "phase21", f"score_snap_{w}.json"))) for w in ("v8", "v8w1002")}
    out = dict(stamp=st, inputs={w: S[w]["stamp"]["code_commit"] for w in S}, arms={}, rules={}, gates={})
    for task in ("arrival", "fill"):
        for arm in ("L4", "L5", "L5_perm", "ack"):
            if arm not in S["v8"][task]["bands"] or arm not in S["v8w1002"][task]["bands"]:
                continue
            row = {}
            for m, hi in DIR[task].items():
                g = {}
                for w in S:
                    a, b = S[w][task]["bands"][arm][m], S[w][task]["bands"]["base"][m]
                    g[w] = dict(gain=(a[1] - b[1]) if hi else (b[1] - a[1]), vs_base=cmp(a, b, hi))
                s1, s2 = np.sign(g["v8"]["gain"]), np.sign(g["v8w1002"]["gain"])
                v2 = g["v8w1002"]["vs_base"]
                verdict = ("REPLICATES" if (v2 != "undetermined" and s1 == s2) else "DOES NOT" if v2 != "undetermined" else "UNDETERMINED")
                row[m] = dict(v8=g["v8"], v8w1002=g["v8w1002"], verdict=verdict,
                              ratio_w2_over_v8=(g["v8w1002"]["gain"] / g["v8"]["gain"]) if g["v8"]["gain"] else None)
            out["arms"][f"{task}|{arm}"] = row
        for w in S:
            det = S[w][task]["deterministic"]
            key = "rule vs base_ens" if task == "arrival" else "standalone_dist vs base_ens"
            out["rules"].setdefault(f"{task}", {})[w] = {m: dict(point=det["point"][m]["rule" if task == "arrival" else "standalone_dist"],
                                                                 base_ens=det["point"][m]["base_ens"], diff=det["diff"][m][key])
                                                          for m in DIR[task]}
            out["gates"].setdefault(task, {})[w] = {k: v.get("verdict", v) for k, v in S[w][task]["gates"].items()}
    C.dump(out, "phase21/stage7_replication.json")
    for k, v in out["arms"].items():
        print(k, {m: (r["verdict"], round(r["v8"]["gain"], 4), round(r["v8w1002"]["gain"], 4), r["v8"]["vs_base"], r["v8w1002"]["vs_base"],
                      None if r["ratio_w2_over_v8"] is None else round(r["ratio_w2_over_v8"], 2)) for m, r in v.items()})
    print(json.dumps(out["rules"], indent=0)[:3000])


if __name__ == "__main__":
    main()
