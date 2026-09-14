"""Phase 5 — score every prediction set with 1,000-resample bootstrap intervals.

Old heads are read from the Phase 2 / closeout artifacts unchanged (same encoder h0, same data, same
split); new heads from phase5_grid*.json; floors from phase5_baselines.json and run9's BASE files.
Incremental: a file already scored at the same mtime is not re-scored.
"""
from __future__ import annotations
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "models")]
import numpy as np
import phase5_metrics as M
from metrics import roc_auc
from config import ARTIFACTS

OUT = os.path.join(ARTIFACTS, "phase5_scores.json")
A = lambda *p: os.path.join(ARTIFACTS, *p)


def entries():
    E = []
    run9 = {(r["world"], r["task"], r["arch"], r["depth"], r["seed"]): r for r in json.load(open(A("run9_grid.json")))}
    for w in ("v6", "v7"):
        r = run9[(w, "arrival_week", "none", 0, 7)]
        E.append(dict(label=f"{w}|arrival_week|OLD_mse_h0", task="arrival_week", kind="old_mse", file=r["preds"],
                      ymu=r["ymu"], ysd=r["ysd"]))
        for b in ("naive", "lightgbm"):
            E.append(dict(label=f"{w}|arrival_week|BASE_{b}", task="arrival_week", kind="point",
                          file=A("run9_preds", f"BASE_{w}_arrival_week_{b}.npz")))
        E.append(dict(label=f"{w}|fill_rate|OLD_ce20_h0", task="fill_rate", kind="legacy20",
                      file=A("run10_preds", f"{w}_fill_rate_none_h0_s7.npz")))
        E.append(dict(label=f"{w}|fill_rate|BASE_naive_run9", task="fill_rate", kind="legacy20",
                      file=A("run9_preds", f"BASE_{w}_fill_rate_naive.npz")))
        E.append(dict(label=f"{w}|fill_rate|BASE_lightgbm_run9", task="fill_rate", kind="legacy20",
                      file=A("run9_preds", f"BASE_{w}_fill_rate_lightgbm.npz")))
        for b, k in (("lightgbm20", "legacy20"), ("lightgbm22", "cells22")):
            E.append(dict(label=f"{w}|fill_rate|BASE_{b}", task="fill_rate", kind=k,
                          file=A("phase5_preds", f"BASE_{w}_fill_rate_{b}.npz")))
        for b in ("naive_global", "naive_supplier", "naive_channel", "lightgbm"):
            E.append(dict(label=f"{w}|capacity_strain|BASE_{b}", task="capacity_strain", kind="quantile",
                          file=A("phase5_preds", f"BASE_{w}_capacity_strain_{b}.npz")))
        E.append(dict(label=f"{w}|shortage_qty|OLD_bce_h0_floor", task="shortage_qty", kind="prob",
                      file=A("run10_preds", f"{w}_shortage_qty_none_h0_s7.npz")))
        for b in ("naive", "lightgbm"):
            E.append(dict(label=f"{w}|shortage_qty|BASE_{b}", task="shortage_qty", kind="prob",
                          file=A("run9_preds", f"BASE_{w}_shortage_qty_{b}.npz")))
    for g in sorted(glob.glob(A("phase5_grid*.json"))):
        for r in json.load(open(g)):
            lab = (f"{r['world']}|{r['task']}|NEW_{r['head']}_{r['arch']}_h{r['depth']}_s{r['seed']}"
                   f"_lr{r['lr']:g}_g{r['gate']}_w{r['wsla']}"
                   + (f"_loss-{r['fill_loss']}" if r.get("fill_loss") else ""))
            kind = {"hazard": "hazard", "cdf22": "cells22", "quantile": "quantile", "binary": "prob"}[r["head"]]
            E.append(dict(label=lab, task=r["task"], kind=kind, file=r["preds"]))
    return E


def score(e):
    z = np.load(e["file"])
    P, Y, EV = z["P"], z["Y"], z["EV"].astype(bool)
    AUX = z["AUX"] if "AUX" in z else np.full(len(Y), np.nan)
    t, k = e["task"], e["kind"]
    if t == "arrival_week":
        if k == "old_mse":
            ET = P * e["ysd"] + e["ymu"]
            s = M.arrival_scores(ET, Y, EV, AUX)
            m = EV & np.isfinite(AUX); yl = (Y[m] > AUX[m]).astype(int)
            # the figure phase1_2 printed: standardised prediction minus a promise in weeks
            s["roc_auc_late_as_phase1_2_units_mismatch"] = (roc_auc(yl, P[m] - AUX[m]),)
            s["roc_auc_late_promise_only"] = (roc_auc(yl, -AUX[m]),)
            return s
        if k == "point":
            return M.arrival_scores(P, Y, EV, AUX)
        return M.arrival_scores(P, Y, EV, AUX, S=z["S"], pT=z["pT"])
    if t == "fill_rate":
        return M.fill_scores(P, Y, k)
    if t == "capacity_strain":
        return M.capacity_scores(P, Y)
    return M.shortage_scores(P, Y)


def main():
    S = json.load(open(OUT)) if os.path.exists(OUT) else {}
    ref_Y = {}
    for e in entries():
        if not os.path.exists(e["file"]):
            print(f"  missing {e['file']}"); continue
        mt = os.path.getmtime(e["file"])
        z = np.load(e["file"]); Y = z["Y"]
        w, t = e["label"].split("|")[:2]
        if (w, t) in ref_Y:
            same = len(Y) == len(ref_Y[(w, t)]) and np.array_equal(Y, ref_Y[(w, t)])
        else:
            ref_Y[(w, t)] = Y; same = True
        if e["label"] in S and S[e["label"]].get("_mtime") == mt:
            continue
        print(f"  scoring {e['label']}", flush=True)
        s = score(e)
        s["_mtime"] = mt; s["_file"] = e["file"]; s["_rows_identical_to_first_entry"] = bool(same)
        S[e["label"]] = s
        json.dump(S, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"scored {len(S)} entries -> {OUT}")


if __name__ == "__main__":
    main()
