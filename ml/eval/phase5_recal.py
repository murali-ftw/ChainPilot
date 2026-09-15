"""Phase 5 §6.6 — does validation-fold recalibration close fill's calibration gap?

§6.4 measured the defect as a stable per-cell bias: the point-mass head cannot fit its own training
marginal on the rare interior cells. A stable per-cell bias is what post-hoc recalibration corrects.
Everything here is FITTED ON THE 2024 VALIDATION FOLD ONLY and applied unchanged to 2025 test.

  none  the head as trained
  mm    per-cell moment matching: multiplicative weights r_b, renormalised per row, iterated until the
        mean recalibrated distribution on validation equals validation's observed cell frequencies
  vs    vector scaling: logits' = log P / T + beta_b, one temperature and 22 biases, fitted by
        minimising validation log score (LBFGS) -- a proper score, so it corrects conditional
        calibration too, not only the marginal ECE measures

Selection among none / mm / vs is on VALIDATION log score. Test scores of the methods not selected
are reported beside it. One caveat is stated, not hidden: the validation fold also chose each model's
early-stopping epoch, so it is used twice -- for the neural head and for LightGBM alike.

  python ml/eval/phase5_recal.py --lgbm    refit LightGBM-22 emitting validation predictions
  python ml/eval/phase5_recal.py --fit     fit, apply, score, print the tables
"""
from __future__ import annotations
import os, sys, re, json, glob, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "models"),
                os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "baselines"),
                os.path.join(HERE, "..", "data")]
# torch and lightgbm each bundle an OpenMP runtime, and on macOS one process must not do heavy work in
# both. Measured: torch imported first -> LightGBM training segfaults (exit 139). So --lgbm imports
# lightgbm FIRST and does no torch work, and --fit (torch LBFGS) never imports lightgbm at all.
# Run the two modes as separate invocations.
if "--lgbm" in sys.argv:
    import lightgbm  # noqa: F401
import numpy as np, torch, torch.nn.functional as F
import phase5_metrics as M
from heads import fill_cell
from config import ARTIFACTS

RDIR = os.path.join(ARTIFACTS, "phase5_preds_recal")
OUT = os.path.join(ARTIFACTS, "phase5_recal_scores.json")
EPS = 1e-7


# ------------------------------------------------------------------ methods
def fit_mm(Pv, cv, iters=100):
    K = Pv.shape[1]
    obs = np.maximum(np.bincount(cv, minlength=K) / len(cv), EPS)     # never zero a cell outright
    r = np.ones(K)
    for _ in range(iters):
        Q = Pv * r; Q /= Q.sum(1, keepdims=True)
        r *= obs / np.maximum(Q.mean(0), EPS)
    return r


def apply_mm(P, r):
    Q = P * r
    return Q / Q.sum(1, keepdims=True)


def fit_vs(Pv, cv):
    z = torch.log(torch.from_numpy(np.clip(Pv, EPS, 1.0)).double())
    y = torch.from_numpy(cv.astype(np.int64))
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(z.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT, b], max_iter=500, tolerance_grad=1e-10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        L = F.cross_entropy(z * torch.exp(-logT) + b, y)
        L.backward()
        return L
    opt.step(closure)
    return float(torch.exp(logT).detach()), b.detach().numpy()


def apply_vs(P, T, b):
    z = np.log(np.clip(P, EPS, 1.0)) / T + b
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def log_score(P, c):
    return float(-np.log(np.clip(P[np.arange(len(c)), c], EPS, 1.0)).mean())


# ------------------------------------------------------------------ LightGBM-22 with validation output
def refit_lightgbm22():
    import lightgbm as lgb
    import phase5_heads as P5, temporal_share as TS, phase5_baselines as B
    os.makedirs(RDIR, exist_ok=True)
    for w in ("v6", "v7"):
        lb = P5.labels(w, "fill_rate")
        tr, va, te = TS.fold(lb.snapshot_date)
        X, _ = B.features(w, lb, with_ids=True)
        y = lb.label_value.to_numpy(float); c = fill_cell(y)
        m = lgb.LGBMClassifier(**{**B.GBM, "objective": "multiclass", "num_class": 22})
        m.fit(X[tr], c[tr], eval_set=[(X[va], c[va])], callbacks=[lgb.early_stopping(40, verbose=False)])
        for suffix, mask in (("", te), ("_val", va)):
            o = P5.ordered(lb, mask)
            np.savez_compressed(os.path.join(RDIR, f"BASE_{w}_fill_rate_lightgbm22{suffix}.npz"),
                                P=m.predict_proba(X.iloc[o]), Y=y[o],
                                EV=~lb.label_censored.to_numpy(bool)[o], AUX=np.full(len(o), np.nan))
        print(f"  {w} LightGBM-22 refit, best iteration {m.best_iteration_}", flush=True)


# ------------------------------------------------------------------ fit / apply / score
def fit_all():
    R = {}
    # the re-trained h0 cells and LightGBM-22 live in RDIR; Phase 4's fill depth cells (h1, h4) save their
    # validation predictions beside their test predictions in phase5_preds/
    srcs = glob.glob(os.path.join(RDIR, "*_val.npz")) + \
        glob.glob(os.path.join(ARTIFACTS, "phase5_preds", "*_fill_rate_cdf22_*_h[14]_*_val.npz"))
    for fv in sorted(srcs):
        ft = fv.replace("_val.npz", ".npz")
        name = os.path.basename(ft)[:-4]
        if name.startswith("RECAL_") or not os.path.exists(ft):
            continue
        zv, zt = np.load(fv), np.load(ft)
        Pv, Yv, Pt, Yt = zv["P"].astype(float), zv["Y"], zt["P"].astype(float), zt["Y"]
        cv, ct = fill_cell(Yv), fill_cell(Yt)
        r = fit_mm(Pv, cv)
        T, b = fit_vs(Pv, cv)
        variants = {"none": (Pv, Pt), "mm": (apply_mm(Pv, r), apply_mm(Pt, r)),
                    "vs": (apply_vs(Pv, T, b), apply_vs(Pt, T, b))}
        entry = {"_params": {"mm_ratio": [float(x) for x in r], "vs_T": T, "vs_bias": [float(x) for x in b]}}
        # DIAGNOSTIC, not selectable: fit on training-fold + validation predictions together. The 2024
        # fold differs from 2025 more than the training years do, so a validation-only fit can import
        # 2024's quirks; the training fold carries the model's own bias without them. No clean fold is
        # left to choose between the two sources, so the validation fit stays the pre-declared primary.
        ftr = ft.replace(".npz", "_train.npz")
        if os.path.exists(ftr):
            ztr = np.load(ftr)
            Ptv = np.concatenate([ztr["P"].astype(float), Pv]); ctv = np.concatenate([fill_cell(ztr["Y"]), cv])
            r_tv = fit_mm(Ptv, ctv); T_tv, b_tv = fit_vs(Ptv, ctv)
            variants["mm_tv"] = (apply_mm(Pv, r_tv), apply_mm(Pt, r_tv))
            variants["vs_tv"] = (apply_vs(Pv, T_tv, b_tv), apply_vs(Pt, T_tv, b_tv))
            entry["_params"].update(vs_tv_T=T_tv)
        for meth, (V, Tt) in variants.items():
            entry[meth] = {
                "val": {"log_score": log_score(V, cv), "ece22": M.ece_marginal(V, cv)[0],
                        "ece20": M.ece_marginal(M.fill_to_legacy(V) if hasattr(M, "fill_to_legacy") else
                                                __import__("heads").fill_to_legacy(V), M.legacy_bin(Yv))[0],
                        "crps_exact": float(M.crps_exact_rows(V, Yv).mean())},
                "test": M.fill_scores(Tt, Yt, "cells22"),
                "test_log_score": log_score(Tt, ct)}
            if meth != "none":
                np.savez_compressed(os.path.join(RDIR, f"RECAL_{meth}_{name}.npz"), P=Tt, Y=Yt, EV=zt["EV"], AUX=zt["AUX"])
        entry["selected_on_validation"] = min(("none", "mm", "vs"), key=lambda k: entry[k]["val"]["log_score"])
        R[name] = entry
        print(f"  {name}: selected {entry['selected_on_validation']} (T = {T:.3f})", flush=True)
    json.dump(R, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return R


def tables(R):
    out = []
    P = lambda s="": out.append(s)
    f = lambda e, m: (f"{e[m][0]:.4f} [{e[m][1]:.4f}, {e[m][2]:.4f}]" if len(e[m]) >= 3 else f"{e[m][0]:.4f}")
    P("| world | model | method | val log score | val ECE 22 | **test ECE 20** | test ECE 22 | test reliability P(f=1) | test CRPS exact | test P(complete) / obs | selected on val |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, e in R.items():
        w = re.search(r"v[67]", name).group(0)
        model = "LightGBM-22" if name.startswith("BASE_") else f"neural s{name.split('_s')[1].split('_')[0]}"
        for meth in ("none", "mm", "vs", "mm_tv", "vs_tv"):
            if meth not in e:
                continue
            t = e[meth]["test"]; v = e[meth]["val"]
            sel = "**✔**" if e["selected_on_validation"] == meth else ("diagnostic" if meth.endswith("_tv") else "")
            P(f"| {w} | {model} | {meth} | {v['log_score']:.4f} | {v['ece22']:.4f} | {f(t, 'ece20')} | {f(t, 'ece22')} | "
              f"{f(t, 'rel_one')} | {f(t, 'crps_exact')} | {t['p_complete_pred'][0]:.4f} / {t['p_complete_obs'][0]:.4f} | {sel} |")
    P()
    P("Seed band after recalibration, v6 neural, seeds 7 / 17 / 27 (max − min):")
    for meth in ("none", "mm", "vs", "mm_tv", "vs_tv"):
        rows = [e[meth]["test"] for n, e in R.items() if n.startswith("v6_fill_rate_cdf22") and meth in e]
        if len(rows) == 3:
            P(f"- {meth}: " + ", ".join(f"{m} {max(r[m][0] for r in rows) - min(r[m][0] for r in rows):.4f}"
                                        for m in ("ece20", "ece22", "rel_one", "crps_exact")))
    txt = "\n".join(out)
    open(os.path.join(ARTIFACTS, "phase5_recal_tables.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgbm", action="store_true")
    ap.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    if a.lgbm:
        refit_lightgbm22()
    if a.fit:
        tables(fit_all())
