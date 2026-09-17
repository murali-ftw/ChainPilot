"""Phase 8D — corrections and the extrapolation check. No model, no training: label tables and finished artifacts only.

8D.1  recount the interim's period-split and B5/coverage tallies from phase8b_capacity.json
8D.2  per origin x world: the training window's capacity_strain range, the fraction of evaluation labels outside it,
      the evaluation mean's z-score; correlated against the h4-minus-h0 pinball margin over all 16 cells
8D.3  mean capacity_strain per snapshot over the whole fit window, both worlds, with trend / step / ceiling tests
8D.4  the same series read against the exceedance deterioration, which begins at origin 6

Reads:
  db/gen_v{6,7}/seed_1001/training_labels.csv   task == capacity_strain only (the label table every phase reads)
  ml/artifacts/backtest/phase8b_capacity.json   the 8B scores

  python ml/eval/phase8d_extrapolation.py
"""
from __future__ import annotations
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "train")]
import numpy as np, pandas as pd
from scipy import stats
from config import WORLDS, FIT_WINDOW, ARTIFACTS
import folds as FO

BT = os.path.join(ARTIFACTS, "backtest")
TASK = "capacity_strain"
H4, H0, B5 = "mp_h4_lr0.00025", "none_h0_lr0.00025", "b5flat_q"
VERDICT = {"a better": "h4", "b better": "h0", "not distinguishable": "tie"}


def labels(w):
    lb = pd.read_csv(os.path.join(WORLDS[w], "training_labels.csv"), usecols=["snapshot_date", "task", "label_value"])
    lb = lb[lb.task == TASK].drop(columns="task")
    lb["snapshot_date"] = pd.to_datetime(lb.snapshot_date)
    return lb[(lb.snapshot_date >= FIT_WINDOW[0]) & (lb.snapshot_date <= FIT_WINDOW[1])].reset_index(drop=True)


def recount(rows):
    """8D.1 -- every tally the interim quotes, recomputed from the verdicts."""
    def tally(sel, key):
        v = [VERDICT[r[key]["verdict"]] for r in rows if sel(r)]
        return {x: v.count(x) for x in ("h4", "tie", "h0")}
    early, late = (lambda r: r["origin"] <= 6), (lambda r: r["origin"] >= 7)
    out = dict(eval_origins_1_6=tally(early, "h4_vs_h0"), eval_origins_7_8=tally(late, "h4_vs_h0"),
               eval_all=tally(lambda r: True, "h4_vs_h0"),
               h4_vs_b5_origins_1_6=tally(early, "h4_vs_b5"), h4_vs_b5_origins_7_8=tally(late, "h4_vs_b5"),
               val_eval_agree=sum(r["h4_vs_h0"]["verdict"] == r["h4_vs_h0_validation"]["verdict"] for r in rows))
    cov = {}
    for c, lab in ((H4, "h4"), (H0, "h0"), (B5, "B5")):
        m = [r["coverage80"][c]["mean"] for r in rows]
        cov[lab] = dict(min=min(m), max=max(m), below_nominal=sum(x < 0.80 for x in m), n=len(m))
    out["coverage80_mean_per_window"] = cov
    ex = {}
    for c, lab in ((H4, "h4"), (H0, "h0"), (B5, "B5")):
        for name, sel in (("o1_5", lambda r: r["origin"] <= 5), ("o6_8", lambda r: r["origin"] >= 6)):
            m = [r["exceed_p90"][c]["mean"] for r in rows if sel(r)]
            ex[f"{lab}|{name}"] = (min(m), max(m))
    out["exceed_p90_mean_range"] = ex
    return out


def extrapolation(lb, w, k):
    tr, va, te = FO.rolling_split(lb.snapshot_date, k)
    y_tr, y_va, y_te = lb.label_value[tr].to_numpy(), lb.label_value[va].to_numpy(), lb.label_value[te].to_numpy()
    p1, p99 = np.percentile(y_tr, [1, 99])
    y_seen = np.concatenate([y_tr, y_va])                     # everything the model's fit touched (train + validation)
    return dict(world=w, origin=k,
                train=dict(n=len(y_tr), min=float(y_tr.min()), p1=float(p1), mean=float(y_tr.mean()), p99=float(p99),
                           max=float(y_tr.max()), sd=float(y_tr.std())),
                test=dict(n=len(y_te), mean=float(y_te.mean()), p99=float(np.percentile(y_te, 99)), max=float(y_te.max())),
                frac_outside_train_minmax=float(((y_te < y_tr.min()) | (y_te > y_tr.max())).mean()),
                frac_above_train_max=float((y_te > y_tr.max()).mean()),
                frac_outside_train_p1_p99=float(((y_te < p1) | (y_te > p99)).mean()),
                frac_above_train_p99=float((y_te > p99).mean()),
                frac_below_train_p1=float((y_te < p1).mean()),
                frac_above_seen_p99=float((y_te > np.percentile(y_seen, 99)).mean()),
                z_test_mean=float((y_te.mean() - y_tr.mean()) / y_tr.std()),
                test_minus_train_p99=float(np.percentile(y_te, 99) - p99))


def snapshot_series(lb):
    g = lb.groupby("snapshot_date").label_value
    s = pd.DataFrame(dict(mean=g.mean(), p90=g.quantile(0.9), p99=g.quantile(0.99), frac_ge_1=g.apply(lambda v: (v >= 1.0).mean()),
                          max=g.max(), n=g.size())).reset_index()
    return s


def trend_tests(s):
    """Trend, step or ceiling. A linear trend is fitted on 2019 -> 2024-06-30 (everything before origin 7's
    evaluation) and extrapolated; late residuals show whether 2025 continues it. A step is tested as the best single
    change point in the mean (two-segment least squares)."""
    t = (s.snapshot_date - s.snapshot_date.min()).dt.days.to_numpy() / 365.25
    y = s["mean"].to_numpy()
    early = (s.snapshot_date <= "2024-06-30").to_numpy()
    lr_all = stats.linregress(t, y)
    lr_early = stats.linregress(t[early], y[early])
    resid = y - (lr_early.intercept + lr_early.slope * t)
    sd_early = resid[early].std(ddof=2)
    sse = []
    for i in range(6, len(y) - 3):
        sse.append((((y[:i] - y[:i].mean()) ** 2).sum() + ((y[i:] - y[i:].mean()) ** 2).sum(), i))
    best_sse, bi = min(sse)
    sse0 = ((y - y.mean()) ** 2).sum()
    sse_lin = ((y - (lr_all.intercept + lr_all.slope * t)) ** 2).sum()
    yr = s.assign(year=s.snapshot_date.dt.year).groupby("year")["mean"].agg(["mean", "min", "max", "size"])
    return dict(slope_per_year_all=lr_all.slope, r2_all=lr_all.rvalue ** 2, p_all=lr_all.pvalue,
                slope_per_year_2019_2024H1=lr_early.slope, r2_early=lr_early.rvalue ** 2, p_early=lr_early.pvalue,
                late_residuals_vs_early_trend=[dict(date=str(d.date()), mean=float(m), resid=float(r), z=float(r / sd_early))
                                               for d, m, r in zip(s.snapshot_date[~early], y[~early], resid[~early])],
                early_resid_sd=float(sd_early),
                best_step=dict(at=str(s.snapshot_date.iloc[bi].date()), before=float(y[:bi].mean()), after=float(y[bi:].mean()),
                               sse_reduction_vs_flat=float(1 - best_sse / sse0), sse_reduction_linear_vs_flat=float(1 - sse_lin / sse0)),
                yearly={int(k): dict(mean=float(v["mean"]), min=float(v["min"]), max=float(v["max"]), n=int(v["size"])) for k, v in yr.iterrows()})


def main(out_path):
    cap = json.load(open(os.path.join(BT, "phase8b_capacity.json")))
    rows = cap["rows"]
    R = dict(recount=recount(rows), cells=[], series={}, trend={})
    margin = {(r["world"], r["origin"]): r for r in rows}
    for w in ("v6", "v7"):
        lb = labels(w)
        for k in range(1, 9):
            e = extrapolation(lb, w, k)
            r = margin[(w, k)]
            e.update(h4_minus_h0=r["h4_vs_h0"]["margin"], h4_minus_b5=r["h4_vs_b5"]["margin"],
                     verdict=VERDICT[r["h4_vs_h0"]["verdict"]], verdict_b5=VERDICT[r["h4_vs_b5"]["verdict"]].replace("h0", "B5"),
                     label_shift_train_to_test=r["label_shift_train_to_test"])
            R["cells"].append(e)
        s = snapshot_series(lb)
        R["series"][w] = [dict(date=str(d.date()), **{c: float(s[c].iloc[i]) for c in ("mean", "p90", "p99", "frac_ge_1", "max", "n")})
                          for i, d in enumerate(s.snapshot_date)]
        R["trend"][w] = trend_tests(s)

    C = pd.DataFrame(R["cells"])
    corr = {}
    for x in ("frac_outside_train_minmax", "frac_above_train_max", "frac_outside_train_p1_p99", "frac_above_train_p99",
              "frac_above_seen_p99", "z_test_mean", "label_shift_train_to_test", "test_minus_train_p99"):
        for yv in ("h4_minus_h0", "h4_minus_b5"):
            pr, pp = stats.pearsonr(C[x], C[yv]); sr, sp = stats.spearmanr(C[x], C[yv])
            corr[f"{x}~{yv}"] = dict(pearson=pr, pearson_p=pp, spearman=sr, spearman_p=sp)
        # within-world, so the world's level offset cannot manufacture a correlation
        for w in ("v6", "v7"):
            c = C[C.world == w]
            sr, sp = stats.spearmanr(c[x], c["h4_minus_h0"])
            corr[f"{x}~h4_minus_h0|{w}"] = dict(spearman=sr, spearman_p=sp, n=len(c))
    R["correlation"] = corr
    # rank of the two o8 cells on each predictor, within their world
    ranks = {}
    for x in ("frac_outside_train_p1_p99", "frac_above_train_p99", "z_test_mean", "frac_outside_train_minmax"):
        for w in ("v6", "v7"):
            c = C[C.world == w].sort_values(x, ascending=False).origin.tolist()
            ranks[f"{x}|{w}"] = c
    R["rank_order_desc"] = ranks
    json.dump(R, open(out_path, "w"), indent=1, default=float)

    print("8D.1 recount:", json.dumps(R["recount"], indent=1, default=float))
    print(f"\n{'w':3s} {'o':>2s} {'tr min':>7s} {'tr p1':>7s} {'tr mean':>7s} {'tr p99':>7s} {'tr max':>7s} {'te mean':>7s} "
          f"{'z':>6s} {'out mm':>7s} {'>max':>6s} {'out p1-99':>9s} {'>p99':>6s} {'<p1':>6s} {'h4-h0':>8s} {'v':4s} {'h4-B5':>8s} {'vB5':4s}")
    for e in R["cells"]:
        t = e["train"]
        print(f"{e['world']:3s} {e['origin']:2d} {t['min']:7.4f} {t['p1']:7.4f} {t['mean']:7.4f} {t['p99']:7.4f} {t['max']:7.4f} "
              f"{e['test']['mean']:7.4f} {e['z_test_mean']:+6.2f} {e['frac_outside_train_minmax']:7.4f} {e['frac_above_train_max']:6.4f} "
              f"{e['frac_outside_train_p1_p99']:9.4f} {e['frac_above_train_p99']:6.4f} {e['frac_below_train_p1']:6.4f} "
              f"{e['h4_minus_h0']:+8.4f} {e['verdict']:4s} {e['h4_minus_b5']:+8.4f} {e['verdict_b5']:4s}")
    print("\ncorrelations:")
    for k, v in corr.items():
        print(f"  {k:55s} " + " ".join(f"{a}={b:+.3f}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
    print("\nrank order (desc):", json.dumps(ranks))
    for w in ("v6", "v7"):
        print(f"\n== {w} snapshot series")
        for r in R["series"][w]:
            print(f"  {r['date']} mean {r['mean']:.4f} p90 {r['p90']:.4f} p99 {r['p99']:.4f} >=1 {r['frac_ge_1']:.4f} max {r['max']:.3f} n {int(r['n'])}")
        T = R["trend"][w]
        print(" trend:", json.dumps({k: v for k, v in T.items() if k != "late_residuals_vs_early_trend"}, indent=1, default=float))
        for r in T["late_residuals_vs_early_trend"]:
            print(f"   {r['date']} mean {r['mean']:.4f} resid {r['resid']:+.4f} z {r['z']:+.2f}")
    print(f"\n-> {out_path}")




def exceedance_check(out_path):
    """8D.4 -- does any label-side property of the window track P90 exceedance (which steps up at origin 6)?"""
    R = json.load(open(out_path))
    cap = {(r["world"], r["origin"]): r for r in json.load(open(os.path.join(BT, "phase8b_capacity.json")))["rows"]}
    C = pd.DataFrame(R["cells"])
    C["shift_val_to_test"] = [cap[(w, k)]["label_shift_validation_to_test"] for w, k in zip(C.world, C.origin)]
    C["late"] = (C.origin >= 6).astype(float)
    out = {}
    for c, lab in ((H4, "h4"), (H0, "h0"), (B5, "B5")):
        C[f"exceed_{lab}"] = [cap[(w, k)]["exceed_p90"][c]["mean"] for w, k in zip(C.world, C.origin)]
        for x in ("z_test_mean", "frac_above_train_p99", "label_shift_train_to_test", "shift_val_to_test", "late"):
            sr, sp = stats.spearmanr(C[x], C[f"exceed_{lab}"])
            out[f"exceed_{lab}~{x}"] = dict(spearman=float(sr), p=float(sp))
    R["exceedance_check"] = out
    json.dump(R, open(out_path, "w"), indent=1, default=float)
    print("\n8D.4 exceedance vs window properties (Spearman, n=16):")
    for k, v in out.items():
        print(f"  {k:45s} rho={v['spearman']:+.3f} p={v['p']:.3f}")
    print(C[["world", "origin", "z_test_mean", "shift_val_to_test", "exceed_h4", "exceed_h0", "exceed_B5"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(BT, "phase8d_extrapolation.json"))
    a = ap.parse_args()
    main(a.out)
    exceedance_check(a.out)
