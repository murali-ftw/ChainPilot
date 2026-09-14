"""Phase 5 — floors for fill (point-mass partition) and capacity (never measured before).

fill      LightGBM multiclass on the LEGACY 20 bins -- a reproduction of phase1_2's 0.0107 / 0.0141
          ECE, run first so the 22-cell number below is known to come from the same model -- and on
          the 22 CELLS with the endpoints separate.
capacity  naive: global / per-supplier / per-channel empirical P10-P50-P90 of training labels
          (per-channel falls back to supplier, then global); LightGBM quantile regression x3.

Features are the as-of channel row at t0 read from the Phase 1.5 cache -- panel[channel, t0], which
is exactly the row `merge_asof(direction='backward')` on week_start <= snapshot_date returns -- with
NULLs restored from the observed-indicators, plus the static channel attributes.

The fill reproduction keeps phase1_2's feature list INCLUDING the supplier / part / plant integer
codes, so its target number is the one being reproduced. The capacity LightGBM does not use them:
the new task is held to the project's inductive rule. The per-supplier and per-channel naive floors
key on identity by definition and are labelled as such.
"""
from __future__ import annotations
import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "train"), os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "models"), os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd, lightgbm as lgb
import phase5_heads as P5
import temporal_share as TS
from heads import fill_cell
from phase5_metrics import legacy_bin, QS
from loader import read_df
from config import WORLDS

GBM = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.9,
           subsample_freq=1, colsample_bytree=0.9, random_state=7, verbose=-1, n_jobs=6)


def features(w, lb, with_ids):
    W = TS.load_world(w)
    cols, null = W["meta"]["cols"], W["meta"]["nullable"]
    t0 = ((lb.snapshot_date - W["w0"]).dt.days // 7).to_numpy()
    ci = lb.key.map(W["cidx"]).to_numpy(np.int64)
    X = pd.DataFrame(W["panel"][ci, t0, :], columns=cols)
    for j, c in enumerate(null):
        X.loc[W["miss"][ci, t0, j] == 0, c] = np.nan
    ch = read_df(WORLDS[w], "sourcing_channels",
                 usecols=["channel_id", "supplier_id", "part_id", "plant_id",
                          "contracted_lead_time_days", "transport_mode", "transport_distance_km"])
    ch["mode_i"] = ch.transport_mode.astype("category").cat.codes
    st = ch.set_index("channel_id").loc[lb.key.to_numpy()]
    X["contracted_lead_time_days"] = st.contracted_lead_time_days.to_numpy()
    X["transport_distance_km"] = st.transport_distance_km.to_numpy()
    X["mode_i"] = st.mode_i.to_numpy()
    if with_ids:
        for c, n in (("supplier_id", "si"), ("part_id", "pi"), ("plant_id", "li")):
            u = {v: i for i, v in enumerate(sorted(ch[c].unique()))}
            X[n] = st[c].map(u).to_numpy()
    return X.astype(np.float32), st.supplier_id.to_numpy()


def save(w, task, name, P, lb, order, extra=None):
    os.makedirs(P5.PRED, exist_ok=True)
    f = os.path.join(P5.PRED, f"BASE_{w}_{task}_{name}.npz")
    np.savez_compressed(f, P=P, Y=lb.label_value.to_numpy(float)[order],
                        EV=~lb.label_censored.to_numpy(bool)[order], AUX=np.full(len(order), np.nan))
    print(f"    -> {os.path.basename(f)}", flush=True)
    return f


def fill(w, out):
    lb = P5.labels(w, "fill_rate")
    tr, va, te = TS.fold(lb.snapshot_date)
    order = P5.ordered(lb, te)
    X, _ = features(w, lb, with_ids=True)
    y = lb.label_value.to_numpy(float)
    for name, target, K in (("lightgbm20", legacy_bin(y), 20), ("lightgbm22", fill_cell(y), 22)):
        t = time.time()
        m = lgb.LGBMClassifier(**{**GBM, "objective": "multiclass", "num_class": K})
        m.fit(X[tr], target[tr], eval_set=[(X[va], target[va])],
              callbacks=[lgb.early_stopping(40, verbose=False)])
        P = m.predict_proba(X.iloc[order])
        assert P.shape[1] == K
        out[f"{w}|fill_rate|{name}"] = dict(best_iter=int(m.best_iteration_ or 400), seconds=time.time() - t,
                                            preds=save(w, "fill_rate", name, P, lb, order))


def capacity(w, out):
    lb = P5.labels(w, "capacity_strain")
    tr, va, te = TS.fold(lb.snapshot_date)
    order = P5.ordered(lb, te)
    X, sup = features(w, lb, with_ids=False)
    y = lb.label_value.to_numpy(float)
    lb["supplier_id"] = sup
    # ---- naive floors
    gq = np.quantile(y[tr], QS)
    trd = lb[tr]
    by_s = trd.groupby("supplier_id").label_value.quantile(list(QS)).unstack()
    by_c = trd.groupby("key").label_value.quantile(list(QS)).unstack()
    te_rows = lb.iloc[order]
    Qg = np.tile(gq, (len(order), 1))
    Qs = by_s.reindex(te_rows.supplier_id).to_numpy()
    Qs = np.where(np.isnan(Qs), Qg, Qs)
    Qc = by_c.reindex(te_rows.key).to_numpy()
    n_c_fallback = int(np.isnan(Qc[:, 0]).sum())
    Qc = np.where(np.isnan(Qc), Qs, Qc)
    # the "mean strain" point forecasts the brief names, for MAE, beside the quantile versions
    mean_c = trd.groupby("key").label_value.mean().reindex(te_rows.key).to_numpy()
    mean_s = trd.groupby("supplier_id").label_value.mean().reindex(te_rows.supplier_id).to_numpy()
    mean_s = np.where(np.isnan(mean_s), y[tr].mean(), mean_s)
    mean_c = np.where(np.isnan(mean_c), mean_s, mean_c)
    for name, Q, pt in (("naive_global", Qg, np.full(len(order), y[tr].mean())),
                        ("naive_supplier", Qs, mean_s), ("naive_channel", Qc, mean_c)):
        f = save(w, "capacity_strain", name, Q, lb, order)
        yt = y[order]
        out[f"{w}|capacity_strain|{name}"] = dict(preds=f, mae_mean_forecast=float(np.abs(yt - pt).mean()),
                                                  channel_fallbacks=n_c_fallback if name == "naive_channel" else None)
    # ---- LightGBM, one quantile model per level
    Q = np.zeros((len(order), 3))
    iters = []
    t = time.time()
    for j, a in enumerate(QS):
        m = lgb.LGBMRegressor(**{**GBM, "objective": "quantile", "alpha": a})
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="quantile",
              callbacks=[lgb.early_stopping(40, verbose=False)])
        Q[:, j] = m.predict(X.iloc[order]); iters.append(int(m.best_iteration_ or 400))
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    out[f"{w}|capacity_strain|lightgbm"] = dict(preds=save(w, "capacity_strain", "lightgbm", Q, lb, order),
                                                best_iters=iters, seconds=time.time() - t,
                                                top_features_p90=imp.head(6).to_dict())
    # the probe's own statistic on exactly these test rows, for the "does rho translate" question
    from scipy.stats import spearmanr
    out[f"{w}|capacity_strain|probe_load_ratio"] = dict(
        spearman_test=float(spearmanr(X.load_ratio.to_numpy()[order], y[order], nan_policy="omit")[0]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="v6,v7")
    ap.add_argument("--tasks", default="fill_rate,capacity_strain")
    ap.add_argument("--out", default=os.path.join(P5.ARTIFACTS, "phase5_baselines.json"))
    a = ap.parse_args()
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for w in a.worlds.split(","):
        for task in a.tasks.split(","):
            print(f"=== {w} {task}", flush=True)
            (fill if task == "fill_rate" else capacity)(w, out)
            json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("DONE", flush=True)
