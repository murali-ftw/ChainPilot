"""Phase 7 — fit every baseline (guide 7.1-7.2) in ONE torch-free process.

LightGBM and torch must not share a process on this machine: whichever library's threads run second segfaults
(measured in Phase 5). So nothing here imports torch — asserted at start-up and at the end — and all scoring,
recalibration and drift happen in `ml/eval/phase7_score.py`, a separate process.

The label table, fixed split and emission order are rebuilt with pandas and `folds.py` only, and were shown to be
identical to the Phase 6 bundles' test rows (labels and entity order, every task, both worlds). The scorer asserts it
again for every prediction file.

What is fitted (validation AND test predictions saved for every one, so recalibration and drift need no refit):

  Stage 2 — the LightGBM-22 refit band, on Addendum A's exact feature set (as-of channel row + static + identity codes)
    raw     seed 7, refitted five times       -> is the quoted single fit even reproducible?
    refit   seeds 7, 17, 27, 37, 47           -> fit noise, no recalibration
    (val-fitted arm: the refit arm's five fits put through the neural head's MM/VS protocol, applied by the scorer)
  LightGBM-20, seeds 7 / 17 / 27 / 37 / 47    -> the legacy partition every earlier report quoted

  Guide B5 — inductive flat-graph LightGBM, NO identity keys, seeds 7 / 17 / 27 / 37 / 47, every task

  Naive floors (deterministic)
    arrival   global / per-channel / per-lane median week (observed training rows), training marginal 13-cell
              distribution, promise-date-only
    fill      global 22-cell CDF, per-channel 22-cell CDF, per-channel mean fill, rolling-52 as-of active-week CDF (B2)
    capacity  global / per-supplier / per-channel empirical P10-P50-P90
    shortage  global base rate, per-part-plant historical rate
"""
from __future__ import annotations
import lightgbm as lgb                      # FIRST — before anything that could pull in torch
import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "data")]
import numpy as np, pandas as pd
from config import WORLDS, FIT_WINDOW, CACHE, ARTIFACTS
import folds as FO
from cache import load_panel
assert "torch" not in sys.modules, "torch must not be imported in the LightGBM process"

OUT = os.path.join(ARTIFACTS, "phase7_preds")
LOG = os.path.join(ARTIFACTS, "phase7_fit.json")
GBM = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.9,
           subsample_freq=1, colsample_bytree=0.9, verbose=-1, n_jobs=6)
SEEDS = [7, 17, 27, 37, 47]
W12 = 12
EDGES = np.linspace(0, 1, 21)
QS = (0.1, 0.5, 0.9)
ID_TASKS = {"arrival_week", "fill_rate"}


# ================================================================== numpy copies of the head's partitions
def fill_cell(y):
    y = np.asarray(y, float)
    inner = 1 + np.clip(np.digitize(y, EDGES[1:-1]), 0, 19)
    return np.where(y <= 0, 0, np.where(y >= 1, 21, inner)).astype(np.int64)


def legacy_bin(y):
    return np.clip(np.digitize(np.asarray(y, float), EDGES[1:-1]), 0, 19)


def cells13(Y, EV):
    return np.where(np.asarray(EV, bool), np.clip(np.asarray(Y, float), 1, W12).astype(int) - 1, W12)


# ================================================================== labels, split, order (torch-free)
def labels(w, task):
    D = WORLDS[w]
    lb = pd.read_csv(D + "/training_labels.csv", usecols=["snapshot_date", "entity_id", "task", "label_value", "label_censored"])
    lb = lb[lb.task == task].copy()
    lb["snapshot_date"] = pd.to_datetime(lb.snapshot_date)
    lb = lb[(lb.snapshot_date >= FIT_WINDOW[0]) & (lb.snapshot_date <= FIT_WINDOW[1])]
    lb["label_censored"] = lb.label_censored.astype(str).str.lower().isin(["true", "1"])
    if task in ("fill_rate", "arrival_week"):
        cols = ["po_line_id", "channel_id"] + (["original_promise_date"] if task == "arrival_week" else [])
        lb = lb.merge(pd.read_csv(D + "/po_lines.csv", usecols=cols), left_on="entity_id", right_on="po_line_id", how="inner")
        lb["key"] = lb.channel_id
        if task == "arrival_week":
            lb["promise_week"] = (pd.to_datetime(lb.original_promise_date, errors="coerce") - lb.snapshot_date).dt.days / 7.0
    elif task == "shortage_qty":
        lb[["part_id", "plant_id"]] = lb.entity_id.str.split("|", expand=True)
        lb["key"] = lb.part_id + "|" + lb.plant_id
        lb["label_value"] = (lb.label_value > 0).astype(float)
    else:
        lb["key"] = lb.entity_id
    return lb.reset_index(drop=True)


def ordered(lb, mask):
    idx = np.flatnonzero(mask)
    return idx[np.argsort(lb.snapshot_date.values[idx], kind="stable")]


# ================================================================== world: panel + graph tables
class World:
    def __init__(self, w):
        self.w = w
        panel, miss, active, meta = load_panel(os.path.join(CACHE, w))
        self.panel, self.miss, self.active, self.meta = panel, miss, active, meta
        self.cols, self.null = meta["cols"], meta["nullable"]
        self.w0 = pd.Timestamp(meta["week0"])
        D = WORLDS[w]
        ch = pd.read_csv(D + "/sourcing_channels.csv", usecols=["channel_id", "supplier_id", "part_id", "plant_id",
                                                                 "contracted_lead_time_days", "transport_mode", "transport_distance_km"])
        self.ch = ch
        self.cidx = {c: i for i, c in enumerate(ch.channel_id)}
        su = pd.read_csv(D + "/suppliers.csv", usecols=["supplier_id", "supplier_group_id"])
        up = pd.read_csv(D + "/supplier_upstream.csv", usecols=["supplier_id", "is_sole_source"])
        up["sole"] = up.is_sole_source.astype(str).str.lower().isin(["true", "1"])
        alt = pd.read_csv(D + "/alternate_sources.csv", usecols=["part_id", "plant_id"])
        g = ch.copy()
        g["mode_i"] = g.transport_mode.astype("category").cat.codes
        g["channels_per_supplier"] = g.groupby("supplier_id").channel_id.transform("size")
        g["channels_per_part"] = g.groupby("part_id").channel_id.transform("size")
        g["channels_per_plant"] = g.groupby("plant_id").channel_id.transform("size")
        g["suppliers_per_part"] = g.groupby("part_id").supplier_id.transform("nunique")
        g["suppliers_per_part_plant"] = g.groupby(["part_id", "plant_id"]).supplier_id.transform("nunique")
        g["sole_source_part_plant"] = (g.suppliers_per_part_plant == 1).astype(float)
        g = g.merge(su, on="supplier_id", how="left")
        g["suppliers_per_group"] = g.groupby("supplier_group_id").supplier_id.transform("nunique")
        ups = up.groupby("supplier_id").agg(upstream_links=("sole", "size"), upstream_sole_links=("sole", "sum")).reset_index()
        g = g.merge(ups, on="supplier_id", how="left").fillna({"upstream_links": 0, "upstream_sole_links": 0})
        alts = alt.groupby(["part_id", "plant_id"]).size().rename("alternate_sources_part_plant").reset_index()
        g = g.merge(alts, on=["part_id", "plant_id"], how="left").fillna({"alternate_sources_part_plant": 0})
        for c, n in (("supplier_id", "si"), ("part_id", "pi"), ("plant_id", "li")):
            u = {v: i for i, v in enumerate(sorted(ch[c].unique()))}
            g[n] = g[c].map(u)
        g = g.set_index("channel_id").loc[ch.channel_id]           # channel order preserved
        self.g = g.reset_index()
        self.FLAT = ["channels_per_supplier", "channels_per_part", "channels_per_plant", "suppliers_per_part",
                     "suppliers_per_part_plant", "sole_source_part_plant", "suppliers_per_group", "upstream_links",
                     "upstream_sole_links", "alternate_sources_part_plant"]
        self.STATIC = ["contracted_lead_time_days", "transport_distance_km", "mode_i"]
        self.IDS = ["si", "pi", "li"]
        pk = (ch.part_id + "|" + ch.plant_id).to_numpy()
        self.pp_keys = sorted(set(pk))
        self.pp_of_chan = np.array([{k: i for i, k in enumerate(self.pp_keys)}[k] for k in pk])

    def t0(self, dates):
        return ((pd.to_datetime(pd.Series(dates)) - self.w0).dt.days // 7).to_numpy()

    def asof(self, ci, t0):
        X = pd.DataFrame(np.asarray(self.panel[ci, t0, :], np.float32), columns=self.cols)
        mv = np.asarray(self.miss[ci, t0, :])
        for j, c in enumerate(self.null):
            X.loc[mv[:, j] == 0, c] = np.nan
        return X

    def channel_features(self, lb, with_ids, flat):
        ci = lb.key.map(self.cidx).to_numpy(np.int64)
        X = self.asof(ci, self.t0(lb.snapshot_date))
        cols = self.STATIC + (self.IDS if with_ids else []) + (self.FLAT if flat else [])
        G = self.g.iloc[ci][cols].reset_index(drop=True)
        return pd.concat([X, G], axis=1).astype(np.float32)

    def part_plant_features(self, lb):
        """Shortage: as-of channel rows aggregated to part x plant (mean, min), plus part-plant graph counts."""
        ppidx = {k: i for i, k in enumerate(self.pp_keys)}
        rows = []
        n_pp = len(self.pp_keys)
        cnt = np.bincount(self.pp_of_chan, minlength=n_pp).astype(float)[:, None]
        cache = {}
        t0s = self.t0(lb.snapshot_date)
        for t in np.unique(t0s):
            A = np.asarray(self.panel[:, t, :], np.float64)
            M = np.asarray(self.miss[:, t, :])
            for j, c in enumerate(self.null):
                A[M[:, j] == 0, self.cols.index(c)] = np.nan
            s = np.zeros((n_pp, A.shape[1])); c_ = np.zeros((n_pp, A.shape[1]))
            np.add.at(s, self.pp_of_chan, np.nan_to_num(A)); np.add.at(c_, self.pp_of_chan, ~np.isnan(A))
            mean = np.where(c_ > 0, s / np.maximum(c_, 1), np.nan)
            mn = np.full((n_pp, A.shape[1]), np.inf)
            np.minimum.at(mn, self.pp_of_chan, np.where(np.isnan(A), np.inf, A))
            mn[np.isinf(mn)] = np.nan
            cache[t] = np.concatenate([mean, mn, cnt], 1)
        pi = lb.key.map(ppidx).to_numpy(np.int64)
        X = np.stack([cache[t][p] for t, p in zip(t0s, pi)])
        names = [f"{c}_mean" for c in self.cols] + [f"{c}_min" for c in self.cols] + ["channels_in_part_plant"]
        Xd = pd.DataFrame(X.astype(np.float32), columns=names)
        pg = self.g.assign(pp=self.g.part_id + "|" + self.g.plant_id).groupby("pp")[
            ["suppliers_per_part_plant", "sole_source_part_plant", "alternate_sources_part_plant", "channels_per_part",
             "suppliers_per_part"]].first()
        Xd = pd.concat([Xd, pg.loc[lb.key.to_numpy()].reset_index(drop=True).astype(np.float32)], axis=1)
        return Xd

    def rolling52_fill(self, lb, mask):
        """B2 for fill, as-of: 22-cell histogram of the channel's ACTIVE-week fill_rate in the 52 weeks ending at t0."""
        o = ordered(lb, mask)
        ci = lb.key.map(self.cidx).to_numpy(np.int64)[o]
        t0 = self.t0(lb.snapshot_date.iloc[o])
        fc = self.cols.index("fill_rate"); fo = self.null.index("fill_rate")
        H = np.zeros((len(o), 22))
        for t in np.unique(t0):
            sel = np.flatnonzero(t0 == t)
            lo = max(0, t - 51)
            v = np.asarray(self.panel[ci[sel], lo:t + 1, fc])
            ok = (np.asarray(self.active[ci[sel], lo:t + 1]) > 0) & (np.asarray(self.miss[ci[sel], lo:t + 1, fo]) > 0)
            cells = fill_cell(v)
            rr = np.repeat(np.arange(len(sel))[:, None], v.shape[1], 1)
            block = np.zeros((len(sel), 22))
            np.add.at(block, (rr[ok], cells[ok]), 1.0)    # H[sel] would be a COPY -- accumulate, then assign
            H[sel] = block
        n_active = H.sum(1)
        return H, n_active


# ================================================================== save
def save(w, task, name, fold, P, lb, order, extra=None):
    os.makedirs(OUT, exist_ok=True)
    d = dict(P=np.asarray(P), Y=lb.label_value.to_numpy(float)[order], EV=~lb.label_censored.to_numpy(bool)[order],
             AUX=(lb.promise_week.to_numpy(float)[order] if "promise_week" in lb else np.full(len(order), np.nan)),
             entity=lb.entity_id.to_numpy().astype(str)[order])
    d.update(extra or {})
    np.savez_compressed(os.path.join(OUT, f"{w}_{task}_{name}_{fold}.npz"), **d)


def emit(w, task, name, lb, va, te, fn, log, meta=None):
    t = time.time()
    for fold, mask in (("val", va), ("test", te)):
        o = ordered(lb, mask)
        P, extra = fn(o)
        save(w, task, name, fold, P, lb, o, extra)
    log[f"{w}|{task}|{name}"] = dict(seconds=time.time() - t, **(meta or {}))


# ================================================================== the fits
def lgbm_fit(kind, X, y, tr, va, seed, K=None, alpha=None, rows_tr=None, rows_va=None):
    rows_tr = tr if rows_tr is None else rows_tr
    rows_va = va if rows_va is None else rows_va
    if kind == "multi":
        m = lgb.LGBMClassifier(**{**GBM, "random_state": seed, "objective": "multiclass", "num_class": K})
    elif kind == "binary":
        m = lgb.LGBMClassifier(**{**GBM, "random_state": seed, "objective": "binary"})
    elif kind == "quantile":
        m = lgb.LGBMRegressor(**{**GBM, "random_state": seed, "objective": "quantile", "alpha": alpha})
    else:
        m = lgb.LGBMRegressor(**{**GBM, "random_state": seed, "objective": "l2"})
    kw = dict(eval_metric="quantile") if kind == "quantile" else {}
    m.fit(X[rows_tr], y[rows_tr], eval_set=[(X[rows_va], y[rows_va])], callbacks=[lgb.early_stopping(40, verbose=False)], **kw)
    return m


def run_world(w, log, only=None):
    Wd = World(w)
    want = lambda k: only is None or k in only

    # ---------------------------------------------------------------- fill
    if want("fill"):
        lb = labels(w, "fill_rate"); tr, va, te = FO.fixed_split(lb.snapshot_date)
        y = lb.label_value.to_numpy(float); c22 = fill_cell(y); c20 = legacy_bin(y)
        Xid = Wd.channel_features(lb, with_ids=True, flat=False)          # Addendum A's feature set
        Xflat = Wd.channel_features(lb, with_ids=False, flat=True)        # guide B5
        for rep in range(5):                                              # raw arm: identical seed, refitted
            t = time.time(); m = lgbm_fit("multi", Xid, c22, tr, va, 7, K=22)
            emit(w, "fill_rate", f"lgbm22_id_raw_r{rep}", lb, va, te, lambda o: (m.predict_proba(Xid.iloc[o]), None), log,
                 dict(best_iter=int(m.best_iteration_ or 400), fit_seconds=time.time() - t, seed=7))
        for s in SEEDS:                                                   # refit arm: different seeds
            m = lgbm_fit("multi", Xid, c22, tr, va, s, K=22)
            emit(w, "fill_rate", f"lgbm22_id_s{s}", lb, va, te, lambda o: (m.predict_proba(Xid.iloc[o]), None), log,
                 dict(best_iter=int(m.best_iteration_ or 400), seed=s))
            m = lgbm_fit("multi", Xid, c20, tr, va, s, K=20)
            emit(w, "fill_rate", f"lgbm20_id_s{s}", lb, va, te, lambda o: (m.predict_proba(Xid.iloc[o]), None), log,
                 dict(best_iter=int(m.best_iteration_ or 400), seed=s))
            m = lgbm_fit("multi", Xflat, c22, tr, va, s, K=22)
            emit(w, "fill_rate", f"b5flat22_s{s}", lb, va, te, lambda o: (m.predict_proba(Xflat.iloc[o]), None), log,
                 dict(best_iter=int(m.best_iteration_ or 400), seed=s))
        # naive floors
        glob22 = np.bincount(c22[tr], minlength=22) / tr.sum()
        emit(w, "fill_rate", "naive_global_cdf", lb, va, te, lambda o: (np.tile(glob22, (len(o), 1)), None), log)
        trd = pd.DataFrame({"k": lb.key[tr].to_numpy(), "c": c22[tr], "y": y[tr]})
        H = trd.groupby(["k", "c"]).size().unstack(fill_value=0).reindex(columns=range(22), fill_value=0)
        Hc = H.div(H.sum(1), axis=0)
        mean_fill = trd.groupby("k").y.mean()
        def per_channel_cdf(o):
            P = Hc.reindex(lb.key.iloc[o].to_numpy()).to_numpy()
            return np.where(np.isnan(P), glob22, P), None
        emit(w, "fill_rate", "naive_channel_cdf", lb, va, te, per_channel_cdf, log, dict(identity_keyed=True))
        def per_channel_mean(o):
            mu = mean_fill.reindex(lb.key.iloc[o].to_numpy()).fillna(float(y[tr].mean())).to_numpy()
            return np.eye(22)[fill_cell(mu)], None                        # a point mass at the channel's mean fill
        emit(w, "fill_rate", "naive_channel_mean", lb, va, te, per_channel_mean, log, dict(identity_keyed=True))
        for fold, mask in (("val", va), ("test", te)):
            t = time.time(); o = ordered(lb, mask)
            Hr, n_act = Wd.rolling52_fill(lb, mask)
            P = np.where(n_act[:, None] > 0, Hr / np.maximum(n_act[:, None], 1), glob22)
            save(w, "fill_rate", "b2_rolling52_cdf", fold, P, lb, o, dict(n_active_weeks=n_act))
        log[f"{w}|fill_rate|b2_rolling52_cdf"] = dict(note="as-of, active weeks only; global CDF where none")

    # ---------------------------------------------------------------- arrival
    if want("arrival"):
        lb = labels(w, "arrival_week"); tr, va, te = FO.fixed_split(lb.snapshot_date)
        y = lb.label_value.to_numpy(float); ev = ~lb.label_censored.to_numpy(bool)
        obs_tr, obs_va = tr & ev, va & ev
        Xid = Wd.channel_features(lb, with_ids=True, flat=False)
        Xflat = Wd.channel_features(lb, with_ids=False, flat=True)
        for s in SEEDS:
            m = lgbm_fit("l2", Xflat, y, tr, va, s, rows_tr=obs_tr, rows_va=obs_va)
            emit(w, "arrival_week", f"b5flat_reg_s{s}", lb, va, te, lambda o: (m.predict(Xflat.iloc[o]), None), log,
                 dict(best_iter=int(m.best_iteration_ or 400), seed=s, trained_on="observed rows only"))
            m = lgbm_fit("l2", Xid, y, tr, va, s, rows_tr=obs_tr, rows_va=obs_va)
            emit(w, "arrival_week", f"lgbm_id_reg_s{s}", lb, va, te, lambda o: (m.predict(Xid.iloc[o]), None), log,
                 dict(best_iter=int(m.best_iteration_ or 400), seed=s, trained_on="observed rows only", identity_keyed=True))
        gmed = float(np.median(y[obs_tr]))
        emit(w, "arrival_week", "naive_global_median", lb, va, te, lambda o: (np.full(len(o), gmed), None), log)
        chm = pd.Series(y[obs_tr]).groupby(lb.key[obs_tr].to_numpy()).median()
        emit(w, "arrival_week", "naive_channel_median", lb, va, te,
             lambda o: (chm.reindex(lb.key.iloc[o].to_numpy()).fillna(gmed).to_numpy(), None), log, dict(identity_keyed=True))
        lane = Wd.g.set_index("channel_id")[["supplier_id", "plant_id"]]
        lk = (lane.supplier_id + "|" + lane.plant_id).reindex(lb.key.to_numpy()).to_numpy()
        lnm = pd.Series(y[obs_tr]).groupby(lk[obs_tr]).median()
        emit(w, "arrival_week", "naive_lane_median", lb, va, te,
             lambda o: (pd.Series(lk[o]).map(lnm).fillna(gmed).to_numpy(), None), log, dict(identity_keyed=True))
        marg = np.bincount(cells13(y[tr], ev[tr]), minlength=13) / tr.sum()       # training marginal, censored rows in
        def marginal(o):
            P13 = np.tile(marg, (len(o), 1)); S = 1 - np.cumsum(P13[:, :W12], 1)
            return 1 + S.sum(1), dict(S=S, pT=P13[:, :W12])
        emit(w, "arrival_week", "naive_marginal13", lb, va, te, marginal, log, dict(note="replaces guide B3"))
        pw = lb.promise_week.to_numpy(float); pmed = float(np.nanmedian(pw[tr]))
        emit(w, "arrival_week", "promise_only", lb, va, te,
             lambda o: (np.where(np.isfinite(pw[o]), pw[o], pmed), None), log,
             dict(note="score = weeks from snapshot to original promise date; what a planner holds without a model"))

    # ---------------------------------------------------------------- capacity
    if want("capacity"):
        lb = labels(w, "capacity_strain"); tr, va, te = FO.fixed_split(lb.snapshot_date)
        y = lb.label_value.to_numpy(float)
        Xflat = Wd.channel_features(lb, with_ids=False, flat=True)
        for s in SEEDS:
            ms = [lgbm_fit("quantile", Xflat, y, tr, va, s, alpha=a) for a in QS]
            emit(w, "capacity_strain", f"b5flat_q_s{s}", lb, va, te,
                 lambda o: (np.stack([m.predict(Xflat.iloc[o]) for m in ms], 1), None), log,
                 dict(best_iters=[int(m.best_iteration_ or 400) for m in ms], seed=s))
        sup = Wd.g.set_index("channel_id").supplier_id.reindex(lb.key.to_numpy()).to_numpy()
        gq = np.quantile(y[tr], QS)
        emit(w, "capacity_strain", "naive_global", lb, va, te, lambda o: (np.tile(gq, (len(o), 1)), None), log)
        sq = pd.DataFrame({"s": sup[tr], "y": y[tr]}).groupby("s").y.quantile(list(QS)).unstack()
        cq = pd.DataFrame({"k": lb.key[tr].to_numpy(), "y": y[tr]}).groupby("k").y.quantile(list(QS)).unstack()
        def per(keys, table, fallback):
            Q = table.reindex(keys).to_numpy()
            return np.where(np.isnan(Q), fallback, Q)
        emit(w, "capacity_strain", "naive_supplier", lb, va, te,
             lambda o: (per(sup[o], sq, gq), None), log, dict(identity_keyed=True))
        emit(w, "capacity_strain", "naive_channel", lb, va, te,
             lambda o: (per(lb.key.iloc[o].to_numpy(), cq, per(sup[o], sq, gq)), None), log, dict(identity_keyed=True))

    # ---------------------------------------------------------------- shortage (DIAGNOSTIC)
    if want("shortage"):
        lb = labels(w, "shortage_qty"); tr, va, te = FO.fixed_split(lb.snapshot_date)
        y = lb.label_value.to_numpy(float)
        Xpp = Wd.part_plant_features(lb)
        for s in SEEDS:
            m = lgbm_fit("binary", Xpp, y.astype(int), tr, va, s)
            emit(w, "shortage_qty", f"b5flat_bin_s{s}", lb, va, te, lambda o: (m.predict_proba(Xpp.iloc[o])[:, 1], None), log,
                 dict(best_iter=int(m.best_iteration_ or 400), seed=s, diagnostic_only=True))
        base = float(y[tr].mean())
        emit(w, "shortage_qty", "naive_global_rate", lb, va, te, lambda o: (np.full(len(o), base), None), log, dict(diagnostic_only=True))
        ppr = pd.Series(y[tr]).groupby(lb.key[tr].to_numpy()).mean()
        emit(w, "shortage_qty", "naive_part_plant_rate", lb, va, te,
             lambda o: (ppr.reindex(lb.key.iloc[o].to_numpy()).fillna(base).to_numpy(), None), log,
             dict(identity_keyed=True, diagnostic_only=True))


# ================================================================== Phase 8.2: capacity B5 on every rolling origin
def run_capacity_origins(w, origins, log):
    """Guide B5 quantile LightGBM, flat graph, no identity keys -- Phase 7's capacity fit, re-fitted per origin with
    early stopping on that origin's own validation slice. Files go to the backtest directory for the single scorer."""
    global OUT
    OUT = os.path.join(ARTIFACTS, "backtest", "preds")
    Wd = World(w)
    lb = labels(w, "capacity_strain")
    y = lb.label_value.to_numpy(float)
    Xflat = Wd.channel_features(lb, with_ids=False, flat=True)
    for k in origins:
        tr, va, te = FO.rolling_split(lb.snapshot_date, k)
        FO.assert_no_leak(lb.snapshot_date, tr, va, te)
        for s in SEEDS:
            ms = [lgbm_fit("quantile", Xflat, y, tr, va, s, alpha=a) for a in QS]
            emit(w, "capacity_strain", f"o{k}_b5flat_q_s{s}", lb, va, te,
                 lambda o: (np.stack([m.predict(Xflat.iloc[o]) for m in ms], 1), None), log,
                 dict(best_iters=[int(m.best_iteration_ or 400) for m in ms], seed=s, origin=k))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="v6,v7")
    ap.add_argument("--only", default=None, help="comma list of fill,arrival,capacity,shortage")
    ap.add_argument("--origins", default=None, help="Phase 8.2: comma list of rolling origins -> capacity B5 per origin")
    a = ap.parse_args()
    if a.origins:
        LOG = os.path.join(ARTIFACTS, "backtest", "phase8_b5_fit.json"); os.makedirs(os.path.dirname(LOG), exist_ok=True)
        blog = json.load(open(LOG)) if os.path.exists(LOG) else {}
        for w in a.worlds.split(","):
            t = time.time(); run_capacity_origins(w, [int(x) for x in a.origins.split(",")], blog)
            print(f"{w} capacity B5 origins {a.origins} done in {time.time() - t:.0f}s", flush=True)
            json.dump(blog, open(LOG, "w"), indent=1)
        assert "torch" not in sys.modules
        sys.exit(0)
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    t_all = time.time()
    for w in a.worlds.split(","):
        t = time.time()
        run_world(w, log, set(a.only.split(",")) if a.only else None)
        print(f"{w} done in {time.time() - t:.0f}s", flush=True)
        json.dump(log, open(LOG, "w"), indent=1)
    assert "torch" not in sys.modules
    print(f"ALL FITS DONE in {time.time() - t_all:.0f}s; torch never imported", flush=True)
