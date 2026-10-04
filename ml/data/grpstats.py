"""Phase 21 -- hierarchical, AS-OF, censoring-aware group statistics (the user's 5-D group idea).

Groups (reports/part2/phase-21-preregistration.md D3):
  L5 = (supplier, part, plant, order month, lane cluster)   L4 = (supplier, part, plant, lane cluster) = the channel
  L3 = (supplier, order month, lane cluster)                L2 = supplier            L1 = global
"Order month" is the month of the year (1-12) of a line's creation. Lane cluster = (transport mode, transit tercile of
the lane's standard_transit_days). Backoff chains: L5 -> L4 -> L2 -> L1 and L3 -> L2 -> L1.

AS-OF (D4). At an instant tau a PO line is in the history only if po_lines.recorded_ts <= tau; its receipt counts only if
grn_lines.recorded_ts <= tau; acknowledgements and zero-fill signals likewise. Every statistic is rebuilt from scratch at
each tau (an expanding window over the whole history), so a training row sees exactly what deployment would see.
`assert_asof` checks every source row that fed a statistic, and that the history is not empty; a failure is a STOP.

SURVIVORSHIP (D5). Lead = first receipt event - created (days); a line with no receipt recorded <= tau is CENSORED at
tau - created. Medians and quantiles are Kaplan-Meier over every line of the group, never completed lines only.
Lateness vs contract = lead - contracted lead, censored the same way; P(late) = KM S_excess(0).

SHRINKAGE (D6) is applied at assembly time (`assemble_*`), so k can be chosen on validation without a rebuild:
shrunk(L) = w raw(L) + (1 - w) shrunk(parent), w = n / (n + k), n = receipted lines (KM events) at L.

Stored per row (raw, unshrunk): L4 / L2 / L1 statistics, and L5 / L3 for ALL 12 months (a "month cube"), so the
permuted-month control reads a donor month without a rebuild. Fill: closed-line histograms (Phase 13's closed rule), and
the acknowledgement gap. Never read: inventory_position_weekly, part_demand_weekly, anything PRIVILEGED.

  python ml/data/grpstats.py build --world v8          # -> ml/artifacts/phase21/grpstats_v8_{snap,place}.npz + .json
  python ml/data/grpstats.py selftest --world v8       # unit tests + future-poison + self-exclusion (with falsification)
"""
from __future__ import annotations
import os, sys, json, time, hashlib, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import numpy as np, pandas as pd
import config
import phase21_paths as PP

LEVELS = ("L5", "L4", "L3", "L2", "L1")
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
ASTATS = [f"lead_q{int(q * 100)}" for q in QS] + [f"exc_q{int(q * 100)}" for q in QS] + ["p_late", "n", "nall", "lb_median"]
FSTATS = ["p_full", "shortfall", "n"]
KSTATS = ["gap_acked", "notfull_acked", "n_acked", "gap_open", "notfull_open", "n_open"]
LOG_FLOOR = -50.0
DAY = np.timedelta64(1, "D")
K_GRID = (1, 3, 10, 30, 100, 300)
OUT = os.path.join(config.ARTIFACTS, "phase21")
VERSION = "grpstats-1"


class AsOfLeak(AssertionError):
    pass


def ns(x):
    return pd.to_datetime(x).to_numpy("datetime64[ns]")


# ================================================================== source tables
class Source:
    """Every line of the world with its as-of timestamps. Arrays are aligned on po_lines' row order."""

    def __init__(self, world):
        self.world = world
        D = config.WORLDS[world]
        ch = pd.read_csv(f"{D}/sourcing_channels.csv", usecols=["channel_id", "supplier_id", "site_id", "part_id", "plant_id",
                                                                 "contracted_lead_time_days", "transport_mode"])
        ln = pd.read_csv(f"{D}/logistics_lanes.csv", usecols=["origin_site_id", "dest_plant_id", "transport_mode",
                                                              "distance_km", "standard_transit_days"])
        # the lane table is one row per channel, in channel order (Stage 1: several channels share a (site, plant, mode)
        # key, so a key join fans out). Aligned by row and every shared attribute asserted equal.
        assert len(ln) == len(ch), "lane table is not one row per channel"
        assert (ln.origin_site_id.to_numpy() == ch.site_id.to_numpy()).all() and \
            (ln.dest_plant_id.to_numpy() == ch.plant_id.to_numpy()).all() and \
            (ln.transport_mode.to_numpy() == ch.transport_mode.to_numpy()).all(), "lane row i is not channel i"
        lanes = ln
        cuts = np.quantile(ln.standard_transit_days.to_numpy(float), [1 / 3, 2 / 3])
        tcls = np.digitize(lanes.standard_transit_days.fillna(np.nanmedian(ln.standard_transit_days)).to_numpy(float), cuts)
        modes = sorted(ch.transport_mode.unique())
        self.lc_names = [f"{m}|t{t}" for m in modes for t in range(3)]
        self.ch = ch
        self.cidx = {c: i for i, c in enumerate(ch.channel_id)}
        sups = sorted(ch.supplier_id.unique()); self.sidx = {s: i for i, s in enumerate(sups)}
        self.n_ch, self.n_sup, self.n_lc = len(ch), len(sups), len(self.lc_names)
        self.ch_sup = ch.supplier_id.map(self.sidx).to_numpy(np.int64)
        self.ch_lc = (pd.Series(ch.transport_mode).map({m: i for i, m in enumerate(modes)}).to_numpy(np.int64) * 3 + tcls)
        self.ch_contract = ch.contracted_lead_time_days.to_numpy(float)
        self.transit_cuts = cuts.tolist()

        pl = pd.read_csv(f"{D}/po_lines.csv", usecols=["po_line_id", "channel_id", "qty_ordered", "original_promise_date",
                                                       "created_ts", "recorded_ts"])
        self.line_id = pl.po_line_id.to_numpy()
        self.lidx = pd.Series(np.arange(len(pl)), index=pl.po_line_id)
        self.chan = pl.channel_id.map(self.cidx).to_numpy(np.int64)
        self.created = ns(pl.created_ts)
        self.line_rec = ns(pl.recorded_ts)
        self.month = (pd.DatetimeIndex(self.created).month.to_numpy() - 1).astype(np.int64)
        self.qty = np.maximum(pl.qty_ordered.to_numpy(float), 1.0)
        self.contract = self.ch_contract[self.chan]
        self.promise = ns(pl.original_promise_date)
        n = len(pl)
        g = pd.read_csv(f"{D}/grn_lines.csv", usecols=["po_line_id", "qty_received", "is_final_receipt", "event_ts", "recorded_ts"])
        g = g.sort_values(["po_line_id", "event_ts"]).drop_duplicates("po_line_id", keep="first")   # first receipt
        gi = self.lidx.reindex(g.po_line_id).to_numpy()
        assert not np.isnan(gi.astype(float)).any(), "a receipt with no PO line"
        gi = gi.astype(np.int64)
        NAT = np.datetime64("NaT", "ns")
        self.g_event = np.full(n, NAT); self.g_rec = np.full(n, NAT)
        self.g_qty = np.zeros(n); self.g_final = np.zeros(n, bool)
        self.g_event[gi] = ns(g.event_ts); self.g_rec[gi] = ns(g.recorded_ts)
        self.g_qty[gi] = g.qty_received.to_numpy(float)
        self.g_final[gi] = g.is_final_receipt.astype(str).str.lower().isin(["true", "1"]).to_numpy()
        a = pd.read_csv(f"{D}/supplier_acknowledgements.csv", usecols=["po_line_id", "ack_qty", "ack_status", "recorded_ts"])
        a = a.drop_duplicates("po_line_id", keep="first")
        ai = self.lidx.reindex(a.po_line_id).to_numpy().astype(np.int64)
        self.k_rec = np.full(n, NAT); self.k_rec[ai] = ns(a.recorded_ts)
        self.k_gap = np.full(n, np.nan); self.k_gap[ai] = np.clip(1 - a.ack_qty.to_numpy(float) / self.qty[ai], 0, 1)
        self.k_notfull = np.zeros(n, bool); self.k_notfull[ai] = (a.ack_status != "full").to_numpy()
        rej = a[a.ack_status == "rejected"][["po_line_id", "recorded_ts"]]
        rv = pd.read_csv(f"{D}/po_line_revisions.csv", usecols=["po_line_id", "field_changed", "new_value", "recorded_ts"])
        rv = rv[(rv.field_changed == "qty") & (pd.to_numeric(rv.new_value, errors="coerce") == 0)][["po_line_id", "recorded_ts"]]
        z = pd.concat([rej, rv]); z["recorded_ts"] = pd.to_datetime(z.recorded_ts)
        z = z.groupby("po_line_id").recorded_ts.min()
        zi = self.lidx.reindex(z.index).to_numpy()
        ok = ~np.isnan(zi.astype(float))
        self.z_rec = np.full(n, NAT); self.z_rec[zi[ok].astype(np.int64)] = z.to_numpy("datetime64[ns]")[ok]
        self.n = n

    # ---- the future-poison test's corruption: every source value recorded AFTER tau is replaced with noise
    def poisoned(self, tau, seed=0):
        import copy
        s = copy.copy(self); r = np.random.default_rng(seed); tau = np.datetime64(tau, "ns")
        for name in ("g_event", "g_rec", "g_qty", "k_rec", "k_gap", "k_notfull", "z_rec", "created", "qty", "month", "chan"):
            setattr(s, name, getattr(self, name).copy())
        fut_line = self.line_rec > tau
        m = fut_line
        s.created[m] = self.created[m] - r.integers(0, 400, m.sum()) * DAY
        s.qty[m] = r.integers(1, 5000, m.sum()).astype(float)
        s.month[m] = r.integers(0, 12, m.sum())
        s.chan[m] = r.integers(0, self.n_ch, m.sum())
        g = ~(self.g_rec <= tau)                     # receipts recorded after tau (or none)
        s.g_event[g] = self.created[g] + r.integers(0, 300, g.sum()) * DAY
        s.g_qty[g] = r.integers(0, 5000, g.sum()).astype(float)
        k = ~(self.k_rec <= tau)
        s.k_gap[k] = r.random(k.sum()); s.k_notfull[k] = r.random(k.sum()) < 0.5
        zz = ~(self.z_rec <= tau)
        s.z_rec[zz] = tau + r.integers(1, 300, zz.sum()) * DAY
        s.contract = self.ch_contract[s.chan]
        return s


# ================================================================== Kaplan-Meier on many groups at once
def km_groups(gid, T, E, G, qs=QS, t_eval=None):
    """Per group g < G: KM quantiles of T (events E, others right-censored), S(t_eval), events, size, lower-bound flag.
    Ties: events before censorings (the standard convention). A quantile KM never reaches is the group's largest time,
    a lower bound, flagged."""
    assert len(gid) > 0, "KM on an empty history"
    o = np.lexsort((~E, T, gid))
    g, t, e = gid[o], T[o], E[o]
    cnt = np.bincount(g, minlength=G)
    start = np.cumsum(cnt) - cnt
    rank = np.arange(len(g)) - start[g]
    nrisk = (cnt[g] - rank).astype(float)
    inc = np.zeros(len(g))
    with np.errstate(divide="ignore"):
        inc[e] = np.log1p(-1.0 / nrisk[e])
    inc = np.maximum(inc, LOG_FLOOR)
    cs = np.cumsum(inc)
    logS = cs - np.r_[0.0, cs][start][g]
    last = np.maximum(start + cnt - 1, 0)
    has = cnt > 0
    out = {}
    for q in qs:
        k = np.bincount(g, weights=(logS > np.log1p(-q) + 1e-12), minlength=G).astype(np.int64)
        ok = k < cnt
        v = np.where(ok, t[np.minimum(start + k, len(t) - 1)], t[last])
        out[q] = np.where(has, v, np.nan)
        if q == 0.5:
            out["lb_median"] = (has & ~ok).astype(float)
    if t_eval is not None:
        j = np.bincount(g, weights=(t <= t_eval), minlength=G).astype(np.int64)
        sv = np.where(j > 0, np.exp(logS[np.maximum(start + j - 1, 0)]), 1.0)
        out["S_eval"] = np.where(has, sv, np.nan)
    out["n"] = np.bincount(g, weights=e.astype(float), minlength=G)
    out["nall"] = cnt.astype(float)
    return out


def km_median_completed_only(gid, T, E, G):
    """The biased estimator the pre-registration forbids, kept only for the censoring unit test."""
    s = pd.Series(T[E]).groupby(gid[E]).median()
    return s.reindex(range(G)).to_numpy(float)


# ================================================================== one tau: every group's raw statistics
class Builder:
    def __init__(self, src: Source, leak=False):
        self.s = src
        self.leak = leak             # the constructed offender: receipts filtered on EVENT time, not recorded time
        self.G = {"L5": src.n_ch * 12, "L4": src.n_ch, "L3": src.n_sup * 12 * src.n_lc, "L2": src.n_sup, "L1": 1}

    def keys(self, chan, month):
        s = self.s
        sup, lc = s.ch_sup[chan], s.ch_lc[chan]
        return {"L5": chan * 12 + month, "L4": chan, "L3": (sup * 12 + month) * s.n_lc + lc, "L2": sup,
                "L1": np.zeros(len(chan), np.int64)}

    def eligible(self, tau):
        return self.s.line_rec <= tau

    def arrival(self, tau):
        s = self.s; tau = np.datetime64(tau, "ns")
        m = self.eligible(tau)
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            raise AsOfLeak(f"history at {tau} is EMPTY")
        rec = (s.g_event[idx] <= tau) if self.leak else (s.g_rec[idx] <= tau)
        cr = s.created[idx]
        lead = np.where(rec, (s.g_event[idx] - cr) / DAY, (tau - cr) / DAY).astype(float)
        exc = lead - s.contract[idx]
        # as-of assertion on every source row that fed the statistics
        if not self.leak:
            if (s.line_rec[idx] > tau).any() or (s.g_rec[idx][rec] > tau).any():
                raise AsOfLeak(f"a source row recorded after {tau} fed the statistics")
        K = self.keys(s.chan[idx], s.month[idx])
        res = {}
        for L in LEVELS:
            a = km_groups(K[L], lead, rec, self.G[L])
            b = km_groups(K[L], exc, rec, self.G[L], t_eval=0.0)
            res[L] = np.stack([a[q] for q in QS] + [b[q] for q in QS] + [b["S_eval"], a["n"], a["nall"], a["lb_median"]], 1)
        meta = dict(n_lines=int(len(idx)), n_receipted=int(rec.sum()),
                    max_line_rec=str(s.line_rec[idx].max()), max_receipt_rec=str(s.g_rec[idx][rec].max()) if rec.any() else None)
        return res, meta, m

    def fill(self, tau):
        s = self.s; tau = np.datetime64(tau, "ns")
        m = self.eligible(tau)
        grec = (s.g_event <= tau) if self.leak else (s.g_rec <= tau)
        closed_r = m & grec & s.g_final
        closed_z = m & (s.z_rec <= tau) & ~grec
        cl = closed_r | closed_z
        idx = np.flatnonzero(cl)
        if len(idx) == 0:
            raise AsOfLeak(f"closed-line history at {tau} is EMPTY")
        if not self.leak:
            dep = np.where(closed_r[idx], s.g_rec[idx], s.z_rec[idx])
            if (dep > tau).any() or (s.line_rec[idx] > tau).any():
                raise AsOfLeak(f"a fill source row recorded after {tau} fed the statistics")
        fv = np.where(closed_r[idx], np.minimum(s.g_qty[idx], s.qty[idx]) / s.qty[idx], 0.0)
        cell = fill_cell(fv)
        K = self.keys(s.chan[idx], s.month[idx])
        res = {}
        for L in LEVELS:
            G = self.G[L]
            n = np.bincount(K[L], minlength=G).astype(float)
            short = np.bincount(K[L], weights=1 - fv, minlength=G)
            pf = np.bincount(K[L], weights=(cell == 21).astype(float), minlength=G)
            with np.errstate(invalid="ignore", divide="ignore"):
                res[L] = np.stack([pf / n, short / n, n], 1)
            if L != "L3":
                H = np.bincount(K[L] * 22 + cell, minlength=G * 22).reshape(G, 22).astype(np.float32)
                res[L + "_hist"] = H
        # acknowledgement gap: over acked lines, and over acked lines still OPEN as-of tau
        krec = (s.k_rec <= tau) & m
        opn = m & ~cl
        for L in ("L4", "L2"):
            Kl = self.keys(s.chan, s.month)[L]
            cols = []
            for sel in (krec, krec & opn):
                i = np.flatnonzero(sel)
                if not self.leak and (s.k_rec[i] > tau).any():
                    raise AsOfLeak("an acknowledgement recorded after tau fed the statistics")
                n = np.bincount(Kl[i], minlength=self.G[L]).astype(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cols += [np.bincount(Kl[i], weights=s.k_gap[i], minlength=self.G[L]) / n,
                             np.bincount(Kl[i], weights=s.k_notfull[i].astype(float), minlength=self.G[L]) / n, n]
            res[L + "_ack"] = np.stack(cols, 1)
        meta = dict(n_closed=int(len(idx)), n_closed_zero=int(closed_z.sum()), n_acked=int(krec.sum()), n_open=int(opn.sum()))
        return res, meta


def fill_cell(y):
    """phase7_fit.fill_cell, copied (that module imports lightgbm): 0 for y <= 0, 21 for y >= 1, 20 interior bins."""
    y = np.asarray(y, float)
    inner = 1 + np.clip(np.digitize(y, np.linspace(0, 1, 21)[1:-1]), 0, 19)
    return np.where(y <= 0, 0, np.where(y >= 1, 21, inner)).astype(np.int64)


# ================================================================== rows
def snapshot_rows(world):
    """phase7_fit.labels(world, 'arrival_week') reproduced without lightgbm: same filter, merge and order."""
    D = config.WORLDS[world]
    lb = pd.read_csv(D + "/training_labels.csv", usecols=["snapshot_date", "entity_id", "task", "label_value", "label_censored"])
    out = {}
    for task in ("arrival_week", "fill_rate"):
        x = lb[lb.task == task].copy()
        x["snapshot_date"] = pd.to_datetime(x.snapshot_date)
        x = x[(x.snapshot_date >= config.FIT_WINDOW[0]) & (x.snapshot_date <= config.FIT_WINDOW[1])]
        x = x.merge(pd.read_csv(D + "/po_lines.csv", usecols=["po_line_id", "channel_id"]), left_on="entity_id",
                    right_on="po_line_id", how="inner")
        out[task] = x.reset_index(drop=True)
    a, f = out["arrival_week"], out["fill_rate"]
    same = len(a) == len(f) and (a.entity_id.to_numpy() == f.entity_id.to_numpy()).all() and \
        (a.snapshot_date.to_numpy() == f.snapshot_date.to_numpy()).all()
    return a, same


def placement_rows(world, src):
    """D10: the distinct PO lines of the arrival label rows, each once, scored at the Monday 00:00 on or before creation."""
    a, _ = snapshot_rows(world)
    ids = pd.unique(a.entity_id)
    li = src.lidx.reindex(ids).to_numpy().astype(np.int64)
    cr = pd.DatetimeIndex(src.created[li])
    keep = (cr >= pd.Timestamp(config.FIT_WINDOW[0])) & (cr <= pd.Timestamp(config.FIT_WINDOW[1]))
    li = li[keep]
    cr = pd.DatetimeIndex(src.created[li])
    tau = (cr.normalize() - pd.to_timedelta(cr.weekday, unit="D")).to_numpy("datetime64[ns]")
    o = np.lexsort((li, tau))
    return li[o], tau[o]


# ================================================================== gather for rows
def build_rows(B: Builder, taus, chan, month, own_lines, want_fill):
    """Rows grouped by tau. Returns dict of arrays aligned to the input order."""
    s = B.s
    n = len(chan)
    nA = len(ASTATS)
    out = {"A_L4": np.full((n, nA), np.nan, np.float32), "A_L2": np.full((n, nA), np.nan, np.float32),
           "A_L1": np.full((n, nA), np.nan, np.float32),
           "A_L5c": np.full((n, 12, nA), np.nan, np.float32), "A_L3c": np.full((n, 12, nA), np.nan, np.float32)}
    if want_fill:
        out.update({f"F_{L}": np.full((n, 3), np.nan, np.float32) for L in ("L4", "L2", "L1")})
        out.update({"F_L5c": np.full((n, 12, 3), np.nan, np.float32), "F_L3c": np.full((n, 12, 3), np.nan, np.float32),
                    "H_L4": np.zeros((n, 22), np.float32), "H_L2": np.zeros((n, 22), np.float32),
                    "H_L1": np.zeros((n, 22), np.float32), "H_L5c": np.zeros((n, 12, 22), np.float32),
                    "K_L4": np.full((n, 6), np.nan, np.float32), "K_L2": np.full((n, 6), np.nan, np.float32)})
    sup, lc = s.ch_sup[chan], s.ch_lc[chan]
    months = np.arange(12)
    metas = {}
    for u in np.unique(taus):
        r = np.flatnonzero(taus == u)
        A, meta, elig = B.arrival(u)
        if own_lines is not None and elig[own_lines[r]].any():           # self-exclusion, asserted per tau
            raise AsOfLeak(f"{int(elig[own_lines[r]].sum())} rows' own lines are in their own history at {u}")
        c = chan[r]
        out["A_L4"][r] = A["L4"][c]; out["A_L2"][r] = A["L2"][sup[r]]; out["A_L1"][r] = A["L1"][0]
        out["A_L5c"][r] = A["L5"][(c[:, None] * 12 + months[None, :])]
        out["A_L3c"][r] = A["L3"][((sup[r][:, None] * 12 + months[None, :]) * s.n_lc + lc[r][:, None])]
        if want_fill:
            F, fmeta = B.fill(u); meta.update(fmeta)
            out["F_L4"][r] = F["L4"][c]; out["F_L2"][r] = F["L2"][sup[r]]; out["F_L1"][r] = F["L1"][0]
            out["F_L5c"][r] = F["L5"][(c[:, None] * 12 + months[None, :])]
            out["F_L3c"][r] = F["L3"][((sup[r][:, None] * 12 + months[None, :]) * s.n_lc + lc[r][:, None])]
            out["H_L4"][r] = F["L4_hist"][c]; out["H_L2"][r] = F["L2_hist"][sup[r]]; out["H_L1"][r] = F["L1_hist"][0]
            out["H_L5c"][r] = F["L5_hist"][(c[:, None] * 12 + months[None, :])]
            out["K_L4"][r] = F["L4_ack"][c]; out["K_L2"][r] = F["L2_ack"][sup[r]]
        metas[str(u)[:10]] = meta
    out["month"] = month.astype(np.int8)
    return out, metas


# ================================================================== assembly with shrinkage (D6, D7)
SHRINK_A = list(range(11))           # lead q*, exc q*, p_late are shrunk; n, nall, lb_median are carried raw


def _shrink(raw, n, parent, k):
    w = np.where(np.isfinite(raw), n / (n + k), 0.0)
    return w * np.nan_to_num(raw) + (1 - w) * parent


def _level_cols(sh, raw, tag):
    """11 columns per level: median / P10 / P90 / IQR of lead and of lateness, P(late), n, nall."""
    q = {name: i for i, name in enumerate(ASTATS)}
    cols = {f"{tag}_lead_med": sh[:, q["lead_q50"]], f"{tag}_lead_p10": sh[:, q["lead_q10"]], f"{tag}_lead_p90": sh[:, q["lead_q90"]],
            f"{tag}_lead_iqr": sh[:, q["lead_q75"]] - sh[:, q["lead_q25"]],
            f"{tag}_exc_med": sh[:, q["exc_q50"]], f"{tag}_exc_p10": sh[:, q["exc_q10"]], f"{tag}_exc_p90": sh[:, q["exc_q90"]],
            f"{tag}_exc_iqr": sh[:, q["exc_q75"]] - sh[:, q["exc_q25"]], f"{tag}_p_late": sh[:, q["p_late"]],
            f"{tag}_n": raw[:, q["n"]], f"{tag}_nall": raw[:, q["nall"]]}
    return cols


def arrival_levels(Z, k, month_key):
    """-> shrunk [rows, 14] per level and the resolved level of each chain. month_key: [rows] month index to read."""
    r = np.arange(len(month_key))
    raw = {"L4": Z["A_L4"].astype(float), "L2": Z["A_L2"].astype(float), "L1": Z["A_L1"].astype(float),
           "L5": Z["A_L5c"][r, month_key].astype(float), "L3": Z["A_L3c"][r, month_key].astype(float)}
    iN = ASTATS.index("n")
    sh = {"L1": raw["L1"].copy()}
    for L, P in (("L2", "L1"), ("L4", "L2"), ("L5", "L4"), ("L3", "L2")):
        x = raw[L].copy()
        n = np.nan_to_num(raw[L][:, iN])[:, None]
        x[:, SHRINK_A] = _shrink(raw[L][:, SHRINK_A], n, sh[P][:, SHRINK_A], k)
        sh[L] = x
    n = {L: np.nan_to_num(raw[L][:, iN]) for L in raw}
    res5 = np.where(n["L5"] >= k, 5, np.where(n["L4"] >= k, 4, np.where(n["L2"] >= k, 2, 1)))
    res4 = np.where(n["L4"] >= k, 4, np.where(n["L2"] >= k, 2, 1))
    return sh, raw, res5, res4


def assemble_arrival(Z, k, arm, month_key=None):
    """arm 'L4' -> L4, L2, L1 columns + resolved level (no-month chain); 'L5' -> + L5, L3 and the L5 chain's level."""
    mk = Z["month"].astype(np.int64) if month_key is None else month_key
    sh, raw, res5, res4 = arrival_levels(Z, k, mk)
    cols = {}
    for L in ("L4", "L2", "L1") + (("L5", "L3") if arm == "L5" else ()):
        cols.update(_level_cols(sh[L], raw[L], f"g{L}"))
    cols["g_resolved"] = (res5 if arm == "L5" else res4).astype(float)
    return pd.DataFrame(cols).astype(np.float32)


def standalone_arrival_weeks(Z, k, month_key=None):
    """D9: the L5 chain's shrunk KM median lead, in weeks (no offset)."""
    mk = Z["month"].astype(np.int64) if month_key is None else month_key
    sh, _, _, _ = arrival_levels(Z, k, mk)
    return sh["L5"][:, ASTATS.index("lead_q50")] / 7.0


def fill_levels(Z, k, month_key):
    r = np.arange(len(month_key))
    H = {"L1": Z["H_L1"].astype(float), "L2": Z["H_L2"].astype(float), "L4": Z["H_L4"].astype(float),
         "L5": Z["H_L5c"][r, month_key].astype(float)}
    F = {"L1": Z["F_L1"].astype(float), "L2": Z["F_L2"].astype(float), "L4": Z["F_L4"].astype(float),
         "L5": Z["F_L5c"][r, month_key].astype(float), "L3": Z["F_L3c"][r, month_key].astype(float)}
    sh = {"L1": H["L1"] / np.maximum(H["L1"].sum(1, keepdims=True), 1)}
    sf = {"L1": np.nan_to_num(F["L1"][:, :2])}
    for L, P in (("L2", "L1"), ("L4", "L2"), ("L5", "L4")):
        n = H[L].sum(1, keepdims=True)
        sh[L] = (H[L] + k * sh[P]) / (n + k)
    for L, P in (("L2", "L1"), ("L4", "L2"), ("L5", "L4"), ("L3", "L2")):
        n = np.nan_to_num(F[L][:, 2])[:, None]
        sf[L] = _shrink(F[L][:, :2], n, sf[P], k)
    n = {L: np.nan_to_num(F[L][:, 2]) for L in F}
    res5 = np.where(n["L5"] >= k, 5, np.where(n["L4"] >= k, 4, np.where(n["L2"] >= k, 2, 1)))
    res4 = np.where(n["L4"] >= k, 4, np.where(n["L2"] >= k, 2, 1))
    return sh, sf, F, res5, res4


def assemble_fill(Z, k, arm, month_key=None):
    mk = Z["month"].astype(np.int64) if month_key is None else month_key
    sh, sf, F, res5, res4 = fill_levels(Z, k, mk)
    cols = {}
    fin = "L5" if arm == "L5" else "L4"
    for j in range(22):
        cols[f"g{fin}_h{j:02d}"] = sh[fin][:, j]
    for L in ("L4", "L2", "L1") + (("L5", "L3") if arm == "L5" else ()):
        cols[f"g{L}_p_full"] = sf[L][:, 0]; cols[f"g{L}_shortfall"] = sf[L][:, 1]; cols[f"g{L}_n"] = np.nan_to_num(F[L][:, 2])
    cols["g_resolved"] = (res5 if arm == "L5" else res4).astype(float)
    return pd.DataFrame(cols).astype(np.float32)


def assemble_ack(Z):
    cols = {}
    for L in ("L4", "L2"):
        for j, nm in enumerate(KSTATS):
            cols[f"ack{L}_{nm}"] = Z[f"K_{L}"][:, j]
    return pd.DataFrame(cols).astype(np.float32)


def standalone_fill(Z, k, month_key=None):
    mk = Z["month"].astype(np.int64) if month_key is None else month_key
    return fill_levels(Z, k, mk)[0]["L5"]


# ================================================================== build + cache
def config_hash():
    src = open(os.path.abspath(__file__), "rb").read()
    return hashlib.sha1(src + VERSION.encode()).hexdigest()[:12]


def build(world, kinds=("snap", "place")):
    sys.path.insert(0, os.path.join(HERE, "..", "eval"))
    import phase12_common as C
    st = C.require_clean()
    os.makedirs(OUT, exist_ok=True)
    src = Source(world); B = Builder(src)
    info = dict(world=world, stamp=st, version=VERSION, config_hash=config_hash(), data=PP.register(),
                hashes=PP.world_hashes(world), lane_clusters=src.lc_names, transit_cuts=src.transit_cuts)
    if "snap" in kinds:
        t = time.time()
        a, same = snapshot_rows(world)
        assert same, "arrival and fill label rows differ -- one store cannot serve both"
        li = src.lidx.reindex(a.entity_id).to_numpy().astype(np.int64)
        taus = a.snapshot_date.to_numpy("datetime64[ns]")
        raised = src.created[li] <= taus
        month = np.where(raised, src.month[li], pd.DatetimeIndex(taus).month.to_numpy() - 1).astype(np.int64)
        chan = src.chan[li]
        assert (chan == a.channel_id.map(src.cidx).to_numpy()).all()
        Z, metas = build_rows(B, taus, chan, month, li, want_fill=True)
        np.savez_compressed(os.path.join(OUT, f"grpstats_{world}_snap.npz"), entity=a.entity_id.to_numpy().astype(str),
                            snapshot=taus.astype("datetime64[D]").astype(str), **Z)
        info["snap"] = dict(rows=len(a), share_raised_at_t0=float(raised.mean()), seconds=time.time() - t, per_tau=metas)
        print(f"snap rows {len(a)} built in {time.time() - t:.0f}s", flush=True)
    if "place" in kinds:
        t = time.time()
        li, taus = placement_rows(world, src)
        Z, metas = build_rows(B, taus, src.chan[li], src.month[li], li, want_fill=False)
        np.savez_compressed(os.path.join(OUT, f"grpstats_{world}_place.npz"), line=li, tau=taus.astype("datetime64[D]").astype(str),
                            entity=src.line_id[li].astype(str), **Z)
        info["place"] = dict(rows=len(li), seconds=time.time() - t, n_tau=len(metas),
                             per_tau_first_last={k: metas[k] for k in (min(metas), max(metas))})
        print(f"place rows {len(li)} built in {time.time() - t:.0f}s", flush=True)
    path = os.path.join(OUT, f"grpstats_{world}.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(info)
    json.dump(old, open(path, "w"), indent=1, default=str)


def load(world, kind):
    Z = dict(np.load(os.path.join(OUT, f"grpstats_{world}_{kind}.npz")))
    info = json.load(open(os.path.join(OUT, f"grpstats_{world}.json")))
    return Z, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--world", default="v8")
    ap.add_argument("--kinds", default="snap,place")
    a = ap.parse_args()
    PP.register()
    build(a.world, tuple(a.kinds.split(",")))
