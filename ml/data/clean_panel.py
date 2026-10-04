"""Phase 22 Stage 1c -- AS-OF-SAFE replacements for the panel columns the leak scan flagged, and the clean worlds.

The scan (ml/eval/phase22_leakscan.py, leakscan_v8.json) flags NINE columns of channel_performance_weekly as leaking under
the panel row's own week (H_week): the generator writes each line's EVENTUAL outcome into the week it was ORDERED
(lead_time_actual_days, lead_time_ratio, otd_rate_last13 from the line's lead; fill_rate, fill_rate_last4/13/52 and
ack_gap_ratio = 1 - fill from the line's delivered fraction), and load_ratio is the supplier's WHOLE-MONTH ordered quantity
over declared capacity, applied to every week of the month.

Replacements, from the emitted CSVs only (po_lines, grn_lines, supplier_acknowledgements, po_line_revisions,
sourcing_channels, supplier_capacity) -- no generator array, nothing PRIVILEGED:
  lead / lead ratio / on-time   a line's lead (first receipt event - created, days) enters the channel's series in the week
                                its receipt AND its order are both visible; forward-filled; ratio = lead / contracted lead;
                                otd_rate_last13 = 13-week rolling mean of the forward-filled on-time flag (generator's roll)
  fill and rolls, ack-gap       the line's delivered fraction min(received / ordered, 1) enters, quantity-weighted, in the
                                week its receipt and order are visible; a line closed at zero (a rejected acknowledgement or a
                                qty -> 0 revision, with no receipt) enters with 0 in the week that signal is visible;
                                forward-filled; rolls 4 / 13 / 52; ack_gap_ratio = 1 - fill (the generator's definition)
  load_ratio                    the supplier's quantity ordered in the CURRENT month and visible by the row's week, divided by
                                the declared capacity for that month recorded by the end of the row's week (else the latest
                                declared month recorded by then)
"Visible" = the generator's visible week max(event week, recorded week) (H_week, pre-registration D1).

Clean worlds: `v8clean` / `v8w1002clean` = the same CSVs, cache dir ml/artifacts/cache/{world}clean/ with these nine columns
(and their missing-indicators) replaced; every other column and file byte-identical to the parent cache. The world name is
in every artifact identity, so nothing stored is overwritten.

`masked_view(panel, miss, meta)`: the feature-view switch -- the leaking columns set to 0 and their indicators to "missing"
for any read of a stored-model input.

Tests (ml/tests/test_phase22_clean_panel.py): FUTURE POISON (every source row visible after the row's week replaced with noise;
the clean columns at that row must not change; the generator-formula columns must) and SELF-EXCLUSION (a snapshot row's own
line, raised after t0, never enters its row; offender: dropping the order-visibility requirement admits it).

  python ml/data/clean_panel.py build --world v8        # -> ml/artifacts/cache/v8clean/
"""
from __future__ import annotations
import os, sys, json, shutil, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import numpy as np, pandas as pd
import config

LEAKING = ["fill_rate", "fill_rate_last4", "fill_rate_last13", "fill_rate_last52", "lead_time_actual_days", "lead_time_ratio",
           "otd_rate_last13", "ack_gap_ratio", "load_ratio"]
W0 = pd.Timestamp("2016-01-04")
NEVER = 10 ** 6


def week_index(ts):
    d = pd.to_datetime(pd.Series(np.asarray(ts))).dt.normalize()
    ws = d - pd.to_timedelta(d.dt.weekday, unit="D")
    out = ((ws - W0).dt.days // 7).to_numpy(float)
    return np.where(np.isfinite(out), out, NEVER).astype(np.int64)


class CsvSources:
    def __init__(self, world, T):
        D = config.WORLDS[world]
        self.T = T
        ch = pd.read_csv(f"{D}/sourcing_channels.csv", usecols=["channel_id", "supplier_id", "contracted_lead_time_days"])
        self.NCH = len(ch); cidx = {c: i for i, c in enumerate(ch.channel_id)}
        sups = sorted(ch.supplier_id.unique()); sidx = {s: i for i, s in enumerate(sups)}
        self.ch_sup = ch.supplier_id.map(sidx).to_numpy(); self.NS = len(sups)
        self.contracted = ch.contracted_lead_time_days.to_numpy(float)
        pl = pd.read_csv(f"{D}/po_lines.csv", usecols=["po_line_id", "channel_id", "qty_ordered", "created_ts", "recorded_ts"])
        self.line_id = pl.po_line_id.to_numpy()
        lidx = pd.Series(np.arange(len(pl)), index=pl.po_line_id)
        self.chan = pl.channel_id.map(cidx).to_numpy()
        self.qty = pl.qty_ordered.to_numpy(float)
        self.created = pd.to_datetime(pl.created_ts).to_numpy()
        self.cweek = week_index(self.created)
        self.vw_ord = np.maximum(self.cweek, week_index(pl.recorded_ts))
        n = len(pl)
        g = pd.read_csv(f"{D}/grn_lines.csv", usecols=["po_line_id", "qty_received", "event_ts", "recorded_ts"])
        g = g.sort_values(["po_line_id", "event_ts"]).drop_duplicates("po_line_id")
        gi = lidx.reindex(g.po_line_id).to_numpy().astype(np.int64)
        self.vw_rec = np.full(n, NEVER, np.int64); self.vw_rec[gi] = np.maximum(week_index(g.event_ts), week_index(g.recorded_ts))
        self.recv = np.zeros(n); self.recv[gi] = g.qty_received.to_numpy(float)
        self.lead = np.full(n, np.nan)
        self.lead[gi] = (pd.to_datetime(g.event_ts).to_numpy() - self.created[gi]) / np.timedelta64(1, "D")
        a = pd.read_csv(f"{D}/supplier_acknowledgements.csv", usecols=["po_line_id", "ack_status", "event_ts", "recorded_ts"])
        a = a[a.ack_status == "rejected"]
        rv = pd.read_csv(f"{D}/po_line_revisions.csv", usecols=["po_line_id", "field_changed", "new_value", "event_ts", "recorded_ts"])
        rv = rv[(rv.field_changed == "qty") & (pd.to_numeric(rv.new_value, errors="coerce") == 0)]
        z = pd.concat([a[["po_line_id", "event_ts", "recorded_ts"]], rv[["po_line_id", "event_ts", "recorded_ts"]]])
        z["vw"] = np.maximum(week_index(z.event_ts), week_index(z.recorded_ts))
        z = z.groupby("po_line_id").vw.min()
        zi = lidx.reindex(z.index).to_numpy()
        ok = ~np.isnan(zi.astype(float))
        self.vw_zero = np.full(n, NEVER, np.int64); self.vw_zero[zi[ok].astype(np.int64)] = z.to_numpy()[ok]
        sc = pd.read_csv(f"{D}/supplier_capacity.csv", usecols=["supplier_id", "capacity_qty_per_month", "effective_from", "recorded_ts"])
        sc = sc[sc.supplier_id.isin(sidx)]
        ef = pd.to_datetime(sc.effective_from)
        self.decl_s = sc.supplier_id.map(sidx).to_numpy(); self.decl_m = ((ef.dt.year - 2016) * 12 + ef.dt.month - 1).to_numpy()
        self.decl_q = sc.capacity_qty_per_month.to_numpy(float); self.decl_vw = week_index(sc.recorded_ts)
        weeks = W0 + pd.to_timedelta(7 * np.arange(T), unit="D")
        self.monthkey = ((weeks.year - 2016) * 12 + weeks.month - 1).to_numpy()


def _ffill(x):
    return pd.DataFrame(x).ffill(axis=1).to_numpy()


def _roll(x, k):
    return pd.DataFrame(x).T.rolling(k, min_periods=1).mean().T.to_numpy()


def clean_columns(S, require_order_visible=True):
    """-> {col: [NCH, T]} for the nine LEAKING columns, as-of the end of each row's week."""
    NCH, T = S.NCH, S.T
    order_ok = (S.vw_ord < T) if require_order_visible else np.ones(len(S.qty), bool)
    # ---- lead / ratio / on-time, keyed to the week receipt AND order are visible
    vis = np.where(require_order_visible, np.maximum(S.vw_rec, S.vw_ord), S.vw_rec)
    m = np.isfinite(S.lead) & (vis < T) & order_ok
    o = np.argsort(vis[m], kind="stable")                 # later rows overwrite earlier ones in the same week
    idx = np.flatnonzero(m)[o]
    leadw = np.full((NCH, T), np.nan); leadw[S.chan[idx], vis[idx]] = S.lead[idx]
    lead_ff = _ffill(leadw)
    otd = np.where(np.isnan(leadw), np.nan, (leadw <= S.contracted[:, None]).astype(float))
    # ---- fill, keyed to the visible receipt (or the visible zero signal for a line closed at zero)
    rec_vis = np.where(require_order_visible, np.maximum(S.vw_rec, S.vw_ord), S.vw_rec)
    zero_vis = np.where(require_order_visible, np.maximum(S.vw_zero, S.vw_ord), S.vw_zero)
    # decided by VISIBILITY only: a line closes at zero if its zero signal is visible before any receipt (it then stays
    # at 0 in the history; a later receipt does not revise it -- stated simplification). A first version also required the
    # line's EVENTUAL received quantity to be 0, which the future-poison test caught as a leak.
    zero = (zero_vis < rec_vis) & (zero_vis < T)
    wk = np.where(zero, zero_vis, np.where(rec_vis < T, rec_vis, NEVER))
    fv = np.where(zero, 0.0, np.minimum(S.recv / np.maximum(S.qty, 1), 1.0))
    k = (wk < T) & order_ok
    cw = S.chan[k] * T + wk[k]
    num = np.bincount(cw, weights=fv[k] * S.qty[k], minlength=NCH * T); den = np.bincount(cw, weights=S.qty[k], minlength=NCH * T)
    fillw = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan).reshape(NCH, T)
    f_ff = _ffill(fillw)
    # ---- load: quantity ordered in the current month and visible by the row's week / declared capacity recorded by then
    cm = S.monthkey[np.clip(S.cweek, 0, T - 1)]
    vo = np.clip(S.vw_ord, 0, NEVER)
    inm = (S.cweek >= 0) & (S.cweek < T) & (vo < T)
    same = inm.copy(); same[inm] = S.monthkey[vo[inm]] == cm[inm]          # visible within its own month
    sup = S.ch_sup[S.chan]
    A = np.bincount(sup[same] * T + vo[same], weights=S.qty[same], minlength=S.NS * T).reshape(S.NS, T)
    cum = np.zeros_like(A)
    for mo in np.unique(S.monthkey):
        cols = np.flatnonzero(S.monthkey == mo)
        cum[:, cols] = np.cumsum(A[:, cols], 1)
    decl = np.full((S.NS, T), np.nan)
    order = np.lexsort((S.decl_vw, S.decl_m))
    for s, mo, q, vw in zip(S.decl_s[order], S.decl_m[order], S.decl_q[order], S.decl_vw[order]):
        cols = np.flatnonzero((S.monthkey >= mo) & (np.arange(T) >= vw))   # usable from the week it is recorded, for its month on
        if len(cols):
            # a later-effective declaration (recorded by then) supersedes an earlier one
            decl[s, cols] = q
    load_s = cum / decl
    load = load_s[S.ch_sup]
    return {"fill_rate": f_ff, "fill_rate_last4": _roll(f_ff, 4), "fill_rate_last13": _roll(f_ff, 13), "fill_rate_last52": _roll(f_ff, 52),
            "lead_time_actual_days": lead_ff, "lead_time_ratio": lead_ff / S.contracted[:, None],
            "otd_rate_last13": _roll(_ffill(otd), 13), "ack_gap_ratio": 1 - f_ff, "load_ratio": load}


def masked_view(panel, miss, meta):
    """Feature-view switch for STORED-model reads: the leaking columns zeroed, their indicators set to missing."""
    P = np.array(panel, np.float32, copy=True); Mi = np.array(miss, np.float32, copy=True)
    for c in LEAKING:
        P[:, :, meta["cols"].index(c)] = 0.0
        if c in meta["nullable"]:
            Mi[:, :, meta["nullable"].index(c)] = 0.0
    return P, Mi


def build(world):
    sys.path.insert(0, os.path.join(HERE, "..", "eval"))
    import phase12_common as C
    import phase21_paths as PP
    PP.register()
    st = C.require_clean()
    src_dir = os.path.join(config.CACHE, world); out_dir = os.path.join(config.CACHE, world + "clean")
    meta = json.load(open(os.path.join(src_dir, "meta.json")))
    panel = np.load(os.path.join(src_dir, "panel.npy")); miss = np.load(os.path.join(src_dir, "miss.npy"))
    S = CsvSources(world, meta["T"])
    assert S.NCH == meta["n_channels"]
    Cc = clean_columns(S)
    diff = {}
    for c, v in Cc.items():
        j = meta["cols"].index(c); jm = meta["nullable"].index(c)
        old = panel[:, :, j].astype(float); oldm = miss[:, :, jm] > 0
        fin = np.isfinite(v)
        diff[c] = dict(observed_share_clean=float(fin.mean()), observed_share_stored=float(oldm.mean()),
                       mean_abs_change_where_both=float(np.abs(old[fin & oldm] - v[fin & oldm]).mean()))
        panel[:, :, j] = np.where(fin, v, 0.0).astype(np.float32)
        miss[:, :, jm] = fin.astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "panel.npy"), panel); np.save(os.path.join(out_dir, "miss.npy"), miss)
    shutil.copyfile(os.path.join(src_dir, "active.npy"), os.path.join(out_dir, "active.npy"))
    meta2 = dict(meta, world=world + "clean", clean_of=world, replaced_columns=LEAKING, built_by="ml/data/clean_panel.py",
                 code_commit=st["code_commit"], replacement_summary=diff)
    json.dump(meta2, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(json.dumps(diff, indent=1))
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["build"]); ap.add_argument("--world", default="v8")
    a = ap.parse_args()
    build(a.world)
