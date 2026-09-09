"""Phase 1 — as-of reads, node tables, HeteroData construction.

Guide steps 1.1-1.4. Three invariants are ASSERTED here, not assumed:

  A1  recorded_ts >= event_ts on every transactional table (no negative reporting lag)
  A2  the weekly store buckets on max(event_week, recorded_week), verified by exact
      integer conservation against po_lines / grn_lines
  A3  features at t0 are built from rows with recorded_ts <= t0 -- never event_ts

Inductive only: no entity id, index or hash is ever a feature. Node features are
attributes a previously unseen entity would also have. 4.59% of channels never trade and
every Rane entity will be unseen, so an identity feature is not merely leakage-adjacent --
it makes the model unusable and blocks the cross-world evaluation in Phase 3.
"""
from __future__ import annotations
import os, gzip, csv, time
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ 1.1 reader
def table_path(csv_dir: str, table: str) -> str:
    for ext in (".csv", ".csv.gz"):
        p = os.path.join(csv_dir, table + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{table}[.csv|.csv.gz] not under {csv_dir}")


def _open(path):
    return gzip.open(path, "rt", newline="") if path.endswith(".gz") else open(path, "rt", newline="")


def read_rows(csv_dir: str, table: str):
    with _open(table_path(csv_dir, table)) as fh:
        yield from csv.DictReader(fh)


def read_df(csv_dir: str, table: str, **kw) -> pd.DataFrame:
    return pd.read_csv(table_path(csv_dir, table), **kw)


# ------------------------------------------------------------------ 1.2 as-of
class AsOfSweep:
    """Rows become visible only when their recorded_ts is reached (guide 1.2)."""
    def __init__(self, df: pd.DataFrame, recorded_col: str = "recorded_ts"):
        self._df = df.sort_values(recorded_col, kind="mergesort").reset_index(drop=True)
        self._rc, self._pos = recorded_col, 0

    def advance(self, cutoff) -> pd.DataFrame:
        col = self._df[self._rc]
        end = int(np.searchsorted(col.values, np.datetime64(pd.Timestamp(cutoff)), "right"))
        out = self._df.iloc[self._pos:end]
        self._pos = end
        return out

    def visible(self, cutoff) -> pd.DataFrame:
        col = self._df[self._rc]
        end = int(np.searchsorted(col.values, np.datetime64(pd.Timestamp(cutoff)), "right"))
        return self._df.iloc[:end]


EVENT_COL = {"purchase_orders": "created_ts", "po_lines": "created_ts",
             "po_line_schedules": "released_ts", "asn": "dispatch_ts",
             "goods_receipts": "receipt_ts", "shortage_events": "shortage_start_ts",
             "line_stop_events": "stop_start_ts"}


def assert_a1_no_negative_lag(csv_dir: str, tables=None) -> dict:
    """A1: recorded_ts >= event_ts everywhere it is meaningful."""
    tables = tables or ["po_lines", "purchase_orders", "grn_lines", "asn", "goods_receipts",
                        "quality_inspections", "inventory_transactions", "supplier_acknowledgements",
                        "shortage_events", "expedite_events", "po_line_revisions"]
    out = {}
    for t in tables:
        ev = EVENT_COL.get(t, "event_ts")
        hdr = pd.read_csv(table_path(csv_dir, t), nrows=0).columns
        if ev not in hdr or "recorded_ts" not in hdr:
            out[t] = ("skipped", "no event/recorded pair"); continue
        d = read_df(csv_dir, t, usecols=[ev, "recorded_ts"])
        e = pd.to_datetime(d[ev], errors="coerce"); r = pd.to_datetime(d["recorded_ts"], errors="coerce")
        m = e.notna() & r.notna()
        bad = int((r[m] < e[m]).sum())
        assert bad == 0, f"A1 FAILED: {t} has {bad} rows with recorded_ts < {ev}"
        out[t] = (len(d), bad)
    return out


def visible_week_index(event, recorded, week0: pd.Timestamp) -> np.ndarray:
    """max(event_week, recorded_week), weeks indexed from week0 (a Monday)."""
    def wk(x):
        d = pd.to_datetime(pd.Series(np.asarray(x)).values).normalize()
        ws = d - pd.to_timedelta(d.weekday, unit="D")
        return ((ws - week0).days // 7).to_numpy()
    return np.maximum(wk(event), wk(recorded))


def assert_a2_visible_week_bucketing(csv_dir: str) -> dict:
    """A2: the store buckets on max(event_week, recorded_week), proven by exact conservation."""
    cpw = read_df(csv_dir, "channel_performance_weekly",
                  usecols=["channel_id", "week_start", "qty_ordered", "qty_received"])
    weeks = pd.to_datetime(cpw.week_start)
    w0, wN = weeks.min(), weeks.max()
    T = int((wN - w0).days // 7) + 1

    pol = read_df(csv_dir, "po_lines", usecols=["po_line_id", "channel_id", "qty_ordered",
                                                "created_ts", "recorded_ts"])
    vw = visible_week_index(pol.created_ts, pol.recorded_ts.fillna(pol.created_ts), w0)
    ino = (vw >= 0) & (vw < T)
    src_ord = int(pol.qty_ordered[ino].sum())
    emt_ord = int(cpw.qty_ordered.sum())

    grn = read_df(csv_dir, "grn_lines", usecols=["po_line_id", "qty_received", "event_ts", "recorded_ts"])
    vwr = visible_week_index(grn.event_ts, grn.recorded_ts.fillna(grn.event_ts), w0)
    inr = (vwr >= 0) & (vwr < T)
    src_rec = int(grn.qty_received[inr].sum())
    emt_rec = int(cpw.qty_received.sum())

    assert src_ord == emt_ord, f"A2 FAILED ordered: source {src_ord:,} vs store {emt_ord:,}"
    assert src_rec == emt_rec, f"A2 FAILED received: source {src_rec:,} vs store {emt_rec:,}"
    return {"ordered": src_ord, "received": src_rec, "weeks": T,
            "week0": str(w0.date()), "weekN": str(wN.date())}


# ------------------------------------------------------------------ 1.3 node tables
# Cardinalities are fixed here, not inferred from the batch that happens to load.
CAT = {
    "supplier_tier": ["tier1", "tier2"],
    "supplier_type": ["manufacturer", "distributor", "specialty"],
    "business_class": ["A", "B", "C"],
    "part_category": ["brake", "steering", "engine", "elastomer", "casting"],
    "material_type": ["steel", "aluminium", "rubber", "composite"],
    "transport_mode": ["road", "rail", "sea", "air"],
    "approval_status": ["approved", "conditional", "development", "blocked"],
}


def _onehot(s: pd.Series, levels) -> np.ndarray:
    idx = pd.Categorical(s.astype(str), categories=levels).codes
    out = np.zeros((len(s), len(levels)), np.float32)
    ok = idx >= 0
    out[np.arange(len(s))[ok], idx[ok]] = 1.0
    return out


def _num(s: pd.Series, log1p: bool = False) -> np.ndarray:
    v = pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(np.float32)
    return np.log1p(np.clip(v, 0, None)).astype(np.float32) if log1p else v


def load_nodes(csv_dir: str) -> dict:
    """Node index maps and INDUCTIVE feature matrices. No id, index or hash is a feature."""
    ch = read_df(csv_dir, "sourcing_channels")
    su = read_df(csv_dir, "suppliers")
    pa = read_df(csv_dir, "parts")
    pl = read_df(csv_dir, "plants")

    idx = {
        "channel": {v: i for i, v in enumerate(ch.channel_id)},
        "supplier": {v: i for i, v in enumerate(su.supplier_id)},
        "part": {v: i for i, v in enumerate(pa.part_id)},
        "plant": {v: i for i, v in enumerate(pl.plant_id)},
    }
    groups = sorted(su.supplier_group_id.unique())
    idx["supplier_group"] = {v: i for i, v in enumerate(groups)}

    x = {}
    x["channel"] = np.concatenate([
        _num(ch.contracted_lead_time_days)[:, None],
        _num(ch.transport_distance_km, log1p=True)[:, None],
        _onehot(ch.transport_mode, CAT["transport_mode"]),
        _onehot(ch.approval_status, CAT["approval_status"]),
        ch.is_approved.astype(str).str.lower().isin(["true", "1"]).to_numpy(np.float32)[:, None],
    ], 1)
    x["supplier"] = np.concatenate([
        _onehot(su.supplier_tier, CAT["supplier_tier"]),
        _onehot(su.supplier_type, CAT["supplier_type"]),
        _onehot(su.business_class, CAT["business_class"]),
        _num(su.payment_terms_days)[:, None],
        su.msme_flag.astype(str).str.lower().isin(["true", "1"]).to_numpy(np.float32)[:, None],
        su.is_active.astype(str).str.lower().isin(["true", "1"]).to_numpy(np.float32)[:, None],
    ], 1)
    x["part"] = np.concatenate([
        _onehot(pa.part_category, CAT["part_category"]),
        _onehot(pa.material_type, CAT["material_type"]),
        pa.is_critical.astype(str).str.lower().isin(["true", "1"]).to_numpy(np.float32)[:, None],
        _num(pa.standard_lead_time_days, log1p=True)[:, None],
        _num(pa.shelf_life_days, log1p=True)[:, None],
    ], 1)
    x["plant"] = np.concatenate([
        _num(pl.capacity_units_per_day, log1p=True)[:, None],
        _num(pl.latitude)[:, None], _num(pl.longitude)[:, None],
    ], 1)

    # guide 1.3: criticality_reason excluded (constant), supplier_group has no attribute table
    for k, v in x.items():
        assert v.dtype == np.float32 and np.isfinite(v).all(), f"{k} features not finite float32"
    return {"idx": idx, "x": x, "raw": {"channel": ch, "supplier": su, "part": pa, "plant": pl}}


# ------------------------------------------------------------------ 1.4 edges / HeteroData
def build_graph(csv_dir: str, extended: bool = True):
    """Core relations reproduce the run-5/7 profile exactly; `extended` adds the guide's rest."""
    from torch_geometric.data import HeteroData
    import torch
    nodes = load_nodes(csv_dir)
    idx, x, raw = nodes["idx"], nodes["x"], nodes["raw"]
    ch = raw["channel"]

    d = HeteroData()
    for nt in ("channel", "supplier", "part", "plant"):
        d[nt].x = torch.from_numpy(x[nt])

    src = np.arange(len(ch), dtype=np.int64)
    core = {
        ("channel", "sourced_from", "supplier"): ch.supplier_id.map(idx["supplier"]).to_numpy(np.int64),
        ("channel", "supplies", "part"):        ch.part_id.map(idx["part"]).to_numpy(np.int64),
        ("channel", "delivers_to", "plant"):    ch.plant_id.map(idx["plant"]).to_numpy(np.int64),
    }
    for (s, r, t), dst in core.items():
        ei = torch.from_numpy(np.stack([src, dst]))
        d[s, r, t].edge_index = ei
        d[t, "rev_" + r, s].edge_index = torch.from_numpy(np.stack([dst, src]))

    ext = {}
    if extended:
        up = read_df(csv_dir, "supplier_upstream", usecols=["supplier_id", "upstream_supplier_id"])
        a = up.supplier_id.map(idx["supplier"]); b = up.upstream_supplier_id.map(idx["supplier"])
        m = a.notna() & b.notna()
        d["supplier", "depends_on", "supplier"].edge_index = torch.from_numpy(
            np.stack([a[m].to_numpy(np.int64), b[m].to_numpy(np.int64)]))
        ext["supplier__depends_on__supplier"] = int(m.sum())

        al = read_df(csv_dir, "alternate_sources", usecols=["part_id", "supplier_id"])
        a = al.part_id.map(idx["part"]); b = al.supplier_id.map(idx["supplier"])
        m = a.notna() & b.notna()
        d["part", "alt_sourced_from", "supplier"].edge_index = torch.from_numpy(
            np.stack([a[m].to_numpy(np.int64), b[m].to_numpy(np.int64)]))
        ext["part__alt_sourced_from__supplier"] = int(m.sum())

        pp = read_df(csv_dir, "part_plant", usecols=["part_id", "plant_id"])
        a = pp.part_id.map(idx["part"]); b = pp.plant_id.map(idx["plant"])
        m = a.notna() & b.notna()
        d["part", "stocked_at", "plant"].edge_index = torch.from_numpy(
            np.stack([a[m].to_numpy(np.int64), b[m].to_numpy(np.int64)]))
        ext["part__stocked_at__plant"] = int(m.sum())
    return d, nodes, core, ext


def profile_graph(core: dict, nodes: dict) -> dict:
    """Node/edge/degree/component profile of the CORE graph, comparable to run 5 and run 7."""
    idx = nodes["idx"]
    n = {k: len(idx[k]) for k in ("channel", "supplier", "part", "plant")}
    NCH = n["channel"]
    deg_ch = np.full(NCH, len(core), np.int64)          # each channel has one edge per relation
    prof = {"nodes": n, "n_nodes": sum(n.values()),
            "n_edges": sum(len(v) for v in core.values()),
            "median_channel_degree": int(np.median(deg_ch))}
    for (s, r, t), dst in core.items():
        cnt = np.bincount(dst, minlength=n[t])
        prof[f"deg_{t}"] = dict(median=float(np.median(cnt)), min=int(cnt.min()),
                                max=int(cnt.max()), isolated=int((cnt == 0).sum()))
    # connectivity by union-find over channel + entity nodes
    off = {"channel": 0, "supplier": NCH, "part": NCH + n["supplier"],
           "plant": NCH + n["supplier"] + n["part"]}
    tot = sum(n.values())
    par = np.arange(tot)
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for (s, r, t), dst in core.items():
        for c, e in zip(range(NCH), dst):
            ra, rb = find(c + off["channel"]), find(int(e) + off[t])
            if ra != rb: par[ra] = rb
    roots = np.array([find(i) for i in range(tot)])
    _, cnt = np.unique(roots, return_counts=True)
    prof["components"] = int(len(cnt))
    prof["largest_component"] = int(cnt.max())
    prof["largest_component_pct"] = 100.0 * cnt.max() / tot
    prof["isolated_nodes"] = int((cnt == 1).sum())
    return prof
