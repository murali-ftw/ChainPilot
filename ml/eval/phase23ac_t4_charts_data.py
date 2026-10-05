"""Phase 23AC T4 -- chart source CSVs (pure: numpy / pandas / json only; no model, no score is computed here).

Every chart rendered by `phase23ac_t4_charts.py` reads exactly one CSV written here, into
reports/part2/phase23ac/t4/charts/data/. The metrics script (`phase23ac_t4_metrics.py`) builds the payload and calls
`build_all`; this module only reshapes numbers it is handed or numbers it reads from named stored artifacts.

  chart1_headline.csv            headline metric per use case, three columns, interval, base rate, Phase 15 class
  chart2_precision_vs_base.csv   precision at the flagged cases (top 5%) against the base rate, three columns
  chart3_capacity_snapshots.csv  capacity precision at 5% per test snapshot, worst snapshot marked, per column
  chart4_order_time.csv          order-time expected-date error vs promise date and channel averages (Track T1's
                                 compare_v8.json); written only if that file exists
  chart5_leak_columns.csv        the nine leaking columns (share of channels changed under future poison, QUOTED
                                 from phase22/leakscan_v8.json)
  chart5_leak_drop.csv           drop per use case when the leak is removed (QUOTED from
                                 phase22/restate_arrival_fill_capacity.json, neural ensemble, block interval)
"""
from __future__ import annotations
import os, re, json
import numpy as np, pandas as pd

COLUMNS = ("INCUMBENT", "BEST PUBLISHED", "CLEAN")
FILES = {"chart1": "chart1_headline.csv", "chart2": "chart2_precision_vs_base.csv", "chart3": "chart3_capacity_snapshots.csv",
         "chart4": "chart4_order_time.csv", "chart5_columns": "chart5_leak_columns.csv", "chart5_drop": "chart5_leak_drop.csv"}

H1 = ["use_case", "metric", "column", "value", "lo", "hi", "interval", "base_rate", "phase15_class", "label", "status",
      "higher_is_better", "source"]
H2 = ["use_case", "column", "precision_at_5pct", "lo", "hi", "interval", "base_rate", "label", "status", "source"]
H3 = ["snapshot", "column", "arm", "precision_at_5pct", "is_worst", "label", "status", "source"]
H4 = ["estimator", "a3_days", "share_within_7d", "signed_bias_days", "json_path", "status", "source"]
H5C = ["column_name", "table", "changes_under_poison_H_week", "share_channels_changed_H_week", "status", "source"]
H5D = ["use_case", "metric", "published", "clean", "delta_clean_minus_published", "lo", "hi", "verdict", "higher_is_better",
       "status", "source"]


def write_csv(rows, path, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    extra = {k for r in rows for k in r} - set(header)
    assert not extra, f"unexpected columns {extra}"
    pd.DataFrame(rows, columns=header).to_csv(path, index=False)
    return path


# ------------------------------------------------------------------ chart 1 / 2 / 3: reshaping the pack's own numbers
def chart1_rows(headlines):
    """headlines: list of dicts with the H1 keys (missing keys -> empty)."""
    out = []
    for h in headlines:
        assert h["column"] in COLUMNS, h["column"]
        out.append({k: h.get(k, "") for k in H1})
    return out


def chart2_rows(prec):
    out = []
    for h in prec:
        assert h["column"] in COLUMNS, h["column"]
        out.append({k: h.get(k, "") for k in H2})
    return out


def chart3_rows(per_snapshot):
    """per_snapshot: list of dict(column, arm, values={snapshot: precision}, label, status, source). The worst snapshot of
    each column is marked."""
    out = []
    for d in per_snapshot:
        vals = {k: float(v) for k, v in d["values"].items() if v is not None and np.isfinite(v)}
        worst = min(vals, key=vals.get) if vals else None
        for snap, v in sorted(d["values"].items()):
            out.append(dict(snapshot=snap, column=d["column"], arm=d["arm"], precision_at_5pct=v, is_worst=int(snap == worst),
                            label=d.get("label", ""), status=d.get("status", "RECOMPUTED"), source=d.get("source", "")))
    return out


# ------------------------------------------------------------------ chart 4: Track T1's comparison (schema-tolerant)
_A3 = re.compile(r"^(a3|a3_days|a3_median_abs_err_days|median_abs_err_days|a3_median_days)$", re.I)


def _first(d, pred):
    for k, v in d.items():
        if pred(k) and isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        if pred(k) and isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], (int, float)):
            return float(v[1] if len(v) == 3 else v[0])
    return None


def find_estimator_rows(obj, path=""):
    """Walk a JSON object; every dict that carries an A3 key is one estimator row (its path names it). Rows under a
    per-month key are dropped (chart 4 is the overall comparison)."""
    rows = []
    if isinstance(obj, dict):
        a3 = _first(obj, lambda k: bool(_A3.match(str(k))))
        if a3 is not None:
            w7 = _first(obj, lambda k: "7" in str(k) and any(t in str(k).lower() for t in ("within", "share", "pm7", "le7")))
            bias = _first(obj, lambda k: "bias" in str(k).lower())
            rows.append(dict(json_path=path, a3_days=a3, share_within_7d=w7, signed_bias_days=bias))
            return rows
        for k, v in obj.items():
            rows += find_estimator_rows(v, f"{path}/{k}" if path else str(k))
    return rows


def chart4_rows(t1_path):
    """-> rows, or None if Track T1's file does not exist (chart 4 is then skipped with a message)."""
    if not os.path.exists(t1_path):
        return None
    rows = [r for r in find_estimator_rows(json.load(open(t1_path)))
            if not re.search(r"month|by_m|per_m|/\d{1,2}$|block|(^|[/_])ci([/_]|$)|interval|w2|world2|v8w1002", r["json_path"], re.I)]
    out = []
    for r in rows:
        out.append(dict(estimator=r["json_path"].split("/")[-1] or r["json_path"], a3_days=r["a3_days"],
                        share_within_7d=r["share_within_7d"], signed_bias_days=r["signed_bias_days"], json_path=r["json_path"],
                        status="QUOTED (Track T1)", source=os.path.relpath(t1_path, _repo())))
    return out


# ------------------------------------------------------------------ chart 5: the leak story (QUOTED from stored artifacts)
def chart5_rows(leakscan_path, restate_path):
    L = json.load(open(leakscan_path))
    cols = []
    for c in L["leaking_H_week"]:
        t = L["table"][c]
        cols.append(dict(column_name=c, table="channel_performance_weekly", changes_under_poison_H_week=bool(t["changes_H_week"]),
                         share_channels_changed_H_week=float(t["share_channels_changed_H_week_mean"]),
                         status="QUOTED", source=os.path.relpath(leakscan_path, _repo())))
    assert len(cols) == 9, f"expected the nine Phase 22 leaking columns, got {len(cols)}"
    R = json.load(open(restate_path))["tasks"]
    pick = (("UC1 late list", "arrival", "uc1_precision_at_5pct", True), ("arrival lateness AUC", "arrival", "lateness_auc", True),
            ("arrival A3 (days)", "arrival", "a3_median_abs_err_days", False), ("UC2 P(full) AUC", "fill", "p_full_auc", True),
            ("UC2b precision @ 5%", "fill", "uc2b_precision_at_5pct", True), ("fill CRPS", "fill", "crps_exact", False),
            ("UC3 precision @ 5%", "capacity", "precision_at_5pct", True))
    drops = []
    for uc, task, m, hi in pick:
        d = R[task]["leak_delta"]["neural"]["block"][m]
        lo_, mid, up = d["block_ci_better_positive"]
        sgn = 1.0 if hi else -1.0                      # stored CI is "better = positive"; restate it as clean - published
        lo, up = sorted((sgn * lo_, sgn * up))
        drops.append(dict(use_case=uc, metric=m, published=d["published"], clean=d["clean"],
                          delta_clean_minus_published=d["delta_clean_minus_published"], lo=lo, hi=up, verdict=d["verdict"],
                          higher_is_better=int(hi), status="QUOTED", source=os.path.relpath(restate_path, _repo())))
    return cols, drops


def _repo():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def build_all(payload, out_dir):
    """payload: dict(headline=[...], precision_vs_base=[...], capacity_snapshots=[...], t1_path, leakscan_path,
    restate_path). -> {chart: path or message}."""
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    res["chart1"] = write_csv(chart1_rows(payload["headline"]), os.path.join(out_dir, FILES["chart1"]), H1)
    res["chart2"] = write_csv(chart2_rows(payload["precision_vs_base"]), os.path.join(out_dir, FILES["chart2"]), H2)
    res["chart3"] = write_csv(chart3_rows(payload["capacity_snapshots"]), os.path.join(out_dir, FILES["chart3"]), H3)
    r4 = chart4_rows(payload["t1_path"])
    if r4 is None:
        p4 = os.path.join(out_dir, FILES["chart4"])
        res["chart4"] = f"SKIPPED: {os.path.relpath(payload['t1_path'], _repo())} does not exist (Track T1 not done)"
        if os.path.exists(p4):
            os.remove(p4)                               # never leave a stale chart 4 source behind
    elif not r4:
        res["chart4"] = "SKIPPED: Track T1's file has no estimator row with an A3 key"
    else:
        res["chart4"] = write_csv(r4, os.path.join(out_dir, FILES["chart4"]), H4)
    cols, drops = chart5_rows(payload["leakscan_path"], payload["restate_path"])
    res["chart5_columns"] = write_csv(cols, os.path.join(out_dir, FILES["chart5_columns"]), H5C)
    res["chart5_drop"] = write_csv(drops, os.path.join(out_dir, FILES["chart5_drop"]), H5D)
    return res
