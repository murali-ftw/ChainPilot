#!/usr/bin/env python3
"""
Gate 1 — deterministic raw-signal check for Mitigation Level. No model, no SHARE, no training.

The cheapest possible test: does a trivial, non-learned function of already-recorded observable
history already explain `mitigation_level`? If it does, Layer 3 does not need to exist for this
state, and Gates 2-3's results must be read in that light.

`x` is defined precisely and deliberately:

    x(s,t) = the supplier's own emitted rolling late rate at t
           = 1 - on_time_rate_90d, read from `supplier_temporal_features`
             (`db/generate_dataset.py:1485`), already a model input feature
             (`ml/data/loader.py:309-315`)

Two exclusions matter for this test to mean anything:
  * `x` is NOT SHARE's embedding -- the point is to have no learned representation in the loop.
  * `x` is NOT mitigation's own lag. Regressing mitigation(t) on mitigation(t-1) would be
    circular and would "pass" for any autocorrelated series.

`shipment_count_180d` is reported alongside because the generator's `seen[0] < 3 -> return 0.0`
branch (`:1220-1222`) ties mitigation to observed shipment COUNT, not only to the late rate.

Levels and deltas are both tested, since Gates 2-3 are about the temporal channel specifically.

    python3 ml/mitigation_raw_signal_gate1.py --variant E --seeds 42,43,44,45,46 --config v1
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.confirm_latent_states import namespace   # noqa: E402
from ml.data.loader import _read                 # noqa: E402


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, tie-aware via average ranks."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="E")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--config", default="v1")
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []

    for dseed in seeds:
        csv_dir = os.path.join(args.csv_root, f"v{args.variant}_seed{dseed}")
        ns = namespace(args.variant, dseed, args.config)
        MH, visible, t0s = ns["MITIGATION_HISTORY"], ns["VISIBLE_SUP"], list(ns["T0S"])

        by_sup: dict = {}
        for (sid, wk), val in MH.items():
            by_sup.setdefault(sid, []).append((wk, val))
        for sid in by_sup:
            by_sup[sid].sort(key=lambda p: p[0])

        stf = _read(csv_dir, "supplier_temporal_features",
                    usecols=["supplier_id", "as_of_date", "on_time_rate_90d",
                             "shipment_count_180d"],
                    parse_dates=("as_of_date",))

        # as-of read, identical to ml/data/loader.py::_supplier_features
        obs = {}
        for t0 in t0s:
            t = pd.Timestamp(t0)
            tz = stf["as_of_date"].dt.tz
            if tz is None and t.tz is not None:
                t = t.tz_localize(None)
            elif tz is not None and t.tz is None:
                t = t.tz_localize("UTC")
            sub = stf[stf["as_of_date"] <= t]
            sub = (sub.sort_values(["supplier_id", "as_of_date"], kind="stable")
                      .groupby("supplier_id", as_index=False).tail(1))
            obs[t0] = sub.set_index("supplier_id")

        def mit_at(sid, t0):
            series = by_sup.get(sid)
            if not series:
                return None
            latest = None
            for wk, val in series:
                if wk <= t0:
                    latest = val
                else:
                    break
            return latest

        # levels, and per-supplier consecutive-snapshot deltas
        lv_x, lv_n, lv_m, dl_x, dl_m = [], [], [], [], []
        for sid in sorted(visible):
            prev_x = prev_m = None
            for t0 in t0s:
                frame = obs[t0]
                if sid not in frame.index:
                    prev_x = prev_m = None
                    continue
                m = mit_at(sid, t0)
                if m is None:
                    prev_x = prev_m = None
                    continue
                otr = frame.loc[sid, "on_time_rate_90d"]
                cnt = frame.loc[sid, "shipment_count_180d"]
                if pd.isna(otr):
                    prev_x = prev_m = None
                    continue
                x = 1.0 - float(otr)               # rolling LATE rate
                lv_x.append(x); lv_m.append(m)
                lv_n.append(float(cnt) if not pd.isna(cnt) else 0.0)
                if prev_x is not None:
                    dl_x.append(x - prev_x); dl_m.append(m - prev_m)
                prev_x, prev_m = x, m

        lv_x, lv_m, lv_n = np.array(lv_x), np.array(lv_m), np.array(lv_n)
        dl_x, dl_m = np.array(dl_x), np.array(dl_m)

        r_level = float(np.corrcoef(lv_x, lv_m)[0, 1])
        rho_level = spearman(lv_x, lv_m)
        r_count = float(np.corrcoef(lv_n, lv_m)[0, 1])
        r_delta = float(np.corrcoef(dl_x, dl_m)[0, 1]) if len(dl_x) > 2 else float("nan")
        rho_delta = spearman(dl_x, dl_m) if len(dl_x) > 2 else float("nan")

        # R^2 of the two-feature non-learned linear fit (late rate + shipment count)
        A = np.column_stack([lv_x, lv_n, np.ones(len(lv_x))])
        w, *_ = np.linalg.lstsq(A, lv_m, rcond=None)
        pred = A @ w
        ss_res = float(((lv_m - pred) ** 2).sum())
        ss_tot = float(((lv_m - lv_m.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        rows.append({"seed": dseed, "n_levels": int(len(lv_x)), "n_deltas": int(len(dl_x)),
                     "pearson_level": r_level, "spearman_level": rho_level,
                     "pearson_level_shipment_count": r_count,
                     "pearson_delta": r_delta, "spearman_delta": rho_delta,
                     "r2_two_feature_linear": r2})
        print(f"  seed {dseed}: n={len(lv_x):,} levels / {len(dl_x):,} deltas  "
              f"r(x,mit)={r_level:+.4f} rho={rho_level:+.4f}  "
              f"r(count,mit)={r_count:+.4f}  "
              f"r(dx,dmit)={r_delta:+.4f} rho={rho_delta:+.4f}  "
              f"R2(2-feat)={r2:.4f}", flush=True)

    mean = lambda k: statistics.fmean(r[k] for r in rows)
    print("\n" + "=" * 92)
    print("GATE 1 — raw-signal check: no model, no SHARE, no training")
    print("=" * 92)
    print(f"  LEVELS  Pearson r(late_rate, mitigation)      : {mean('pearson_level'):+.4f}")
    print(f"          Spearman rho                          : {mean('spearman_level'):+.4f}")
    print(f"          Pearson r(shipment_count, mitigation) : {mean('pearson_level_shipment_count'):+.4f}")
    print(f"          R^2, 2-feature linear fit             : {mean('r2_two_feature_linear'):.4f}")
    print(f"  DELTAS  Pearson r(d late_rate, d mitigation)  : {mean('pearson_delta'):+.4f}")
    print(f"          Spearman rho                          : {mean('spearman_delta'):+.4f}")
    print("\n  (informational gate: reported regardless of outcome; does not itself block Gate 2)")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"variant": args.variant, "config": args.config, "per_seed": rows,
                       "means": {k: mean(k) for k in
                                 ("pearson_level", "spearman_level",
                                  "pearson_level_shipment_count", "pearson_delta",
                                  "spearman_delta", "r2_two_feature_linear")}},
                      f, indent=2, default=str)
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
