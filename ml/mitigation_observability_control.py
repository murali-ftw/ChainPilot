#!/usr/bin/env python3
"""
Control: is `mitigation_level` actually LATENT, or a near-transform of an emitted feature?

The Phase 1 depth sweep estimates `mitigation_level` at AUC ~0.99 **at every depth, including
h^0** -- the raw input projection. A state that is equally recoverable before and after message
passing is a warning sign, not a result, and the generator says why:

    mitigation_level = min(1, observed_late_rate * (0.5 + RESILIENCE[sup]))   (:1191)

`observed_late_rate` is the supplier's own recorded late fraction. `supplier_temporal_features`
**emits** `on_time_rate_30d/90d/180d` (`db/generate_dataset.py:1485`) and `ml/data/loader.py`
feeds them straight in as Supplier node features (`:309-315`). So the dominant factor of this
"latent" state is a near-deterministic function of columns the model already reads -- the same
disqualifier that dropped Inventory Health in `reports/phase1_latent_state.md`.

This script measures that directly, with no encoder in the loop: it scores the binarised
mitigation target using ONLY the emitted `on_time_rate_*` columns. If that alone approaches the
head's AUC, the head is reading back an observable and the state must not be reported as a
latent-state estimation success.

    python3 ml/mitigation_observability_control.py --variant E --seeds 42,43,44,45,46 --config v1
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

from ml.confirm_latent_states import namespace     # noqa: E402
from ml.data.loader import _read                   # noqa: E402
from ml.hypothesis_ranker import roc_auc           # noqa: E402

STF_COLS = ["on_time_rate_30d", "on_time_rate_90d", "on_time_rate_180d",
            "trend_slope", "lateness_variance", "days_since_last_late",
            "shipment_count_180d"]


def observable_frame(csv_dir: str, t0s) -> dict:
    """Latest emitted supplier_temporal_features row at or before each t0 -- exactly the
    as-of read `ml/data/loader.py::_supplier_features` performs."""
    # `_read` appends the extension itself -- passing a name ending in ".csv" silently
    # yields an empty frame rather than an error.
    stf = _read(csv_dir, "supplier_temporal_features",
                usecols=["supplier_id", "as_of_date"] + STF_COLS,
                parse_dates=("as_of_date",))
    if stf.empty:
        raise RuntimeError(f"no supplier_temporal_features under {csv_dir}")
    out = {}
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
        out[t0.isoformat()] = sub.set_index("supplier_id")[STF_COLS]
    return out


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
        MH, visible = ns["MITIGATION_HISTORY"], ns["VISIBLE_SUP"]
        t0s = list(ns["T0S"])

        by_sup: dict = {}
        for (sid, wk), val in MH.items():
            by_sup.setdefault(sid, []).append((wk, val))
        for sid in by_sup:
            by_sup[sid].sort(key=lambda p: p[0])

        obs = observable_frame(csv_dir, t0s)

        # every candidate state, as (name, {sup_id: {t0_iso: value}}) -> aligned rows
        stress, RESILIENCE = ns["stress"], ns["RESILIENCE"]
        sup_by_id = ns["sup_by_id"]

        def mitigation_at(sid, t0):
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

        targets = {
            "mitigation_level": mitigation_at,
            "supply_stress": lambda sid, t0: stress(sid, sup_by_id[sid]["base_rel"], t0),
            "recovery_capability": (lambda sid, t0: RESILIENCE.get(sid)) if RESILIENCE else None,
        }

        for state, fn in targets.items():
            if fn is None:
                continue
            y, feats = [], []
            for t0 in t0s:
                frame = obs[t0.isoformat()]
                for sid in sorted(visible):
                    if sid not in frame.index or sid not in sup_by_id:
                        continue
                    val = fn(sid, t0)
                    if val is None:
                        continue
                    y.append(val)
                    feats.append(frame.loc[sid].to_numpy(dtype=float))
            if not y:
                continue

            y = np.asarray(y, dtype=float)
            X = np.nan_to_num(np.vstack(feats), nan=0.0, posinf=0.0, neginf=0.0)
            yb = (y > np.median(y)).astype(bool)
            if yb.all() or not yb.any():
                continue

            # all emitted stf columns, plain least-squares linear probe, no encoder.
            Xa = np.hstack([X, np.ones((len(X), 1))])
            w, *_ = np.linalg.lstsq(Xa, yb.astype(float), rcond=None)
            auc_all = roc_auc(Xa @ w, yb)

            rows.append({"seed": dseed, "state": state, "n": int(len(y)),
                         "pos": int(yb.sum()), "auc_emitted_only": auc_all})
            print(f"  seed {dseed} {state:<22} n={len(y):>6,} pos={int(yb.sum()):>6,}  "
                  f"AUC(emitted cols only, no encoder)={auc_all:.4f}", flush=True)

    print("\n" + "=" * 92)
    print("OBSERVABLE-ONLY CONTROL — no encoder, no frozen backbone, emitted columns only")
    print("=" * 92)
    print(f"{'state':<24} {'emitted-only AUC':>18} {'backbone head AUC':>19} {'representation adds':>21}")
    print("-" * 92)
    HEAD = {"mitigation_level": 0.9855, "supply_stress": 0.6509,
            "recovery_capability": 0.6530}          # best-depth values, Variant E
    summary = {}
    for state in ("mitigation_level", "supply_stress", "recovery_capability"):
        vals = [r["auc_emitted_only"] for r in rows if r["state"] == state]
        if not vals:
            continue
        m = statistics.fmean(vals)
        summary[state] = {"emitted_only_auc": m, "per_seed": vals,
                          "backbone_head_auc": HEAD.get(state),
                          "representation_adds": (HEAD[state] - m) if state in HEAD else None}
        print(f"{state:<24} {m:>18.4f} {HEAD.get(state, float('nan')):>19.4f} "
              f"{HEAD[state] - m:>+21.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"variant": args.variant, "config": args.config,
                       "per_seed": rows, "summary": summary}, f, indent=2, default=str)
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
