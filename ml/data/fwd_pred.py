"""Phase 20 Stage 3 -- `fwd_pred`: a LEARNED forecast of each channel's and supplier's ordered quantity in weeks 1-4 / 5-8 /
9-13 after t0. Omitted at default: an arm carries these columns only if it names fwd_pred.

This module holds the as-of inputs and the targets; the forecaster itself (LightGBM) is ml/baselines/phase20_forecaster.py.
  history(world)   per snapshot x channel, from po_lines asserted recorded_ts <= t0: ordered quantity of lines raised in the
                   4 / 13 / 52 weeks to t0, and the line count over 52 weeks   (HIST_COLS)
  targets(world)   per snapshot x channel: realised ordered quantity of the channel's lines CREATED in (t0, t0+28d],
                   (t0+28d, t0+56d], (t0+56d, t0+91d]. These are observed labels (PO lines raised later), used only as
                   training targets and for the forecaster's own accuracy report -- never as an input.
  load(world, seed)  the forecaster's predictions [snapshot, channel, 6] (COLS), written by the forecaster.
inventory_position_weekly is never read.
"""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import numpy as np, pandas as pd
import fwd_load as FL
from config import ARTIFACTS

OUT_DIR = os.path.join(ARTIFACTS, "phase20")
WIN = ((0, 28), (28, 56), (56, 91))
WTAG = ("w1_4", "w5_8", "w9_13")
HIST_COLS = ["ord_qty_4w", "ord_qty_13w", "ord_qty_52w", "ord_lines_52w"]
COLS = [f"fwd_pred_ch_{w}" for w in WTAG] + [f"fwd_pred_sup_{w}" for w in WTAG]


def _pol(world):
    S = FL.load_sources(world)
    return S, S["pol"]


def history(world):
    S, pol = _pol(world)
    snaps = FL.snapshot_dates(world); NCH = len(S["ch"])
    H = np.zeros((len(snaps), NCH, len(HIST_COLS)), np.float32); n = 0
    for i, t0 in enumerate(snaps):
        t0 = pd.Timestamp(t0)
        rec = pol[pol.recorded_ts.values <= t0.to_datetime64()]
        n += FL.assert_asof(rec.recorded_ts, t0, "po_lines (ordering history)")
        for j, wk in enumerate((4, 13, 52)):
            m = rec.created_ts.values > (t0 - pd.Timedelta(weeks=wk)).to_datetime64()
            H[i, :, j] = np.bincount(rec.ci.values[m], weights=rec.qty_ordered.values[m].astype(float), minlength=NCH)
            if wk == 52:
                H[i, :, 3] = np.bincount(rec.ci.values[m], minlength=NCH)
    return H, [str(pd.Timestamp(s).date()) for s in snaps], n


def targets(world):
    S, pol = _pol(world)
    snaps = FL.snapshot_dates(world); NCH = len(S["ch"])
    Y = np.zeros((len(snaps), NCH, 3), np.float32)
    ct = pol.created_ts.values
    for i, t0 in enumerate(snaps):
        t0 = pd.Timestamp(t0)
        for j, (a, b) in enumerate(WIN):
            m = (ct > (t0 + pd.Timedelta(days=a)).to_datetime64()) & (ct <= (t0 + pd.Timedelta(days=b)).to_datetime64())
            Y[i, :, j] = np.bincount(pol.ci.values[m], weights=pol.qty_ordered.values[m].astype(float), minlength=NCH)
    return Y, [str(pd.Timestamp(s).date()) for s in snaps]


def path(world, seed):
    return os.path.join(OUT_DIR, f"fwd_pred_{world}_s{seed}.npz")


def load(world, seed):
    z = np.load(path(world, seed), allow_pickle=False)
    return z["X"], list(z["snapshots"]), list(z["channels"]), list(z["cols"])
