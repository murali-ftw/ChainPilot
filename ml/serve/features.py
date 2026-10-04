"""Serving-side as-of feature builder for the Phase 18-19 families: forward season, forward load, cadence.

Built FROM SOURCE ROWS at each requested snapshot t0, by the committed feature modules (no stored feature array is read):
  fwd_load  ml/data/fwd_load.snapshot_features  -- every source row asserted recorded_ts <= t0; part_demand_weekly's recorded
            time is as_of_date + 2 days (deviation 162)
  fwd_season  the NaN-aware network mean of that same fwd_load block (ml/data/fwd_season's definition)
  cadence   ml/data/cadence.snapshot_cadence     -- every source row asserted recorded_ts <= t0
A failing assertion raises fwd_load.AsOfViolation and serving stops.

Then the bundle's OWN scaler (fitted on its training rows, stored in train_log.json `row_family_scaler`) is applied exactly
as ml/train/phase19_bind.RowStore.transform does: z-score, clip +-5, NaN -> 0, missing indicators for the scaler's columns.
inventory_position_weekly is never read.
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "data")]
import numpy as np, pandas as pd
import fwd_load as FL, cadence as CD
import fwd_season as FS

FAMILY_COLS = {"fwdload": list(FL.COLS), "season": list(FS.COLS), "cadence": list(CD.COLS)}


class AsOfFeatures:
    """Per-world source tables, loaded once; blocks built per snapshot on demand and cached."""

    def __init__(self, world="v8"):
        self.world = world
        self.S_load = FL.load_sources(world)
        self.S_cad = CD.load_sources(world)
        self.channels = list(self.S_load["ch"].channel_id)
        self._cache = {}

    def block(self, t0):
        t0 = pd.Timestamp(t0)
        if t0 not in self._cache:
            L, _ = FL.snapshot_features(self.S_load, t0)
            Sn = np.nanmean(L, axis=0).astype(np.float32)
            Cd, _ = CD.snapshot_cadence(self.S_cad, t0)
            self._cache[t0] = dict(fwdload=L, season=Sn, cadence=Cd)
        return self._cache[t0]

    def raw(self, rows, family):
        """rows: labels frame (snapshot_date, key = channel); family: e.g. 'fwdload+season+cadence' -> float64 matrix."""
        cidx = pd.Series(range(len(self.channels)), index=self.channels)
        out = []
        for part in family.split("+"):
            cols = []
            for t0, grp in rows.groupby("snapshot_date", sort=False):
                b = self.block(t0)
                if part == "season":
                    cols.append((grp.index, np.repeat(b["season"][None, :], len(grp), 0)))
                else:
                    ci = cidx.reindex(grp.key.to_numpy()).to_numpy()
                    assert not np.isnan(ci.astype(float)).any(), "a row's channel is not in the world"
                    cols.append((grp.index, b[part][ci.astype(np.int64)]))
            M = np.zeros((len(rows), len(FAMILY_COLS[part])), np.float32)
            pos = pd.Series(range(len(rows)), index=rows.index)
            for idx, block in cols:
                M[pos.loc[idx].to_numpy()] = block
            out.append(M)
        return np.concatenate(out, 1).astype(np.float64)


def transform(R, scaler):
    """Exactly phase19_bind.RowStore.transform, with the bundle's stored scaler."""
    mu, sd = np.asarray(scaler["mu"], float), np.asarray(scaler["sd"], float)
    ind = [scaler["cols"].index(c) for c in scaler["indicator_cols"]]
    Z = np.clip((R - mu) / sd, -5, 5)
    miss = np.isnan(Z)
    Z = np.where(miss, 0.0, Z)
    X = np.concatenate([Z, miss[:, ind].astype(float)], 1).astype(np.float32)
    assert X.shape[1] == scaler["width"], "serving feature width differs from the bundle's"
    return X
