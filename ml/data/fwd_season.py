"""Phase 19 -- `fwd_season`: ONLY the snapshot-level network means of the fwd_load columns.

For each snapshot t0 and each of the 15 `fwd_load.COLS`, the NaN-aware mean over all channels. The result is one 15-value
vector per snapshot, the same for every row of that snapshot: the network-wide forward requirement (the forward season),
with no supplier or channel identity. These are exactly the values Phase 18's diagnostic `fwd_load_netmean` arm read.

As-of: inherited. Every value is a mean of fwd_load cells, each built from rows asserted recorded_ts <= t0.
Before anything is derived, the stored fwd_load arrays are REBUILT from the unchanged ml/data/fwd_load.py and asserted
equal (the brief: reuse Phase 18's arrays, but prove they are Phase 18's arrays).

  python ml/data/fwd_season.py build        # -> ml/artifacts/phase19/fwd_season_v8.npz
"""
from __future__ import annotations
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import numpy as np, pandas as pd
from config import ARTIFACTS
import fwd_load as FL

OUT_DIR = os.path.join(ARTIFACTS, "phase19")
COLS = [f"{c}_season" for c in FL.COLS]


def verify_fwd_load(world="v8"):
    """Rebuild every snapshot's fwd_load block from source and require equality with the stored Phase 18 array."""
    X, snaps, chans, cols, _ = FL.load(world)
    assert cols == FL.COLS, "stored fwd_load columns differ from the module's"
    S = FL.load_sources(world)
    assert list(S["ch"].channel_id) == chans, "channel order differs"
    for i, t0 in enumerate(snaps):
        F, _ = FL.snapshot_features(S, pd.Timestamp(str(t0)))
        assert np.array_equal(F, X[i], equal_nan=True), f"fwd_load at {t0} differs from the stored Phase 18 array"
    return dict(snapshots=len(snaps), channels=len(chans), equal=True)


def build(world="v8"):
    t = time.time()
    check = verify_fwd_load(world)
    X, snaps, chans, cols, _ = FL.load(world)
    M = np.nanmean(X, axis=1).astype(np.float32)                 # [snapshot, col]
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"fwd_season_{world}.npz")
    np.savez_compressed(path, X=M, snapshots=np.array(snaps), cols=np.array(COLS))
    meta = dict(world=world, fwd_load_rebuilt_equal=check, n_snapshots=len(snaps), cols=COLS, seconds=round(time.time() - t, 1))
    json.dump(meta, open(path.replace(".npz", ".json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    return path


def load(world="v8"):
    z = np.load(os.path.join(OUT_DIR, f"fwd_season_{world}.npz"), allow_pickle=False)
    return z["X"], list(z["snapshots"]), list(z["cols"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build"])
    ap.add_argument("--world", default="v8")
    a = ap.parse_args()
    build(a.world)
