"""Phase 22 Stage 3 -- per-(snapshot, channel) family tables and the per-row store the LightGBM and neural fill arms read.

Families (pre-registration D7), every one as-of t0 (each source module asserts recorded_ts <= t0):
  season    ml/data/fwd_season.py (Phase 19; network means of the forward plan, as-of with deviation 162's +2 d bound)
  cadence   ml/data/cadence.py (Phase 19)
  ack       acknowledgement gap at the channel (L4) and supplier (L2) level, from ml/data/grpstats.py Builder.fill(t0):
            mean 1 - ack_qty / ordered and share not-full over acknowledged lines, and over acknowledged lines still open
  L4        the channel-level fill block of grpstats (shrunk 22-bin history at L4, P(fill = 1), shortfall and n at L4 / L2 / L1,
            k = 100 as chosen on validation in Phase 21), no month
The ack and L4 tables are computed per (snapshot, channel) for EVERY channel at every snapshot, so the controls are exact:
a snapshot permutation reads the SAME channel at a donor snapshot of the same split (keeps the channel, breaks time); a
cross-channel shuffle reads ANOTHER channel at the same snapshot (keeps the season, breaks the channel).

  python ml/data/phase22_rows.py build --world v8      -> ml/artifacts/phase22/rows_{world}.npz
"""
from __future__ import annotations
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd
import config
import phase21_paths as PP

OUT = os.path.join(config.ARTIFACTS, "phase22")
K_FILL = 100
ACK_COLS = [f"ack{L}_{n}" for L in ("L4", "L2") for n in ("gap_acked", "notfull_acked", "n_acked", "gap_open", "notfull_open", "n_open")]


def build(world):
    import grpstats as GS
    import phase12_common as C
    PP.register()
    st = C.require_clean()
    t = time.time()
    src = GS.Source(world); B = GS.Builder(src)
    a, _ = GS.snapshot_rows(world)
    snaps = sorted(pd.unique(a.snapshot_date))
    NCH = src.n_ch
    chan = np.arange(NCH); sup = src.ch_sup
    ack = np.zeros((len(snaps), NCH, 12), np.float32)
    l4 = None; cols = None
    for i, s in enumerate(snaps):
        tau = np.datetime64(pd.Timestamp(s), "ns")
        F, _ = B.fill(tau)
        ack[i] = np.concatenate([F["L4_ack"][chan], F["L2_ack"][sup]], 1)
        Z = {"H_L4": F["L4_hist"][chan], "H_L2": F["L2_hist"][sup], "H_L1": np.repeat(F["L1_hist"][[0]], NCH, 0),
             "F_L4": F["L4"][chan], "F_L2": F["L2"][sup], "F_L1": np.repeat(F["L1"][[0]], NCH, 0),
             "H_L5c": np.zeros((NCH, 12, 22), np.float32), "F_L5c": np.full((NCH, 12, 3), np.nan), "F_L3c": np.full((NCH, 12, 3), np.nan),
             "month": np.zeros(NCH, np.int8)}
        blk = GS.assemble_fill(Z, K_FILL, "L4")
        if l4 is None:
            l4 = np.zeros((len(snaps), NCH, blk.shape[1]), np.float32); cols = list(blk.columns)
        l4[i] = blk.to_numpy(np.float32)
    np.savez_compressed(os.path.join(OUT, f"rows_{world}.npz"), snaps=np.array([str(pd.Timestamp(s).date()) for s in snaps]),
                        channels=src.ch.channel_id.to_numpy().astype(str), ack=ack, l4=l4, l4_cols=np.array(cols), ack_cols=np.array(ACK_COLS))
    json.dump(dict(stamp=st, world=world, k_fill=K_FILL, snapshots=len(snaps), seconds=time.time() - t,
                   note="ack and L4 fill tables for every channel at every snapshot; grpstats Builder.fill (as-of asserted)"),
              open(os.path.join(OUT, f"rows_{world}.json"), "w"), indent=1)
    print(f"rows_{world}: {len(snaps)} snapshots x {NCH} channels in {time.time() - t:.0f}s", flush=True)


def load(world):
    z = np.load(os.path.join(OUT, f"rows_{world}.npz"))
    return dict(snaps=list(z["snaps"]), channels=list(z["channels"]), ack=z["ack"], l4=z["l4"], l4_cols=list(z["l4_cols"]),
                ack_cols=list(z["ack_cols"]))


def split_of(snaps):
    d = pd.to_datetime(pd.Series(snaps))
    te = d > pd.Timestamp(config.SPLIT["val_end"]); va = (d > pd.Timestamp(config.SPLIT["train_end"])) & ~te
    return np.where(te, 2, np.where(va, 1, 0))


def derange(groups, rng):
    """donor[i] != i, same group (a cyclic shift of a random order)."""
    donor = np.arange(len(groups))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        p = idx[rng.permutation(len(idx))]
        donor[p] = np.roll(p, -1)
    return donor


def block(fam, world, lb, seed, perm=None):
    """-> DataFrame for lb's rows (snapshot_date, key). perm: None | 'sperm' (snapshot derangement within split, same
    channel) | 'xsh' (another channel, same snapshot)."""
    if fam in ("season", "cadence"):
        import fwd_season as FS, cadence as CD, phase19_proxy as P19
        if fam == "season":
            X, snaps, cols = FS.load(world)
            if perm == "sperm":
                X = X[P19.derangement_within_split(snaps, seed)]
            si, _ = P19.rows_index(lb, snaps)
            return pd.DataFrame(X[si], columns=cols)
        X, snaps, chans, cols = CD.load(world)
        si, ci = P19.rows_index(lb, snaps, chans)
        return pd.DataFrame(X[si, ci], columns=cols)
    R = load(world)
    X = R["ack"] if fam == "ack" else R["l4"]
    cols = R["ack_cols"] if fam == "ack" else R["l4_cols"]
    snap = pd.to_datetime(lb.snapshot_date).dt.strftime("%Y-%m-%d").to_numpy()
    si = pd.Series(range(len(R["snaps"])), index=R["snaps"]).reindex(snap).to_numpy()
    ci = pd.Series(range(len(R["channels"])), index=R["channels"]).reindex(lb.key.to_numpy()).to_numpy()
    assert not np.isnan(si.astype(float)).any() and not np.isnan(ci.astype(float)).any(), "rows missing from the family table"
    si, ci = si.astype(np.int64), ci.astype(np.int64)
    if perm == "sperm":
        donor = derange(split_of(R["snaps"]), np.random.default_rng(90_000 + seed))
        si = donor[si]
    elif perm == "xsh":
        rng = np.random.default_rng(80_000 + seed)
        ci = (ci + rng.integers(1, len(R["channels"]), len(ci))) % len(R["channels"])
    return pd.DataFrame(X[si, ci], columns=[f"{c}" for c in cols])


class RowStore22:
    """Per-row features for the neural binding: z-scored on training rows, clipped +-5, NaN -> 0, plus a missing indicator
    for every column ever missing on training rows (Phase 19's pre-registered scaling)."""

    def __init__(self, fams, world="v8", task="fill_rate"):
        self.fams, self.world = fams, world

    def raw(self, rows):
        rows = rows.copy()
        if "key" not in rows:
            rows["key"] = rows.channel_id
        return pd.concat([block(f, self.world, rows.reset_index(drop=True), 0) for f in self.fams], axis=1).to_numpy(np.float64)

    def fit_scaler(self, lb, tr):
        R = self.raw(lb[np.asarray(tr)])
        self.cols = [c for f in self.fams for c in block(f, self.world, lb.iloc[:2].assign(key=lb.iloc[:2].get("key", lb.iloc[:2].get("channel_id"))).reset_index(drop=True), 0).columns]
        self.mu = np.nanmean(R, 0); self.sd = np.nanstd(R, 0)
        self.sd = np.where(np.isfinite(self.sd) & (self.sd > 0), self.sd, 1.0); self.mu = np.where(np.isfinite(self.mu), self.mu, 0.0)
        self.ind = np.flatnonzero(np.isnan(R).any(0))
        self.width = R.shape[1] + len(self.ind)
        return dict(cols=self.cols, mu=self.mu.tolist(), sd=self.sd.tolist(), indicator_cols=[self.cols[i] for i in self.ind],
                    width=self.width, fitted_on=f"training rows ({int(np.asarray(tr).sum())})")

    def transform(self, rows):
        R = self.raw(rows)
        Z = np.clip((R - self.mu) / self.sd, -5, 5)
        miss = np.isnan(Z)
        Z = np.where(miss, 0.0, Z)
        return np.concatenate([Z, miss[:, self.ind].astype(float)], 1).astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["build"]); ap.add_argument("--world", default="v8")
    a = ap.parse_args()
    build(a.world)
