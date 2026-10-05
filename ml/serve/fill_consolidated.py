"""Phase 23AC T2 -- DORMANT fill service: the five Phase 22 neural season + cadence bundles, served on CPU. No training.

Status: DORMANT. Nothing in ml/configs/shipped.json references this module. It serves only under the PROPOSED exception
docs/decisions/proposed_exception_part_demand_weekly.md (NOT ACTIVE until the owner approves in writing). The LightGBM
consolidated model is not served: it was never persisted, and persisting it needs a refit (deviation 225).

Members (world v8clean, fill h0, lr 0.000125, row family season+cadence; trained by ml/train/phase22_train.py bind):
  ml/artifacts/phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s{7,17,27,37,47}_rfseason+cadence

  register_clean()   in-process only, as phase22_train.register_clean: v8clean reads v8's CSVs, its cache is its own
  check_bundle()     the identity guard; raises ml/serve/guard.IdentityMismatch on: a missing bundle dir / config, an
                     incomplete bundle, a missing model file, phase19_identity.bundle_name(cfg) != the expected name, a
                     directory name != the expected name, world != v8clean, row_family != season+cadence, scaler columns
                     != the config's, or a checkpoint.pt / train_log.json SHA-1 != the one pinned below (a tampered model
                     file). A checkpoint that does not load is also re-raised as IdentityMismatch.
  load_member()      exactly as ml/serve/service.Server.load_member: loop.load_bundle, phase5_heads.device_inputs(v8clean,
                     training snapshots, cfg["wsla"]), loop._net(cfg)(..., n_row_feats=cfg["row_family_width"],
                     fill_head=cfg.get("fill_head", "cells22")), load_state_dict(map_location="cpu")
  features()         per row, FROM SOURCE ROWS at its snapshot t0:
                       season   NaN-aware channel mean of the fwd_load block built through ml/serve/_plan_reader.py
                                (the bound as_of_date + 2 d <= t0 asserted on every plan row, then fwd_load.snapshot_features)
                       cadence  ml/data/cadence.snapshot_cadence on sources taken from the SAME reader's load (po_lines sorted
                                exactly as cadence.load_sources sorts them, the plan dropped) -- so no second read of the
                                plan table happens anywhere in this service
                     assembled in the bundle's train_log["row_family_scaler"]["cols"] order (the mapping is asserted to
                     cover every scaler column), then z-score, clip +-5, NaN -> 0, missing indicators (ml/serve/features.transform)
  serve()            per requested test snapshot, P (22 fill cells) per seed and the 5-seed mean, with the label-row indices
                     and their positions in the stored preds_test.npz order (phase5_heads.ordered over the test mask)

    HADES_DEVICE=cpu python ml/serve/fill_consolidated.py --snapshots 0 --rows 5     # serve the first test snapshot
"""
from __future__ import annotations
import os, sys, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "models"),
                os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd, torch
import config
import phase21_paths as PP
import loop as L, phase5_heads as P5, folds
import phase19_identity as PI
import guard as G
import _plan_reader as PR
import cadence as CD
import fwd_season as FS

WORLD = "v8clean"
TASK = "fill_rate"
ROW_FAMILY = "season+cadence"
SEEDS = (7, 17, 27, 37, 47)
ROOT = os.path.join(config.ARTIFACTS, "phase22", "bundles", TASK)
REQUIRED = ("config.json", "checkpoint.pt", "train_log.json", "normaliser.npz", "recalibration.json", "drift_baseline.json")
# pinned at Phase 23AC from the stored Phase 22 bundles (written 2026-10-05, code 2cee433); a changed byte raises
PINNED_SHA1 = {
    7: dict(checkpoint="f811ec2a7e6e69a6a9f275ec75d2213d0574f3a1", train_log="6303a81e3fbd4416a4791df3f6fa3ba0b4f4fffd"),
    17: dict(checkpoint="184db1aede2f8a7f719e92ff2e8fdd244d4d56fa", train_log="703ec2220f6270a4990770105cef0cef4c409dc7"),
    27: dict(checkpoint="645ec0f3a96a8ec73b06de32c12984d13961f727", train_log="2583ab30d96279ba7202c3f7e8effa4275510028"),
    37: dict(checkpoint="7f0d42ceff974d21ba50183706a84ed28bfa4343", train_log="68b03eabba692c1b94d6865b868244e1fef34241"),
    47: dict(checkpoint="8cd4b92695a9d75610ed9685494f7f57629b8aad", train_log="44cb7976a5924ccd6d1216e525c418378a8f0511"),
}
COLMAP = {**{c: ("season", j) for j, c in enumerate(FS.COLS)}, **{c: ("cadence", j) for j, c in enumerate(CD.COLS)}}


def expected_name(seed):
    return f"{WORLD}_none_h0_lr0.000125_s{seed}_rf{ROW_FAMILY}"


def register_clean():
    """In-process only (phase22_train.register_clean, v8clean part): same CSVs as v8, own cache dir."""
    PP.register()
    config.WORLDS[WORLD] = config.WORLDS["v8"]
    config.EXPECTED_PANEL_D[WORLD] = 15
    return config.WORLDS[WORLD]


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_bundle(bdir, name, pinned=None):
    """-> cfg, or raise guard.IdentityMismatch. name: the expected bundle name; pinned: {checkpoint, train_log} SHA-1s."""
    path = os.path.join(bdir, "config.json")
    if not os.path.isdir(bdir) or not os.path.exists(path):
        raise G.IdentityMismatch(f"no bundle at {bdir}")
    cfg = json.load(open(path))
    if not cfg.get("complete"):
        raise G.IdentityMismatch(f"{bdir} is not a complete bundle")
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(bdir, f))]
    if missing:
        raise G.IdentityMismatch(f"{bdir}: missing {missing}")
    c = dict(cfg); c.setdefault("origin", None)
    diffs = {}
    got = PI.bundle_name(c)
    if got != name:
        diffs["name"] = (name, got)
    if os.path.basename(os.path.normpath(bdir)) != name:
        diffs["directory"] = (name, os.path.basename(os.path.normpath(bdir)))
    if cfg.get("world") != WORLD:
        diffs["world"] = (WORLD, cfg.get("world"))
    if cfg.get("row_family") != ROW_FAMILY:
        diffs["row_family"] = (ROW_FAMILY, cfg.get("row_family"))
    if cfg.get("task") != TASK:
        diffs["task"] = (TASK, cfg.get("task"))
    tl = json.load(open(os.path.join(bdir, "train_log.json")))
    sc = tl.get("row_family_scaler") or {}
    if sc.get("cols") != cfg.get("row_family_cols") or sc.get("width") != cfg.get("row_family_width"):
        diffs["scaler"] = ((cfg.get("row_family_cols"), cfg.get("row_family_width")), (sc.get("cols"), sc.get("width")))
    if pinned is not None:
        for key, fn in (("checkpoint", "checkpoint.pt"), ("train_log", "train_log.json")):
            h = sha1(os.path.join(bdir, fn))
            if h != pinned[key]:
                diffs[f"sha1:{fn}"] = (pinned[key], h)
    if diffs:
        raise G.IdentityMismatch(f"refusing to serve {bdir}: expected != on disk for {sorted(diffs)}: {diffs}")
    return cfg


def cadence_sources(S):
    """cadence.load_sources' result, taken from the plan reader's load: po_lines sorted (ci, created_ts) stably, the plan
    removed (snapshot_cadence never reads it; a KeyError would show if it ever did)."""
    Sc = {k: v for k, v in S.items() if k != "pdw"}
    Sc["pol"] = S["pol"].sort_values(["ci", "created_ts"], kind="stable").reset_index(drop=True)
    return Sc


def transform(R, scaler):
    """Copy of ml/serve/features.transform (= phase19_bind.RowStore.transform = phase22_rows.RowStore22.transform)."""
    mu, sd = np.asarray(scaler["mu"], float), np.asarray(scaler["sd"], float)
    ind = [scaler["cols"].index(c) for c in scaler["indicator_cols"]]
    Z = np.clip((R - mu) / sd, -5, 5)
    miss = np.isnan(Z)
    Z = np.where(miss, 0.0, Z)
    X = np.concatenate([Z, miss[:, ind].astype(float)], 1).astype(np.float32)
    assert X.shape[1] == scaler["width"], "serving feature width differs from the bundle's"
    return X


class FillConsolidatedService:
    def __init__(self, root=ROOT, seeds=SEEDS, names=None, pinned=True):
        register_clean()
        self.root, self.seeds = root, tuple(seeds)
        self.names = dict(names) if names else {s: expected_name(s) for s in self.seeds}
        self.pinned = pinned
        self._lb = self._D = self._S = self._Sc = self._order = None
        self._blocks = {}
        self.members = None

    # ------------------------------------------------------------- guard + load
    def check(self, seed):
        """The configured name is checked against the bundle on disk, then against the Phase 22 name for this seed."""
        bdir = os.path.join(self.root, self.names[seed])
        cfg = check_bundle(bdir, self.names[seed], PINNED_SHA1[seed] if self.pinned else None)
        if self.names[seed] != expected_name(seed):
            raise G.IdentityMismatch(f"configured {self.names[seed]} is not the dormant member {expected_name(seed)}")
        return bdir, cfg

    def labels(self):
        if self._lb is None:
            lb = P5.labels(WORLD, TASK); tr, va, te = folds.fixed_split(lb.snapshot_date)
            order = P5.ordered(lb, te)
            pos = np.full(len(lb), -1, np.int64); pos[order] = np.arange(len(order))
            self._lb = (lb, tr, va, te); self._order = (order, pos)
        return self._lb

    def inputs(self, cfg):
        if self._D is None:
            lb, tr, _, _ = self.labels()
            self._D = P5.device_inputs(WORLD, np.sort(lb.snapshot_date[tr].unique()), cfg["wsla"],
                                       graph_shuffle=cfg.get("graph_shuffle"), drop_relation=cfg.get("drop_relation"))
        return self._D

    def load_member(self, seed):
        bdir, cfg = self.check(seed)                     # raises IdentityMismatch before anything is loaded
        B = L.load_bundle(bdir); D = self.inputs(cfg)
        m = L._net(cfg)(D["X"].shape[2], cfg["task"], cfg["arch"], cfg["depth"], fill_loss=cfg["fill_loss"],
                        n_row_feats=int(cfg["row_family_width"]), fill_head=cfg.get("fill_head", "cells22")).to(L.DEV)
        try:
            m.load_state_dict(torch.load(os.path.join(bdir, "checkpoint.pt"), map_location="cpu"))
        except Exception as e:                          # a tampered file that slipped past an unpinned guard
            raise G.IdentityMismatch(f"{bdir}: checkpoint does not load into the configured model: {type(e).__name__}: {e}")
        tl = json.load(open(os.path.join(bdir, "train_log.json")))
        sc = tl["row_family_scaler"]
        missing = [c for c in sc["cols"] if c not in COLMAP]
        assert not missing, f"scaler columns with no source family: {missing}"
        return dict(seed=seed, model=m.eval(), cfg=cfg, D=D, tl=tl, B=B, dir=bdir, scaler=sc)

    def load(self):
        if self.members is None:
            self.members = [self.load_member(s) for s in self.seeds]
        return self.members

    # ------------------------------------------------------------- features (as-of, from source rows)
    def sources(self):
        if self._S is None:
            self._S = PR.load_sources(WORLD)
            self._Sc = cadence_sources(self._S)
        return self._S, self._Sc

    def block(self, t0):
        t0 = pd.Timestamp(t0)
        if t0 not in self._blocks:
            S, Sc = self.sources()
            b = PR.snapshot_blocks(S, t0)
            Cd, n_cad = CD.snapshot_cadence(Sc, t0)
            self._blocks[t0] = dict(season=b["season"], cadence=Cd, n_plan_rows_asserted=b["n_plan_rows_asserted"],
                                    n_cadence_rows_asserted=int(n_cad))
        return self._blocks[t0]

    def raw(self, rows, cols):
        """rows: label rows (snapshot_date, key = channel_id) -> float64 [n, len(cols)] in `cols` order."""
        S, _ = self.sources()
        missing = [c for c in cols if c not in COLMAP]
        assert not missing, f"no source family for scaler columns {missing}"
        R = np.full((len(rows), len(cols)), np.nan)
        snap = pd.to_datetime(rows.snapshot_date).to_numpy()
        ci = pd.Series(S["cidx"]).reindex(rows.key.to_numpy()).to_numpy()
        assert not np.isnan(ci.astype(float)).any(), "a row's channel is not in the world"
        ci = ci.astype(np.int64)
        for s in np.unique(snap):
            r = np.flatnonzero(snap == s); b = self.block(s)
            for k, c in enumerate(cols):
                fam, j = COLMAP[c]
                R[r, k] = b["season"][j] if fam == "season" else b["cadence"][ci[r], j]
        return R

    def features(self, rows, scaler):
        return transform(self.raw(rows, scaler["cols"]), scaler)

    # ------------------------------------------------------------- predict
    def predict(self, M, snapshots, n_per_snapshot=None):
        """Server.predict for fill_rate. The model runs on the whole snapshot (as phase5_heads.predict did when the
        stored preds were written); the first n_per_snapshot rows of each snapshot are returned."""
        lb, tr, va, te = self.labels()
        order, pos = self._order
        want = set(pd.to_datetime(list(snapshots)))
        P, rows = [], []
        with torch.no_grad():
            for t0, tg, ii in P5.batches(TASK, lb, te, M["D"]["W"]):
                s = pd.Timestamp(lb.snapshot_date.values[ii[0]])
                if s not in want:
                    continue
                xr = torch.from_numpy(self.features(lb.iloc[ii], M["scaler"])).to(L.DEV)
                z = P5.forward(M["model"], M["D"], t0, tg["idx"], xrow=xr)
                p = type(M["model"].head).probs(z).float().cpu().numpy()
                k = len(ii) if n_per_snapshot is None else int(n_per_snapshot)
                P.append(p[:k]); rows.append(ii[:k])
        rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        return dict(P=np.concatenate(P) if P else np.zeros((0, 22), np.float32), rows=rows, pos=pos[rows])

    def serve(self, snapshots, n_per_snapshot=None):
        preds = [self.predict(M, snapshots, n_per_snapshot) for M in self.load()]
        rows = preds[0]["rows"]
        assert all(np.array_equal(p["rows"], rows) for p in preds), "members disagree on rows"
        lb = self.labels()[0]
        return dict(per_seed={M["seed"]: p["P"] for M, p in zip(self.members, preds)},
                    mean=np.mean([p["P"] for p in preds], 0), rows=rows, pos=preds[0]["pos"],
                    entity=lb.entity_id.to_numpy()[rows], snapshot=lb.snapshot_date.to_numpy()[rows],
                    members=[M["dir"] for M in self.members], device=str(L.DEV))

    def test_snapshots(self):
        lb, _, _, te = self.labels()
        return sorted(pd.unique(lb.snapshot_date[te]))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default="0", help="comma list of sorted test-snapshot indices")
    ap.add_argument("--rows", type=int, default=5)
    a = ap.parse_args()
    svc = FillConsolidatedService()
    snaps = [svc.test_snapshots()[int(i)] for i in a.snapshots.split(",")]
    out = svc.serve(snaps, a.rows)
    np.set_printoptions(precision=4, suppress=True)
    print(dict(device=out["device"], snapshots=[str(pd.Timestamp(s).date()) for s in snaps], rows=out["rows"].tolist()))
    print("5-seed mean P(fill = 1):", out["mean"][:, -1])
