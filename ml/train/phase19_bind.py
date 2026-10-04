"""Phase 19 Stage 3 -- neural binding of the families that PASSED Gate v2, through the existing per-row input.

The incumbent's own loop.train and loop.finish_bundle run unchanged. Three things are bound into this process only:
  * phase5_heads.HeadNet  -> RowFamilyHeadNet, a NEW subclass that builds the incumbent with n_row_feats = the bound width
    (per-row features concatenated to the channel encoding before the head -- the path Phase 11 / 13 already used)
  * phase5_heads.batches  -> a wrapper that calls the original and attaches xrow to every batch
  * loop.bundle_dir       -> phase19_identity.bundle_dir (axis `row_family`, omitted at default; separate bundle root)
No existing class or file is edited.

What is bound per use case = the union of the families that PASSED Gate v2 for it (ml/artifacts/phase19/gate_v2.json):
  fwd_load_supplier_specific -> fwd_load's 15 columns; fwd_season -> its 15; cadence -> its 5.
Scaling (pre-registered): z-score with the TRAINING rows' NaN-aware mean / sd, clip +-5, NaN -> 0, and one missing
indicator for every column with any NaN on training rows. Every source was asserted recorded_ts <= t0 when it was built.

  python ml/train/phase19_bind.py handshake                  # Stage -1: stored arrival lite h4 s7, 3 epochs
  python ml/train/phase19_bind.py train --task capacity --seed 7
  python ml/train/phase19_bind.py queue --tasks capacity,arrival,fill --deadline "2026-10-04 06:26"
"""
from __future__ import annotations
import os, sys, json, time, argparse, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "eval"), os.path.join(HERE, "..", "data")]
import phase12_common as C
import numpy as np, pandas as pd

TASK = {"arrival": "arrival_week", "fill": "fill_rate", "capacity": "capacity_strain"}
INCUMBENT = {"arrival": ("lite", 4, 2.5e-4), "fill": ("none", 0, 1.25e-4), "capacity": ("mp", 4, 2.5e-4)}
STORED = {"arrival": "arrival_week/v8_lite_h4_lr0.00025_s{s}", "fill": "fill_rate/v8_none_h0_lr0.000125_s{s}",
          "capacity": "capacity_strain/v8_mp_h4_lr0.00025_s{s}"}
SHORT = {"fwd_load_supplier_specific": "fwdload", "fwd_season": "season", "cadence": "cadence"}
CONCURRENCY = 1
STATE = os.path.join(C.ART, "phase19")


def passing_families(task):
    g = json.load(open(os.path.join(STATE, "gate_v2.json")))["gate_v2"]
    return [fam for fam in ("fwd_load_supplier_specific", "fwd_season", "cadence") if g[f"{task}|{fam}"]["verdict"] == "PASS"]


class RowStore:
    """Raw per-row features for any labels frame (snapshot_date, key = channel), and the training-row scaler."""

    def __init__(self, fams):
        import fwd_load as FL, fwd_season as FS, cadence as CD
        self.parts = []
        for fam in fams:
            if fam == "fwd_load_supplier_specific":
                X, snaps, chans, cols, _ = FL.load(); self.parts.append(("chan", X, snaps, chans, list(cols)))
            elif fam == "fwd_season":
                X, snaps, cols = FS.load(); self.parts.append(("snap", X, snaps, None, list(cols)))
            else:
                X, snaps, chans, cols = CD.load(); self.parts.append(("chan", X, snaps, chans, list(cols)))
        self.cols = [c for p in self.parts for c in p[4]]

    def raw(self, rows):
        snap = pd.to_datetime(rows.snapshot_date).dt.strftime("%Y-%m-%d").to_numpy()
        out = []
        for kind, X, snaps, chans, cols in self.parts:
            si = pd.Series(range(len(snaps)), index=[str(s) for s in snaps]).reindex(snap).to_numpy()
            assert not np.isnan(si.astype(float)).any(), "a row's snapshot is missing from the feature store"
            si = si.astype(np.int64)
            if kind == "snap":
                out.append(X[si])
            else:
                ci = pd.Series(range(len(chans)), index=chans).reindex(rows.key.to_numpy()).to_numpy()
                assert not np.isnan(ci.astype(float)).any(), "a row's channel is missing from the feature store"
                out.append(X[si, ci.astype(np.int64)])
        return np.concatenate(out, 1).astype(np.float64)

    def fit_scaler(self, lb, tr):
        R = self.raw(lb[tr])
        self.mu = np.nanmean(R, 0); self.sd = np.nanstd(R, 0)
        self.sd = np.where(np.isfinite(self.sd) & (self.sd > 0), self.sd, 1.0); self.mu = np.where(np.isfinite(self.mu), self.mu, 0.0)
        self.ind = np.flatnonzero(np.isnan(R).any(0))
        self.width = R.shape[1] + len(self.ind)
        return dict(cols=self.cols, mu=self.mu.tolist(), sd=self.sd.tolist(), indicator_cols=[self.cols[i] for i in self.ind],
                    width=self.width, fitted_on=f"training rows ({int(tr.sum())})")

    def transform(self, rows):
        R = self.raw(rows)
        Z = np.clip((R - self.mu) / self.sd, -5, 5)
        miss = np.isnan(Z)
        Z = np.where(miss, 0.0, Z)
        return np.concatenate([Z, miss[:, self.ind].astype(float)], 1).astype(np.float32)


def bind(task, seed, max_epochs=None, families=None):
    import torch
    import loop as L, phase5_heads as P5, folds
    import phase19_identity as PI
    st = C.require_clean()
    fams = families or passing_families(task)
    assert fams, f"{task}: no family passed Gate v2 -- no neural run"
    store = RowStore(fams)
    arch, depth, lr = INCUMBENT[task]
    args = argparse.Namespace(config=os.path.join(C.ML, "configs", "shipped.json"), task=TASK[task], world="v8", seed=seed,
                              arch=arch, depth=depth, lr=lr, max_epochs=max_epochs, tag="")
    cfg = L.resolve(args)
    cfg["row_family"] = "+".join(SHORT[f] for f in fams)
    L.bundle_dir = PI.bundle_dir                                    # this process only
    out = L.bundle_dir(cfg)
    if os.path.exists(os.path.join(out, "config.json")) and json.load(open(os.path.join(out, "config.json"))).get("complete"):
        print(f"[skip] complete bundle exists: {out}", flush=True); return out
    lb0 = P5.labels("v8", TASK[task]); tr, _, _ = folds.fixed_split(lb0.snapshot_date)
    scaler = store.fit_scaler(lb0, tr)
    cfg["row_family_cols"] = scaler["cols"]; cfg["row_family_width"] = scaler["width"]
    Base, orig_batches = P5.HeadNet, P5.batches
    width = scaler["width"]

    class RowFamilyHeadNet(Base):
        """The incumbent HeadNet with the Phase 19 per-row families concatenated before the head."""
        def __init__(self, d_in, task_, arch="none", depth=0, **kw):
            kw["n_row_feats"] = width
            super().__init__(d_in, task_, arch, depth, **kw)

    def batches(task_, lb, mask, W, ymu=0.0, ysd=1.0):
        bs = orig_batches(task_, lb, mask, W, ymu, ysd)
        for t0, tg, ii in bs:
            assert "xrow" not in tg, "the incumbent batch already carries per-row features"
            tg["xrow"] = torch.from_numpy(store.transform(lb.iloc[ii])).to(P5.DEV)
        return bs
    P5.HeadNet, P5.batches = RowFamilyHeadNet, batches
    t0 = time.time()
    model, D, lb, split, log = L.train(cfg)
    assert isinstance(model, RowFamilyHeadNet) and model.n_row_feats == width, "loop.train did not build the bound model"
    log["row_family"] = cfg["row_family"]; log["row_family_scaler"] = scaler
    out = L.finish_bundle(cfg, model, D, lb, split, log, trained_by="ml/train/phase19_bind.py (loop.train with RowFamilyHeadNet)")
    tl = json.load(open(os.path.join(out, "train_log.json")))
    tl.update(concurrency_level=CONCURRENCY, wall_seconds_total=time.time() - t0, code_commit=st["code_commit"])
    json.dump(tl, open(os.path.join(out, "train_log.json"), "w"), indent=1)
    print(f"  {task} s{seed} [{log['stop']}] best_ep {log['best_epoch']} of {log['epochs_run']}  val {log['best_val']:.5f}  "
          f"{(time.time() - t0) / 60:.1f} min  {log['sec_per_epoch']:.1f}s/ep  width {width}  -> {out}", flush=True)
    P5.HeadNet, P5.batches = Base, orig_batches
    return out


def handshake():
    """Stage -1: retrain the stored arrival lite h4 s7 config for 3 epochs on this machine; nothing is written."""
    import loop as L
    st = C.require_clean()
    args = argparse.Namespace(config=os.path.join(C.ML, "configs", "shipped.json"), task="arrival_week", world="v8", seed=7,
                              arch="lite", depth=4, lr=2.5e-4, max_epochs=3, tag="")
    cfg = L.resolve(args)
    t = time.time()
    _, _, _, _, log = L.train(cfg)
    stored = [json.load(open(os.path.join(C.BUND, STORED["arrival"].format(s=s), "train_log.json")))["vals"][:3] for s in C.V8_SEEDS]
    rows = []
    for e in range(3):
        b = [v[e] for v in stored]
        rows.append(dict(epoch=e, here=log["vals"][e], stored_s7=stored[0][e], abs_diff=abs(log["vals"][e] - stored[0][e]),
                         band=[min(b), max(b)], inside=bool(min(b) <= log["vals"][e] <= max(b))))
    res = dict(stamp=st, device=str(L.DEV), seconds=time.time() - t, epochs=rows, inside_band=all(r["inside"] for r in rows),
               loss_here=log["losses"], loss_stored_s7=json.load(open(os.path.join(C.BUND, STORED["arrival"].format(s=7), "train_log.json")))["losses"][:3])
    C.dump(res, "phase19/handshake.json")
    print(json.dumps(res, indent=1))
    assert res["inside_band"], "handshake OUTSIDE the stored band -- label results 'within-machine comparison' and retrain incumbents"


def queue(tasks, deadline):
    """One neural cell at a time. A cell starts only if its estimated time (stored incumbent mean x 1.3) fits before the
    deadline; otherwise it and everything after it is recorded not_run in phase-19-STOPPED.json."""
    dl = datetime.datetime.strptime(deadline, "%Y-%m-%d %H:%M")
    done, not_run = [], []
    for task in tasks:
        est = 1.3 * np.mean([json.load(open(os.path.join(C.BUND, STORED[task].format(s=s), "train_log.json")))["seconds"]
                             for s in C.V8_SEEDS]) / 60
        for s in C.V8_SEEDS:
            if not_run or datetime.datetime.now() + datetime.timedelta(minutes=est) > dl:
                not_run.append(f"{task} s{s}"); continue
            bind(task, s); done.append(f"{task} s{s}")
    if not_run:
        path = os.path.join(C.REPO, "reports", "part2", "phase-19-STOPPED.json")
        json.dump(dict(completed=done, not_run=not_run, reason=f"wall-clock stop {deadline} IST: the next cell's estimate did not fit",
                       written=datetime.datetime.now().isoformat(timespec="seconds")), open(path, "w"), indent=1)
        print(f"STOPPED: {len(not_run)} cells not run -> {path}", flush=True)
    print("queue done:", done, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["handshake", "train", "queue"])
    ap.add_argument("--task", choices=list(TASK))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--tasks", default="capacity,arrival,fill")
    ap.add_argument("--deadline", default="2026-10-04 06:26")
    a = ap.parse_args()
    if a.mode == "handshake":
        handshake()
    elif a.mode == "train":
        bind(a.task, a.seed, a.max_epochs)
    else:
        queue(a.tasks.split(","), a.deadline)
