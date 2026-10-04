"""Phase 17 B4 -- the lean encoder on arrival_week (B4b) against the stored incumbent SHARE-lite h4 (B4a).

B4b runs the INCUMBENT'S OWN training loop and bundle writer -- loop.train and loop.finish_bundle -- with one change:
the HeadNet it builds carries share_lean.SHARELean instead of share.SHARE. The swap is a subclass (LeanHeadNet) bound
into phase5_heads for this process only; no incumbent class is edited. Config axis lean_encoder=True (omitted at its
default) names the bundle *_lean, so it can never collide with the incumbent.

  python ml/train/phase17_b4.py equiv               (CPU: lean == C3 part-dropped incumbent on the real graph; falsified)
  python ml/train/phase17_b4.py train --seed 7      (GPU, one cell)
  python ml/train/phase17_b4.py score
"""
from __future__ import annotations
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "eval")]
import phase12_common as C

TASK = "arrival_week"
SEEDS = C.V8_SEEDS
CONCURRENCY = 1


def equiv():
    import temporal_share as TS
    from share_lean import assert_equivalent_to_c3
    st = C.require_clean()
    W = TS.load_world("v8")
    r = assert_equivalent_to_c3(W, TS.DEV)
    r["stamp"] = st; r["device"] = str(TS.DEV)
    print(json.dumps(r, indent=1)); C.dump(r, "phase17/b4_equivalence.json")


def train(seed, max_epochs=None, bundle_root=None):
    import torch
    import loop as L, phase5_heads as P5, temporal_share as TS
    from share_lean import SHARELean
    st = C.require_clean()
    if bundle_root:
        L.BUNDLES = bundle_root                           # smoke runs must not occupy real bundle paths
    Base = P5.HeadNet

    class LeanHeadNet(Base):
        def __init__(self, d_in, task, arch="none", depth=0, **kw):
            super().__init__(d_in, task, arch, depth, **kw)
            assert arch == "lite" and depth == TS.HP["share_layers"], "the lean encoder replaces SHARE-lite h4 only"
            self.enc = SHARELean(TS.HP["tcn_hidden"], TS.HP["graph_hidden"], n_rel=6, n_layers=TS.HP["share_layers"])

    args = argparse.Namespace(config=os.path.join(C.ML, "configs", "shipped.json"), task=TASK, world="v8", seed=seed,
                              arch="lite", depth=4, lr=2.5e-4, max_epochs=max_epochs, tag="")
    cfg = L.resolve(args); cfg["lean_encoder"] = True
    out = L.bundle_dir(cfg)
    if os.path.exists(os.path.join(out, "config.json")) and json.load(open(os.path.join(out, "config.json"))).get("complete"):
        print(f"[skip] complete bundle exists: {out}"); return
    P5.HeadNet = LeanHeadNet                              # this process only: loop.train builds the lean model
    t0 = time.time()
    model, D, lb, split, log = L.train(cfg)
    assert isinstance(model.enc, SHARELean), "loop.train did not build the lean encoder"
    log["encoder"] = "share_lean.SHARELean"; log["graph_encoder_params"] = sum(p.numel() for p in model.enc.parameters())
    out = L.finish_bundle(cfg, model, D, lb, split, log, trained_by="ml/train/phase17_b4.py (loop.train with LeanHeadNet)")
    tl = json.load(open(os.path.join(out, "train_log.json"))); tl["concurrency_level"] = CONCURRENCY
    tl["wall_seconds_total"] = time.time() - t0
    json.dump(tl, open(os.path.join(out, "train_log.json"), "w"), indent=1)
    print(f"  [{log['stop']}] best_ep {log['best_epoch']} of {log['epochs_run']}  val {log['best_val']:.5f}  "
          f"{log['seconds']:.0f}s  {log['sec_per_epoch']:.1f}s/ep  graph params {log['graph_encoder_params']}  -> {out}", flush=True)


def score():
    import numpy as np, pandas as pd
    import phase5_heads as P5, folds
    from metrics import cindex
    from phase11b_lateness import build_reference
    from sklearn.metrics import roc_auc_score
    st = C.require_clean()
    lb = P5.labels("v8", TASK); tr, va, te = folds.fixed_split(lb.snapshot_date)
    R, _ = build_reference("asof_channel_lead", "v8", lb, tr); Rv, Rt = R[P5.ordered(lb, va)], R[P5.ordered(lb, te)]
    BA = os.path.join(C.BUND, TASK)

    def lateness_auc(z, Rx):                      # docs/specs/lateness_metric.md: label Y > R, score P - R, uncensored rows
        ev = z["EV"].astype(bool); y = (z["Y"][ev] > Rx[ev]).astype(int)
        return float(roc_auc_score(y, (z["P"] - Rx)[ev]))
    out = dict(stamp=st, concurrency_level_B4b=CONCURRENCY, arms={})
    for arm, suf in (("B4a_incumbent", ""), ("B4b_lean", "_lean")):
        rows = []
        for s in SEEDS:
            d = os.path.join(BA, f"v8_lite_h4_lr0.00025_s{s}{suf}")
            if not (os.path.exists(os.path.join(d, "config.json")) and json.load(open(os.path.join(d, "config.json"))).get("complete")):
                continue
            zv, zt = dict(np.load(os.path.join(d, "preds_val.npz"))), dict(np.load(os.path.join(d, "preds_test.npz")))
            tl = json.load(open(os.path.join(d, "train_log.json")))
            rows.append(dict(seed=s, test_cindex=float(cindex(zt["P"], zt["Y"], zt["EV"].astype(bool))),
                             val_cindex=float(cindex(zv["P"], zv["Y"], zv["EV"].astype(bool))),
                             test_lateness_auc_asof=lateness_auc(zt, Rt), val_lateness_auc_asof=lateness_auc(zv, Rv),
                             epochs=tl.get("epochs_run"), best_epoch=tl.get("best_epoch"), sec_per_epoch=tl.get("sec_per_epoch"),
                             seconds=tl.get("seconds"), device=tl.get("device", "mps (stored, Mac)"),
                             concurrency_level=tl.get("concurrency_level", "not recorded (stored)"), params=tl.get("params")))
        b = lambda k: [float(min(r[k] for r in rows)), float(np.mean([r[k] for r in rows])), float(max(r[k] for r in rows))] if rows else None
        out["arms"][arm] = dict(seeds=[r["seed"] for r in rows], per_seed=rows,
                                **{k: b(k) for k in ("test_cindex", "val_cindex", "test_lateness_auc_asof", "val_lateness_auc_asof")})
    a, l = out["arms"]["B4a_incumbent"], out["arms"]["B4b_lean"]
    if l["seeds"]:
        verdict = {}
        for k in ("test_cindex", "test_lateness_auc_asof", "val_cindex"):
            verdict[k] = "GAIN (disjoint)" if l[k][0] > a[k][2] else "WORSE (disjoint)" if l[k][2] < a[k][0] else "TIE (bands overlap)"
        out["verdict"] = verdict
    # same-machine wall clock for the incumbent: this machine's full-length h4 runs (ml/artifacts_win, concurrency 1)
    win = []
    for s in (7, 17, 27):
        p = os.path.join(C.REPO, "ml", "artifacts_win", "bundles", TASK, f"v8_lite_h4_lr0.00025_s{s}", "train_log.json")
        if os.path.exists(p):
            t = json.load(open(p)); win.append(dict(seed=s, sec_per_epoch=t["sec_per_epoch"], seconds=t["seconds"], epochs=t["epochs_run"]))
    out["incumbent_wall_clock_this_machine"] = dict(source="ml/artifacts_win (Phase 1 re-run on this machine, concurrency 1, same low-VRAM mode)",
                                                    per_seed=win)
    C.dump(out, "phase17/b4_score.json")
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1))
    for k, v in out["arms"].items():
        print(k, v["seeds"], "test C", v["test_cindex"], "lateness", v["test_lateness_auc_asof"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["equiv", "train", "score"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--bundle-root", default=None, help="smoke runs only")
    a = ap.parse_args()
    {"equiv": equiv, "score": score}.get(a.mode, lambda: train(a.seed, a.max_epochs, a.bundle_root))()
