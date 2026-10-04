"""Phase 22 Stages -1, 1e, 3b -- neural runs on CLEAN inputs. MPS, concurrency 1, incumbent architectures and LRs.

  handshake   Stage -1: the stored arrival lite h4 s7 config retrained 3 epochs on this machine (world v8). Inside the stored
              5-seed band at every epoch -> proceed. Constructed failing case, run here: the same check against the stored
              values shifted by 0.05 must report OUTSIDE (pre-registration D4).
  clean       Stage 1e: an incumbent config trained on world `v8clean` (same CSVs; cache panel with the leaking columns
              replaced, ml/data/clean_panel.py). The world name is in the identity, so the bundle name is new:
              ml/artifacts/phase22/bundles/{task}/v8clean_{config_name}.  loop.train / finish_bundle run unchanged.
  bind        Stage 3b (and the clean Phase 19 fill arm): fill h0 on v8clean with per-row families through n_row_feats
              (Phase 19's RowFamilyHeadNet pattern, a NEW subclass in this process only); bundle name gets
              phase19_identity's `_rf{families}` suffix, under the Phase 22 root.
  queue       one cell at a time; a cell starts only if (stored incumbent mean minutes x 1.3) fits before the deadline,
              else it and every later cell are listed in reports/part2/phase22/STOPPED.json.

  HADES_DATA_ROOT=<main checkout> python ml/train/phase22_train.py handshake
  ... python ml/train/phase22_train.py queue --cells clean:arrival,clean:capacity,clean:fill --deadline "2026-10-05 14:53"
"""
from __future__ import annotations
import os, sys, json, time, argparse, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "eval"), os.path.join(HERE, "..", "data"), HERE]
import phase12_common as C
import numpy as np, pandas as pd
import config
import phase21_paths as PP

TASK = {"arrival": "arrival_week", "fill": "fill_rate", "capacity": "capacity_strain"}
INCUMBENT = {"arrival": ("lite", 4, 2.5e-4), "fill": ("none", 0, 1.25e-4), "capacity": ("mp", 4, 2.5e-4)}
STORED = {"arrival": "arrival_week/v8_lite_h4_lr0.00025_s{s}", "fill": "fill_rate/v8_none_h0_lr0.000125_s{s}",
          "capacity": "capacity_strain/v8_mp_h4_lr0.00025_s{s}"}
CONCURRENCY = 1
ROOT22 = os.path.join(C.ART, "phase22", "bundles")
CLEAN = {"v8": "v8clean", "v8w1002": "v8w1002clean"}


def register_clean():
    """In-process only: the clean worlds read the same CSVs as their parent; their cache dir is their own."""
    PP.register()
    for parent, w in CLEAN.items():
        config.WORLDS[w] = config.WORLDS[parent]
        config.EXPECTED_PANEL_D[w] = 15
    return {w: config.WORLDS[w] for w in CLEAN.values()}


def bundle_dir22(cfg):
    import phase19_identity as PI
    assert not cfg.get("origin")
    return os.path.join(ROOT22, cfg["task"], PI.bundle_name(cfg))


def _resolve(task, seed, world, max_epochs=None):
    import loop as L
    arch, depth, lr = INCUMBENT[task]
    args = argparse.Namespace(config=os.path.join(C.ML, "configs", "shipped.json"), task=TASK[task], world=world, seed=seed,
                              arch=arch, depth=depth, lr=lr, max_epochs=max_epochs, tag="")
    return L.resolve(args)


def handshake():
    import loop as L
    register_clean()
    st = C.require_clean()
    cfg = _resolve("arrival", 7, "v8", max_epochs=3)
    t = time.time()
    _, _, _, _, log = L.train(cfg)
    stored = [json.load(open(os.path.join(C.BUND, STORED["arrival"].format(s=s), "train_log.json")))["vals"][:3] for s in C.V8_SEEDS]

    def check(vals):
        rows = []
        for e in range(3):
            b = [v[e] for v in stored]
            rows.append(dict(epoch=e, here=vals[e], stored_s7=stored[0][e], abs_diff=abs(vals[e] - stored[0][e]),
                             band=[min(b), max(b)], inside=bool(min(b) <= vals[e] <= max(b))))
        return rows
    rows = check(log["vals"])
    falsify = check([v + 0.05 for v in log["vals"]])          # constructed failing case: must be OUTSIDE
    res = dict(stamp=st, device=str(L.DEV), seconds=time.time() - t, epochs=rows, inside_band=all(r["inside"] for r in rows),
               falsification_shifted_0p05=dict(rows=falsify, outside=not all(r["inside"] for r in falsify)),
               loss_here=log["losses"],
               loss_stored_s7=json.load(open(os.path.join(C.BUND, STORED["arrival"].format(s=7), "train_log.json")))["losses"][:3])
    os.makedirs(os.path.join(C.ART, "phase22"), exist_ok=True)
    C.dump(res, "phase22/handshake.json")
    print(json.dumps(res, indent=1, default=str))
    assert res["falsification_shifted_0p05"]["outside"], "the handshake check cannot fail -- invalid"
    assert res["inside_band"], "handshake OUTSIDE the stored band -- label results 'within-machine comparison'"


def _finish(L, cfg, model, D, lb, split, log, st, t0, by):
    out = L.finish_bundle(cfg, model, D, lb, split, log, trained_by=by)
    tl = json.load(open(os.path.join(out, "train_log.json")))
    tl.update(concurrency_level=CONCURRENCY, wall_seconds_total=time.time() - t0, code_commit=st["code_commit"],
              machine="Apple M4 Pro", per_seed_minutes=(time.time() - t0) / 60)
    json.dump(tl, open(os.path.join(out, "train_log.json"), "w"), indent=1)
    print(f"  {cfg['task']} {cfg['world']} s{cfg['seed']} [{log['stop']}] best_ep {log['best_epoch']} of {log['epochs_run']} "
          f"val {log['best_val']:.5f} {(time.time() - t0) / 60:.1f} min {log['sec_per_epoch']:.1f}s/ep -> {out}", flush=True)
    return out


def clean(task, seed, world="v8clean", max_epochs=None):
    import loop as L
    register_clean()
    st = C.require_clean()
    cfg = _resolve(task, seed, world, max_epochs)
    L.bundle_dir = bundle_dir22
    out = L.bundle_dir(cfg)
    if os.path.exists(os.path.join(out, "config.json")) and json.load(open(os.path.join(out, "config.json"))).get("complete"):
        print(f"[skip] complete bundle exists: {out}", flush=True); return out
    t0 = time.time()
    model, D, lb, split, log = L.train(cfg)
    return _finish(L, cfg, model, D, lb, split, log, st, t0, "ml/train/phase22_train.py clean (loop.train, world v8clean)")


def bind(task, seed, families, world="v8clean", max_epochs=None):
    """families: comma list from {season, cadence, ack, L4} -> per-row columns (phase22_rows.RowStore22)."""
    import torch
    import loop as L, phase5_heads as P5, folds
    import phase22_rows as R22
    register_clean()
    st = C.require_clean()
    fams = families.split(",")
    cfg = _resolve(task, seed, world, max_epochs)
    cfg["row_family"] = "+".join(fams)
    L.bundle_dir = bundle_dir22
    out = L.bundle_dir(cfg)
    if os.path.exists(os.path.join(out, "config.json")) and json.load(open(os.path.join(out, "config.json"))).get("complete"):
        print(f"[skip] complete bundle exists: {out}", flush=True); return out
    store = R22.RowStore22(fams, world=world.replace("clean", ""), task=TASK[task])
    lb0 = P5.labels(world, TASK[task]); tr, _, _ = folds.fixed_split(lb0.snapshot_date)
    scaler = store.fit_scaler(lb0, tr)
    cfg["row_family_cols"] = scaler["cols"]; cfg["row_family_width"] = scaler["width"]
    Base, orig_batches = P5.HeadNet, P5.batches
    width = scaler["width"]

    class RowFamilyHeadNet22(Base):
        """The incumbent HeadNet with Phase 22 per-row families concatenated before the head (n_row_feats)."""
        def __init__(self, d_in, task_, arch="none", depth=0, **kw):
            kw["n_row_feats"] = width
            super().__init__(d_in, task_, arch, depth, **kw)

    def batches(task_, lb, mask, W, ymu=0.0, ysd=1.0):
        bs = orig_batches(task_, lb, mask, W, ymu, ysd)
        for t0_, tg, ii in bs:
            assert "xrow" not in tg
            tg["xrow"] = torch.from_numpy(store.transform(lb.iloc[ii])).to(P5.DEV)
        return bs
    P5.HeadNet, P5.batches = RowFamilyHeadNet22, batches
    t0 = time.time()
    model, D, lb, split, log = L.train(cfg)
    assert isinstance(model, RowFamilyHeadNet22) and model.n_row_feats == width
    log["row_family"] = cfg["row_family"]; log["row_family_scaler"] = scaler
    out = _finish(L, cfg, model, D, lb, split, log, st, t0, "ml/train/phase22_train.py bind (loop.train with RowFamilyHeadNet22)")
    P5.HeadNet, P5.batches = Base, orig_batches
    return out


def queue(cells, deadline):
    dl = datetime.datetime.strptime(deadline, "%Y-%m-%d %H:%M")
    done, not_run = [], []
    for cell in cells:
        kind, task = cell.split(":")[:2]
        fams = cell.split(":")[2] if kind == "bind" else None
        est = 1.3 * np.mean([json.load(open(os.path.join(C.BUND, STORED[task].format(s=s), "train_log.json")))["seconds"]
                             for s in C.V8_SEEDS]) / 60
        for s in C.V8_SEEDS:
            if not_run or datetime.datetime.now() + datetime.timedelta(minutes=est) > dl:
                not_run.append(f"{cell} s{s}"); continue
            import subprocess
            cmd = [sys.executable, os.path.abspath(__file__), kind, "--task", task, "--seed", str(s)] + (["--families", fams] if fams else [])
            r = subprocess.run(cmd)                       # one process per cell: no state carried between cells
            if r.returncode != 0:
                raise SystemExit(f"cell failed: {cell} s{s}")
            done.append(f"{cell} s{s}")
    if not_run:
        path = os.path.join(C.REPO, "reports", "part2", "phase22", "STOPPED.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(dict(completed=done, not_run=not_run, reason=f"wall-clock stop {deadline} IST: the next cell's estimate did not fit",
                       written=datetime.datetime.now().isoformat(timespec="seconds")), open(path, "w"), indent=1)
        print(f"STOPPED: {len(not_run)} cells not run -> {path}", flush=True)
    print("queue done:", done, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["handshake", "clean", "bind", "queue"])
    ap.add_argument("--task", choices=list(TASK))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--families", default=None)
    ap.add_argument("--cells", default="")
    ap.add_argument("--deadline", default="2026-10-05 14:53")
    a = ap.parse_args()
    if a.mode == "handshake":
        handshake()
    elif a.mode == "clean":
        clean(a.task, a.seed)
    elif a.mode == "bind":
        bind(a.task, a.seed, a.families)
    else:
        queue(a.cells.split(","), a.deadline)
