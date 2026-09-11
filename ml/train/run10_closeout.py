"""Closeout item 3 — re-run fill and shortage to convergence at per-task learning rates.

Arrival is NOT re-run: all 14 of its cells converged in run 9 and its rate is not retuned, so
phase1_2's arrival numbers carry forward unchanged and the two reports stay comparable there.

Cap 120, patience 8, restore-best. Any cell still reaching the cap is still a floor and is
labelled one.
"""
from __future__ import annotations
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "models"), os.path.join(HERE, "..", "eval")]
import numpy as np, torch
import temporal_share as TS
from device import peak_rss_gb

PRED = "ml/artifacts/run10_preds"
ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--lr", type=float, required=True)
ap.add_argument("--configs", default="none:0,mp:1,mp:4,share:1,share:4")
ap.add_argument("--worlds", default="v6,v7")
ap.add_argument("--seeds", default="7")
ap.add_argument("--out", default="ml/artifacts/run10_grid.json")
a = ap.parse_args()

TS.HP.update(lr=a.lr, max_epochs=120, patience=8)
os.makedirs(PRED, exist_ok=True)
rows = json.load(open(a.out)) if os.path.exists(a.out) else []
done = {(r["world"], r["task"], r["arch"], r["depth"], r["seed"]) for r in rows}

for seed in [int(s) for s in a.seeds.split(",")]:
    for w in a.worlds.split(","):
        for cfg in a.configs.split(","):
            arch, depth = cfg.split(":"); depth = int(depth)
            key = (w, a.task, arch, depth, seed)
            if key in done:
                print(f"[skip] {key}", flush=True); continue
            print(f"\n=== {w} | {a.task} | {arch} h{depth} | seed {seed} | lr {a.lr:g} ===",
                  flush=True)
            t0 = time.time()
            m, nz, info = TS.train_cell(w, a.task, arch, depth, seed=seed, verbose=False)
            P, Y, EV, AUX = TS.predict_test(m, nz, w, a.task)
            f = f"{PRED}/{w}_{a.task}_{arch}_h{depth}_s{seed}.npz"
            np.savez_compressed(f, P=P, Y=Y, EV=EV, AUX=AUX)
            rows.append(dict(world=w, task=a.task, arch=arch, depth=depth, seed=seed,
                             preds=f, wall=time.time() - t0, rss=peak_rss_gb(), lr=a.lr,
                             **{k: v for k, v in info.items() if k != "losses"}))
            json.dump(rows, open(a.out, "w"), indent=1)
            print(f"  [{info['stop']}] best_ep {info['best_epoch']} of {info['epochs_run']}  "
                  f"val {info['best_val']:.4f}  {time.time()-t0:.0f}s  RSS {peak_rss_gb():.2f} GB",
                  flush=True)
            del m
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
print("\nDONE", flush=True)
