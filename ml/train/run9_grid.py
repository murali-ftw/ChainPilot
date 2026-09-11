"""Run 9 — WITHIN-WORLD grid. Train v6 test v6, train v7 test v7. Nothing else.

Saves raw test predictions per cell so every metric, and its bootstrap interval, can be
computed offline without retraining.

h0 builds NO graph module, so it is one model shared by both architecture columns. It is
trained once per (world, task) and additionally re-trained under each architecture label so
the "do the two h0 runs match" check is measured rather than asserted.
"""
from __future__ import annotations
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "models"), os.path.join(HERE, "..", "eval")]
import numpy as np, torch
import temporal_share as TS
from device import peak_rss_gb

TASKS = ["arrival_week", "fill_rate", "shortage_qty"]
WORLDS = ["v6", "v7"]
CONFIGS = [("none", 0), ("mp", 1), ("mp", 4), ("share", 1), ("share", 4)]
PRED = "ml/artifacts/run9_preds"

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="ml/artifacts/run9_grid.json")
ap.add_argument("--lr", type=float, default=None, help="frozen lr from the v6 tuning pass")
ap.add_argument("--tasks", default=",".join(TASKS))
ap.add_argument("--seeds", default="7", help="comma list; >1 measures device/seed noise")
ap.add_argument("--noise-cell", default="", help="world:task:arch:depth to repeat over seeds")
a = ap.parse_args()
if a.lr:
    TS.HP["lr"] = a.lr
os.makedirs(PRED, exist_ok=True)

rows = json.load(open(a.out)) if os.path.exists(a.out) else []
done = {(r["world"], r["task"], r["arch"], r["depth"], r["seed"]) for r in rows}
save = lambda: json.dump(rows, open(a.out, "w"), indent=1)


def run(world, task, arch, depth, seed, tag=""):
    key = (world, task, arch, depth, seed)
    if key in done:
        print(f"[skip] {key}", flush=True); return
    print(f"\n=== {world} | {task} | {arch} h{depth} | seed {seed} {tag}===", flush=True)
    t0 = time.time()
    m, nz, info = TS.train_cell(world, task, arch, depth, seed=seed, verbose=False)
    P, Y, EV, AUX = TS.predict_test(m, nz, world, task)
    f = f"{PRED}/{world}_{task}_{arch}_h{depth}_s{seed}.npz"
    np.savez_compressed(f, P=P, Y=Y, EV=EV, AUX=AUX)
    rows.append(dict(world=world, task=task, arch=arch, depth=depth, seed=seed,
                     preds=f, wall=time.time() - t0, rss=peak_rss_gb(), lr=TS.HP["lr"],
                     **{k: v for k, v in info.items() if k != "losses"}))
    print(f"  [{info['stop']}] best_ep {info['best_epoch']} of {info['epochs_run']}  "
          f"val {info['best_val']:.4f}  {time.time()-t0:.0f}s  RSS {peak_rss_gb():.2f} GB  "
          f"-> {os.path.basename(f)}", flush=True)
    save()
    del m
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


t_all = time.time()
if a.noise_cell:
    w, tk, ar, dp = a.noise_cell.split(":")
    for s in [int(x) for x in a.seeds.split(",")]:
        run(w, tk, ar, int(dp), s, tag="[NOISE] ")
else:
    for task in a.tasks.split(","):
        for w in WORLDS:
            for arch, depth in CONFIGS:
                run(w, task, arch, depth, 7)
    # P2.4 h0 agreement check, on arrival in both worlds. Net builds no graph module at
    # depth 0, so a "share h0" and an "mp h0" request are the SAME model -- these two runs
    # measure whether independent trainings of it agree, rather than asserting that they do.
    for w in WORLDS:
        run(w, "arrival_week", "share", 0, 7, tag="[h0 CHECK] ")
        run(w, "arrival_week", "mp", 0, 7, tag="[h0 CHECK] ")

print(f"\nCOMPLETE  {len(rows)} trainings  {time.time()-t_all:.0f}s  "
      f"peak RSS {peak_rss_gb():.2f} GB")
