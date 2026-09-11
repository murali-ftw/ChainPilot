"""Run 8 — the grid. train v6/v7 x test v6/v7, three tasks, two architectures, h0/h1/h4.

h0 has NO graph module constructed, so it is the same model whichever architecture asked for
it. It is trained twice, once per architecture sweep, with the same seed -- the two numbers
must agree, and their spread IS the MPS run-to-run noise band (P2.6).
"""
from __future__ import annotations
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "models")]
import numpy as np, torch
import temporal_share as TS
from device import peak_rss_gb

TASKS = ["arrival_week", "fill_rate", "shortage_qty"]
WORLDS = ["v6", "v7"]
CONFIGS = [("none", 0), ("share", 1), ("share", 4), ("mp", 1), ("mp", 4)]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="ml/artifacts/run8_grid.json")
ap.add_argument("--tasks", default=",".join(TASKS))
ap.add_argument("--repeats", type=int, default=0, help="extra seeds on the noise-band cell")
a = ap.parse_args()
tasks = a.tasks.split(",")

rows = []
t_all = time.time()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
if os.path.exists(a.out):
    rows = json.load(open(a.out))
done = {(r["train"], r["task"], r["arch"], r["depth"], r.get("rep", 0)) for r in rows}

def save():
    json.dump(rows, open(a.out, "w"), indent=1)

for task in tasks:
    for tw in WORLDS:
        for arch, depth in CONFIGS:
            key = (tw, task, arch, depth, 0)
            if key in done:
                print(f"[skip] {key}", flush=True); continue
            print(f"\n=== train {tw} | {task} | {arch} h{depth} ===", flush=True)
            t0 = time.time()
            m, nz, info = TS.train_cell(tw, task, arch, depth, seed=7, verbose=True)
            ev = {w: TS.eval_on(m, nz, w, task) for w in WORLDS}
            rows.append(dict(train=tw, task=task, arch=arch, depth=depth, rep=0,
                             test=ev, wall=time.time() - t0, rss=peak_rss_gb(),
                             **{k: v for k, v in info.items() if k != "losses"}))
            print(f"  -> test {ev}  [{info['stop']}] best_ep {info['best_epoch']} "
                  f"of {info['epochs_run']}  {time.time()-t0:.0f}s  RSS {peak_rss_gb():.2f} GB",
                  flush=True)
            save()
            del m
            if DEV_MPS := torch.backends.mps.is_available():
                torch.mps.empty_cache()

# --- MPS run-to-run spread: repeat one cell at the SAME seed -----------------
for rep in range(1, a.repeats + 1):
    key = ("v7", "arrival_week", "share", 4, rep)
    if key in done: continue
    print(f"\n=== NOISE BAND repeat {rep}: v7 | arrival | share h4 (same seed 7) ===", flush=True)
    t0 = time.time()
    m, nz, info = TS.train_cell("v7", "arrival_week", "share", 4, seed=7, verbose=False)
    ev = {w: TS.eval_on(m, nz, w, "arrival_week") for w in WORLDS}
    rows.append(dict(train="v7", task="arrival_week", arch="share", depth=4, rep=rep,
                     test=ev, wall=time.time() - t0, rss=peak_rss_gb(),
                     **{k: v for k, v in info.items() if k != "losses"}))
    print(f"  -> {ev}  {time.time()-t0:.0f}s", flush=True)
    save()

print(f"\nGRID COMPLETE  {len(rows)} trainings  {time.time()-t_all:.0f}s  "
      f"peak RSS {peak_rss_gb():.2f} GB")
