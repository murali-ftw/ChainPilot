"""Phase 6 — Stages 3, 4 and 5 as two MPS queues.

  python ml/train/phase6_queue.py --queue A
  python ml/train/phase6_queue.py --queue B

Stage 3 (the reproduction gate) runs FIRST in both queues. Phase 5 cells that already have a restored-best checkpoint and
validation predictions are CONVERTED into bundles (steps 3-8, no retraining). The rest are seed-triple cells, ordered so
the brief's cut order is respected if the budget runs out: fill v7 last, capacity v7 before it, arrival never cut.

Every cell skips itself if its bundle is complete, so a killed queue resumes where it stopped. After each cell the queue
prints its measured mean cell time and a projection for what remains.
"""
from __future__ import annotations
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import loop as L
from config import ARTIFACTS

SHIPPED = "ml/configs/shipped.json"
P5P = os.path.join(ARTIFACTS, "phase5_preds")
P5R = os.path.join(ARTIFACTS, "phase5_preds_recal")


def cfg(task, world, seed, arch=None, depth=None, lr=None, tag=""):
    ns = argparse.Namespace(config=SHIPPED, task=task, world=world, seed=seed, arch=arch, depth=depth, lr=lr,
                            tag=tag, max_epochs=None)
    return L.resolve(ns)


# (kind, cfg, source-for-convert)
REPRO = {
    "R1": ("train", lambda: cfg("arrival_week", "v6", 7, "none", 0, 2e-3, "stage3-R1"), None),
    "R2": ("train", lambda: cfg("arrival_week", "v6", 7, tag="stage3-R2 + stage4"), None),
    "R3": ("train", lambda: cfg("fill_rate", "v6", 7, tag="stage3-R3 + stage4"), None),
    "R4": ("train", lambda: cfg("capacity_strain", "v6", 7, "none", 0, 2.5e-4, "stage3-R4 + capacity h0 reference"), None),
    "R5": ("train", lambda: cfg("capacity_strain", "v7", 7, tag="stage3-R5 + stage4"), None),
}
QUEUES = {
    "A": [
        ("convert", lambda: cfg("fill_rate", "v6", 17, tag="stage4 (Addendum A cell)"), f"{P5R}/v6_fill_rate_cdf22_none_h0_s17_lr0.000125_g0_w0.npz"),
        ("convert", lambda: cfg("fill_rate", "v6", 27, tag="stage4 (Addendum A cell)"), f"{P5R}/v6_fill_rate_cdf22_none_h0_s27_lr0.000125_g0_w0.npz"),
        ("convert", lambda: cfg("fill_rate", "v7", 7, tag="stage4 (Addendum A cell)"), f"{P5R}/v7_fill_rate_cdf22_none_h0_s7_lr0.000125_g0_w0.npz"),
        ("convert", lambda: cfg("arrival_week", "v7", 7, "none", 0, 2.5e-4, "arrival h0 reference v7 (Addendum B cell)"),
         f"{P5P}/v7_arrival_week_hazard_none_h0_s7_lr0.00025_g0_w0.npz"),
        REPRO["R2"], REPRO["R1"],
        ("train", lambda: cfg("arrival_week", "v6", 7, "none", 0, 2.5e-4, "stage5 arrival h0 reference v6"), None),
        ("train", lambda: cfg("arrival_week", "v7", 7, tag="stage4"), None),
        ("train", lambda: cfg("arrival_week", "v6", 17, tag="stage4"), None),
        ("train", lambda: cfg("arrival_week", "v7", 17, tag="stage4"), None),
        ("train", lambda: cfg("capacity_strain", "v7", 17, tag="stage4 (cut 2nd)"), None),
        ("train", lambda: cfg("fill_rate", "v7", 17, tag="stage4 (cut 1st)"), None),
    ],
    "B": [
        REPRO["R3"], REPRO["R4"], REPRO["R5"],
        ("train", lambda: cfg("capacity_strain", "v6", 7, tag="stage4"), None),
        ("train", lambda: cfg("arrival_week", "v6", 27, tag="stage4"), None),
        ("train", lambda: cfg("arrival_week", "v7", 27, tag="stage4"), None),
        ("train", lambda: cfg("capacity_strain", "v6", 17, tag="stage4"), None),
        ("train", lambda: cfg("capacity_strain", "v6", 27, tag="stage4"), None),
        ("train", lambda: cfg("capacity_strain", "v7", 27, tag="stage4 (cut 2nd)"), None),
        ("train", lambda: cfg("fill_rate", "v7", 27, tag="stage4 (cut 1st)"), None),
    ],
}


def main(q):
    log_path = os.path.join(ARTIFACTS, f"phase6_queue_{q}.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else []
    jobs = QUEUES[q]
    times = []
    for i, (kind, mk, src) in enumerate(jobs):
        c = mk()
        t0 = time.time()
        out = L.run_train(c) if kind == "train" else L.run_convert(c, src)
        dt = time.time() - t0
        if kind == "train" and dt > 120:
            times.append(dt)
        tl = json.load(open(os.path.join(out, "train_log.json")))
        log.append(dict(queue=q, i=i, kind=kind, task=c["task"], world=c["world"], arch=c["arch"], depth=c["depth"],
                        lr=c["lr"], seed=c["seed"], tag=c["tag"], bundle=out, wall=dt, stop=tl.get("stop"),
                        epochs=tl.get("epochs_run"), best_epoch=tl.get("best_epoch"), best_val=tl.get("best_val"),
                        peak_rss_gb=tl.get("peak_rss_gb")))
        json.dump(log, open(log_path, "w"), indent=1)
        left = sum(1 for k, *_ in jobs[i + 1:] if k == "train")
        if times:
            print(f"[queue {q}] {i + 1}/{len(jobs)} done; mean trained cell {sum(times) / len(times) / 60:.1f} min; "
                  f"{left} training cells left -> projected {left * sum(times) / len(times) / 3600:.1f} h", flush=True)
    print(f"[queue {q}] QUEUE DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, choices=["A", "B"])
    main(ap.parse_args().queue)
