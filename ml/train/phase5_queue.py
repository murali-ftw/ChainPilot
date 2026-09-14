"""Phase 5 — everything downstream of a finished learning-rate sweep, for one queue.

Per task, in order:
  1. select the rate on the VALIDATION fold from its v6 / h0 sweep; if the optimum sits on a boundary
     of the five-point grid, extend one point outward and re-select (the closeout left shortage on a
     boundary and said so -- this does not repeat that)
  2. v7 at the selected rate                                   (within-world, both worlds)
  3. seeds 17 and 27 on v6 at the selected rate                (the per-metric noise band)
  4. arrival and fill only: the 5.0 arms -- reconstructed weeks_since_last_activity without the gate,
     and with it -- both worlds, same head, same rate
  5. fill only: the loss ablation -- cross-entropy and RPS + cross-entropy on the 22 cells, both worlds

Every selection is written to phase5_selection.json with the whole sweep beside it, so the test score
of each rate NOT chosen can be stated. Test scores are never read here.
"""
from __future__ import annotations
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import phase5_heads as P5

LRS = [2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4]
SEL = os.path.join(P5.ARTIFACTS, "phase5_selection.json")


def grid_path(task):
    return os.path.join(P5.ARTIFACTS, f"phase5_grid_{task.split('_')[0]}.json")


def sweep_rows(task):
    g = grid_path(task)
    rows = json.load(open(g)) if os.path.exists(g) else []
    return [r for r in rows if r["world"] == "v6" and r["seed"] == 7 and r["gate"] == 0 and r["wsla"] == 0
            and r["arch"] == "none" and r["depth"] == 0 and not r.get("fill_loss")]


def select(task, max_epochs, patience):
    lrs = list(LRS)
    while True:
        rows = sweep_rows(task)
        have = {r["lr"]: r for r in rows}
        missing = [lr for lr in lrs if lr not in have]
        for lr in missing:
            P5.run("v6", task, lr, 7, "none", 0, False, False, "sweep-ext", grid_path(task), max_epochs, patience)
        rows = sweep_rows(task); have = {r["lr"]: r for r in rows}
        pts = sorted((lr for lr in lrs if lr in have), reverse=True)
        better = max if P5.HIGHER[task] else min
        best = better(pts, key=lambda lr: have[lr]["best_val"])
        edge = best in (pts[0], pts[-1])
        if edge and len(lrs) < 7:
            lrs.append(best * 2 if best == pts[0] else best / 2)
            print(f"[select] {task}: optimum {best:g} on the boundary -> extending to {lrs[-1]:g}", flush=True)
            continue
        path = SEL.replace(".json", f"_{task}.json")      # one file per task: two queues write concurrently
        sel = {}
        sel[task] = dict(chosen=best, boundary_after_extension=bool(edge), metric_higher_is_better=P5.HIGHER[task],
                         sweep=[dict(lr=lr, val=have[lr]["best_val"], best_epoch=have[lr]["best_epoch"],
                                     epochs=have[lr]["epochs_run"], stop=have[lr]["stop"],
                                     sec_per_epoch=have[lr]["sec_per_epoch"], preds=have[lr]["preds"]) for lr in pts])
        json.dump(sel, open(path, "w"), indent=1)
        print(f"[select] {task}: chosen {best:g} (val {have[best]['best_val']:.5f})", flush=True)
        return best


def wait_for_sweep(task, n=5, poll=30):
    while len(sweep_rows(task)) < n:
        time.sleep(poll)


def downstream(task, max_epochs, patience):
    wait_for_sweep(task)
    lr = select(task, max_epochs, patience)
    g = grid_path(task)
    P5.run("v7", task, lr, 7, "none", 0, False, False, "main", g, max_epochs, patience)
    for s in (17, 27):
        P5.run("v6", task, lr, s, "none", 0, False, False, "noise", g, max_epochs, patience)
    if task in ("arrival_week", "fill_rate"):
        for w in ("v6", "v7"):
            P5.run(w, task, lr, 7, "none", 0, False, True, "gate-off+wsla", g, max_epochs, patience)
            P5.run(w, task, lr, 7, "none", 0, True, True, "gate-on+wsla", g, max_epochs, patience)
    if task == "fill_rate":
        for loss in ("ce", "rps+ce"):
            for w in ("v6", "v7"):
                P5.run(w, task, lr, 7, "none", 0, False, False, f"loss-{loss}", g, max_epochs, patience, loss)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--max-epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=8)
    a = ap.parse_args()
    for t in a.tasks.split(","):
        downstream(t, a.max_epochs, a.patience)
    print("QUEUE DONE", flush=True)
