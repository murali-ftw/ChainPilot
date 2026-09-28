"""Phase 13 training queues. Each cell runs `loop.py train` in a subprocess; its completion marker is
artifact_identity.marker_name(resolved config) -- a function of the FULL identity (P2), so two arms can never share a
marker (deviation 73 lost two of three arms that way). Refuses to start on a dirty ml/ (P3).

  python ml/train/phase13_queue.py f1a      # F1: boundary-weight grid at seed 7, then the Beta head x5
  python ml/train/phase13_queue.py f1b      # F1: the selected boundary weight x4 more seeds, regression x5
  python ml/train/phase13_queue.py f2       # F2 arm (c): ratio as head input, K4 hier / K1 ps / K2 sp / K3 psp, x5
  python ml/train/phase13_queue.py f2a      # Stage 2A: shrunk ratio as the Beta centre, x5
"""
from __future__ import annotations
import os, sys, json, subprocess, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "eval")]
import artifact_identity as AI
import phase12_common as C

SEEDS = (7, 17, 27, 37, 47)
BASE = ["--config", "ml/configs/shipped.json", "--task", "fill_rate", "--world", "v8",
        "--arch", "none", "--depth", "0", "--lr", "0.000125"]
LOG = os.path.join(C.ART, "phase13_logs")


def resolved(extra, seed):
    import loop as L
    ap = argparse.Namespace(config="ml/configs/shipped.json", task="fill_rate", world="v8", seed=seed, arch="none",
                            depth=0, lr=0.000125, tag="", origin=None, graph_shuffle=None, drop_relation=None,
                            row_features=None, train_snapshots=None, max_epochs=None, fill_head=None, fill_loss=None,
                            ratio_key=None, ratio_est="ros")
    it = iter(extra)
    for a in it:
        setattr(ap, a[2:].replace("-", "_"), next(it))
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        return L.resolve(ap)
    finally:
        os.chdir(cwd)


def cell(extra, seed):
    cfg = resolved(extra, seed)
    mk = os.path.join(LOG, AI.marker_name(cfg))
    if os.path.exists(mk):
        print(f"[skip] {AI.bundle_path_key(cfg)}", flush=True); return
    while C.stamp()["code_dirty"]:          # never start a cell on uncommitted ml/ -- wait, do not die
        print("   ml/ is dirty; waiting to start", flush=True); time.sleep(60)
    log = mk.replace(".done", ".log")
    t = time.strftime("%H:%M:%S")
    print(f"=== START {AI.bundle_path_key(cfg)}  {t}", flush=True)
    r = subprocess.run([os.path.join(REPO, "venv", "bin", "python"), "-u", "ml/train/loop.py", "train", *BASE,
                        "--seed", str(seed), *extra], cwd=REPO, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    tail = open(log).read().strip().splitlines()[-2:] if os.path.exists(log) else []
    if r.returncode == 0:
        json.dump(dict(identity=AI.identity_of(cfg), stamp=C.stamp()), open(mk, "w"))
    print(f"=== END   {AI.bundle_path_key(cfg)}  {time.strftime('%H:%M:%S')}  rc={r.returncode}  {tail[:1]}", flush=True)


def selected_bw():
    p = os.path.join(C.ART, "phase13_f1_bw_selection.json")
    assert os.path.exists(p), "boundary weight not selected yet -- run ml/eval/phase13_select_bw.py after f1a"
    return json.load(open(p))["selected_fill_loss"]


def main():
    os.makedirs(LOG, exist_ok=True)
    q = sys.argv[1]
    if q == "f1a":
        for w in (3, 10, 30):
            cell(["--fill-loss", f"rps_bw{w}"], 7)
        for s in SEEDS:
            cell(["--fill-head", "beta3"], s)
    elif q == "f1b":
        fl = selected_bw()
        for s in SEEDS:
            cell(["--fill-loss", fl], s)
        for s in SEEDS:
            cell(["--fill-head", "reg"], s)
    elif q == "f2":
        for key in ("hier", "ps", "sp", "psp"):
            for s in SEEDS:
                cell(["--ratio-key", key], s)
    elif q == "f2a":
        for s in SEEDS:
            cell(["--fill-head", "beta3c", "--ratio-key", "hier"], s)
    print(f"=== QUEUE {q} COMPLETE {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
