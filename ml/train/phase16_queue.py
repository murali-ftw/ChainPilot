"""Phase 16 training queue. ONE job on the GPU at a time: every cell is a blocking subprocess, the stages run in series.

Completion marker = artifact_identity.marker_name(resolved config), a function of the FULL identity including the
Phase 16 axes (S4). Refuses to start a cell on a dirty ml/. 12 h cap per stage: a stage that would start a cell past
its cap writes ml/artifacts/phase16_STOPPED_<stage>.json and the queue moves on to nothing -- it stops.

  python ml/train/phase16_queue.py all

Stages, in order:
  a      Stage 2  arrival  A1 share_traj (Delta present), A2 share_traj (Delta zeroed -- the control)       x5 seeds
  b_arr  Stage 3  arrival  B1 share_pna mmms; B2 bypass k=1; B2 k=2 (seed 7: the validation curve's second point);
                           B3 mmms + the SELECTED k                                                           x5 seeds
  b_cap  Stage 3  capacity B1 heteromp_pna mmms                                                                x5 seeds
  audit  Stage 4B re-run the stored arrival lite h4 s7 and capacity mp h4 s7 end to end, into a separate bundle root

A0 / B0 are the stored incumbent bundles (identity unchanged -- S4); they are NOT retrained.
k SELECTION RULE, fixed here before any B2 number exists: k maximises B2's validation C-index (seed 7, the only seed
at which both k run); if the two edge sets are identical (Stage 0.3 proved they are) or the validation values tie,
k = 1, the smaller intervention. Capacity B2/B3 are not trained: HeteroMP has no softmax, so the bypass is the
identity and B2 = B0, B3 = B1 by construction (deviation 149).
"""
from __future__ import annotations
import os, sys, json, subprocess, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "eval"), os.path.join(HERE, "..", "models"),
                os.path.join(HERE, "..", "data")]
import artifact_identity as AI
import phase12_common as C

SEEDS = (7, 17, 27, 37, 47)
LOG = os.path.join(C.ART, "phase16_logs")
AUDIT_ROOT = os.path.join(C.ART, "phase16_audit_bundles")
CAP_H = 12.0
BASE = {"arrival_week": ["--arch", "lite", "--depth", "4", "--lr", "0.00025"],
        "capacity_strain": ["--arch", "mp", "--depth", "4", "--lr", "0.00025"]}


def resolved(task, extra, seed):
    import loop as L
    b = BASE[task]
    ap = argparse.Namespace(config="ml/configs/shipped.json", task=task, world="v8", seed=seed, arch=b[1],
                            depth=int(b[3]), lr=float(b[5]), tag="", origin=None, graph_shuffle=None, drop_relation=None,
                            row_features=None, train_snapshots=None, max_epochs=None, fill_head=None, fill_loss=None,
                            ratio_key=None, ratio_est="ros", encoder_variant=None, traj_depths=None, traj_delta=None,
                            pna_aggregators=None, low_degree_k=None)
    it = iter(extra)
    for a in it:
        if a == "--bundle-root":
            next(it); continue
        v = next(it)
        setattr(ap, a[2:].replace("-", "_"), int(v) if a == "--low-degree-k" else v)
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        return L.resolve(ap)
    finally:
        os.chdir(cwd)


class Stage:
    def __init__(self, name):
        self.name, self.t0 = name, time.time()

    def over(self):
        return (time.time() - self.t0) / 3600 > CAP_H


def cell(stage, task, extra, seed, tag=""):
    cfg = resolved(task, extra, seed)
    mk = os.path.join(LOG, (tag + "__" if tag else "") + AI.marker_name(cfg))
    if os.path.exists(mk):
        print(f"[skip] {AI.bundle_path_key(cfg)} {tag}", flush=True); return True
    if stage.over():
        json.dump(dict(stage=stage.name, stopped_before=AI.bundle_path_key(cfg), cap_h=CAP_H, stamp=C.stamp()),
                  open(os.path.join(C.ART, f"phase16_STOPPED_{stage.name}.json"), "w"), indent=1)
        print(f"=== STOPPED {stage.name}: 12 h cap reached before {AI.bundle_path_key(cfg)}", flush=True)
        return False
    while C.stamp()["code_dirty"]:
        print("   ml/ is dirty; waiting to start", flush=True); time.sleep(60)
    log = mk.replace(".done", ".log")
    print(f"=== START {AI.bundle_path_key(cfg)} {tag}  {time.strftime('%H:%M:%S')}", flush=True)
    r = subprocess.run([os.path.join(REPO, "venv", "bin", "python"), "-u", "ml/train/loop.py", "train",
                        "--config", "ml/configs/shipped.json", "--task", task, "--world", "v8", *BASE[task],
                        "--seed", str(seed), *extra], cwd=REPO, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    tail = open(log).read().strip().splitlines()[-2:] if os.path.exists(log) else []
    if r.returncode == 0:
        json.dump(dict(identity=AI.identity_of(cfg), stamp=C.stamp(), tag=tag), open(mk, "w"))
    print(f"=== END   {AI.bundle_path_key(cfg)} {tag}  {time.strftime('%H:%M:%S')}  rc={r.returncode}  {tail[:1]}",
          flush=True)
    return True


def val_of(task, extra, seed):
    import loop as L
    cfg = resolved(task, extra, seed)
    p = os.path.join(L.bundle_dir(cfg), "train_log.json")
    return json.load(open(p))["best_val"] if os.path.exists(p) else None


def select_k():
    v1 = val_of("arrival_week", ["--encoder-variant", "share_pna", "--pna-aggregators", "sum", "--low-degree-k", "1"], 7)
    v2 = val_of("arrival_week", ["--encoder-variant", "share_pna", "--pna-aggregators", "sum", "--low-degree-k", "2"], 7)
    st0 = json.load(open(os.path.join(C.ART, "phase16_stage0.json")))
    identical = st0["degree_spectrum"]["bypass_k1_equals_k2"]
    k = 1 if (identical or v1 is None or v2 is None or v1 >= v2) else 2
    rec = dict(val_c_index_seed7={"k1": v1, "k2": v2}, edge_sets_identical=identical, selected_k=k,
               rule="argmax validation C-index at seed 7; identical edge sets or a tie -> k = 1", stamp=C.stamp())
    json.dump(rec, open(os.path.join(C.ART, "phase16_k_selection.json"), "w"), indent=1)
    print("k selection:", rec, flush=True)
    return k


def main():
    os.makedirs(LOG, exist_ok=True)
    q = sys.argv[1]
    order = ["a", "b_arr", "b_cap", "audit"] if q == "all" else [q]
    for name in order:
        st = Stage(name); ok = True
        if name == "a":
            for arm in (["--encoder-variant", "share_traj"],
                        ["--encoder-variant", "share_traj", "--traj-delta", "zeroed"]):
                for s in SEEDS:
                    ok = ok and cell(st, "arrival_week", arm, s)
        elif name == "b_arr":
            b1 = ["--encoder-variant", "share_pna", "--pna-aggregators", "mmms"]
            b2 = ["--encoder-variant", "share_pna", "--pna-aggregators", "sum", "--low-degree-k", "1"]
            for s in SEEDS:
                ok = ok and cell(st, "arrival_week", b2, s)
            ok = ok and cell(st, "arrival_week", ["--encoder-variant", "share_pna", "--pna-aggregators", "sum",
                                                   "--low-degree-k", "2"], 7)
            k = select_k()
            for s in SEEDS:
                ok = ok and cell(st, "arrival_week", b1, s)
            for s in SEEDS:
                ok = ok and cell(st, "arrival_week", b1 + ["--low-degree-k", str(k)], s)
        elif name == "b_cap":
            for s in SEEDS:
                ok = ok and cell(st, "capacity_strain", ["--encoder-variant", "heteromp_pna", "--pna-aggregators",
                                                         "mmms"], s)
        elif name == "audit":
            for task in ("arrival_week", "capacity_strain"):
                ok = ok and cell(st, task, ["--bundle-root", AUDIT_ROOT], 7, tag="audit")
        print(f"=== STAGE {name} {'COMPLETE' if ok else 'STOPPED'} {time.strftime('%H:%M:%S')}", flush=True)
        if not ok:
            break
    print(f"=== QUEUE {q} DONE {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
