"""Phase 24 Stage 1 -- artifact and input check (read-only). New file.

Every path is taken from the reports that name it (phase-13/14/15/17/22/23b and phase23b/stage3_simulation.md); no globs.
  a. the six Phase 22 CLEAN bundles the simulation needs (arrival s7; fill s7..s47) and the six LEAKY bundles the harness
     hard-codes (ml/sim/montecarlo.read_heads): present, complete, directory == the identity module's bundle_name(cfg),
     config fields as expected; checkpoint.pt SHA-1 PINNED here for the Stage 2 wrapper's identity gate (G1).
  b. ml/artifacts/phase23b (Phase 23B's clean B1a bundles and predictions, written on the other machine).
  c. the stored files Stage 2 must reproduce, and the clean cache Stage 6 reads.
Writes ml/artifacts/phase24/inputs.json. Reads no CSV, runs no model.

    python ml/eval/phase24_inputs.py
"""
from __future__ import annotations
import os, sys, json, hashlib, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import phase12_common as C
import artifact_identity as AI
import phase19_identity as PI

SEEDS = (7, 17, 27, 37, 47)
LEAKY = {("arrival_week", 7): "bundles/arrival_week/v8_lite_h4_lr0.00025_s7",
         **{("fill_rate", s): f"bundles/fill_rate/v8_none_h0_lr0.000125_s{s}" for s in SEEDS}}
CLEAN = {("arrival_week", 7): "phase22/bundles/arrival_week/v8clean_lite_h4_lr0.00025_s7",
         **{("fill_rate", s): f"phase22/bundles/fill_rate/v8clean_none_h0_lr0.000125_s{s}" for s in SEEDS}}
EXPECT = {"arrival_week": dict(arch="lite", depth=4, lr=0.00025), "fill_rate": dict(arch="none", depth=0, lr=0.000125)}
NEEDED = ("checkpoint.pt", "normaliser.npz", "recalibration.json", "config.json")      # what loop.load_bundle reads
STORED = ["phase13_s1.json", "phase14_sim.json", "phase15.json", "phase15_sim_gate.json", "phase12_b2_validate.json",
          "phase12_b2_seeds.json"] + [f"phase15_sim/{f}_s{s}.npz" for f in ("val", "test") for s in SEEDS]
P23B = ["phase23b/rescue_configured.json"] + [f"phase23b/bundles/rescue_lgbm_v8clean_s{s}/{x}" for s in SEEDS
                                              for x in ("model.txt", "identity.json")]


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_bundle(rel, task, seed, world, namer):
    p = os.path.join(C.ART, rel)
    r = dict(path=f"ml/artifacts/{rel}", task=task, seed=seed, world=world)
    missing = [f for f in NEEDED if not os.path.exists(os.path.join(p, f))]
    if missing:
        r.update(status="MISSING: " + ", ".join(f"ml/artifacts/{rel}/{f}" for f in missing)); return r
    cfg = json.load(open(os.path.join(p, "config.json")))
    want = dict(task=task, world=world, seed=seed, **EXPECT[task])
    fields_ok = all(cfg.get(k) == v for k, v in want.items())
    name_ok = os.path.basename(p) == namer(cfg)
    r.update(status="OK" if (fields_ok and name_ok and cfg.get("complete")) else "IDENTITY MISMATCH",
             complete=bool(cfg.get("complete")), fields_match=fields_ok, name_is_identity=name_ok,
             identity_module=namer.__module__, checkpoint_sha1=sha1(os.path.join(p, "checkpoint.pt")),
             normaliser_sha1=sha1(os.path.join(p, "normaliser.npz")), recalibration_sha1=sha1(os.path.join(p, "recalibration.json")))
    return r


def main():
    st = C.require_clean()
    R = dict(stamp=st)
    R["leaky_bundles"] = {f"{t}|s{s}": check_bundle(rel, t, s, "v8", AI.bundle_name) for (t, s), rel in LEAKY.items()}
    R["clean_bundles"] = {f"{t}|s{s}": check_bundle(rel, t, s, "v8clean", PI.bundle_name) for (t, s), rel in CLEAN.items()}
    R["stored_files"] = {f: (dict(status="OK", sha1=sha1(os.path.join(C.ART, f))) if os.path.exists(os.path.join(C.ART, f))
                             else dict(status=f"MISSING: ml/artifacts/{f}")) for f in STORED}
    # the pinned simulation code (deviation 122): the cached copy must equal git show 9e2d59d
    src = subprocess.check_output(["git", "-C", C.REPO, "show", "9e2d59d:ml/opt/order_policy.py"], text=True)
    cp = os.path.join(C.ART, "cache", "order_policy_9e2d59d.py")
    R["order_policy_9e2d59d"] = dict(git_sha1=hashlib.sha1(src.encode()).hexdigest(),
                                     cached_copy_equal=os.path.exists(cp) and open(cp).read() == src)
    present = os.path.isdir(os.path.join(C.ART, "phase23b"))
    R["phase23b"] = dict(present=present, files={f: os.path.exists(os.path.join(C.ART, f)) for f in P23B},
                         note=None if present else "ml/artifacts/phase23b ABSENT: Phase 23B's clean rescue figures are QUOTED from "
                                                   "reports/part2/phase-23b.md, not recomputed")
    R["phase17_artifacts"] = dict(present=os.path.isdir(os.path.join(C.ART, "phase17")),
                                  note="b1_rows.npz is rebuilt for Stage 6 under ml/artifacts/phase24/ (gated against phase15_sim)")
    meta = json.load(open(os.path.join(C.ART, "cache", "v8clean", "meta.json")))
    R["v8clean_cache"] = dict(clean_of=meta.get("clean_of"), replaced_columns=meta.get("replaced_columns"))
    ok_l = all(v["status"] == "OK" for v in R["leaky_bundles"].values())
    ok_c = all(v["status"] == "OK" for v in R["clean_bundles"].values())
    ok_s = all(v["status"] == "OK" for v in R["stored_files"].values())
    R["can_run"] = {"stage2_leaky_reproduction": ok_l and ok_s and R["order_policy_9e2d59d"]["cached_copy_equal"],
                    "stage2_clean_arm": ok_l and ok_c and ok_s,
                    "stage3_rescue_row_recomputed": present, "stage3_rescue_row_quoted": True,
                    "stage6_b1b": bool(R["v8clean_cache"]["clean_of"]) and R["stored_files"]["phase15_sim/test_s7.npz"]["status"] == "OK"}
    os.makedirs(os.path.join(C.ART, "phase24"), exist_ok=True)
    print(json.dumps({k: v for k, v in R.items() if k not in ("stamp",)}, indent=1, default=str)[:6000])
    print(C.dump(R, "phase24/inputs.json"))


if __name__ == "__main__":
    main()
