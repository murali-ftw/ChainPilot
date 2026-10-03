"""Phase 18 Stage 2 -- PRIVILEGED. Re-run the v8 generator UNMODIFIED into a scratch directory, verify every emitted
file equals the stored world, and only then save the latents the generator does not persist.

Hidden generator state is never a model feature. This file and everything it writes live under
reports/phase18/oracle/ with the PRIVILEGED__ prefix; ml/tests/test_phase18_isolation.py asserts no ml/ module imports it.

How the run stays byte-identical: the generator's source is exec'd as-is, with `__file__` set to the real path, so its
code-commit stamp (sha1 of its own bytes, written into snapshots.csv and manifest.json) is unchanged and no RNG draw is
added or moved. The latents are read from the finished namespace afterwards.

What `_sim.npz` already persists (joined directly, no re-run needed): sup_state [week, supplier], regime [week],
K / util / ordered / DECL / delivered [month, supplier], and per PO line pt / pch / pq / pd / pa.
What it does NOT persist, and is the reason for this re-run: transit_state [week, plant], dem_state [week, plant].

  python reports/phase18/oracle/PRIVILEGED__regen_latents.py --scratch <dir>      # ~generator runtime, ~10 GB peak
  python reports/phase18/oracle/PRIVILEGED__regen_latents.py --scratch <dir> --verify-only
"""
from __future__ import annotations
import os, sys, json, hashlib, argparse, time
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
GEN = os.path.join(REPO, "db", "gen_v8", "generator_v8.py")
STORED = os.path.join(REPO, "db", "gen_v8", "seed_1001")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PRIVILEGED__latents_v8s1001.npz")
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PRIVILEGED__regen_check.json")
# Files carrying a wall-clock write time are compared by CONTENT with that time removed; every other file by SHA-1:
#   manifest.json / parameters_v8.json  generated_at          dataset_coverage.csv  last column (per-table write time)
#   _sim.npz / _events.npz              zip member headers carry the write time -> every array compared
# level4.json is written by validator_v8.py (l. 1365), not by the generator, so a generator re-run cannot produce it.
NOT_GENERATOR = {"level4.json"}


def sha1(path, buf=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                return h.hexdigest()
            h.update(b)


def npz_content_equal(a, b):
    za, zb = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    if sorted(za.files) != sorted(zb.files):
        return False, "member list differs"
    for k in za.files:
        if za[k].shape != zb[k].shape or not np.array_equal(za[k], zb[k]):
            return False, f"array {k} differs"
    return True, f"{len(za.files)} arrays equal"


def strip_time(x):
    if isinstance(x, dict):
        return {k: strip_time(v) for k, v in x.items() if k != "generated_at"}
    return [strip_time(v) for v in x] if isinstance(x, list) else x


def verify(scratch_world):
    names = sorted(os.listdir(STORED))
    names = [n for n in names if not n.startswith("._")]
    rows, ok_all = [], True
    for n in names:
        a, b = os.path.join(STORED, n), os.path.join(scratch_world, n)
        if n in NOT_GENERATOR:
            rows.append(dict(file=n, equal=None, how="validator output, not generator output: not compared")); continue
        if not os.path.exists(b):
            rows.append(dict(file=n, equal=False, how="missing in re-run")); ok_all = False; continue
        if n.endswith(".npz"):
            eq, how = npz_content_equal(a, b)
        elif n in ("manifest.json", "parameters_v8.json"):
            ja, jb = strip_time(json.load(open(a))), strip_time(json.load(open(b)))
            eq, how = ja == jb, "json equal except generated_at (at any depth)"
        elif n == "dataset_coverage.csv":
            la = [l.rsplit(",", 1)[0] for l in open(a).read().splitlines()]
            lb_ = [l.rsplit(",", 1)[0] for l in open(b).read().splitlines()]
            eq, how = la == lb_, "equal except the per-table write-time column"
        else:
            ha, hb = sha1(a), sha1(b)
            eq, how = ha == hb, f"sha1 {ha[:12]}" + ("" if ha == hb else f" vs {hb[:12]}")
        rows.append(dict(file=n, equal=bool(eq), how=how)); ok_all &= bool(eq)
    extra = sorted(set(os.listdir(scratch_world)) - set(names))
    return ok_all, rows, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    world = os.path.join(a.scratch, "seed_1001")
    ns = None
    if not a.verify_only:
        t = time.time()
        src = open(GEN, "rb").read()
        assert hashlib.sha1(src).hexdigest()[:12] == "71de78afa645", "generator source is not the manifest's 71de78a"
        sys.argv = [GEN, "--seed", "1001", "--out", a.scratch]
        ns = {"__name__": "__main__", "__file__": GEN, "__builtins__": __builtins__}
        cwd = os.getcwd(); os.chdir(REPO)
        try:
            exec(compile(src, GEN, "exec"), ns)
        finally:
            os.chdir(cwd)
        print(f"generator re-run done in {time.time() - t:.0f}s", flush=True)
    ok, rows, extra = verify(world)
    rep = dict(stored=os.path.relpath(STORED, REPO), rerun=world, generator_sha1_12="71de78afa645", all_equal=ok,
               n_files=sum(r["equal"] is not None for r in rows), n_equal=sum(r["equal"] is True for r in rows),
               not_compared=[r["file"] for r in rows if r["equal"] is None], files=rows, extra_in_rerun=extra)
    json.dump(rep, open(REPORT, "w"), indent=1)
    print(f"verify: {rep['n_equal']}/{rep['n_files']} files equal; all_equal={ok}", flush=True)
    if not ok:
        print("BLOCKED: the re-run does not reproduce the stored world; no latent is extracted.", flush=True)
        sys.exit(2)
    if ns is not None:
        np.savez_compressed(OUT, PRIVILEGED__transit_state=np.asarray(ns["transit_state"]),
                            PRIVILEGED__dem_state=np.asarray(ns["dem_state"]),
                            PRIVILEGED__plant_f=np.asarray(ns["plant_f"]), PRIVILEGED__rate=np.asarray(ns["rate"]),
                            CL=np.asarray(ns["CL"]), CS=np.asarray(ns["CS"]), MONTHKEY=np.asarray(ns["MONTHKEY"]))
        print("latents saved ->", OUT, flush=True)


if __name__ == "__main__":
    main()
