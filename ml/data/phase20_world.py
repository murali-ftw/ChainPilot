"""Phase 20 Stage 2 -- the second world, generated OUTSIDE db/ and registered in-process.

  generate  exec the UNMODIFIED generator_v8.py (its own __file__, so its self-hash 71de78a and every output byte are
            unchanged) with --seed 1002 --out data_worlds/v8_seed1002. Afterwards: `git status -- db` must be empty,
            and SHA-1 of the key tables is recorded (and compared, as a diagnostic, with the stored db/gen_v8/seed_1002).
  register  in-process only: config.WORLDS["v8w1002"] = data_worlds/v8_seed1002/seed_1002 and its panel width (15,
            asserted by cache.build_panel against the MEASURED width, never borrowed). No existing loader is edited.
  cache     the existing ml/data/cache.py builder -> ml/artifacts/cache/v8w1002/
  features  the unchanged Phase 18-19 feature builders for this world: fwd_load, fwd_season, cadence (each asserts as-of)

  python ml/data/phase20_world.py generate
  python ml/data/phase20_world.py cache
  python ml/data/phase20_world.py features
"""
from __future__ import annotations
import os, sys, json, hashlib, subprocess, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import config

WORLD = "v8w1002"
REPO = config.REPO
OUT_ROOT = os.path.join(REPO, "data_worlds", "v8_seed1002")
PATH = os.path.join(OUT_ROOT, "seed_1002")
GEN = os.path.join(REPO, "db", "gen_v8", "generator_v8.py")
KEY_TABLES = ["po_lines.csv", "grn_lines.csv", "training_labels.csv", "channel_performance_weekly.csv", "part_demand_weekly.csv",
              "sourcing_channels.csv", "inventory_transactions.csv", "manifest.json"]


def register():
    config.WORLDS[WORLD] = PATH
    config.EXPECTED_PANEL_D[WORLD] = 15
    return PATH


def sha1(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def generate():
    t = time.time()
    src = open(GEN, "rb").read()
    self_hash = hashlib.sha1(src).hexdigest()[:12]
    assert self_hash == "71de78afa645", f"generator source is not 71de78a ({self_hash})"
    if not os.path.exists(os.path.join(PATH, "manifest.json")):
        os.makedirs(OUT_ROOT, exist_ok=True)
        argv = sys.argv
        sys.argv = [GEN, "--seed", "1002", "--out", OUT_ROOT]
        ns = {"__name__": "__main__", "__file__": GEN, "__builtins__": __builtins__}
        cwd = os.getcwd(); os.chdir(REPO)
        try:
            exec(compile(src, GEN, "exec"), ns)
        finally:
            os.chdir(cwd); sys.argv = argv
    db_status = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", "db"], text=True).strip()
    assert not db_status, f"db/ changed by the generation: {db_status}"
    stored = os.path.join(REPO, "db", "gen_v8", "seed_1002")
    rows = {}
    for f in KEY_TABLES:
        a = sha1(os.path.join(PATH, f))
        b = sha1(os.path.join(stored, f)) if os.path.exists(os.path.join(stored, f)) else None
        rows[f] = dict(sha1=a, equals_stored_db_seed_1002=(a == b) if b else None)
    man = json.load(open(os.path.join(PATH, "manifest.json")))
    rep = dict(world=WORLD, path=os.path.relpath(PATH, REPO), generator_self_hash=self_hash, manifest_code_commit=man.get("code_commit"),
               seed=man.get("seed"), db_status_after="clean", key_tables=rows, seconds=round(time.time() - t, 1))
    os.makedirs(os.path.join(config.ARTIFACTS, "phase20"), exist_ok=True)
    json.dump(rep, open(os.path.join(config.ARTIFACTS, "phase20", "world2_generation.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def cache():
    register()
    import cache as CA
    out = os.path.join(config.CACHE, WORLD)
    meta = CA.build_panel(PATH, out, world=WORLD)
    print(json.dumps({k: v for k, v in meta.items() if k in ("world", "cols", "shape", "dropped_constant_zero")}, indent=1, default=str))


def features():
    register()
    import fwd_load as FL, fwd_season as FS, cadence as CD
    FL.build(WORLD); FS.build(WORLD); CD.build(WORLD)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["generate", "cache", "features"])
    a = ap.parse_args()
    {"generate": generate, "cache": cache, "features": features}[a.mode]()
