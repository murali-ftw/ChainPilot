"""Phase 21 -- where the worlds' CSVs live, in-process only.

Phase 21 runs in a separate git worktree. The generated worlds (`db/gen_v8/seed_*`, `data_worlds/`) are gitignored and
exist only in the main checkout, so `HADES_DATA_ROOT` names that checkout and `register()` points `config.WORLDS` at it.
Unset, every path is the repo's own and nothing changes. No identity axis: the path decides where identical bytes are
read from, and `world_hashes()` records them so a run can prove which bytes it read.
"""
from __future__ import annotations
import os, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import config

DATA_ROOT = os.environ.get("HADES_DATA_ROOT", config.REPO)
WORLD2 = "v8w1002"


def register():
    config.WORLDS["v8"] = os.path.join(DATA_ROOT, "db", "gen_v8", "seed_1001")
    for s in (1002, 1003, 1004, 1005):
        config.WORLDS[f"v8s{s}"] = os.path.join(DATA_ROOT, "db", "gen_v8", f"seed_{s}")
    config.WORLDS[WORLD2] = os.path.join(DATA_ROOT, "data_worlds", "v8_seed1002", "seed_1002")
    config.EXPECTED_PANEL_D[WORLD2] = 15
    return dict(data_root=DATA_ROOT, v8=config.WORLDS["v8"], v8w1002=config.WORLDS[WORLD2])


def sha1(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def world_hashes(world, tables=("po_lines.csv", "grn_lines.csv", "training_labels.csv", "supplier_acknowledgements.csv",
                                "sourcing_channels.csv", "logistics_lanes.csv", "po_line_revisions.csv")):
    D = config.WORLDS[world]
    return {t: sha1(os.path.join(D, t)) for t in tables}
