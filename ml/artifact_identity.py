"""Artifact identity — one place that names a bundle, a prediction file and an index key.

Deviation 28: `loop.bundle_dir` and `backtest.export` each built the same name independently, they drifted, and a
truncated-history cell silently overwrote a full-history cell's predictions. It was caught by luck. The rule now:

    artifact identity must be ASSERTED UNIQUE AT WRITE TIME, never assumed from a naming convention.

So every writer derives its name here, and every write goes through `guard_write`, which refuses to overwrite an
artifact whose recorded provenance differs.
"""
from __future__ import annotations
import json, os

MANIFEST = "_manifest.json"


def config_name(cfg) -> str:
    """The configuration's name: architecture, depth, learning rate, seed, and any suffix that makes it a DIFFERENT
    configuration. Anything that changes what was trained must appear here, or two cells collide."""
    name = f"{cfg['arch']}_h{cfg['depth']}_lr{cfg['lr']:g}_s{cfg['seed']}"
    if cfg.get("train_snapshots"):
        name += f"_tr{cfg['train_snapshots']}"
    if cfg.get("row_features"):
        name += "_rowfeat"                          # Phase 11 Stage 2: promise_week + line age as head inputs
    if cfg.get("graph_shuffle") is not None:
        name += f"_shuf{cfg['graph_shuffle']}"      # Phase 11A Stage 1: the shuffled-graph control arm
    if cfg.get("drop_relation"):
        name += f"_drop{cfg['drop_relation']}"      # Phase 12 C3: relation ablation at fixed depth
    # Phase 13: every axis that varies in the fill reparameterisation and the history-ratio features
    if cfg.get("fill_head"):
        name += f"_head{cfg['fill_head']}"          # F1 arm: cells22 | beta3 | wideatom | reg
    if cfg.get("ratio_key"):
        name += f"_rk{cfg['ratio_key']}"            # F2 key: ps | sp | psp | hier
        name += f"_{cfg.get('ratio_est', 'ros')}"   # ratio_of_sums | mean_of_ratios
        name += "_shrink" if cfg.get("shrink") else "_raw"
        name += f"_use{cfg.get('ratio_use', 'input')}"   # alone | input | centre
    return name


def bundle_name(cfg) -> str:
    return f"{cfg['world']}_{config_name(cfg)}"


def bundle_path_key(cfg) -> str:
    """Where the bundle actually lives: a name is unique only WITHIN its task/origin directory."""
    if cfg.get("origin"):
        return f"{cfg['task']}/o{cfg['origin']}/{bundle_name(cfg)}"
    return f"{cfg['task']}/{bundle_name(cfg)}"


def pred_stem(cfg, fold: str) -> str:
    return f"{cfg['world']}_{cfg['task']}_o{cfg['origin']}_{config_name(cfg)}_{fold}"


def index_key(cfg) -> str:
    return f"{cfg['world']}|{cfg['task']}|o{cfg['origin']}_{config_name(cfg)}"


def identity_of(cfg) -> dict:
    """The fields that make two artifacts the same artifact. A difference in any of them is a different artifact."""
    keys = ("task", "world", "origin", "arch", "depth", "lr", "seed", "train_snapshots", "max_epochs",
            "row_features", "graph_shuffle", "drop_relation",
            "fill_head", "ratio_key", "ratio_est", "shrink", "ratio_use")
    return {k: cfg.get(k) for k in keys}


class CollisionError(AssertionError):
    pass


def assert_unique(cfgs, label="configurations"):
    """Every configuration must map to a distinct bundle name, prediction stem and index key."""
    for fn, what in ((bundle_path_key, "bundle path"), (lambda c: pred_stem(c, "test"), "prediction stem"),
                     (index_key, "index key")):
        seen = {}
        for c in cfgs:
            k = fn(c)
            prev = seen.get(k)
            if prev is not None and identity_of(prev) != identity_of(c):
                raise CollisionError(f"two distinct {label} share one {what} {k!r}: "
                                     f"{identity_of(prev)} vs {identity_of(c)}")
            seen[k] = c
    return True


def load_manifest(directory) -> dict:
    p = os.path.join(directory, MANIFEST)
    return json.load(open(p)) if os.path.exists(p) else {}


def save_manifest(directory, man) -> None:
    json.dump(man, open(os.path.join(directory, MANIFEST), "w"), indent=1, sort_keys=True)


def guard_write(man, directory, filename, identity, owner=None):
    """Refuse to overwrite an artifact whose recorded identity differs from this writer's.

    A file with no manifest entry is adopted (the manifest post-dates the artifacts it protects) and recorded, so the
    second write of a colliding pair is refused even when the first predates this guard.
    """
    prev = man.get(filename)
    if prev is not None and prev.get("identity") != identity:
        raise CollisionError(
            f"refusing to overwrite {os.path.join(directory, filename)}: it holds {prev.get('identity')} "
            f"(written by {prev.get('owner')}), this writer has {identity}. Two configurations resolve to one path.")
    man[filename] = dict(identity=identity, owner=owner)
    return True


# ------------------------------------------------------------------ Phase 13: score files and completion markers
SCORE_AXES = ("world", "task", "origin", "split", "arm", "fill_head", "ratio_key", "ratio_est", "shrink", "ratio_use",
              "calib", "seed", "train_snapshots")


def score_name(ident: dict) -> str:
    """A score/prediction/marker filename that is a function of the FULL identity (Phase 13 P2).

    Deviation 73: a .done marker keyed on a partial identity collided across three arms and silently skipped two.
    Every axis in SCORE_AXES appears, including absent ones as '-', so no two identities can share a name."""
    unknown = set(ident) - set(SCORE_AXES)
    assert not unknown, f"identity carries axes score_name does not name: {sorted(unknown)}"
    assert ident.get("calib") in (None, "raw", "recal"), f"calibration state must be raw|recal, got {ident.get('calib')}"
    return "__".join(f"{k}={ident.get(k, '-') if ident.get(k) is not None else '-'}" for k in SCORE_AXES)


def marker_name(cfg: dict) -> str:
    """Completion marker for a TRAINING cell: the full bundle identity, never a partial one."""
    return bundle_path_key(cfg).replace("/", "__") + ".done"
