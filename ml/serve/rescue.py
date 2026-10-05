"""Phase 23B Stage 4 -- PREDICT-THE-RESCUE as a loadable, identity-guarded service (new file; nothing existing edited).

What it says: P(a transfer-in is recorded at this part-plant in week t0 + 7(w + 1)), w = 0..12 -- Phase 17 B1a, re-fitted on
CLEAN inputs in Phase 23B (ml/eval/phase23b_rescue.py fit, arm `clean`). One LightGBM booster per seed, saved at its best
iteration under ml/artifacts/phase23b/bundles/rescue_lgbm_v8clean_s{seed}/ with identity.json.

Guards:
  identity   load_bundle() recomputes the bundle's identity hash and compares seed, world, GBM-config hash, commit and the full
             feature list with the CONFIGURED member; any difference raises IdentityMismatch. The booster's own feature count and
             tree count must match the identity too.
  clean      features are read only from a CLEAN cache: meta["clean_of"] set and meta["replaced_columns"] == clean_panel.LEAKING
             (the nine generator-form columns are never read; their as-of-safe replacements are). A leaky cache raises NotClean.
  as-of      the panel row used is the one whose week contains t0 (asserted: week start <= t0); every master-table row the feature
             builder reads (sourcing_channels, suppliers, supplier_upstream, alternate_sources) is asserted recorded_ts <= t0 where
             the table carries recorded_ts. A violation raises AsOfViolation.
Features are built by Phase 17's own builder, phase7_fit.World.part_plant_features (per part-plant mean / min of the clean panel
row at t0, plus graph counts) and the week offset -- the code the bundle was trained on, so serving is bit-exact.
inventory_position_weekly and part_demand_weekly are never read (ml/tests/test_phase23b_serve.py scans this file's AST).
"""
from __future__ import annotations
import os, sys, json, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "baselines")]
import numpy as np, pandas as pd
import config

BUNDLES = os.path.join(config.ARTIFACTS, "phase23b", "bundles")
MASTERS = ("sourcing_channels", "suppliers", "supplier_upstream", "alternate_sources")


class IdentityMismatch(AssertionError):
    pass


class NotClean(AssertionError):
    pass


class AsOfViolation(AssertionError):
    pass


def identity_hash(ident):
    return hashlib.sha1(json.dumps({k: ident[k] for k in ("feature_list", "gbm_config_hash", "seed", "world")},
                                   sort_keys=True).encode()).hexdigest()[:16]


CONFIGURED = os.path.join(config.ARTIFACTS, "phase23b", "rescue_configured.json")


def file_sha1(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def record_configured(seeds=(7, 17, 27, 37, 47), bundles=BUNDLES, out=CONFIGURED):
    """Write the configured members ONCE (after the fit): each seed's identity plus the SHA-1 of its booster file."""
    assert not os.path.exists(out), f"refusing to overwrite the configured record {out}"
    rec = {}
    for s in seeds:
        d = os.path.join(bundles, f"rescue_lgbm_v8clean_s{s}")
        ident = json.load(open(os.path.join(d, "identity.json")))
        rec[str(s)] = dict(ident, model_sha1=file_sha1(os.path.join(d, "model.txt")))
    json.dump(rec, open(out, "w"), indent=1)
    return rec


def configured_member(seed, record=CONFIGURED):
    """The configured member for a seed, from the configured record (NOT from the bundle being checked)."""
    return json.load(open(record))[str(seed)]


def load_bundle(path, expected):
    """Identity guard first, then the booster. `expected` = the configured member (dict with the identity fields)."""
    import lightgbm as lgb
    ident = json.load(open(os.path.join(path, "identity.json")))
    if identity_hash(ident) != ident.get("identity_hash"):
        raise IdentityMismatch(f"{path}: identity.json does not hash to its recorded identity_hash")
    for k in ("seed", "world", "gbm_config_hash", "commit", "feature_list", "identity_hash"):
        if ident.get(k) != expected.get(k):
            raise IdentityMismatch(f"{path}: {k} = {ident.get(k)!r}, configured {expected.get(k)!r}")
    if os.path.basename(os.path.normpath(path)) != f"rescue_lgbm_{ident['world']}_s{ident['seed']}":
        raise IdentityMismatch(f"{path}: directory name does not match the identity")
    if "model_sha1" in expected and file_sha1(os.path.join(path, "model.txt")) != expected["model_sha1"]:
        raise IdentityMismatch(f"{path}: model.txt is not the configured booster (SHA-1 differs)")
    booster = lgb.Booster(model_file=os.path.join(path, "model.txt"))
    if booster.num_feature() != len(ident["feature_list"]):
        raise IdentityMismatch(f"{path}: booster has {booster.num_feature()} features, identity {len(ident['feature_list'])}")
    if booster.num_trees() != ident["best_iteration"]:
        raise IdentityMismatch(f"{path}: booster has {booster.num_trees()} trees, identity best_iteration {ident['best_iteration']}")
    return dict(booster=booster, identity=ident, path=path)


def assert_asof(frame, t0, table):
    """Every row a feature builder uses must be recorded by t0 (where the table carries recorded_ts)."""
    if "recorded_ts" not in frame.columns:
        return 0
    late = pd.to_datetime(frame.recorded_ts) > pd.Timestamp(t0)
    if late.any():
        raise AsOfViolation(f"{table}: {int(late.sum())} rows recorded after t0 {pd.Timestamp(t0).date()}")
    return int(len(frame))


def assert_clean(meta):
    from clean_panel import LEAKING
    if not meta.get("clean_of") or list(meta.get("replaced_columns") or []) != list(LEAKING):
        raise NotClean(f"cache {meta.get('world')!r} is not a clean cache (clean_of={meta.get('clean_of')!r})")


class Features:
    """As-of feature builder over a clean world; one World per process (the panel is memory-mapped)."""

    def __init__(self, world="v8clean"):
        import lightgbm  # noqa: F401  -- phase7_fit requires lightgbm before anything else
        import phase21_paths as PP, phase7_fit as P7
        PP.register()
        for p in ("v8", "v8w1002"):
            config.WORLDS.setdefault(p + "clean", config.WORLDS[p]); config.EXPECTED_PANEL_D.setdefault(p + "clean", 15)
        self.P7, self.world = P7, world
        self.Wd = P7.World(world)
        assert_clean(self.Wd.meta)
        self.masters = {t: pd.read_csv(os.path.join(config.WORLDS[world], f"{t}.csv"), nrows=None) for t in MASTERS}

    def build(self, part_ids, plant_ids, t0, week_offsets):
        t0 = pd.Timestamp(t0)
        row = int((t0 - self.Wd.w0).days // 7)
        if not (0 <= row < self.Wd.panel.shape[1]) or self.Wd.w0 + pd.Timedelta(days=7 * row) > t0:
            raise AsOfViolation(f"t0 {t0.date()}: panel row {row} is not the week containing t0")
        for t, f in self.masters.items():
            assert_asof(f, t0, t)
        keys = np.char.add(np.char.add(np.asarray(part_ids, str), "|"), np.asarray(plant_ids, str))
        lb = pd.DataFrame({"key": keys, "snapshot_date": [t0] * len(keys)})
        X = self.Wd.part_plant_features(lb).to_numpy(np.float32)
        return np.concatenate([X, np.asarray(week_offsets, np.float32)[:, None]], 1)


def predict(member, X):
    b = member["booster"]
    return b.predict(X, num_iteration=member["identity"]["best_iteration"]).astype(np.float32)


def serve(part_ids, plant_ids, t0, week_offsets, seeds=(7, 17, 27, 37, 47), feats=None, bundles=BUNDLES, record=CONFIGURED):
    """5-seed mean P(transfer-in that week), built strictly as-of t0. Returns (mean, per-seed array)."""
    feats = feats or Features()
    X = feats.build(part_ids, plant_ids, t0, week_offsets)
    per = np.stack([predict(load_bundle(os.path.join(bundles, f"rescue_lgbm_v8clean_s{s}"), configured_member(s, record)), X)
                    for s in seeds])
    return per.mean(0), per
