"""Deviation 28's regression test: two distinct configurations must never resolve to one artifact.

The defect: `loop.bundle_dir` carried the truncation suffix, `backtest.export` did not, so a truncated-history cell
overwrote the full-history cell's predictions and index entry. Nothing failed; it was caught by noticing a number had
moved. These tests fail if that class of collision becomes possible again.

    python ml/tests/test_artifact_identity.py        (or: pytest ml/tests/test_artifact_identity.py)
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..")]
import artifact_identity as AI

FULL = dict(task="arrival_week", world="v6", origin=7, arch="lite", depth=4, lr=0.00025, seed=7)
TRUNC = dict(FULL, train_snapshots=18)


def test_truncation_suffix_separates_every_name():
    """The exact deviation-28 pair: same task/world/origin/arch/depth/lr/seed, different training history."""
    assert AI.bundle_name(FULL) != AI.bundle_name(TRUNC)
    assert AI.bundle_path_key(FULL) != AI.bundle_path_key(TRUNC)
    assert AI.pred_stem(FULL, "test") != AI.pred_stem(TRUNC, "test")
    assert AI.pred_stem(FULL, "val") != AI.pred_stem(TRUNC, "val")
    assert AI.index_key(FULL) != AI.index_key(TRUNC)


def test_assert_unique_accepts_distinct_and_rejects_collisions():
    AI.assert_unique([FULL, TRUNC, dict(FULL, seed=17)])
    # a name is unique only within its task/origin directory: the same name under a different origin is NOT a collision
    AI.assert_unique([FULL, dict(FULL, origin=1, max_epochs=120)])
    try:
        AI.assert_unique([FULL, dict(FULL, max_epochs=200)])       # same name, different identity
    except AI.CollisionError:
        pass
    else:
        raise AssertionError("two configurations differing only in a field absent from the name must collide loudly")


def test_every_identity_field_either_names_or_collides():
    """A field that changes what was trained must change the name, or assert_unique must refuse the pair."""
    for field, value in (("arch", "mp"), ("depth", 0), ("lr", 0.002), ("seed", 17), ("train_snapshots", 18),
                         ("origin", 8), ("world", "v7"), ("task", "fill_rate"), ("max_epochs", 200)):
        other = dict(FULL, **{field: value})
        if AI.identity_of(other) == AI.identity_of(FULL):
            continue
        named = AI.pred_stem(other, "test") != AI.pred_stem(FULL, "test")
        if named:
            continue
        try:
            AI.assert_unique([FULL, other])
        except AI.CollisionError:
            continue
        raise AssertionError(f"{field} changes the artifact but neither names it nor collides")


def test_guard_write_refuses_a_differing_overwrite():
    man = {}
    AI.guard_write(man, "/tmp/preds", "x_test.npz", AI.identity_of(FULL), owner="bundleA")
    AI.guard_write(man, "/tmp/preds", "x_test.npz", AI.identity_of(FULL), owner="bundleA")   # idempotent re-export
    try:
        AI.guard_write(man, "/tmp/preds", "x_test.npz", AI.identity_of(TRUNC), owner="bundleB")
    except AI.CollisionError:
        return
    raise AssertionError("guard_write must refuse an overwrite whose identity differs")


def test_guard_write_adopts_unmanifested_files():
    """Artifacts written before the manifest existed are adopted, then protected."""
    man = {}
    AI.guard_write(man, "/tmp/preds", "legacy_test.npz", AI.identity_of(FULL), owner="bundleA")
    assert man["legacy_test.npz"]["identity"] == AI.identity_of(FULL)


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); n += 1; print(f"  PASS {name}")
    print(f"{n} passed")
