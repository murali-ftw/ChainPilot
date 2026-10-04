"""Serving identity guard. The service REFUSES (raises) when a loaded bundle is not the configured model.

Closes the deviation-28 class (two configurations silently sharing a path) and the deviation-46 class (shipped.json naming
a model the serving path cannot load, while something else is served). A configured member carries:
  name         the bundle name (phase19_identity.bundle_name: artifact_identity's name + _rf<family> when the axis is set)
  identity     artifact_identity.identity_of(cfg) as recorded when the decision was made
  row_family   the Phase 19 axis (absent for an incumbent). identity_of does NOT carry this axis, so an incumbent and its
               row-family sibling share identity_of -- comparing identity_of alone would let one silently stand in for the
               other. The guard therefore checks name AND identity AND row_family AND the directory name.
"""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "train")]
import artifact_identity as AI
import phase19_identity as PI


class IdentityMismatch(RuntimeError):
    """The bundle on disk is not the model the configuration names. Serving stops."""


class NotServable(RuntimeError):
    """The configuration declares a model the serving path cannot load (e.g. a LightGBM model that was never persisted)."""


def describe(cfg):
    cfg = dict(cfg); cfg.setdefault("origin", None)
    return dict(name=PI.bundle_name(cfg), identity=AI.identity_of(cfg), row_family=cfg.get("row_family"))


def check(bundle_dir, member):
    """member: the configured dict {name, identity, row_family}. Returns the loaded cfg or raises IdentityMismatch."""
    path = os.path.join(bundle_dir, "config.json")
    if not os.path.exists(path):
        raise IdentityMismatch(f"no bundle at {bundle_dir}")
    cfg = json.load(open(path))
    if not cfg.get("complete"):
        raise IdentityMismatch(f"{bundle_dir} is not a complete bundle")
    got = describe(cfg)
    want = dict(name=member["name"], identity=member["identity"], row_family=member.get("row_family"))
    diffs = {k: (want[k], got[k]) for k in want if want[k] != got[k]}
    if os.path.basename(os.path.normpath(bundle_dir)) != member["name"]:
        diffs["directory"] = (member["name"], os.path.basename(os.path.normpath(bundle_dir)))
    if diffs:
        raise IdentityMismatch(f"refusing to serve {bundle_dir}: configured != loaded on {sorted(diffs)}: {diffs}")
    return cfg
