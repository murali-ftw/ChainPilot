"""Phase 12 shared helpers: path setup, the code stamp every artifact carries, and a row bootstrap.

The stamp refuses to be quiet about a dirty tree: `code_dirty` is recorded in every Phase 12 JSON
and `require_clean()` raises, so a result cannot be produced from uncommitted code and then quoted
("commit before running; nothing stamped +dirty").
"""
from __future__ import annotations
import os, sys, json, subprocess, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..")
REPO = os.path.abspath(os.path.join(ML, ".."))
for p in (ML, os.path.join(ML, "train"), os.path.join(ML, "eval"), os.path.join(ML, "data"),
          os.path.join(ML, "models"), os.path.join(ML, "sim"), os.path.join(ML, "opt")):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np

ART = os.path.join(ML, "artifacts")
BUND = os.path.join(ART, "bundles")
V8_SEEDS = (7, 17, 27, 37, 47)


def stamp():
    try:
        commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", "ml"],
                                             text=True).strip())
    except Exception:
        commit, dirty = "unknown", True
    return dict(code_commit=commit, code_dirty=dirty,
                code_version=f"{commit}{'+dirty' if dirty else ''}",
                run_ts=datetime.datetime.now().isoformat(timespec="seconds"))


def require_clean():
    s = stamp()
    if s["code_dirty"]:
        raise SystemExit(f"refusing to run: ml/ has uncommitted changes ({s['code_version']}). Commit first.")
    return s


def dump(obj, name):
    path = os.path.join(ART, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return path


def boot_mean_ci(x, B=2000, seed=0, alpha=0.05):
    """Row bootstrap of a mean. Rows are treated as independent -- callers must say where that is
    optimistic (rows inside one snapshot share that snapshot's shocks)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (B, len(x)))
    bm = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(bm, alpha / 2)), float(np.quantile(bm, 1 - alpha / 2)), float(bm.std())
