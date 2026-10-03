"""Phase 19 -- the `row_family` identity axis, OMITTED AT ITS DEFAULT (deviation 110).

ml/artifact_identity.py is protected in this phase, so the axis cannot be added there. This module wraps it:
  config_name(cfg) = artifact_identity.config_name(cfg)                    when cfg has no row_family (the default)
                   = artifact_identity.config_name(cfg) + "_rf" + family    otherwise
and Phase 19 bundles live under their own root, ml/artifacts/phase19/bundles/{task}/, so a row-family cell can never
occupy an incumbent's path even if the suffix were lost. ml/tests/test_phase19_isolation.py checks both properties:
every stored config is named identically by this wrapper, and a set axis gives a distinct name.
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..")]
import artifact_identity as AI
from config import ARTIFACTS

BUNDLES19 = os.path.join(ARTIFACTS, "phase19", "bundles")


def config_name(cfg) -> str:
    n = AI.config_name(cfg)
    fam = cfg.get("row_family")
    return n + f"_rf{fam}" if fam else n


def bundle_name(cfg) -> str:
    return f"{cfg['world']}_{config_name(cfg)}"


def bundle_dir(cfg) -> str:
    assert not cfg.get("origin"), "Phase 19 binds the fixed split only"
    assert cfg.get("row_family"), "a Phase 19 bundle must carry its row_family"
    return os.path.join(BUNDLES19, cfg["task"], bundle_name(cfg))
