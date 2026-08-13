#!/usr/bin/env python3
"""
Privileged ground truth: per-supplier hidden resilience and Mechanism J tier depth.

Mechanism E's `resilience` is never emitted to CSV -- that is the point of it, and
`ml/data/loader.py::verify_no_hidden_state` asserts it stays that way. Question 3 of
the gate diagnostic asks whether the nodes a depth gate disagrees on share a hidden
property, which is a question about the *dataset*, not about what a model could
learn. Answering it therefore requires reading the generator's internal state
directly.

This script does that by executing `db/generate_dataset.py` in-process with `write()`
stubbed out -- the same mechanism `db/run_benchmark.py --stats-only` uses -- and
lifting `RESILIENCE` and `SUP_CHAIN` out of the resulting namespace. Nothing it
produces may ever be fed to a model; it exists to characterise the world, and every
report that uses it says so.

    python3 ml/extract_hidden_state.py --variant J --seed 42 --config v1 \\
        --out out/hidden/J_42.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import contextlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(REPO, "db", "generate_dataset.py")


def extract(variant: str, seed: int, config: str = "v1") -> dict:
    src = open(GEN).read()
    src = src.replace("def write(name, header, rows):",
                      "def write(name, header, rows):\n    return", 1)
    src = src.replace(
        'with open(os.path.join(OUT, "resolved_config.json"), "w") as _f:\n'
        '    json.dump(_manifest, _f, indent=2, sort_keys=True)',
        "MANIFEST_OUT = _manifest", 1)
    argv = ["generate_dataset.py", "--variant", variant, "--config", config, "--seed", str(seed)]
    saved, sys.argv = sys.argv, argv
    # A real module object registered in sys.modules, not a bare dict: `@dataclass`
    # resolves `cls.__module__` through sys.modules, and fails on a namespace that
    # isn't registered there.
    import types
    mod = types.ModuleType("__hidden_extract__")
    mod.__file__ = GEN
    sys.modules["__hidden_extract__"] = mod
    ns = mod.__dict__
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(compile(src, GEN, "exec"), ns)
            except SystemExit:
                pass
    finally:
        sys.argv = saved

    # Mechanism B's Type A/B/C groups: `HP_GROUPS` is internal state only -- no CSV
    # carries group id, type or membership, which is what makes the discovery question
    # meaningful in the first place. Read here for VALIDATION only, never fed to a model.
    hp_groups = [{"index": i, "type": g["type"], "members": list(g["members"])}
                 for i, g in enumerate(ns.get("HP_GROUPS", []) or [])]
    resilience = dict(ns.get("RESILIENCE", {}))
    chain = ns.get("SUP_CHAIN", {}) or {}
    tier = ns.get("SUP_TIER", {}) or {}
    visible = set(ns.get("VISIBLE_SUP", set()) or set())
    # Chain length = how many upstream hops Mechanism J gave this supplier's head.
    chain_len = {head: len(ch) for head, ch in chain.items()}
    return {
        "variant": variant, "seed": seed, "config": config,
        "resilience": resilience,                  # {} when Mechanism E is off
        "hp_groups": hp_groups,                    # [] when Mechanism B is off
        "coparent_coupling": float(ns.get("COPARENT_COUPLING", 0.0)),
        "alpha": float(ns.get("CFG").alpha) if ns.get("CFG") is not None else None,
        "mechanisms": list(ns.get("MECHS", ()) or ()),
        "chain_len": chain_len,                    # {} when Mechanism J is off
        "tier": {k: int(v) for k, v in tier.items()},
        "visible_suppliers": sorted(visible),
        "n_suppliers": len(ns.get("suppliers", [])),
        # Shipment -> supplier, so a Shipment-level diagnostic can inherit its
        # supplier's hidden properties.
        "shipment_supplier": {sh["id"]: sh["supplier_id"]
                              for sh in ns.get("shipments", []) if sh.get("supplier_id")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    blob = extract(args.variant, args.seed, args.config)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(blob, fh)
    print(f"variant {args.variant} seed {args.seed}: "
          f"{len(blob['resilience'])} resilience values, "
          f"{len(blob['chain_len'])} chains, {len(blob['shipment_supplier']):,} shipment links")
    return 0


if __name__ == "__main__":
    sys.exit(main())
