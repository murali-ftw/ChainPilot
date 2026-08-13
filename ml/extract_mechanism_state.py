#!/usr/bin/env python3
"""
Privileged mechanism state, per variant-seed — the multi-label ground truth §10 ranks against.

`ml/extract_hidden_state.py` lifts `HP_GROUPS` (and Mechanism E/J state) out of the
generator's namespace for validation. §10's hypothesis module needs a strictly larger read:
it labels *supplier pairs* by which generator mechanism, if any, actually couples them, and
`HP_GROUPS` is only one of those mechanisms. This script lifts the rest by the same method
-- executing `db/generate_dataset.py` in-process with `write()` stubbed -- and emits, per
variant-seed:

* `hp_groups`          -- Mechanism B's Type A/B/C groups (same read as the sibling script;
                          cross-checked against `out/hidden_mid/` on load, see
                          `ml/hypothesis_labels.py`).
* `base_pools`         -- the base world's shared hidden-factor event pools, `H_PORT`,
                          `H_TRUCK`, `H_CUSTOMS`, `H_POLYMER`. Never emitted to CSV; two of
                          the three are *defined* by `country` and the third by the `sea`
                          flag, which is why §9.7.1 found them observable-adjacent.
* `cs_rewires`         -- Mechanism C's sourcing-graph rewiring events. Expected empty on
                          Variants B and D; the point of reading it is to confirm that
                          rather than assume it (`reports/layer3_testing.md` §10.1).
* `shock_events`       -- Mechanism H. Same reason.
* `coparents`          -- the `component_suppliers` co-parent graph, whose `COPARENT_COUPLING`
                          term is a live causal bleed-through on B and D.
* `idio`               -- per-supplier idiosyncratic outage windows. These are what makes the
                          "unknown / coincidental" class a real mechanism rather than a
                          leftover bucket.
* `sea`, `country`, `lead_time_days` -- per supplier, so the report can state exactly how much
                          of each pool is observable from emitted columns.

**This output is training-time and evaluation-time only and is never a model input**, the
same disclosure `reports/layer3_testing.md` §9.1 made for Stage 1's contrastive supervision.
Unlike Stage 1, nothing here reaches a gradient of SHARE itself: it is the *target* of the
§10 ranking head, which is a separate model reading observable pair features only.

    python3 ml/extract_mechanism_state.py --variants B,D --seeds 42,43,44,45,46 \
        --config <mid.json> --out-dir out/hidden_mid_ext
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(REPO, "db", "generate_dataset.py")


def _namespace(variant: str, seed: int, config: str) -> tuple[dict, list[str]]:
    """Run the generator in-process with `write()` stubbed, return its namespace.

    Byte-identical to `ml/extract_hidden_state.py::extract`'s mechanism, including the
    real `types.ModuleType` registration that `@dataclass` needs -- copied rather than
    imported so a change to one script's read cannot silently alter the other's.
    """
    src = open(GEN).read()
    src = src.replace("def write(name, header, rows):",
                      "def write(name, header, rows):\n    return", 1)
    src = src.replace(
        'with open(os.path.join(OUT, "resolved_config.json"), "w") as _f:\n'
        '    json.dump(_manifest, _f, indent=2, sort_keys=True)',
        "MANIFEST_OUT = _manifest", 1)
    argv = ["generate_dataset.py", "--variant", variant, "--config", config,
            "--seed", str(seed)]
    saved, sys.argv = sys.argv, argv
    mod = types.ModuleType("__mech_extract__")
    mod.__file__ = GEN
    sys.modules["__mech_extract__"] = mod
    ns = mod.__dict__
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                exec(compile(src, GEN, "exec"), ns)
            except SystemExit:
                # Two very different things raise SystemExit here and the sibling script's
                # bare `except SystemExit: pass` cannot tell them apart. `resolve_config`
                # exits BEFORE the world is built -- one stray key in a JSON override is
                # enough -- and swallowing that hands back an EMPTY namespace that reads as
                # "this variant has no mechanisms" rather than as a crash. The check suite
                # exits AFTER everything is built, so a failure there leaves the state
                # perfectly readable and is a fact about the dataset worth recording. The
                # `suppliers` test below separates them.
                pass
    finally:
        sys.argv = saved
    if not ns.get("suppliers"):
        raise RuntimeError(
            f"generator produced no suppliers for variant {variant} seed {seed} "
            f"(config={config}); stderr: {err.getvalue()[:400] or out.getvalue()[-400:]}")
    failed = [line.split("]", 1)[1].strip() for line in out.getvalue().splitlines()
              if line.strip().startswith("[FAIL]")]
    return ns, failed


def extract(variant: str, seed: int, config: str) -> dict:
    ns, failed_checks = _namespace(variant, seed, config)
    suppliers = ns.get("suppliers", [])
    coparents = {k: sorted(v) for k, v in (ns.get("coparents", {}) or {}).items() if v}
    return {
        "variant": variant, "seed": seed, "config": os.path.basename(config),
        "mechanisms": list(ns.get("MECHS", ()) or ()),
        "hp_alpha": float(ns.get("HP_ALPHA", 0.0)),
        "coparent_coupling": float(ns.get("COPARENT_COUPLING", 0.0)),
        # Supplier order is the generator's own insertion order, which is also the emitted
        # CSV row order -- asserted in `ml/hypothesis_labels.py` rather than trusted.
        "supplier_ids": [s["id"] for s in suppliers],
        "country": {s["id"]: s["country"] for s in suppliers},
        "sea": sorted(s["id"] for s in suppliers if s.get("sea")),
        "lead_time_days": {s["id"]: int(s["lead_time_days"]) for s in suppliers},
        "hp_groups": [{"index": i, "type": g["type"], "members": sorted(g["members"])}
                      for i, g in enumerate(ns.get("HP_GROUPS", []) or [])],
        "base_pools": {name: sorted(ns.get(name, set()) or set())
                       for name in ("H_PORT", "H_TRUCK", "H_CUSTOMS", "H_POLYMER")},
        "cs_rewires": [{"component_id": c, "old": o, "new": n,
                        "when": str(w), "kind": k}
                       for c, o, n, w, k in (ns.get("CS_REWIRES", []) or [])],
        "n_shock_events": len(ns.get("SHOCK_EVENTS", []) or []),
        "shock_members": [sorted(m) for m, *_ in (ns.get("SHOCK_EVENTS", []) or [])],
        "coparents": coparents,
        # (start, peak, end) per supplier that drew an idiosyncratic outage; the coincidental
        # co-degradation these produce is the substance of the "unknown" class.
        "idio": {k: [str(v[0]), str(v[1]), str(v[2]), float(v[3])]
                 for k, v in (ns.get("IDIO", {}) or {}).items() if v},
        "n_hp_b_events": len(ns.get("HP_B_EVENTS", []) or []),
        # The generator's own acceptance checks, run on this exact world. Recorded rather
        # than assumed to pass: seed 44 fails "Type A/B/C indistinguishable from TOPOLOGY",
        # which is a property of that dataset that §10's report has to disclose, not a
        # property of anything §10 built.
        "generator_checks_failed": failed_checks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="B,D")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--config", required=True,
                    help="preset name or JSON override path; must be the one that "
                         "produced the CSVs being labelled")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "out", "hidden_mid_ext"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
            blob = extract(variant, seed, args.config)
            path = os.path.join(args.out_dir, f"{variant}_{seed}.json")
            with open(path, "w") as fh:
                json.dump(blob, fh)
            pools = {k: len(v) for k, v in blob["base_pools"].items()}
            print(f"v{variant} s{seed}: {len(blob['hp_groups'])} hp_groups, pools {pools}, "
                  f"{len(blob['cs_rewires'])} rewires, {blob['n_shock_events']} shocks, "
                  f"{sum(len(v) for v in blob['coparents'].values()) // 2} coparent edges",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
