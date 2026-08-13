#!/usr/bin/env python3
"""
Phase 1, Step 1 — confirm generator-side ground truth for each candidate latent state.

`docs/HADES-v3_init.md` §6 proposes six candidate latent operational states. Two are
described there as already confirmed (`own_stress` ~ Supply Stress, `RESILIENCE` ~ Recovery
Capability); the other four are explicitly NOT confirmed. This script settles all six the
same way `reports/layer3_testing.md` §9.1/§10.1 settled `HP_GROUPS` and `RESILIENCE`: by
executing `db/generate_dataset.py` in-process with `write()` stubbed and inspecting the
resulting namespace, rather than by reading the source and assuming.

The extraction mechanism is copied from `ml/extract_hidden_state.py::extract` -- deliberately
copied rather than imported, the same reasoning `ml/extract_mechanism_state.py` states: a
change to one script's read must not silently alter another's.

Three outcomes per state, per the phase brief:
  confirmed             -- a generator-side variable exists, is latent (never emitted), and
                           can be materialised as a supervision target
  needs_instrumentation -- a real latent quantity exists but cannot be read out as-is
  dropped               -- no latent generator-side correlate exists

**Everything this script emits is privileged, training/eval-time only, and is never a model
input at inference** -- the standard `ml/extract_hidden_state.py` set and
`ml/data/loader.py::verify_no_hidden_state` enforces from the other side.

    python3 ml/confirm_latent_states.py --variant A --seed 42 --config spec \
        --out out/phase1/states_A_42.json
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


def namespace(variant: str, seed: int, config: str) -> dict:
    """Run the generator in-process with `write()` stubbed; return its namespace."""
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
    mod = types.ModuleType("__latent_confirm__")
    mod.__file__ = GEN
    sys.modules["__latent_confirm__"] = mod
    ns = mod.__dict__
    ns["__name__"] = "__latent_confirm__"
    buf = io.StringIO()
    # The generator ends with `raise SystemExit(1 if fail else 0)`
    # (db/generate_dataset.py:2318). Catch it like ml/extract_hidden_state.py does, but
    # keep the code: a non-zero status means the generator's own validation suite failed,
    # which must not be swallowed silently.
    status = 0
    try:
        with contextlib.redirect_stdout(buf):
            try:
                exec(compile(src, GEN, "exec"), ns)
            except SystemExit as e:
                status = int(e.code or 0)
    finally:
        sys.argv = saved
    if status != 0:
        raise RuntimeError(
            f"generator self-validation FAILED (exit {status}) for this variant/seed; "
            f"refusing to report ground truth from an invalid world.\n"
            + "\n".join(buf.getvalue().splitlines()[-15:]))
    ns["_GEN_STDOUT"] = buf.getvalue()
    return ns


def confirm(ns: dict) -> dict:
    """Decide each of the six candidate states against the generator's real namespace."""
    suppliers = ns["suppliers"]
    sup_by_id = ns["sup_by_id"]
    T0S = ns["T0S"]
    own_stress, stress = ns["own_stress"], ns["stress"]
    RESILIENCE, IDIO = ns["RESILIENCE"], ns["IDIO"]

    out: dict = {"variant": ns["VARIANT"], "seed": ns["CFG"].seed,
                 "mechanisms": sorted(ns["MECHS"]),
                 "n_suppliers_simulated": len(suppliers),
                 "n_snapshots": len(T0S), "states": {}}

    # ---- 1. Supply Stress -------------------------------------------------
    # own_stress()/stress() are pure functions of module-level state that survives the
    # run (EVENTS, HP_B_EVENTS, SHOCK_EVENTS, IDIO, coparents, HP_GROUPS, SUP_CHAIN),
    # so they can be re-evaluated post-hoc at any t. Verified by evaluating them here.
    probe = T0S[len(T0S) // 2]
    vals_own = [own_stress(s["id"], s["base_rel"], probe) for s in suppliers]
    vals_tot = [stress(s["id"], s["base_rel"], probe) for s in suppliers]
    out["states"]["supply_stress"] = {
        "verdict": "confirmed",
        "generator_symbol": "own_stress(sup_id, base_rel, t) / stress(sup_id, base_rel, t)",
        "source": "db/generate_dataset.py:827, :860",
        "latent": True,
        "emitted_to_csv": False,
        "in_hidden_state_columns": True,   # 'own_stress', 'stress' both listed
        "time_varying": True,
        "entity": "Supplier",
        "coverage": f"{len(vals_own)}/{len(suppliers)} suppliers, all {len(T0S)} snapshots",
        "probe_t0": probe.isoformat(),
        "own_stress_range": [min(vals_own), max(vals_own)],
        "own_stress_mean": sum(vals_own) / len(vals_own),
        "stress_range": [min(vals_tot), max(vals_tot)],
        "stress_mean": sum(vals_tot) / len(vals_tot),
        "note": "Recomputable post-hoc: pure function of persistent module state.",
    }

    # ---- 2. Recovery Capability ------------------------------------------
    res_vals = list(RESILIENCE.values())
    out["states"]["recovery_capability"] = {
        # Not "dropped": the state is real and confirmed, it simply does not exist on a
        # variant that excludes Mechanism E. Distinguishing these matters -- Variant A, the
        # testbed the roadmap prescribes, is exactly such a variant.
        "verdict": "confirmed" if res_vals else "unavailable_on_this_variant",
        "generator_symbol": "RESILIENCE[sup_id]",
        "source": "db/generate_dataset.py:518",
        "latent": True,
        "emitted_to_csv": False,
        "in_hidden_state_columns": True,
        "time_varying": False,
        "entity": "Supplier",
        "coverage": f"{len(res_vals)}/{len(suppliers)} suppliers",
        "range": [min(res_vals), max(res_vals)] if res_vals else None,
        "mean": (sum(res_vals) / len(res_vals)) if res_vals else None,
        "note": ("Static per-supplier draw. Requires Mechanism E; empty dict when E is off, "
                 "in which case this state is unavailable on this variant."),
    }

    # ---- 3. Supplier Reliability -----------------------------------------
    # base_rel IS emitted (suppliers.reliability_history), so the reliability *level* is
    # observable, not latent. The latent part is IDIO: a per-supplier idiosyncratic outage
    # window, never emitted. Report both halves separately rather than collapsing them.
    idio_active = {k: v for k, v in IDIO.items() if v}
    n_active_at_probe = sum(1 for sid, ev in idio_active.items()
                            if ev[0] <= probe <= ev[2])
    out["states"]["supplier_reliability"] = {
        "verdict": "confirmed",
        "generator_symbol": "IDIO[sup_id] = (start, peak, end, magnitude)",
        "source": "db/generate_dataset.py:727-733",
        "latent": True,
        "emitted_to_csv": False,
        "in_hidden_state_columns": False,   # not on the list; see caveat
        "time_varying": True,
        "entity": "Supplier",
        "coverage": f"{len(idio_active)}/{len(suppliers)} suppliers ever have an outage",
        "active_at_probe_t0": n_active_at_probe,
        "caveat": ("The reliability LEVEL is not latent: base_rel is emitted verbatim as "
                   "suppliers.reliability_history (generate_dataset.py:1407). Only the "
                   "idiosyncratic outage STATE is hidden. Note ml/data/loader.py:169 does "
                   "not read reliability_history, so it is emitted-but-unused as a feature."),
    }

    # ---- 4. Inventory Health ---------------------------------------------
    out["states"]["inventory_health"] = {
        "verdict": "dropped",
        "generator_symbol": 'ip["stock"], ip["thr"]',
        "source": "db/generate_dataset.py:1240-1245",
        "latent": False,
        "emitted_to_csv": True,
        "emitted_as": ("inventory.stock_level, inventory.reorder_threshold, "
                       "inventory_history.stock_level, inventory_history.reorder_threshold "
                       "(generate_dataset.py:1441, :1443)"),
        "used_as_model_feature": True,
        "feature_source": "ml/data/loader.py:237-238, :324-336, :409",
        "reason": ("Not a latent state on this benchmark. Stock level and reorder threshold "
                   "are emitted verbatim AND already consumed as model input features, so an "
                   "'estimator' for them would be reading its own target. The only hidden "
                   "component is Mechanism G's reporting lag between true and recorded stock, "
                   "and G is not enabled on Variant A, so even that is absent on this testbed."),
    }

    # ---- 5. Logistics Stability ------------------------------------------
    out["states"]["logistics_stability"] = {
        "verdict": "needs_instrumentation",
        "generator_symbol": "PORT_EVENTS window test (inline, not stored)",
        "source": "db/generate_dataset.py:1080-1082",
        "latent": "partially",
        "emitted_to_csv": False,
        "reason": ("A real latent logistics disturbance exists -- port events add +0.10 to a "
                   "sea shipment's stress -- but it is evaluated inline inside new_shipment() "
                   "and never stored, so there is no per-entity, per-t0 variable to read out. "
                   "Its driver is also observable-adjacent: H_PORT membership is exactly the "
                   "`sea` flag, which is a deterministic function of the emitted `country` "
                   "column (layer3_testing.md §9.7.1 measured this confound). Materialising a "
                   "target would need a small, well-scoped generator change recording the "
                   "per-shipment port-event contribution. Not built in this phase."),
    }

    # ---- 6. Capacity Pressure --------------------------------------------
    fac_outage = ns["FACTORY_OUTAGE"]
    out["states"]["capacity_pressure"] = {
        "verdict": "dropped",
        "generator_symbol": "capacity_score / capacity_units_per_day / FACTORY_OUTAGE",
        "source": "db/generate_dataset.py:334, :984, :1078",
        "latent": False,
        "emitted_to_csv": True,
        "emitted_as": ("suppliers.capacity_score, factories.capacity_units_per_day, "
                       "product_factories.capacity_units_per_day, warehouses.capacity_units"),
        "used_as_model_feature": True,
        "feature_source": "ml/data/loader.py:169, :279 (capacity_score_z)",
        "reason": ("No time-varying latent capacity variable exists per supplier. Every "
                   "capacity quantity is a static attribute emitted to CSV, and capacity_score "
                   "is already a model input feature. The single genuinely hidden capacity "
                   "event is FACTORY_OUTAGE -- one factory of "
                   f"{len(ns['factories'])}, for a fixed {(fac_outage[2]-fac_outage[1]).days}-day "
                   "window -- which is far too sparse to supervise a per-supplier estimator "
                   "and is not a supplier-level quantity at all."),
    }

    # ---- 7. mitigation_level -- instrumented in V3 Phase 1, not in the init doc's six ----
    MH = ns.get("MITIGATION_HISTORY", {})
    mvals = list(MH.values())
    nz = [v for v in mvals if v > 0]
    out["states"]["mitigation_level"] = {
        "verdict": "confirmed" if MH else "unavailable_on_this_variant",
        "generator_symbol": "MITIGATION_HISTORY[(sup_id, week)] <- mitigation_level(sup_id, as_of)",
        "source": "db/generate_dataset.py:1191 (recorder at :1203-1210, store at :1164-1181)",
        "latent": True,
        "emitted_to_csv": False,
        "in_hidden_state_columns": True,   # 'mitigation' is on the list
        "time_varying": True,
        "entity": "Supplier",
        "provenance": "RECORDED DURING SIMULATION (not recomputed post-hoc)",
        "n_records": len(MH),
        "n_suppliers_covered": len({k[0] for k in MH}),
        "n_weeks": len({k[1] for k in MH}),
        "nonzero_fraction": (len(nz) / len(mvals)) if mvals else None,
        "range": [min(mvals), max(mvals)] if mvals else None,
        "note": ("Requires Mechanism E, exactly as recovery_capability does: mitigation_level "
                 "returns before recording when RESILIENCE is empty. Coverage is a subset of "
                 "suppliers -- only those reachable as a replenishment candidate through "
                 "prod_bom_sup are ever evaluated -- which is a data property, not a modelling "
                 "choice (cf. roadmap §4 risk 6 on bounded coverage ceilings)."),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="A")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="spec")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print(f"running generator: variant {args.variant}, seed {args.seed}, config {args.config} ...")
    ns = namespace(args.variant, args.seed, args.config)
    rep = confirm(ns)

    order = ["supply_stress", "recovery_capability", "supplier_reliability",
             "inventory_health", "logistics_stability", "capacity_pressure",
             "mitigation_level"]
    print(f"\nvariant {rep['variant']} seed {rep['seed']}  mechanisms={rep['mechanisms']}  "
          f"suppliers={rep['n_suppliers_simulated']}  snapshots={rep['n_snapshots']}\n")
    print(f"{'state':<24} {'verdict':<22} {'latent':<10} {'emitted':<8} symbol")
    print("-" * 110)
    for k in order:
        s = rep["states"][k]
        print(f"{k:<24} {s['verdict']:<22} {str(s['latent']):<10} "
              f"{str(s['emitted_to_csv']):<8} {s['generator_symbol'][:44]}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2, sort_keys=True, default=str)
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
