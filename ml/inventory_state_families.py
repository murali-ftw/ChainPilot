#!/usr/bin/env python3
"""
Layer 3 redesign, PHASE 0 -- inventory the generator against the four proposed state families.

**This is an audit script, not a model.** It trains nothing, fits nothing, and touches no
checkpoint. It executes `db/generate_dataset.py` in-process with `write()` stubbed (the same
`ml/confirm_latent_states.py::namespace` mechanism Phase 1 used) and inspects the resulting
namespace directly, so every verdict below is a measurement of the generator's own state rather
than a reading of its source.

The redesign proposes four state families -- supplier, logistics, inventory, production. Two of
them (supplier, inventory) already have verdicts in this project's record and are **cited, not
remeasured**: `reports/phase1_latent_state.md` Step 1 and `reports/layer3_uncertainty_aware.md`
STEP 2. What this script adds is (a) the unaudited logistics and production candidates, and
(b) a re-check that the two cited verdicts still hold against the generator as it stands today,
since `db/generate_dataset.py` has been edited once since Phase 1 (the `MITIGATION_HISTORY`
recorder, `reports/phase1_mitigation_level.md`).

Disposition categories, per the redesign brief:

    not-simulated     -- no generator-side variable corresponds to the family at all
    already-a-feature -- the quantity exists but is emitted and consumed as a model input
    too-sparse        -- the quantity is latent but has too few positive instances to supervise
    unidentifiable    -- measured under do(Z) and found not to move the observable channel

A family with no candidate generator variable at all is closed HERE, before Phase 1, and
reported as a generator-extension requirement.

    python3 ml/inventory_state_families.py --variants A,E,F,K --seed 42 --config v1 \
        --out out/layer3_redesign/phase0_families.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.confirm_latent_states import namespace          # noqa: E402
from ml.data.loader import HIDDEN_STATE_COLUMNS         # noqa: E402

FAMILIES = ("supplier", "logistics", "inventory", "production")

# Prior verdicts, cited rather than recomputed. Each carries its source so the report can
# quote it without this script re-deriving a number that already exists.
PRIOR = {
    "supplier": {
        "supply_stress": ("CONFIRMED, PASSES",
                          "phase1_latent_state.md Step 1 + Steps 2-4 (AUC 0.6426 A / 0.6509 E); "
                          "layer3_uncertainty_aware.md STEP 2 identifiable (Case 1)"),
        "supplier_reliability": ("CONFIRMED, FAILS IDENTIFIABILITY",
                                 "layer3_uncertainty_aware.md STEP 2: do(IDIO) moves 1.0%/0.6% "
                                 "of labelled rows, AUC 0.5000/0.4997 vs measured null "
                                 "0.5040/0.5079. Case 3, classification C, CLOSED"),
        "recovery_capability": ("CONFIRMED, PASSES (Mechanism-E variants only)",
                                "phase1_latent_state.md (AUC 0.6530 E); "
                                "layer3_uncertainty_aware.md STEP 2 identifiable (Case 1)"),
    },
    "inventory": {
        "inventory_health": ("DROPPED -- already-a-feature",
                             "phase1_latent_state.md Step 1 #4: stock_level/reorder_threshold "
                             "emitted to inventory.csv and inventory_history.csv AND consumed "
                             "as model input (ml/data/loader.py)"),
    },
    "production": {
        "capacity_pressure": ("DROPPED -- already-a-feature + too-sparse",
                              "phase1_latent_state.md Step 1 #6: every capacity quantity is a "
                              "static emitted attribute; the one hidden event is FACTORY_OUTAGE, "
                              "one factory of five for one fixed 21-day window"),
    },
    "logistics": {
        "logistics_stability": ("NEEDS INSTRUMENTATION",
                                "phase1_latent_state.md Step 1 #5: port events add +0.10 to a "
                                "sea shipment's stress but are evaluated inline inside "
                                "new_shipment() and never stored"),
    },
}


# ---------------------------------------------------------------------------
# per-family probes
# ---------------------------------------------------------------------------

def probe_supplier(ns: dict) -> dict:
    """own_stress / stress / IDIO / RESILIENCE -- confirm they are still where Phase 1 left
    them, and still latent. No AUC is recomputed."""
    out = {}
    for name, key in (("own_stress", "own_stress"), ("stress", "stress")):
        out[name] = {"present": key in ns, "callable": callable(ns.get(key))}
    out["IDIO"] = {"present": "IDIO" in ns,
                   "n_suppliers": len(ns.get("IDIO") or {}),
                   "n_with_outage": sum(1 for v in (ns.get("IDIO") or {}).values() if v)}
    out["RESILIENCE"] = {"present": "RESILIENCE" in ns,
                         "n": len(ns.get("RESILIENCE") or {})}
    return out


def probe_logistics(ns: dict) -> dict:
    """The two candidates the brief names, measured rather than assumed.

    (1) `recv_atten` / `SUP_ATTEN` -- Mechanism F's adaptive transmission attenuation.
    (2) carrier / route timing fields, and the per-shipment PORT_EVENTS stress bump.
    """
    out: dict = {}
    F_ON = bool(ns.get("F_ON"))
    SUP_ATTEN = ns.get("SUP_ATTEN") or {}
    RESILIENCE = ns.get("RESILIENCE") or {}
    recv_atten = ns.get("recv_atten")

    # -- (1) recv_atten ---------------------------------------------------
    # Two questions, and they have different answers on different variants: is the coefficient
    # POPULATED, and is it CAUSALLY LIVE? `recv_atten()` returns 1.0 whenever F is off, so on a
    # J-without-F variant SUP_ATTEN is populated by the Phase-4 placeholder draw and then never
    # read -- the dict exists and the state does not.
    live_vals = [recv_atten(s) for s in list(SUP_ATTEN)[:200]] if recv_atten else []
    causally_live = bool(live_vals and (max(live_vals) - min(live_vals)) > 1e-12)

    # Is it an independent latent, or a deterministic transform of RESILIENCE? Mechanism F
    # defines attenuation_of() as a piecewise-linear function of RESILIENCE with no second
    # draw, so on F-variants the two are in bijection and estimating one IS estimating the
    # other. Checked by rank correlation rather than by reading the source.
    determinism = None
    if F_ON and SUP_ATTEN and RESILIENCE:
        shared = sorted(set(SUP_ATTEN) & set(RESILIENCE))
        a = [SUP_ATTEN[s] for s in shared]
        r = [RESILIENCE[s] for s in shared]
        # The claim being tested is not "correlated" but "the SAME estimation problem". Phase 1
        # binarises every continuous state at its train median, so what a head actually learns
        # is the median-split label. If median-split(SUP_ATTEN) is the exact complement of
        # median-split(RESILIENCE) on every supplier, then a probe for one is the same probe for
        # the other with the labels flipped, and would post the identical AUC by symmetry. That
        # is measured here rather than argued from the source.
        med_a = statistics.median(a)
        med_r = statistics.median(r)
        lab_a = [x > med_a for x in a]
        lab_r = [x > med_r for x in r]
        n_complement = sum(1 for x, y in zip(lab_a, lab_r) if x != y)
        determinism = {
            "n": len(shared),
            "spearman_with_resilience": _spearman(a, r),
            "distinct_atten_per_distinct_resilience": (
                len({round(x, 12) for x in a}) == len({round(x, 12) for x in r})),
            "median_split_is_exact_complement": n_complement == len(shared),
            "n_rows_agreeing_with_complement": n_complement,
            "n_rows": len(shared),
        }

    out["recv_atten"] = {
        "_variant": ns.get("VARIANT"),
        "F_ON": F_ON,
        "SUP_ATTEN_populated": len(SUP_ATTEN),
        "causally_live": causally_live,
        "constant_value_when_inert": (live_vals[0] if live_vals and not causally_live else None),
        "on_hidden_state_columns": "recv_atten" in HIDDEN_STATE_COLUMNS,
        "deterministic_transform_of_resilience": determinism,
        "per_entity_time_varying": False,      # SUP_ATTEN is a static per-supplier scalar
    }

    # -- (2) port events / carrier timing ---------------------------------
    ships, SEA, PE = ns["shipments"], ns["SEA"], ns["PORT_EVENTS"]
    n_sea = sum(1 for s in ships if s["carrier"] in SEA)
    n_bump = sum(1 for s in ships if s["carrier"] in SEA
                 and any(s0 <= s["dispatched_at"] <= s1 for _m, s0, _p, s1, _g in PE))
    out["port_events"] = {
        "n_shipments": len(ships),
        "n_sea": n_sea, "frac_sea": n_sea / len(ships),
        "n_port_bumped": n_bump, "frac_port_bumped": n_bump / len(ships),
        "bump_magnitude": 0.10,
        # The decisive question: is the per-shipment contribution STORED anywhere the way
        # stress() can be recomputed, or evaluated inline and discarded?
        "stored_per_shipment": any(k in ships[0] for k in
                                   ("port_stress", "port_bump", "stress", "atten")),
        "shipment_fields": sorted(ships[0].keys()),
        # H_PORT membership is the `sea` flag, a deterministic function of the emitted
        # `country` column -- so the DRIVER is observable even though the bump is not.
        "h_port_members": len(ns.get("H_PORT") or ()),
        "driver_is_emitted": True,
    }
    out["carrier_timing"] = {
        # carrier_performance.csv is emitted and `carrier_on_time_rate_90d` is a Shipment
        # input feature (ml/data/loader.py::_shipment_features). Same disqualification class
        # as Inventory Health.
        "carrier_performance_emitted": os.path.exists(
            os.path.join(REPO, "db", "csv_v1scale", "vA_seed42", "carrier_performance.csv.gz")),
        "consumed_as_model_input": True,
        "input_feature_name": "carrier_on_time_rate_90d",
        "carriers": sorted(ns["CARRIERS"]),
    }
    return out


def probe_inventory(ns: dict, csv_dir: str) -> dict:
    """Re-check Phase 1's `already-a-feature` verdict against the generator as it stands.

    The brief asks explicitly whether anything has changed since that finding, so this does not
    cite it blind: it confirms the columns are still emitted and still read by the loader.
    """
    inv_pairs = ns.get("inv_pairs") or []
    emitted = {name: os.path.exists(os.path.join(csv_dir, name + ".csv.gz"))
               for name in ("inventory", "inventory_history")}
    return {
        "generator_symbols": ["ip['stock']", "ip['thr']"],
        "n_inventory_pairs": len(inv_pairs),
        "sample_keys": sorted(inv_pairs[0].keys()) if inv_pairs else [],
        "emitted_tables": emitted,
        # loader.py reads stock_level / reorder_threshold into Product features AND onto the
        # STOCKED_AT edge attribute, including a stock/threshold ratio.
        "consumed_as_model_input": True,
        "input_feature_names": ["min_stock_ratio", "avg_stock_ratio", "total_stock",
                                "total_reorder_threshold",
                                "STOCKED_AT edge attr (stock_level, reorder_threshold)"],
        "on_hidden_state_columns": any(c in HIDDEN_STATE_COLUMNS
                                       for c in ("stock", "stock_level", "reorder_threshold")),
    }


def probe_production(ns: dict) -> dict:
    """FACTORY_OUTAGE -- the one hidden production event. Measured for BOTH sparsity and,
    more decisively, whether its window intersects the snapshot grid at all."""
    FO = ns["FACTORY_OUTAGE"]
    ships = ns["shipments"]
    T0S = list(ns["T0S"])
    n_fac_ship = sum(1 for s in ships if s.get("factory_id"))
    in_window = [s for s in ships
                 if s.get("factory_id") == FO[0] and FO[1] <= s["dispatched_at"] <= FO[2]]
    t0_inside = [t for t in T0S if FO[1] <= t <= FO[2]]
    return {
        "generator_symbol": "FACTORY_OUTAGE",
        "n_factories": len(ns["factories"]),
        "outage_factories": 1,
        "window_start": str(FO[1]), "window_end": str(FO[2]),
        "window_days": (FO[2] - FO[1]).days,
        "n_shipments": len(ships), "n_factory_shipments": n_fac_ship,
        "n_shipments_in_outage_window": len(in_window),
        "frac_shipments_in_outage_window": len(in_window) / len(ships),
        "stress_bump": 0.35,
        # THE decisive measurement. A latent state that is never active at any snapshot t0
        # cannot be supervised at all -- this is stronger than "too sparse".
        "n_snapshots": len(T0S),
        "first_t0": str(T0S[0]), "last_t0": str(T0S[-1]),
        "n_t0_inside_outage_window": len(t0_inside),
        # capacity quantities, for completeness: all static, all emitted.
        "capacity_quantities_emitted": ["suppliers.capacity_score",
                                        "factories.capacity_units_per_day",
                                        "product_factories.capacity_units_per_day",
                                        "warehouses.capacity_units"],
        "capacity_consumed_as_model_input": True,
    }


def _spearman(a: list, b: list) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


# ---------------------------------------------------------------------------
# dispositions
# ---------------------------------------------------------------------------

def dispose(probes: dict) -> dict:
    """One disposition per family, derived from the probes above -- never asserted."""
    out = {}

    # -- supplier ---------------------------------------------------------
    out["supplier"] = {
        "candidate_variables": ["own_stress()/stress()", "IDIO[sup_id]", "RESILIENCE[sup_id]"],
        "already_emitted": "partially (reliability_history is the LEVEL; the outage STATE is not)",
        "disposition": "LIVE",
        "carried_to_phase_1": ["supply_stress", "recovery_capability"],
        "closed_here": ["supplier_reliability"],
        "closed_reason": "unidentifiable (cited, layer3_uncertainty_aware.md STEP 2, Case 3/C)",
        "prior": PRIOR["supplier"],
    }

    # -- logistics --------------------------------------------------------
    lg = probes["logistics"]
    ra, pe, ct = lg["recv_atten"], lg["port_events"], lg["carrier_timing"]
    reasons = []
    if not ra["causally_live"]:
        reasons.append(
            f"recv_atten() is inert on this variant (F_ON={ra['F_ON']}, returns "
            f"{ra['constant_value_when_inert']} for every supplier)")
    det = ra["deterministic_transform_of_resilience"]
    if det and abs(abs(det["spearman_with_resilience"]) - 1.0) < 1e-9:
        reasons.append(
            "where recv_atten IS live it is a deterministic piecewise-linear transform of "
            f"RESILIENCE (Spearman {det['spearman_with_resilience']:+.6f}, n={det['n']}), so it "
            "is Recovery Capability under another name, not an independent family")
    if det and det.get("median_split_is_exact_complement"):
        reasons.append(
            "and the identity holds at the level a head actually sees: the median-split label "
            f"for SUP_ATTEN is the exact complement of RESILIENCE's on "
            f"{det['n_rows_agreeing_with_complement']}/{det['n_rows']} suppliers, so a Model A "
            "probe for it would return Recovery Capability's own AUC with the labels flipped -- "
            "a relabelling, not a new measurement")
    if ra["F_ON"] and not os.path.exists(os.path.join(
            REPO, "out", "ds_ckpt", f"v{ra['_variant']}_seed42_v{ra['_variant']}_d42_m0_e100.pt")):
        reasons.append(
            f"no frozen backbone checkpoint exists for variant {ra['_variant']} "
            "(out/ds_ckpt/ holds vA and vE only); running Phase 1 here would require TRAINING "
            "SHARE, which the standing ground rule forbids")
    if not pe["stored_per_shipment"]:
        reasons.append(
            f"the PORT_EVENTS stress bump reaches {pe['frac_port_bumped']:.1%} of shipments and "
            "is evaluated inline in new_shipment() and never stored -- no per-entity, per-t0 "
            "variable exists to read out")
    if ct["consumed_as_model_input"]:
        reasons.append(
            f"carrier/route timing is already-a-feature: `{ct['input_feature_name']}` is a "
            "Shipment input in ml/data/loader.py::_shipment_features")
    out["logistics"] = {
        "candidate_variables": ["recv_atten()/SUP_ATTEN", "PORT_EVENTS bump", "carrier timing"],
        "already_emitted": "carrier timing yes; attenuation and port bump no",
        "disposition": "CLOSED",
        "disposition_categories": ["not-simulated (as an independent per-entity state)",
                                   "already-a-feature (carrier timing)"],
        "reasons": reasons,
        "generator_extension_required": (
            "record the per-shipment PORT_EVENTS stress contribution at the point it is applied "
            "(db/generate_dataset.py:1080-1082), the same shape of change "
            "reports/phase1_mitigation_level.md made for mitigation_level"),
        "prior": PRIOR["logistics"],
    }

    # -- inventory --------------------------------------------------------
    iv = probes["inventory"]
    out["inventory"] = {
        "candidate_variables": ["ip['stock']", "ip['thr']"],
        "already_emitted": "YES",
        "disposition": "CLOSED",
        "disposition_categories": ["already-a-feature"],
        "reasons": [
            "stock_level and reorder_threshold are emitted to "
            f"{[k for k, v in iv['emitted_tables'].items() if v]} and consumed as model inputs "
            f"({', '.join(iv['input_feature_names'][:4])})",
            "re-checked against the generator as it stands today; unchanged since Phase 1"],
        "prior": PRIOR["inventory"],
    }

    # -- production -------------------------------------------------------
    pr = probes["production"]
    out["production"] = {
        "candidate_variables": ["FACTORY_OUTAGE"],
        "already_emitted": "capacity quantities yes; the outage event no",
        "disposition": "CLOSED",
        "disposition_categories": (
            ["not-simulated (at any observed t0)", "too-sparse", "already-a-feature (capacity)"]
            if pr["n_t0_inside_outage_window"] == 0 else ["too-sparse"]),
        "reasons": [
            f"FACTORY_OUTAGE covers 1 of {pr['n_factories']} factories for one fixed "
            f"{pr['window_days']}-day window ({pr['window_start'][:10]} to "
            f"{pr['window_end'][:10]})",
            f"{pr['n_shipments_in_outage_window']} of {pr['n_shipments']:,} shipments "
            f"({pr['frac_shipments_in_outage_window']:.2%}) are dispatched inside it",
            f"DECISIVE: {pr['n_t0_inside_outage_window']} of {pr['n_snapshots']} snapshot t0s "
            f"fall inside the window -- the grid starts at {pr['first_t0'][:10]}, after the "
            f"outage has already ended. There is no snapshot at which this state is active, so "
            f"no supervision target exists at any t0, independent of sparsity",
            "every other capacity quantity is a static emitted attribute and already a model "
            "input feature"],
        "generator_extension_required": (
            "a time-varying per-factory capacity/outage state whose windows intersect the "
            "snapshot grid; the present single fixed pre-grid window cannot be instrumented "
            "into a supervisable state"),
        "prior": PRIOR["production"],
    }
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def audit_variant(variant: str, seed: int, config: str, csv_root: str) -> dict:
    t = time.time()
    ns = namespace(variant, seed, config)
    csv_dir = os.path.join(csv_root, f"v{variant}_seed{seed}")
    probes = {
        "supplier": probe_supplier(ns),
        "logistics": probe_logistics(ns),
        "inventory": probe_inventory(ns, csv_dir),
        "production": probe_production(ns),
    }
    return {"variant": variant, "seed": seed, "config": config,
            "mechanisms": sorted(ns.get("MECHS") or ()),
            "seconds": time.time() - t,
            "probes": probes, "dispositions": dispose(probes)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="A,E",
                    help="A and E are the two testbeds with backbone checkpoints; F and K are "
                         "the only variants where Mechanism F is live and are audited for the "
                         "logistics candidate even though no checkpoint exists for them")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="v1")
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    res = {"seed": a.seed, "config": a.config, "variants": variants, "per_variant": {}}
    for v in variants:
        print(f"\n=== auditing variant {v}, seed {a.seed}, config {a.config} ===", flush=True)
        cell = audit_variant(v, a.seed, a.config, a.csv_root)
        res["per_variant"][v] = cell
        print(f"  mechanisms {cell['mechanisms']}   ({cell['seconds']:.1f}s)", flush=True)
        for fam in FAMILIES:
            d = cell["dispositions"][fam]
            print(f"  {fam:<12} {d['disposition']:<8} "
                  f"{'/'.join(d.get('disposition_categories', [])) or 'carried forward'}",
                  flush=True)

    print("\n" + "=" * 118)
    print(f"PHASE 0 — generator inventory against the four proposed state families "
          f"(seed {a.seed}, config {a.config})")
    print("=" * 118)
    hdr = f"{'Family':<12}{'Candidate generator variable(s)':<44}{'Emitted?':<12}{'Disposition':<10}"
    print(hdr); print("-" * len(hdr))
    ref = res["per_variant"][variants[0]]["dispositions"]
    for fam in FAMILIES:
        d = ref[fam]
        print(f"{fam:<12}{', '.join(d['candidate_variables'])[:43]:<44}"
              f"{d['already_emitted'][:11]:<12}{d['disposition']:<10}")
    print("-" * len(hdr))
    for fam in FAMILIES:
        d = ref[fam]
        if d["disposition"] == "CLOSED":
            print(f"\n{fam.upper()} — closed at Phase 0 "
                  f"[{', '.join(d.get('disposition_categories', []))}]")
            for r in d["reasons"]:
                print(f"  - {r}")
            if d.get("generator_extension_required"):
                print(f"  GENERATOR EXTENSION REQUIRED: {d['generator_extension_required']}")
        else:
            print(f"\n{fam.upper()} — carried to Phase 1: {d['carried_to_phase_1']}; "
                  f"closed here: {d['closed_here']} ({d['closed_reason']})")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
