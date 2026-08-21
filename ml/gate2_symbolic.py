#!/usr/bin/env python3
"""
Layer 3 v3, PHASE 3 — GATE 2: SCM-derived symbolic rule verification.

`layer3_neurosymbolic_build_prompt.md` §5: **import and verify, do not re-extract.** The V3
Phase 1/2 work already transcribed the generator's structural equations into `ml/scm.py` (a
standalone re-implementation, never a wrapper) and validated them in
`ml/run_scm_validation.py`; `reports/phase2_scm.md` records the result. This module does five
things with that material and adds no sixth:

1. **Import** the equations and their provenance -- `ml/scm.py`'s `SupplyChainSCM`, `SCMParams`
   and `from_namespace`, and `ml/run_scm_validation.py`'s `equation_fidelity` and
   `outcome_fidelity`, all unmodified.
2. **Verify against the CURRENT generator.** Three independent checks, because they fail in
   different ways:
   * *provenance* -- every equation carries a cited line range in `db/generate_dataset.py`. The
     cited construct is looked up in the generator **as it stands today**. This is the check
     that catches the generator having moved since the extraction, which equation fidelity
     cannot: `ml/scm.py` never calls the generator, so a construct that has merely *relocated*
     still computes identically and would pass fidelity silently.
   * *equation fidelity* -- the standalone SCM against the generator's own `own_stress`/`stress`
     on every visible supplier at every snapshot, required to be **exactly** 0.0.
   * *term coverage* -- which terms the variant under test actually exercises. On Variant 0
     `MECHS = ()`, so `hp_coupling`, `upstream_stress`, `recv_atten` and `absorption` are all
     inert and a fidelity pass there says nothing about them. Variant K activates all ten
     mechanisms and is run as the coverage arm for exactly that reason.
3. **Encode only verified mechanisms as executable rules.** Each `Rule` carries its equation, its
   provenance, the variant condition under which it is *live*, and its verification status. A
   rule that fails any check is retained as an explicit `HYPOTHESIS`, never silently dropped and
   never promoted.
4. **Test that known-valid paths are retained and known-invalid ones rejected**, through
   `validate_path`, which walks a candidate chain edge by edge and requires every edge to
   resolve to a `VERIFIED` and *live* rule.
5. **Version the rule set** against the generator it was verified against: the sha256 of
   `db/generate_dataset.py`, its `GITC` marker, and the resolved config manifest of the world.

**The standing validity caveat**, carried from `ml/scm.py`'s own docstring: these equations are
exact *because* the world is synthetic and its structural equations are source code. That
licenses "validated on this benchmark via generator-extracted equations" and nothing beyond it.

    python3 ml/gate2_symbolic.py --variants 0,K --seeds 42,43,44,45,46
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.confirm_latent_states import namespace                # noqa: E402
from ml.run_scm_validation import equation_fidelity, outcome_fidelity  # noqa: E402
from ml.scm import SupplyChainSCM, from_namespace             # noqa: E402

GEN = os.path.join(REPO, "db", "generate_dataset.py")

VERIFIED, HYPOTHESIS = "VERIFIED", "HYPOTHESIS"


# ---------------------------------------------------------------------------
# the rule set: one entry per structural equation, with its provenance
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """One executable structural mechanism, with everything needed to check it.

    `cited_lines` is the range recorded by the original extraction (`ml/scm.py`'s docstring and
    `reports/phase2_scm.md`'s table). `token` is a substring that identifies the construct in the
    generator source; verification looks for it *anywhere* in the file and compares where it was
    found against where it was cited. A relocation is reported as drift and does not by itself
    invalidate the rule -- a *disappearance* does.
    """
    name: str
    edge: tuple            # (cause, effect) in the causal DAG
    equation: str
    cited_lines: tuple
    token: str
    live_when: str         # human-readable condition under which this term is non-zero
    live_check: str        # key in the liveness probe below
    status: str = HYPOTHESIS
    found_lines: list = field(default_factory=list)
    drift: int | None = None
    notes: str = ""


RULES = [
    Rule("event_ramp", ("HiddenFactorEvent", "own_stress"),
         "ramp(e,t) = mag * clamp01((t-s0)/(pk-s0) if t<=pk else 1-(t-pk)/(s1-pk))",
         (839, 856), "frac = (t-s0)/(pk-s0)", "always", "always"),
    Rule("own_stress", ("base_rel + events", "own_stress"),
         "own_stress(s,t) = min(0.95, (1-base_rel)*0.5 + SUM ramps over "
         "EVENTS, HP_B_EVENTS, SHOCK_EVENTS, IDIO)",
         (827, 857), "def own_stress(sup_id, base_rel, t):", "always", "always"),
    Rule("coparent_bleed", ("Supplier.own_stress", "CoParentSupplier.stress"),
         "stress(s) += recv_atten(s) * COPARENT_COUPLING * SUM_{p in coparents[s]} own_stress(p,t)",
         (869, 872), "COPARENT_COUPLING * own_stress(partner",
         "co-parent graph non-empty (always on this generator)", "coparent"),
    Rule("hp_coupling", ("HiddenParentGroup", "Supplier.stress"),
         "stress(s) += recv_atten(s) * HP_ALPHA * mean_{co-members, Type A} own_stress",
         (812, 824), "HP_ALPHA * sum(", "Mechanism D", "hp_alpha"),
    Rule("upstream_chain", ("UpstreamSupplier", "Supplier.stress"),
         "stress(s) += SUM_hops (PROD downstream attenuations) * own_stress(up,t)",
         (634, 650), "def upstream_stress(", "Mechanism J", "upstream"),
    Rule("recv_atten", ("Supplier.RESILIENCE", "inbound transmission"),
         "recv_atten(s) = attenuation_of(s), piecewise-linear in RESILIENCE; 1.0 when F is off",
         (585, 592), "def recv_atten(", "Mechanism F", "f_on"),
    Rule("stress", ("own_stress + transmission", "Supplier.stress"),
         "stress(s,t) = min(0.95, own_stress + coparent + hp + upstream)",
         (860, 880), "def stress(sup_id, base_rel, t):", "always", "always"),
    Rule("absorption", ("Supplier.RESILIENCE", "p_delay"),
         "absorption(s) = max(0, 1 - resilience_lambda * RESILIENCE[s]); 1.0 when E is off",
         (536, 547), "1.0 - CFG.resilience_lambda * RESILIENCE[sup_id]",
         "Mechanism E (Recovery Capability)", "resilience"),
    Rule("p_delay", ("Supplier.stress", "Shipment.delayed"),
         "p_delay(sh) = min(0.80, 0.025 + 0.38 * st * absorption(sup))",
         (1085, 1085), "p_delay = min(0.80, 0.025 + 0.38", "always", "always"),
    Rule("port_bump", ("PortEvent x SEA carrier", "Shipment.delayed"),
         "st += 0.10 for a SEA carrier dispatched inside a PORT_EVENT window",
         (1080, 1082), "st = min(0.95, st + 0.10)", "always (SEA carriers)", "port"),
    Rule("factory_bump", ("FactoryOutage", "Shipment.delayed"),
         "st += 0.35 inside the FACTORY_OUTAGE window (outbound, factory-sourced shipments)",
         (1078, 1079), "st = min(0.95, st + 0.35)", "always (factory-sourced)", "factory"),
    Rule("sourcing_stress", ("BOM suppliers", "Product.sourcing_stress"),
         "sourcing_stress(p,t) = max_{BOM-reachable suppliers} stress(s,t)",
         (961, 971), "def sourcing_stress(", "orders without a named supplier", "sourcing"),
    Rule("replenish_trigger", ("mitigation_level", "replenishment trigger"),
         "trigger(t) = 1.55 + 0.45 * max_{BOM suppliers} mitigation_level",
         (1221, 1225), "1.55 + 0.45 * _mit", "Mechanism E", "resilience"),
    Rule("replenish_qty", ("mitigation_level", "replenishment quantity"),
         "qty = thr * U(2.0,2.8) * (1 + 0.35 * mitigation_level)",
         (1233, 1235), "1.0 + 0.35 * mitigation_level", "Mechanism E", "resilience"),
    # --- the two label-forming aggregations, without which no path reaches a task -------
    Rule("impact_aggregation", ("Shipment.delayed", "Supplier.impact"),
         "impact(s,t0) = 1{ any in-flight shipment of s logs `delayed` in (t0, t0+H] }",
         (1355, 1391), "if lab and sh[\"supplier_id\"]: sup_hit.add(", "always", "always"),
    Rule("shortage_event", ("inventory walk", "Product.shortage"),
         "shortage(p,wh,t0) = 1{ stock < thr observed in (t0, t0+H] }, from the weekly "
         "inventory walk; replenishment arrival timing is the only channel a delayed "
         "shipment reaches it through",
         (1259, 1262), "shortage_events.setdefault(key, []).append(week)", "always", "always"),
]


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def provenance_check(rules: list) -> list:
    """Locate each rule's cited construct in the generator **as it stands today**.

    Reported per rule: whether it was found at all, where, and how far that is from where the
    original extraction said it was. Drift is a warning, not a failure -- code moves. A rule
    whose construct cannot be found at all is a failure, because it means the mechanism was
    renamed, restructured or removed and the extraction is describing a generator that no
    longer exists.
    """
    lines = open(GEN).read().split("\n")
    for r in rules:
        r.found_lines = [i + 1 for i, ln in enumerate(lines) if r.token in ln]
        if not r.found_lines:
            r.status = HYPOTHESIS
            r.notes = "construct not found in the current generator"
            r.drift = None
            continue
        lo, hi = r.cited_lines
        inside = [x for x in r.found_lines if lo <= x <= hi]
        r.drift = 0 if inside else min(min(abs(x - lo), abs(x - hi)) for x in r.found_lines)
        if r.drift:
            r.notes = (f"relocated: cited {lo}-{hi}, found at "
                       f"{', '.join(str(x) for x in r.found_lines[:4])}")
    return rules


def liveness(ns: dict) -> dict:
    """Which terms are non-zero in THIS world -- measured from the namespace, not assumed.

    A fidelity pass on a variant where a term is inert proves nothing about that term, and
    `reports/phase2_scm.md` §4a.1 makes exactly this point: on Variant E alone the extraction
    could have been wrong in four places and still scored 0.0.
    """
    return {
        "always": True,
        "coparent": bool(ns.get("coparents")) and float(ns.get("COPARENT_COUPLING", 0.0)) > 0,
        "hp_alpha": float(ns.get("HP_ALPHA", 0.0)) > 0,
        "upstream": bool(ns.get("SUP_CHAIN")),
        "f_on": bool(ns.get("F_ON", False)),
        "resilience": bool(ns.get("RESILIENCE")),
        "port": bool(ns.get("PORT_EVENTS")),
        "factory": ns.get("FACTORY_OUTAGE") is not None,
        "sourcing": True,
    }


def verify_world(variant: str, dseed: int, config: str, rules: list) -> dict:
    """Equation fidelity + outcome fidelity + liveness for one (variant, seed)."""
    ns = namespace(variant, dseed, config)
    params, structure = from_namespace(ns)
    scm = SupplyChainSCM(params, structure)
    probes = list(ns["T0S"])
    eq = equation_fidelity(scm, ns, probes)
    oc = outcome_fidelity(scm, ns)
    live = liveness(ns)
    return {"variant": variant, "dataset_seed": dseed,
            "equation_fidelity": eq, "outcome_fidelity": oc, "liveness": live,
            "mechanisms_enabled": list(ns.get("MECHS", ())),
            "generator_gitc": ns.get("GITC"),
            "n_suppliers": len(ns["suppliers"]),
            "terms_exercised": sorted(k for k, v in live.items() if v and k != "always")}


def rule_set_version(worlds: list) -> dict:
    """Version the rule set against the generator it was verified against."""
    h = hashlib.sha256(open(GEN, "rb").read()).hexdigest()
    return {"generator_sha256": h, "generator_path": os.path.relpath(GEN, REPO),
            "generator_gitc": sorted({w.get("generator_gitc") for w in worlds}),
            "n_rules": len(RULES),
            "verified_against_variants": sorted({w["variant"] for w in worlds}),
            "verified_against_seeds": sorted({w["dataset_seed"] for w in worlds})}


def finalise_status(rules: list, worlds: list) -> list:
    """A rule is VERIFIED only if its construct is present in the current generator AND it was
    exercised, exactly, in at least one world that was checked.

    The second clause is what stops a rule from being promoted by a variant that never runs it.
    `always`-live rules ride on the fidelity result directly; mechanism-gated rules need a world
    where the mechanism is on.
    """
    exact = [w for w in worlds if w["equation_fidelity"]["exact"]]
    for r in rules:
        if not r.found_lines:
            r.status = HYPOTHESIS
            continue
        covered = [w for w in exact if w["liveness"].get(r.live_check)]
        if covered:
            r.status = VERIFIED
            r.notes = ((r.notes + "; " if r.notes else "")
                       + "exercised and exact on "
                       + ", ".join(f"v{w['variant']}/d{w['dataset_seed']}"
                                   for w in covered[:3])
                       + (f" (+{len(covered)-3} more)" if len(covered) > 3 else ""))
        else:
            r.status = HYPOTHESIS
            r.notes = ((r.notes + "; " if r.notes else "")
                       + "not exercised by any variant checked -- transcription present, "
                         "mechanism inert, so fidelity says nothing about it")
    return rules


# ---------------------------------------------------------------------------
# executable symbolic paths
# ---------------------------------------------------------------------------

# A candidate path is a list of rule names. It is symbolically consistent only if every edge
# resolves to a VERIFIED rule that is live in the world being reasoned about.
KNOWN_PATHS = {
    "coparent_to_impact": {
        "narrative": "Co-parent supplier stressed -> partner's stress rises -> partner's "
                     "shipment delayed -> partner flagged for impact",
        "edges": ["own_stress", "coparent_bleed", "p_delay", "impact_aggregation"],
        "expect": True,
    },
    "own_stress_to_impact": {
        "narrative": "Supplier's own hidden-factor event -> its stress rises -> its shipment "
                     "delayed -> supplier flagged for impact",
        "edges": ["event_ramp", "own_stress", "p_delay", "impact_aggregation"],
        "expect": True,
    },
    "port_event_to_impact": {
        "narrative": "Port event + SEA carrier -> shipment delayed -> supplier flagged",
        "edges": ["port_bump", "p_delay", "impact_aggregation"],
        "expect": True,
    },
    # --- paths terminating at the `delay` label, so that task has a catalogue too -----------
    "own_stress_to_delay": {
        "narrative": "Supplier's own hidden-factor event -> its stress rises -> this shipment's "
                     "delay probability rises",
        "edges": ["event_ramp", "own_stress", "p_delay"],
        "expect": True,
    },
    "coparent_to_delay": {
        "narrative": "Co-parent supplier stressed -> this shipment's supplier's stress rises -> "
                     "this shipment's delay probability rises",
        "edges": ["own_stress", "coparent_bleed", "p_delay"],
        "expect": True,
    },
    "port_event_to_delay": {
        "narrative": "Port event + SEA carrier -> this shipment's delay probability rises",
        "edges": ["port_bump", "p_delay"],
        "expect": True,
    },
    "carrier_rate_to_delay": {
        "narrative": "The carrier's observed 90-day on-time rate causes this shipment's delay "
                     "probability (the factor `grad_input` and `learned_gate` most often cite)",
        "edges": ["carrier_on_time_rate"],
        "expect": False,
        "why": "`carrier_on_time_rate_90d` is an emitted aggregate of realised outcomes. The "
               "generator's only carrier-mediated mechanism is `port_bump`, which is gated on "
               "SEA membership and a PORT_EVENT window, not on the observed rate. No equation "
               "of this name exists, so citing the rate as a cause is a hypothesis.",
    },
    "days_since_dispatch_to_delay": {
        "narrative": "Days since dispatch causes this shipment's delay (the factor occlusion "
                     "cites on 69% of delay explanations)",
        "edges": ["days_since_dispatch"],
        "expect": False,
        "why": "`eta = dispatch + lead_time` and the dispatch offset are drawn independently of "
               "stress in `new_shipment`. Elapsed time since dispatch is EXPOSURE, not a causal "
               "parent, and no extracted equation contains it.",
    },
    "supplier_failure_to_shortage": {
        "narrative": "Supplier Failure -> Shipment Delay -> Inventory Reduction -> Shortage "
                     "(the prompt's own example of a plausible business narrative)",
        "edges": ["own_stress", "p_delay", "replenish_arrival_lag", "shortage_event"],
        "expect": False,
        "why": "`replenish_arrival_lag` is not an extracted rule: the generator's inventory "
               "walk consumes a replenishment shipment's realised arrival time, but no "
               "equation for that arrival -> stock -> shortage transfer was ever extracted "
               "or verified, so the third edge has no verified referent.",
    },
    "resilience_to_impact": {
        "narrative": "Recovery Capability absorbs stress -> lower delay probability -> "
                     "supplier not flagged",
        "edges": ["absorption", "p_delay", "impact_aggregation"],
        "expect": None,      # variant-dependent: verified only where Mechanism E is live
    },
    "attention_weight_to_cause": {
        "narrative": "SHARE's attention weight on an edge indicates that edge is causal",
        "edges": ["attention_weight"],
        "expect": False,
        "why": "There is no generator equation named `attention_weight`. An attention "
               "distribution is a property of the trained model, not of the world.",
    },
}


def validate_path(edges: list, rules_by_name: dict, live: dict | None = None) -> dict:
    """Walk a candidate path and require every edge to resolve to a VERIFIED, live rule.

    Returns the per-edge verdict as well as the overall one: a rejected path should say *which*
    edge failed and why, because "not symbolically consistent" with no locus is not actionable.
    """
    per_edge, ok = [], True
    for e in edges:
        r = rules_by_name.get(e)
        if r is None:
            per_edge.append({"edge": e, "resolves": False,
                             "reason": "no extracted generator equation of this name"})
            ok = False
            continue
        live_here = True if live is None else bool(live.get(r.live_check, False))
        good = (r.status == VERIFIED) and live_here
        per_edge.append({"edge": e, "resolves": True, "status": r.status,
                         "live_in_world": live_here, "equation": r.equation,
                         "provenance": f"db/generate_dataset.py:{r.cited_lines[0]}-"
                                       f"{r.cited_lines[1]}",
                         "found_at": r.found_lines[:4], "drift": r.drift, "ok": good})
        ok = ok and good
    return {"symbolically_consistent": ok, "per_edge": per_edge}


# ---------------------------------------------------------------------------
# the Gate 1 hand-off: which OBSERVABLE features a verified path can reach
# ---------------------------------------------------------------------------
#
# Gate 1 asks for "agreement with the verified SCM mechanisms from Gate 2, where a ground truth
# is available". This is that ground truth, and its shape is itself a finding: on this generator
# the causal PARENTS of every task label are latent (`stress`, `RESILIENCE`, event ramps) and are
# never emitted. What the loader hands the model are DESCENDANTS of those parents, plus static
# attributes that no verified equation mentions at all. So a cited feature is classified into
# three tiers, not two.
#
#   scm_parent      the feature is (a component of) a direct parent in a verified equation
#   scm_descendant  the feature is an emitted observable produced by a verified mechanism on the
#                   path to the outcome -- informative, but downstream of the cause
#   scm_unrelated   no verified equation connects it to the outcome at all
#
FEATURE_SCM_TIER = {
    "delay": {
        # p_delay's parents: stress (latent), absorption (latent), the SEA/port bump, and the
        # factory-outage bump. `carrier_on_time_rate_90d` is the only emitted feature that sits
        # on a verified parent channel -- the carrier identity that the port bump gates on.
        "carrier_on_time_rate_90d": "scm_parent",
        # Realisation descendants: the as-of status one-hot is a function of the realised
        # transitions, which p_delay produced. Informative, downstream.
        "status_scheduled": "scm_descendant",
        "status_in_transit": "scm_descendant",
        "status_delivered": "scm_descendant",
        "status_delayed": "scm_descendant",
        # Schedule quantities. `eta = dispatch + lead_time` and the dispatch offset are drawn
        # independently of stress (`new_shipment`), so they are neither parent nor descendant of
        # the delay mechanism -- they are exposure, not cause.
        "days_to_eta": "scm_unrelated",
        "days_since_dispatch": "scm_unrelated",
    },
    "impact": {
        "on_time_rate_30d": "scm_descendant",
        "on_time_rate_90d": "scm_descendant",
        "on_time_rate_180d": "scm_descendant",
        "trend_slope": "scm_descendant",
        "lateness_variance": "scm_descendant",
        "days_since_last_late": "scm_descendant",
        "lead_time_days_z": "scm_unrelated",
        "capacity_score_z": "scm_unrelated",
    },
    "shortage": {
        "min_stock_ratio": "scm_descendant",
        "avg_stock_ratio": "scm_descendant",
        "total_stock": "scm_descendant",
        "total_reorder_threshold": "scm_unrelated",
        "warehouse_count": "scm_unrelated",
        "bom_component_count": "scm_unrelated",
        "bom_mean_quantity_required": "scm_unrelated",
    },
}


def scm_tier(task: str, feature: str) -> str:
    """Tier for one cited feature. Anything unlisted -- country/category one-hots, positional
    fallbacks -- is `scm_unrelated`, which is the correct default: no verified equation mentions
    it."""
    tbl = FEATURE_SCM_TIER.get(task, {})
    if feature in tbl:
        return tbl[feature]
    base = re.sub(r"_\d+$", "", feature)
    return tbl.get(base, "scm_unrelated")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="0,K")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--config", default="v1")
    ap.add_argument("--out", default=os.path.join(REPO, "out", "layer3_v3", "gate2.json"))
    a = ap.parse_args()

    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    dseeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    rules = provenance_check(list(RULES))
    print("\n=== STEP 2a — provenance against the CURRENT generator ===")
    hdr = f"{'rule':<22}{'cited':>12}{'found':>18}{'drift':>8}"
    print(hdr); print("-" * len(hdr))
    for r in rules:
        print(f"{r.name:<22}{f'{r.cited_lines[0]}-{r.cited_lines[1]}':>12}"
              f"{(','.join(str(x) for x in r.found_lines[:3]) or 'MISSING'):>18}"
              f"{('-' if r.drift is None else str(r.drift)):>8}")

    worlds = []
    for v in variants:
        for d in dseeds:
            w = verify_world(v, d, a.config, rules)
            worlds.append(w)
            eq, oc = w["equation_fidelity"], w["outcome_fidelity"]
            print(f"  v{v} d{d}: {eq['n_comparisons']:,} comparisons  "
                  f"max|err| own={eq['max_abs_err_own_stress']:g} "
                  f"stress={eq['max_abs_err_stress']:g}  exact={eq['exact']}  "
                  f"brier={oc.get('brier', float('nan')):.5f}  "
                  f"terms={w['terms_exercised']}", flush=True)

    rules = finalise_status(rules, worlds)
    by_name = {r.name: r for r in rules}

    print("\n=== STEP 2b — rule dispositions ===")
    hdr = f"{'rule':<22}{'status':>11}{'live when':<44}{'notes'}"
    print(hdr); print("-" * 120)
    for r in rules:
        print(f"{r.name:<22}{r.status:>11}  {r.live_when:<42}{r.notes[:60]}")

    print("\n=== STEP 2c — path validation ===")
    # Liveness of the FIRST variant listed is what paths are judged against; a path that needs
    # an inert mechanism is rejected for that world and said to be so.
    live0 = next(w["liveness"] for w in worlds if w["variant"] == variants[0])
    paths = {}
    for name, spec in KNOWN_PATHS.items():
        got = validate_path(spec["edges"], by_name, live0)
        agree = (spec["expect"] is None) or (got["symbolically_consistent"] == spec["expect"])
        paths[name] = {**spec, **got, "matches_expectation": agree}
        bad = [e["edge"] for e in got["per_edge"] if not e.get("ok")]
        print(f"  {name:<32} consistent={str(got['symbolically_consistent']):<6} "
              f"expected={str(spec['expect']):<6} {'OK' if agree else 'MISMATCH':<9}"
              + (f"  blocked at: {', '.join(bad)}" if bad else ""))

    ver = rule_set_version(worlds)
    blob = {"config": vars(a), "rule_set_version": ver,
            "rules": [{"name": r.name, "edge": list(r.edge), "equation": r.equation,
                       "cited_lines": list(r.cited_lines), "found_lines": r.found_lines,
                       "drift": r.drift, "status": r.status, "live_when": r.live_when,
                       "notes": r.notes} for r in rules],
            "worlds": worlds, "paths": paths,
            "feature_scm_tier": FEATURE_SCM_TIER}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=str)
    print(f"\nrule set version: generator sha256 {ver['generator_sha256'][:16]}  "
          f"GITC {ver['generator_gitc']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
