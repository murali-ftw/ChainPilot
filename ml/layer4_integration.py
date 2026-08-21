#!/usr/bin/env python3
"""
Layer 3 v3, PHASE 5 — Layer 4 integration, claim-gated.

`layer3_neurosymbolic_build_prompt.md` §7. **Wire in only what passed.** Six fields, kept
strictly separate:

    1  prediction probability   always, from SHARE + the Markov readout
    2  evidence status          always, a categorical summary of WHICH GATES PASSED
    3  entity relevance         only if Gate 1 passed
    4  symbolic consistency     only if Gate 2 passed
    5  causal effect            only if Gate 3 passed
    6  confidence / uncertainty  always, and never substituted for evidence status

**The two mandatory distinctions are enforced in code, not in prose.**

    gate value  !=  causal effect
    probability !=  evidence status  !=  confidence

`Field` carries a `kind` and a `derived_from` provenance string, and `assert_distinctions`
refuses to emit a record in which (a) a relevance/gate score has been written into field 5,
(b) a confidence number has been derived from the probability alone, or (c) the evidence status
is a function of the probability or the confidence rather than of the gate outcomes. Each of
those three is a specific way this collapse happens in practice, and each is checked separately.

Fields 3, 4 and 5 carry the **Unsupported** text the prompt specifies when their gate did not
pass, rather than being omitted -- a missing field reads as "not applicable here", and the
finding is "not established anywhere".

**Confidence is built from the axis the variance actually lives on.** Model-init ensembling alone
would measure the smaller term: `reports/decision_support_build.md` §2.2 measured dataset-seed
variance at **98.7% / 67.9% / 70.8%** of total AUC variance on delay / shortage / impact. Field 6
therefore reports the per-entity model-seed spread *and* the population-level cross-world term,
labelled as the different objects they are, plus the cross-world isotonic recalibration
(`ml/hypothesis_ranker.py`'s PAVA, unmodified) which §2.3 measured as the component actually
doing the work (ECE 9x-17x better; model-init ensembling did not clear its own floor).

    python3 ml/layer4_integration.py --task impact --dseed 42 --top 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.ds_backbone import get_backbone, predict                       # noqa: E402
from ml.evaluate import collect_predictions                            # noqa: E402
from ml.hypothesis_ranker import isotonic_apply, isotonic_fit          # noqa: E402
from ml.models.depth import TASK_ENTITY_TYPE                           # noqa: E402

OUT_DIR = os.path.join(REPO, "out", "layer3_v3")

UNSUPPORTED = {
    "relevance": ("Entity-level explanation: Unsupported\n"
                  "Reason: relevance signal did not meet coverage, faithfulness, or "
                  "reproducibility criteria."),
    "symbolic": ("Symbolic mechanism: Unsupported\n"
                 "Reason: proposed pathway is not verified against the generator SCM."),
    "causal": ("Causal validation: Unsupported\n"
               "Causal-effect and root-cause claims withheld pending validated "
               "intervention-effect estimation."),
}


@dataclass
class Field:
    """One output field, with the provenance that lets the distinctions be checked.

    `kind` is the semantic type -- `probability`, `evidence_status`, `relevance`, `symbolic`,
    `causal_effect`, `confidence`. `derived_from` names the quantities the value was computed
    from. Both exist so `assert_distinctions` can be a real check rather than a comment.
    """
    kind: str
    value: object
    derived_from: str
    supported: bool = True
    note: str = ""


def assert_distinctions(rec: dict) -> None:
    """Refuse to emit a record that has collapsed two of the six fields into one.

    Three checks, one per way this actually goes wrong:

    1. A **gate value is not a causal effect.** Field 5 may only be derived from an estimator
       that passed Gate 3. A relevance score, an attention weight or a learned gate written into
       field 5 is the single most common form of this error and is rejected by provenance.
    2. **Confidence is not the probability.** Field 6 must be derived from the seed grid, not
       from field 1's value -- a max(p, 1-p) "confidence" is a restatement of the probability.
    3. **Evidence status is not confidence and not probability.** Field 2 must be derived from
       gate outcomes only.
    """
    causal = rec["causal_effect"]
    if causal.supported and not causal.derived_from.startswith("gate3:"):
        raise AssertionError(
            f"field 5 (causal effect) derived from `{causal.derived_from}` -- a causal effect "
            f"may only come from an estimator that passed Gate 3. Gate value != causal effect.")
    conf = rec["confidence"]
    if "probability" in conf.derived_from and "seed_grid" not in conf.derived_from:
        raise AssertionError(
            f"field 6 (confidence) derived from `{conf.derived_from}` -- confidence restated "
            f"from the probability is not confidence.")
    ev = rec["evidence_status"]
    if not ev.derived_from.startswith("gates:"):
        raise AssertionError(
            f"field 2 (evidence status) derived from `{ev.derived_from}` -- evidence status is "
            f"a summary of which gates passed, not of how confident or how likely.")


# --------------------------------------------------------------------------- gate reading

def read_gate_outcomes(task: str, gate_dir: str) -> dict:
    """Load each gate's verdict for one task, from the JSON that gate actually wrote.

    A gate whose file is absent is `not_run`, which is distinct from `failed` and is reported as
    such -- "we did not test this" and "we tested this and it did not hold" are different claims
    and Layer 4 must not merge them.
    """
    out = {}

    g0 = os.path.join(gate_dir, "gate0.json")
    if os.path.exists(g0):
        b = json.load(open(g0))
        cell = b.get("summary", {}).get(task, {})
        tiers = {t: (v.get("identifiable") if isinstance(v, dict) else None)
                 for t, v in cell.items()}
        out["gate0"] = {"status": ("passed" if any(v for v in tiers.values())
                                   else ("failed" if tiers else "not_computable")),
                        "tiers": tiers, "detail": cell}
    else:
        out["gate0"] = {"status": "not_run"}

    g1 = os.path.join(gate_dir, "gate1.json")
    if os.path.exists(g1):
        b = json.load(open(g1))
        band = b.get("summary", {}).get(task, {}).get("band", {})
        passed = {m: bool(v.get("clears_floor_p") and v.get("sign_consistent_p")
                          and (v.get("coverage", {}).get("mean") or 0) > 0.5)
                  for m, v in band.items()}
        out["gate1"] = {"status": ("passed" if any(passed.values()) else "failed"),
                        "per_method": passed, "detail": band}
    else:
        out["gate1"] = {"status": "not_run"}

    g2 = os.path.join(gate_dir, "gate2.json")
    if os.path.exists(g2):
        b = json.load(open(g2))
        verified = [r["name"] for r in b["rules"] if r["status"] == "VERIFIED"]
        out["gate2"] = {"status": "passed" if verified else "failed",
                        "verified_rules": verified,
                        "hypothesis_rules": [r["name"] for r in b["rules"]
                                             if r["status"] != "VERIFIED"],
                        "rule_set_version": b["rule_set_version"],
                        "paths": b["paths"]}
    else:
        out["gate2"] = {"status": "not_run"}

    g3 = os.path.join(gate_dir, "gate3.json")
    if os.path.exists(g3):
        b = json.load(open(g3))
        arms = b.get("step4_6_dual_floor", {})
        passed = {}
        for arm, m in arms.items():
            s = m.get("sign_agreement", {})
            passed[arm] = bool(s.get("available")
                               and (s["mean"] - 0.5) > s["floor"]
                               and all(v > 0.5 for v in s["per_world_mean"]))
        # **Two arms are ceilings, not methods, and both are excluded by construction.**
        # `A3_oracle` and `A3_grid` consume the generator's TRUE latent `own_stress`, which no
        # deployment has; they exist to prove the extracted equations are right and to price time
        # discretisation. Only `naive` and `A3_est` are things a deployment could run.
        CEILINGS = ("A3_oracle", "A3_grid")
        CONTROL = "A3_const"
        deployable = {k: v for k, v in passed.items()
                      if k not in CEILINGS and k != CONTROL}
        # `A3_const` is the no-information control: the same equations driven by a constant. An
        # arm that does not separate from it is measuring the intervention's structure, not the
        # state, so it cannot carry a causal claim about a specific entity.
        sep = {}
        for arm in deployable:
            a = arms.get(arm, {}).get("sign_agreement", {})
            c = arms.get(CONTROL, {}).get("sign_agreement", {})
            if a.get("available") and c.get("available"):
                sep[arm] = bool((a["mean"] - c["mean"]) > max(a["floor"], c["floor"]))
        ident = b.get("step1_identifiability", {})
        ok = bool(ident.get("identifiable")
                  and any(deployable.get(k) and sep.get(k) for k in deployable))
        out["gate3"] = {"status": "passed" if ok else "failed",
                        "per_arm": passed, "separates_from_control": sep,
                        "deployable_arms": sorted(deployable),
                        "detail": arms, "identifiability": ident,
                        "note": ("Gate 3 requires (a) the effect to be identifiable at Gate 0, "
                                 "(b) a DEPLOYABLE arm to clear chance and the dual floor, and "
                                 "(c) that arm to separate from the constant-state control. "
                                 "A3_oracle and A3_grid consume the generator's true latent "
                                 "state and are ceilings, not methods.")}
    else:
        out["gate3"] = {"status": "not_run"}
    return out


def evidence_status(gates: dict) -> tuple:
    """The categorical summary. A function of gate outcomes and of nothing else."""
    g = {k: v["status"] for k, v in gates.items()}
    if g["gate0"] not in ("passed",):
        tier = "Prediction only — effect not identifiable"
    elif g["gate1"] == "passed" and g["gate2"] == "passed" and g["gate3"] == "passed":
        tier = "Validated causal effect"
    elif g["gate1"] == "passed" and g["gate2"] == "passed":
        tier = "Validated relevance + symbolic mechanism"
    elif g["gate2"] == "passed":
        tier = "Prediction + verified mechanism catalogue only"
    else:
        tier = "Prediction only"
    return tier, g


# --------------------------------------------------------------------------- field 6

def confidence_grid(csv_dir: str, variant: str, dseed: int, mseeds: list[int], task: str,
                    calib_dir: str, calib_dseed: int) -> dict:
    """Per-entity ensemble mean and model-seed spread, plus cross-world isotonic recalibration.

    The isotonic map is fitted on a **different world** and applied here, matching
    `reports/decision_support_build.md` §2.3's protocol: fitting within-world would be optimistic
    in exactly the place the variance decomposition says the variance lives.
    """
    # Per snapshot, not one big array: `Shipment` node counts differ from snapshot to snapshot
    # (the graph grows as shipments are created), so the grid is ragged along that axis and
    # stacking it silently requires an equality that does not hold.
    per_seed, bundles = [], None
    for m in mseeds:
        model, meta = get_backbone(csv_dir, variant, dseed, m, device="cpu", verbose=False)
        bundles = meta["test_bundles"]
        per_seed.append([predict(model, b)[task] for b in bundles])
    mu, sd = [], []
    for si in range(len(bundles)):
        Q = np.stack([per_seed[k][si] for k in range(len(mseeds))])   # [M, N_si]
        mu.append(Q.mean(axis=0))
        sd.append(Q.std(axis=0, ddof=1) if len(mseeds) > 1 else np.zeros_like(Q[0]))

    cal = {"fitted": False}
    cp, cy = [], []
    for m in mseeds:
        model, meta = get_backbone(calib_dir, variant, calib_dseed, m, device="cpu",
                                   verbose=False)
        pr = collect_predictions(model, meta["test_bundles"])[task]
        cp.append(pr["p"]); cy.append(pr["y"])
    if cp and len(cp[0]):
        curve = isotonic_fit(np.mean(np.stack(cp), axis=0), cy[0])
        cal = {"fitted": True, "calibration_world": calib_dseed, "curve": curve}
    return {"mu": mu, "sd": sd, "bundles": bundles, "isotonic": cal,
            "n_model_seeds": len(mseeds)}


# --------------------------------------------------------------------------- record assembly

def build_record(entity_id: str, task: str, p_raw: float, p_cal: float, sd_model: float,
                 gates: dict, relevance: dict | None, symbolic: dict | None,
                 causal: dict | None, pop_cross_world_sd: float | None) -> dict:
    tier, gstat = evidence_status(gates)
    rec = {
        "entity_id": entity_id,
        "task": task,
        "probability": Field("probability", {"raw": p_raw, "calibrated": p_cal},
                             "share+markov forward pass; isotonic fitted on a held-out world"),
        "evidence_status": Field("evidence_status", {"tier": tier, "gates": gstat},
                                 "gates:" + ",".join(f"{k}={v}" for k, v in gstat.items())),
        "entity_relevance": (Field("relevance", relevance, "gate1:passed")
                             if gates["gate1"]["status"] == "passed" and relevance
                             else Field("relevance", UNSUPPORTED["relevance"],
                                        "gate1:" + gates["gate1"]["status"], supported=False)),
        "symbolic_consistency": (Field("symbolic", symbolic, "gate2:passed")
                                 if gates["gate2"]["status"] == "passed" and symbolic
                                 else Field("symbolic", UNSUPPORTED["symbolic"],
                                            "gate2:" + gates["gate2"]["status"],
                                            supported=False)),
        "causal_effect": (Field("causal_effect", causal, "gate3:passed")
                          if gates["gate3"]["status"] == "passed" and causal
                          else Field("causal_effect", UNSUPPORTED["causal"],
                                     "gate3:" + gates["gate3"]["status"], supported=False)),
        "confidence": Field("confidence",
                            {"model_seed_sd": sd_model,
                             "cross_world_population_sd": pop_cross_world_sd,
                             "identifiability_note":
                                 "model-seed spread is per entity; the cross-world term is "
                                 "population-level only -- entities do not persist across "
                                 "dataset seeds on this generator"},
                            "seed_grid: model-init ensemble spread + cross-world population "
                            "term"),
    }
    assert_distinctions(rec)
    return rec


def render(rec: dict) -> str:
    p = rec["probability"].value
    ev = rec["evidence_status"].value
    c = rec["confidence"].value
    lines = [
        f"entity: {rec['entity_id']}   task: {rec['task']}",
        f"1. Prediction probability : {p['raw']:.4f} raw / {p['calibrated']:.4f} calibrated",
        f"2. Evidence status        : {ev['tier']}",
        "                            " + "  ".join(f"{k}={v}" for k, v in ev["gates"].items()),
    ]
    for n, key in ((3, "entity_relevance"), (4, "symbolic_consistency"), (5, "causal_effect")):
        f = rec[key]
        label = {"entity_relevance": "Entity relevance", "symbolic_consistency":
                 "Symbolic consistency", "causal_effect": "Causal effect"}[key]
        if f.supported:
            lines.append(f"{n}. {label:<23}: {json.dumps(f.value, default=str)}")
        else:
            body = f.value.split("\n")
            lines.append(f"{n}. {label:<23}: {body[0]}")
            for extra in body[1:]:
                lines.append(f"                            {extra}")
    sdw = c["cross_world_population_sd"]
    lines.append(f"6. Confidence / uncertainty: model-seed sd {c['model_seed_sd']:.5f}"
                 + (f"; cross-world population sd {sdw:.5f}" if sdw is not None else ""))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--variant", default="0")
    ap.add_argument("--task", default="impact")
    ap.add_argument("--dseed", type=int, default=42)
    ap.add_argument("--calib-dseed", type=int, default=43)
    ap.add_argument("--mseeds", default="0,1,2")
    ap.add_argument("--snapshot", type=int, default=-1)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--all-nodes", action="store_true",
                    help="rank over every node of the task's entity type rather than over the "
                         "scored (labelled) population; see the comment at the ranking step")
    ap.add_argument("--gate-dir", default=OUT_DIR)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "layer4_records.json"))
    a = ap.parse_args()

    mseeds = [int(x) for x in a.mseeds.split(",") if x.strip()]
    csv_dir = os.path.join(a.csv_root, f"v{a.variant}_seed{a.dseed}")
    calib_dir = os.path.join(a.csv_root, f"v{a.variant}_seed{a.calib_dseed}")

    gates = read_gate_outcomes(a.task, a.gate_dir)
    print("gate outcomes: " + "  ".join(f"{k}={v['status']}" for k, v in gates.items()))

    grid = confidence_grid(csv_dir, a.variant, a.dseed, mseeds, a.task, calib_dir,
                           a.calib_dseed)
    si = a.snapshot if a.snapshot >= 0 else len(grid["bundles"]) - 1
    bundle = grid["bundles"][si]
    et = TASK_ENTITY_TYPE[a.task]
    ids = bundle.data[et].node_id
    mu, sd = grid["mu"][si], grid["sd"][si]
    p_cal = (isotonic_apply(grid["isotonic"]["curve"], mu) if grid["isotonic"]["fitted"]
             else mu)

    # The cross-world term is population-level by construction and is computed as such.
    pop = []
    for d in (a.dseed, a.calib_dseed):
        _m, meta = get_backbone(os.path.join(a.csv_root, f"v{a.variant}_seed{d}"), a.variant,
                                d, mseeds[0], device="cpu", verbose=False)
        pop.append(float(collect_predictions(_m, meta["test_bundles"])[a.task]["p"].mean()))
    pop_sd = float(np.std(pop, ddof=1)) if len(pop) > 1 else None

    # Fields 3/4/5 content, only ever filled where the gate passed.
    relevance = symbolic = causal = None
    if gates["gate2"]["status"] == "passed":
        # The field carries the PATHWAYS that are symbolically consistent in this world, plus a
        # pointer to the versioned rule set -- not the rule catalogue itself. A user reading one
        # entity's record needs to know which mechanism could carry this outcome, not that
        # sixteen equations exist.
        # Only paths that TERMINATE at this task's label rule. A supplier-impact pathway is not
        # an explanation for a product shortage, and listing it under one would be exactly the
        # "plausible business narrative" failure Gate 2 exists to block.
        terminal = {"delay": "p_delay", "shortage": "shortage_event",
                    "impact": "impact_aggregation"}[a.task]
        symbolic = {"consistent_paths": [k for k, v in gates["gate2"]["paths"].items()
                                         if v["symbolically_consistent"]
                                         and v["edges"][-1] == terminal],
                    "n_verified_rules": len(gates["gate2"]["verified_rules"]),
                    "n_unverified_rules": len(gates["gate2"]["hypothesis_rules"]),
                    "rule_set_version":
                        gates["gate2"]["rule_set_version"]["generator_sha256"][:16]}

    # **Rank within the SCORED population by default.** Phase 0 measured that 96.4% of
    # `Shipment` nodes sit at p >= 0.999 and that 99.8%+ of those are already DELIVERED --
    # shipments the generator's own label walk skips and no planner would open. Ranking over all
    # nodes returns those and nothing else. `--all-nodes` reproduces the unscoped behaviour.
    cand = np.arange(len(mu))
    if not a.all_nodes:
        idx, _y = bundle.labels[a.task]
        if idx.numel():
            cand = idx.detach().cpu().numpy()
    order = cand[np.argsort(-mu[cand])][:a.top]
    recs = []
    for i in order:
        rec = build_record(ids[int(i)], a.task, float(mu[i]), float(p_cal[i]), float(sd[i]),
                           gates, relevance, symbolic, causal, pop_sd)
        recs.append(rec)
        print("\n" + render(rec))

    blob = {"config": vars(a), "gates": gates,
            "records": [{k: (v.__dict__ if isinstance(v, Field) else v)
                         for k, v in r.items()} for r in recs],
            "calibration": {"fitted_on_world": a.calib_dseed,
                            "isotonic_available": grid["isotonic"]["fitted"]}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=str)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
