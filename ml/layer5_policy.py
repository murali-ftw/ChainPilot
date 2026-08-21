#!/usr/bin/env python3
"""
Layer 3 v3, PHASE 6 — Layer 5 decision policy, tiered by evidence.

`layer3_neurosymbolic_build_prompt.md` §8. The action a decision-support system is allowed to
recommend is a function of **which claims were validated**, not of how high the probability is or
how confident the model feels. Three tiers, and the language of each is fixed here rather than
composed at call time, so a stronger tier's phrasing cannot leak into a weaker one's output:

| tier | admitted when | what may be said |
|---|---|---|
| **prediction-only** | Gate 1 and Gate 2 did not both pass | monitor / contingency language only |
| **relevance + mechanism** | Gate 1 and Gate 2 passed, Gate 3 did not | investigate the named candidates through their verified path |
| **validated causal effect** | Gate 3 passed for this effect | intervention-oriented recommendation, with stated uncertainty and business constraints |

**The rule the whole phase exists for.** An intervention -- "activate the alternate supplier" --
may be framed as *causally justified* only where Gate 3 passed **for that effect**. Everywhere
else it is presented strictly as a **contingency option under the prediction-only tier**: the same
action may still be sensible, but the justification on offer is "risk is elevated", not "doing
this will reduce the risk by X". `assert_no_causal_language` enforces that mechanically, because
the failure mode is a phrasing failure and phrasing is easy to get wrong by accident.

Confidence never promotes a tier. A high-confidence prediction with no validated relevance is
still prediction-only; `reports/decision_support_build.md` §2.3 is the reason to be strict about
this -- recalibration moved ECE by 9x-17x while model-init ensembling did not clear its own
floor, so a confident-looking number here is a statement about calibration, not about evidence.

    python3 ml/layer5_policy.py --task impact --dseed 42
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.layer4_integration import (  # noqa: E402
    OUT_DIR, confidence_grid, evidence_status, read_gate_outcomes)

TIERS = ("prediction_only", "relevance_plus_mechanism", "validated_causal_effect")

# Words that assert a causal claim about an action's consequence. Checked against any
# recommendation emitted below the top tier.
CAUSAL_TOKENS = re.compile(
    r"\b(will reduce|reduces|caused by|root cause|because of|prevents?|mitigates?|"
    r"eliminat\w+|attributable to|responsible for)\b", re.I)


def _both(p_raw, p_cal) -> str:
    return (f"{p_cal:.3f} calibrated ({p_raw:.3f} raw)" if p_raw is not None
            else f"{p_cal:.3f}")


def tier_for(gates: dict) -> str:
    g = {k: v["status"] for k, v in gates.items()}
    if g["gate3"] == "passed" and g["gate1"] == "passed" and g["gate2"] == "passed":
        return "validated_causal_effect"
    if g["gate1"] == "passed" and g["gate2"] == "passed":
        return "relevance_plus_mechanism"
    return "prediction_only"


def assert_no_causal_language(tier: str, text: str) -> None:
    """Below the top tier, no sentence may assert that an action changes the outcome.

    This is a real check and not decoration: the whole gate architecture is defeated if the
    prose promises a causal effect the numbers never established. The check runs on the emitted
    string, which is the artefact a user actually reads.
    """
    if tier == "validated_causal_effect":
        return
    m = CAUSAL_TOKENS.search(text)
    if m:
        raise AssertionError(
            f"tier `{tier}` recommendation contains causal language: {m.group(0)!r}. "
            f"An intervention may be framed as causally justified only where Gate 3 passed "
            f"for that effect.")


def recommend(tier: str, entity_id: str, task: str, p_cal: float, sd: float,
              gates: dict, candidates: list | None = None,
              effect: dict | None = None, p_raw: float | None = None) -> dict:
    """The action text for one entity at its admitted tier.

    Both probabilities are quoted. The isotonic map's top bin is a PAVA step to 1.0, so a
    calibrated-only line reads "risk 1.000" for an entity whose raw score is 0.88 — true to the
    calibration and misleading to a reader.
    """
    if tier == "prediction_only":
        # "we tested it and it did not hold" and "we did not test it" are different claims and
        # the text must not merge them -- a reader who is told a gate failed will conclude the
        # capability is absent, which is not what `not_run` means.
        phrase = {"failed": "did not clear", "not_run": "was not run for this task",
                  "not_computable": "was not computable on this generator"}
        why = []
        for g, label in (("gate0", "Gate 0 (identifiability)"),
                         ("gate1", "Gate 1 (entity relevance)"),
                         ("gate2", "Gate 2 (symbolic mechanism)"),
                         ("gate3", "Gate 3 (causal effect)")):
            st = gates[g]["status"]
            if st != "passed":
                why.append(f"{label} {phrase.get(st, st)}")
        text = (
            f"MONITOR. {task} risk for {entity_id} is estimated at "
            f"{_both(p_raw, p_cal)} (model-seed sd {sd:.4f}). "
            f"No validated explanation is available: "
            + "; ".join(why) + ". "
            "Contingency options such as activating an alternate source may be held ready as "
            "an operational hedge; this system offers no evidence about what such an action "
            "would do to the risk.")
        act = {"action": "monitor", "contingency_options_offered": True,
               "contingency_framing": "hedge under an elevated prediction, not a justified "
                                      "intervention"}
    elif tier == "relevance_plus_mechanism":
        names = ", ".join(str(c["entity"]) for c in (candidates or [])) or "(none named)"
        path = ", ".join((effect or {}).get("path", [])) or "(no path)"
        text = (
            f"INVESTIGATE. {task} risk for {entity_id} is estimated at {p_cal:.3f} "
            f"(model-seed sd {sd:.4f}). Candidates to examine: {names}, through the verified "
            f"pathway [{path}]. The pathway is verified against the generator SCM; the size of "
            f"any effect an action on these candidates would have is not established.")
        act = {"action": "investigate", "candidates": candidates or [], "path": path}
    else:
        text = (
            f"INTERVENE. {task} risk for {entity_id} is estimated at {p_cal:.3f} "
            f"(model-seed sd {sd:.4f}). The validated estimate is that the proposed "
            f"intervention changes the risk by {effect.get('delta'):+.4f} "
            f"[{effect.get('lo'):+.4f}, {effect.get('hi'):+.4f}], through "
            f"[{', '.join(effect.get('path', []))}]. Subject to business constraints: "
            f"{effect.get('constraints', 'lead time, contract, and qualification cost')}.")
        act = {"action": "intervene", "effect": effect}
    assert_no_causal_language(tier, text)
    return {"tier": tier, "text": text, **act}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--variant", default="0")
    ap.add_argument("--task", default="impact")
    ap.add_argument("--dseed", type=int, default=42)
    ap.add_argument("--calib-dseed", type=int, default=43)
    ap.add_argument("--mseeds", default="0,1,2")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--gate-dir", default=OUT_DIR)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "layer5_policy.json"))
    a = ap.parse_args()

    import numpy as np
    from ml.hypothesis_ranker import isotonic_apply
    from ml.models.depth import TASK_ENTITY_TYPE

    mseeds = [int(x) for x in a.mseeds.split(",") if x.strip()]
    gates = read_gate_outcomes(a.task, a.gate_dir)
    tier = tier_for(gates)
    ev_tier, gstat = evidence_status(gates)
    print(f"gates: " + "  ".join(f"{k}={v['status']}" for k, v in gates.items()))
    print(f"evidence status : {ev_tier}")
    print(f"policy tier     : {tier}\n")

    grid = confidence_grid(os.path.join(a.csv_root, f"v{a.variant}_seed{a.dseed}"), a.variant,
                           a.dseed, mseeds, a.task,
                           os.path.join(a.csv_root, f"v{a.variant}_seed{a.calib_dseed}"),
                           a.calib_dseed)
    si = len(grid["bundles"]) - 1
    bundle = grid["bundles"][si]
    ids = bundle.data[TASK_ENTITY_TYPE[a.task]].node_id
    mu, sd = grid["mu"][si], grid["sd"][si]
    p_cal = (isotonic_apply(grid["isotonic"]["curve"], mu) if grid["isotonic"]["fitted"]
             else mu)

    # Ranked within the SCORED population, for the reason `ml/layer4_integration.py` records:
    # 96.4% of Shipment nodes sit at p >= 0.999 and 99.8%+ of those are already delivered.
    cand = np.arange(len(mu))
    idx, _y = bundle.labels[a.task]
    if idx.numel():
        cand = idx.detach().cpu().numpy()

    recs = []
    for i in cand[np.argsort(-mu[cand])][:a.top]:
        r = recommend(tier, ids[int(i)], a.task, float(p_cal[i]), float(sd[i]), gates,
                      p_raw=float(mu[i]))
        recs.append({"entity_id": ids[int(i)], "p_raw": float(mu[i]),
                     "p_calibrated": float(p_cal[i]),
                     "model_seed_sd": float(sd[i]), **r})
        print(r["text"] + "\n")

    blob = {"config": vars(a), "gates": {k: v["status"] for k, v in gates.items()},
            "evidence_status": ev_tier, "policy_tier": tier, "recommendations": recs,
            "tier_table": {t: ("monitor / contingency language only"
                               if t == "prediction_only" else
                               "investigate named candidates through their verified path"
                               if t == "relevance_plus_mechanism" else
                               "intervention-oriented recommendation with stated uncertainty")
                           for t in TIERS}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=str)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
