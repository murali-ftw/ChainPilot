#!/usr/bin/env python3
"""
Layer 3 v3, PHASE 1 — GATE 0: identifiability, before any relevance or causal work.

`layer3_neurosymbolic_build_prompt.md` §3 makes this a **precondition, not one diagnostic among
several**: for every task/effect this build will later make a relevance (Gate 1) or causal-effect
(Gate 3) claim about, first establish

    P(X | do(Z = z1))  !=  P(X | do(Z = z2))

distinguishable from a **measured** empirical null and reproducible across dataset seeds. If the
effect is not identifiable from observations, no downstream work is attempted for it.

---

**Reuse, not rebuild.** Tiers A and B are `ml/task_identifiability_gate.py`'s, imported unmodified
-- which in turn imports `ml/identifiability_check.py`'s `Realiser`, `draw_table`,
`_median_split_levels` and `featurise` unmodified. The CRN protocol, the three-group
entity-disjoint residual test, the exact row-for-row label-rule check and the shortage-feasibility
probe are all called, not re-implemented. What is new here is:

1. **Variant 0 as the testbed.** Every prior number this build is measured against
   (`reports/decision_support_build.md`) was taken on Variant 0, and the previous identifiability
   work ran on Variants A and E. Running Gate 0 anywhere else would leave Gates 1 and 3 gated by
   a world their own baselines were never measured in.
2. **Tier C — `do(co-parent structure)`, which is the effect Gate 3 actually estimates.**
   Tiers A and B move a *level* (`stress`, `RESILIENCE`). Gate 3's three interventions
   (`add_dual_source`, `remove_supplier`, `substitute_supplier`) move the **co-parent graph**,
   and on Variant 0 that graph is the single live structural lever: `MECHS = ()` zeroes
   `hp_coupling` and `upstream_stress` and sets `recv_atten = absorption = 1`, leaving

       stress(s,t) = own_stress(s,t) + 0.35 * SUM_{p in coparents[s]} own_stress(p,t)

   Tier C contrasts `do(coparents[s] = {})` against the factual set, one supplier at a time.
   Isolation is exact for the same reason Tier A's is: `stress(s)` reads its partners' OWN
   stress, which no edit to `coparents[s]` can move, and both task labels read only the entity's
   own shipments.

**One decision in Tier C that changes what the gate means, so it is stated rather than buried.**
The observable block `X` is built from the **factual** co-parent structure in both worlds, even
though the intervention removes it. Letting `X` carry the structural edit would put a perfect
world indicator (partner count 0 vs. k) into the very features `g(X, X_nbr)` is fitted on, and
`g` would then absorb the average label shift as a per-world intercept -- the gate would be
testing whether the edit is visible, which is trivially yes, instead of whether its *effect* on
outcomes is. Holding `X`'s structure factual is the conservative choice: it can only make a Tier C
pass harder to obtain, never easier.

**`shortage` is probed for feasibility before anything is attempted for it**, using
`task_identifiability_gate.shortage_feasibility` unmodified.

    python3 ml/gate0_identifiability.py --variant 0 --seeds 42,43,44,45,46 \
        --out out/layer3_v3/gate0.json
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

from ml.confirm_latent_states import namespace                       # noqa: E402
from ml.task_identifiability_gate import (                           # noqa: E402
    TaskRealiser, label_rule_check, observables, residual_world_test,
    shortage_feasibility, tier_a_worlds, tier_b_worlds,
)

TIER_LABELS = {"A": "do(stress), isolated per supplier",
               "B": "do(joint upstream set), global",
               "C": "do(co-parent structure), isolated per supplier"}


# ---------------------------------------------------------------------------
# Tier C — the structural lever Gate 3 estimates
# ---------------------------------------------------------------------------

def tier_c_worlds(r: TaskRealiser, task: str) -> tuple:
    """`do(coparents[s] = {})` vs. the factual co-parent set, one supplier at a time.

    The generator's `stress()` closes over the module-level `coparents` dict, so installing the
    intervention is a mutation of that dict in place -- the *real* function then computes the
    counterfactual itself, exactly as `ml/identifiability_check.py::DoSpec.world` does for
    `IDIO` and `RESILIENCE`. The factual value is restored in a `finally`, so a raised exception
    cannot leave the namespace edited for the next world.
    """
    cop = r.ns["coparents"]
    scope = [s for s in sorted(r.visible) if r.sup_ship.get(s) and cop.get(s)]
    st_fac = (lambda sid, sh: r.stress(sid, r.sup_by_id[sid]["base_rel"], sh["dispatched_at"]))

    n_partners = [len(cop[s]) for s in scope]
    worlds = []
    for level in ("off", "on"):
        lab, obs = {}, {}
        for sid in scope:
            saved = cop.get(sid)
            if level == "off":
                cop[sid] = ()
            try:
                with r.open_world([sid], st_fac, r.absorption) as ev:
                    lab.update(r.task_labels_for([sid], ev)[task])
                    # X is read under the FACTUAL structure in both worlds; see module
                    # docstring. The re-realised shipment outcomes are still the intervened
                    # ones, which is the part that must move.
                    cop[sid] = saved
                    obs.update(observables(r, task, [sid], ev))
            finally:
                if saved is None:
                    cop.pop(sid, None)
                else:
                    cop[sid] = saved
        worlds.append((lab, obs))
    meta = {"n_suppliers_in_scope": len(scope),
            "mean_partners": (statistics.fmean(n_partners) if n_partners else 0.0),
            "coparent_coupling": float(r.ns.get("COPARENT_COUPLING", 0.0)),
            "isolation": "one supplier at a time",
            "levels": "coparents[s] = {} vs factual",
            "X_structure": "factual in both worlds (see module docstring)"}
    return worlds[0], worlds[1], meta


BUILDERS = {"A": tier_a_worlds, "B": tier_b_worlds, "C": tier_c_worlds}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_seed(variant: str, dseed: int, config: str, tasks: list[str], init_seeds: list[int],
             realisation_seeds: tuple, tiers: dict) -> dict:
    ns = namespace(variant, dseed, config)
    r_main = TaskRealiser(ns, realisation_seeds[0])
    r_null = TaskRealiser(ns, realisation_seeds[1])

    out = {"_label_rule_check": label_rule_check(r_main)}
    lc = out["_label_rule_check"]
    for t in ("delay", "impact"):
        print(f"  label rule [{t}]: {lc[t]['n_compared']:,} compared, "
              f"{lc[t]['n_mismatches']} mismatches, exact={lc[t]['exact']}", flush=True)

    for task in tasks:
        cell = {}
        for tier in tiers.get(task, []):
            builder = BUILDERS[tier]
            w1, w2, levels = builder(r_main, task)
            # NULL: identical latent/structural configuration on both sides, different
            # realisation seed. Every source of apparent separation except the intervention
            # survives, which is the only way to know what the pipeline manufactures on its own.
            n1, _n2, _ = builder(r_null, task)
            sig = residual_world_test(w1, w2, init_seeds, split_seed=dseed)
            nul = residual_world_test(w1, n1, init_seeds, split_seed=dseed)
            cell[tier] = {"levels": levels, "signal": sig, "null": nul}
            print(f"  {task:<9} tier {tier}:  rows {sig.get('n_rows',0):,}  "
                  f"Y differs {sig.get('n_rows_label_differs',0):,}  "
                  f"label {sig.get('label_shift_auc', float('nan')):.4f}  "
                  f"obs {sig.get('observable_shift_auc', float('nan')):.4f}  "
                  f"RESID {sig.get('residual_shift_auc', float('nan')):.4f} "
                  f"(null {nul.get('residual_shift_auc', float('nan')):.4f})", flush=True)
        out[task] = cell
    return out


def summarise(res: dict) -> dict:
    """Per (task, tier): mean residual AUC, the measured null, all three floors, disposition.

    The floor is `max(init-seed spread, dataset-seed spread, null excursion)` -- the same
    three-way maximum `ml/identifiability_check.py::summarise` uses, so this gate's floor is
    the same object every earlier identifiability number in the project was judged against.
    """
    summary = {}
    for task in res["tasks"]:
        summary[task] = {}
        for tier in ("A", "B", "C"):
            cells = [res["per_seed"][str(d)][task][tier] for d in res["dataset_seeds"]
                     if tier in res["per_seed"].get(str(d), {}).get(task, {})]
            if not cells:
                continue
            fin = lambda x: x == x  # noqa: E731  -- NaN filter
            sa = [c["signal"]["residual_shift_auc"] for c in cells
                  if fin(c["signal"].get("residual_shift_auc", float("nan")))]
            na = [c["null"]["residual_shift_auc"] for c in cells
                  if fin(c["null"].get("residual_shift_auc", float("nan")))]
            if not sa:
                summary[task][tier] = {"insufficient": True,
                                       "n_rows": cells[0]["signal"].get("n_rows")}
                continue
            init_floor = max([c["signal"]["init_seed_spread"] for c in cells
                              if c["signal"].get("init_seed_spread") is not None]
                             or [float("nan")])
            dfloor = (max(sa) - min(sa)) if len(sa) > 1 else float("nan")
            null_exc = max(abs(a - 0.5) for a in na) if na else float("nan")
            floor = max(f for f in (init_floor, dfloor, null_exc) if f == f)
            mean = statistics.fmean(sa)
            s = {
                "tier_label": TIER_LABELS[tier],
                "n_dataset_seeds": len(sa),
                "residual_auc_mean": mean, "residual_auc_per_dataset_seed": sa,
                "null_auc_mean": statistics.fmean(na) if na else None,
                "null_auc_per_dataset_seed": na,
                "label_shift_auc": statistics.fmean(
                    [c["signal"]["label_shift_auc"] for c in cells]),
                "observable_shift_auc": statistics.fmean(
                    [c["signal"]["observable_shift_auc"] for c in cells]),
                "mean_rows": statistics.fmean([c["signal"]["n_rows"] for c in cells]),
                "mean_rows_label_differs": statistics.fmean(
                    [c["signal"]["n_rows_label_differs"] for c in cells]),
                "init_seed_floor": init_floor, "dataset_seed_floor": dfloor,
                "null_excursion_floor": null_exc, "reproduction_floor": floor,
                "above_chance_by": mean - 0.5,
                "clears_floor": (mean - 0.5) > floor,
                "sign_consistent": all(a > 0.5 for a in sa) or all(a < 0.5 for a in sa),
                "levels": cells[0]["levels"],
            }
            s["identifiable"] = bool(s["clears_floor"] and s["sign_consistent"])
            summary[task][tier] = s
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="0")
    ap.add_argument("--tasks", default="delay,shortage,impact")
    ap.add_argument("--tiers", default="delay:A,B,C;impact:A,B,C")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--config", default="v1")
    ap.add_argument("--init-seeds", default="0,1,2,3,4")
    ap.add_argument("--realisation-seeds", default="1,2")
    ap.add_argument("--out", default=os.path.join(REPO, "out", "layer3_v3", "gate0.json"))
    a = ap.parse_args()

    dseeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    iseeds = [int(s) for s in a.init_seeds.split(",") if s.strip()]
    rseeds = tuple(int(s) for s in a.realisation_seeds.split(","))
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    tiers = {}
    for part in a.tiers.split(";"):
        if ":" in part:
            k, v = part.split(":")
            tiers[k.strip()] = [x.strip() for x in v.split(",") if x.strip()]

    res = {"variant": a.variant, "config": a.config, "dataset_seeds": dseeds,
           "init_seeds": iseeds, "realisation_seeds": list(rseeds), "tiers": tiers,
           "tier_labels": TIER_LABELS, "per_seed": {}}

    if "shortage" in tasks:
        print("\n=== shortage: is the CRN protocol applicable at all? ===", flush=True)
        t = time.time()
        res["shortage_feasibility"] = shortage_feasibility(a.variant, dseeds[0], a.config)
        f = res["shortage_feasibility"]
        print(f"  shipments {f['n_shipments_factual']:,} -> {f['n_shipments_perturbed']:,} "
              f"({f['shipment_count_delta']:+d})", flush=True)
        print(f"  identical identity tuple at the same ledger index: "
              f"{f['frac_identical_identity_tuple_at_same_index']:.4%}", flush=True)
        print(f"  shortage-event key Jaccard: {f['jaccard_shortage_keys']:.4f}", flush=True)
        print(f"  CRN APPLICABLE: {f['crn_applicable']}   ({time.time()-t:.0f}s)", flush=True)
        tasks = [x for x in tasks if x != "shortage"]

    res["tasks"] = tasks
    for d in dseeds:
        print(f"\n=== variant {a.variant} seed {d} ===", flush=True)
        t = time.time()
        res["per_seed"][str(d)] = run_seed(a.variant, d, a.config, tasks, iseeds, rseeds, tiers)
        print(f"  ({time.time()-t:.0f}s)", flush=True)
    res["summary"] = summarise(res)

    print("\n" + "=" * 132)
    print(f"GATE 0 — identifiability, variant {a.variant}, {len(dseeds)} dataset seeds "
          f"x {len(iseeds)} init seeds")
    print("=" * 132)
    hdr = (f"{'task':<10}{'tier':>5}{'rows':>9}{'Y differs':>11}{'label AUC':>11}"
           f"{'obs AUC':>9}{'RESID':>8}{'null':>8}{'floor':>8}{'clears':>8}"
           f"{'sign':>6}{'IDENT':>7}")
    print(hdr); print("-" * len(hdr))
    for task in tasks:
        for tier in ("A", "B", "C"):
            s = res["summary"].get(task, {}).get(tier)
            if not s:
                continue
            if s.get("insufficient"):
                print(f"{task:<10}{tier:>5}{'insufficient rows':>50}")
                continue
            print(f"{task:<10}{tier:>5}{s['mean_rows']:>9,.0f}"
                  f"{s['mean_rows_label_differs']:>11,.0f}{s['label_shift_auc']:>11.4f}"
                  f"{s['observable_shift_auc']:>9.4f}{s['residual_auc_mean']:>8.4f}"
                  f"{s['null_auc_mean']:>8.4f}{s['reproduction_floor']:>8.4f}"
                  f"{('YES' if s['clears_floor'] else 'no'):>8}"
                  f"{('yes' if s['sign_consistent'] else 'NO'):>6}"
                  f"{('YES' if s['identifiable'] else 'NO'):>7}")
    print("-" * len(hdr))
    for tier, lab in TIER_LABELS.items():
        print(f"  tier {tier}: {lab}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
