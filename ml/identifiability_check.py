#!/usr/bin/env python3
"""
Layer 3, STEP 2A — is the latent state IDENTIFIABLE at all?

Recoverability ("can the current SHARE representation and head get Z out of ordinary
observational data") is already measured: it is Model A's AUC in
`reports/phase1_latent_state.md`, and this module does not remeasure it. This module answers
the *other* question, which no experiment in this project has asked yet:

    Does changing the true Z produce, in principle, a distinguishable observational change?

A NO here means Case 3 -- the observations do not carry the information and no architecture,
calibration or confidence mechanism can help. A YES separates "the information is missing"
from "the information is there and the current model does not extract it", which are different
failures with different remedies.

---

**The intervention is a real do(Z), and it is isolated to one supplier at a time.**

For a target supplier `s`, exactly one latent quantity is overridden in the generator's own
namespace and every other property of the world -- structure, co-parents, schedule, dispatch
times, eta, lead times, carriers, every other supplier's Z -- is held fixed:

| state | do() | channel it acts through |
|---|---|---|
| `supply_stress` | `stress(s,t) := z_lo` vs `:= z_hi` | `p_delay = 0.025 + 0.38*st*absorption` |
| `supplier_reliability` | `IDIO[s] := None` vs factual outage | `own_stress` -> `stress` -> `p_delay` |
| `recovery_capability` | `RESILIENCE[s] := r_lo` vs `:= r_hi` | `absorption(s)` -> `p_delay` |

Isolation matters and is not cosmetic. `stress(s,t)` reads its co-parents' `own_stress`, so a
*global* `IDIO := None` would also move every partner and the measured effect would be
contaminated by a second causal path -- which could only ever flatter identifiability. Because
each supplier's shipments depend on that supplier's own `stress` and `absorption` and nothing
else, overriding one supplier at a time is both exact and sufficient.

**Re-simulation is replaced by re-realisation under common random numbers**, for the reason
`ml/counterfactual_ground_truth.py` measured rather than assumed: the generator draws from one
sequential stream, so an intervention that flips a single shipment desynchronises every later
draw and 100% of shipments end up differing for reasons unrelated to the intervention. Here,
each shipment gets a fixed triple `(u_delay, g_lateness, e_early)` drawn from its own stream,
keyed by shipment index and a realisation seed. Both worlds consume the identical triple, so
the ONLY thing that can move an outcome is Z. Everything else in `new_shipment`'s tail is
reproduced verbatim:

    delayed = u_delay < min(0.80, 0.025 + 0.38*st*absorption(sup))
    late_by = max(1, int(g_lateness * (1 + 2*st))) days           # magnitude also carries Z
    actual  = eta + late_by            if delayed
              eta - e_early hours      otherwise

**The observation is the generator's own emitted feature row, not a reimplementation.**
`sup_features()` is called out of the generator namespace after the supplier's shipment dicts
have been rewritten in place, so the observation vector is exactly
`supplier_temporal_features.csv` -- `on_time_rate_{30,90,180}d`, `trend_slope`,
`lateness_variance`, `days_since_last_late`, `shipment_count_180d` -- which is the entire
time-varying observable channel `ml/data/loader.py` reads for a Supplier. The remaining
supplier inputs (`country`, `capacity_score`, `lead_time_days`) are static and Z-invariant by
construction, so they cannot carry a do(Z) signal and are correctly absent.

**Two numbers are reported, and the first is the one that cannot be argued with.**

1. *Divergence rate* -- the fraction of `(supplier, t0)` observation rows that differ AT ALL
   between the two worlds. This is exact, not statistical. If it is 0.0, Z is unidentifiable
   from this observable channel, full stop, and no classifier result can rescue it.
2. *Classifier AUC above a measured null* -- a head trained to say which world a row came
   from. The null is not assumed to be 0.5: it is measured by running the identical pipeline
   on two worlds with the SAME Z and different realisation seeds, which is the only way to
   know how much apparent separation the pipeline manufactures from realisation noise alone.

The split is supplier-disjoint. Supplier identity is uninformative about the world label by
construction (every supplier appears in both worlds), so this is belt-and-braces rather than
load-bearing, but it costs nothing.

**Stated limitation, because it bounds what a negative result means.** `RESILIENCE` also acts
through `mitigation_level()`, which changes the replenishment trigger and order quantity and
therefore the *schedule*. Holding the schedule fixed is what makes common random numbers work,
so that channel is outside this measurement. Any identifiability figure for
`recovery_capability` here is therefore a LOWER bound on the true one.

    python3 ml/identifiability_check.py --variant E --seeds 42,43,44,45,46 \
        --states supply_stress,supplier_reliability,recovery_capability \
        --out out/layer3/step2_E.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import statistics
import sys
import time
from datetime import timedelta

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.counterfactual_ground_truth import generator_namespace   # noqa: E402
from ml.hypothesis_ranker import roc_auc                         # noqa: E402
from ml.latent_state_head import train_head                      # noqa: E402

STATES = ("supply_stress", "supplier_reliability", "recovery_capability")
# `sup_features` returns (r30, r90, r180, slope, lateness_var, days_since_last_late, n180).
FEATURE_NAMES = ("on_time_rate_30d", "on_time_rate_90d", "on_time_rate_180d", "trend_slope",
                 "lateness_variance", "days_since_last_late", "shipment_count_180d")
N_NULLABLE = 6          # every field except shipment_count_180d can come back None


# ---------------------------------------------------------------------------
# common random numbers
# ---------------------------------------------------------------------------

def draw_table(n_ship: int, realisation_seed: int) -> list:
    """`[(u_delay, g_lateness, e_early_hours, j_jitter_seconds)]`, one tuple per shipment index.

    Keyed by index rather than by shipment id: `hash(str)` is PYTHONHASHSEED-dependent, a trap
    `db/generate_dataset.py` documents having been bitten by, and an index is stable.

    `j_jitter` reproduces `J()` (`db/generate_dataset.py:207`), which the generator applies to
    every recorded delivery timestamp. It looks cosmetic and is not: a NOT-delayed shipment is
    recorded at `J(eta - randint(0,30) hours)`, so the 1-in-31 shipments that draw 0 hours land
    at `eta + up to 45 min` and are counted LATE by `sup_features`' `delivered_at <= eta` test.
    Omitting the jitter moved the pooled on-time rate by +2.6 points -- caught by
    `harness_check`, which is exactly what that check exists for.
    """
    rng = random.Random(0xC12A17 + realisation_seed)
    return [(rng.random(), rng.lognormvariate(1.1, 0.6), rng.randint(0, 30),
             rng.randint(0, 2700)) for _ in range(n_ship)]


class Realiser:
    """Re-realises one supplier's shipments under a supplied `(stress, absorption)` and reads
    the generator's own observable feature row back out."""

    def __init__(self, ns: dict, realisation_seed: int):
        self.ns = ns
        self.suppliers = ns["suppliers"]
        self.sup_by_id = ns["sup_by_id"]
        self.sup_ship = ns["sup_ship"]
        self.T0S = list(ns["T0S"])
        self.T_END = ns["T_END"]
        self.visible = set(ns.get("VISIBLE_SUP") or {s["id"] for s in self.suppliers})
        self.sup_features = ns["sup_features"]
        self.stress = ns["stress"]
        self.absorption = ns["absorption"]
        self.SEA, self.PORT_EVENTS = ns["SEA"], ns["PORT_EVENTS"]
        self.IDIO, self.RESILIENCE = ns["IDIO"], ns["RESILIENCE"]

        self.ship_index = {sh["id"]: i for i, sh in enumerate(ns["shipments"])}
        self.draws = draw_table(len(ns["shipments"]), realisation_seed)
        # Mechanism G's reporting lag, preserved per shipment rather than redrawn -- it is a
        # nuisance quantity, so holding it fixed across worlds is part of the CRN pin.
        self.rec_lag = {}
        for sh in ns["shipments"]:
            d, r = sh.get("delivered_at"), sh.get("delivered_rec")
            self.rec_lag[sh["id"]] = (r - d) if (d and r) else timedelta(0)

    # -- the generator's own p_delay pipeline, with `st` supplied ------------

    def _port_bump(self, sh, st):
        if sh["carrier"] in self.SEA:
            for _m, s0, _pk, s1, _mag in self.PORT_EVENTS:
                if s0 <= sh["dispatched_at"] <= s1:
                    st = min(0.95, st + 0.10)
        return st

    def realise_supplier(self, sup_id: str, st_of, abs_of, probe=None):
        """Rewrite this supplier's shipments in place, return its `(t0 -> feature tuple)`.

        Restores the factual dicts before returning, so the namespace is reusable for the
        next world -- the alternative, deep-copying ~40k shipment dicts per world, is what
        this avoids.
        """
        ships = self.sup_ship.get(sup_id, [])
        if not ships:
            return ([], probe(sup_id)) if probe else []
        saved = [(sh["status"], sh["delivered_at"], sh["delivered_rec"]) for sh in ships]
        ab = abs_of(sup_id)
        for sh in ships:
            u, g, e, j = self.draws[self.ship_index[sh["id"]]]
            st = self._port_bump(sh, st_of(sup_id, sh))
            p = min(0.80, 0.025 + 0.38 * st * ab)
            delayed = u < p
            if delayed:
                actual = sh["eta"] + timedelta(days=max(1, int(g * (1 + 2 * st))))
            else:
                actual = sh["eta"] - timedelta(hours=e)
            actual = actual + timedelta(seconds=j)          # J(), see draw_table
            if actual <= self.T_END:
                sh["status"], sh["delivered_at"] = "delivered", actual
                sh["delivered_rec"] = actual + self.rec_lag[sh["id"]]
            else:
                sh["status"] = "delayed" if delayed else "in_transit"
                sh["delivered_at"] = sh["delivered_rec"] = None
        srow = self.sup_by_id[sup_id]
        rows = [(t0, self.sup_features(srow, t0)) for t0 in self.T0S]
        probed = probe(sup_id) if probe else None
        for sh, (stt, da, dr) in zip(ships, saved):
            sh["status"], sh["delivered_at"], sh["delivered_rec"] = stt, da, dr
        return (rows, probed) if probe else rows


# ---------------------------------------------------------------------------
# the do(Z) definitions
# ---------------------------------------------------------------------------

def _median_split_levels(values: list) -> tuple:
    """`(low_level, high_level)` = the medians of the two halves of a median split.

    This is deliberately the SAME contrast Model A is scored on: Phase 1 binarises every
    continuous state at its train median, so a head at AUC 0.65 is separating exactly these
    two populations. Using quartiles or extremes here would make the identifiability number
    answer a strictly easier question than the recoverability number it is compared against.
    """
    v = sorted(values)
    m = v[len(v) // 2]
    lo = [x for x in v if x <= m] or v
    hi = [x for x in v if x > m] or v
    return lo[len(lo) // 2], hi[len(hi) // 2]


class DoSpec:
    """`(state, world)` -> the `(st_of, abs_of)` pair and the rows that are in scope."""

    def __init__(self, r: Realiser, state: str):
        self.r, self.state = r, state
        self.ns = r.ns
        ids = sorted(s["id"] for s in r.suppliers
                     if s["id"] in r.visible and r.sup_ship.get(s["id"]))
        self.sup_ids = ids
        self.note = ""

        if state == "supply_stress":
            vals = [r.stress(sid, r.sup_by_id[sid]["base_rel"], t0)
                    for sid in ids for t0 in r.T0S]
            self.lo, self.hi = _median_split_levels(vals)
            self.scope = {sid: set(range(len(r.T0S))) for sid in ids}

        elif state == "supplier_reliability":
            if not any(r.IDIO.get(s) for s in ids):
                raise ValueError("no IDIO outages in this world")
            self.lo, self.hi = None, None          # do() is structural, not a level
            # In scope: exactly the (supplier, t0) rows where the outage is ACTIVE at t0 --
            # Model A's target definition, verbatim.
            self.scope = {}
            for sid in ids:
                ev = r.IDIO.get(sid)
                if not ev:
                    continue
                act = {i for i, t0 in enumerate(r.T0S) if ev[0] <= t0 <= ev[2]}
                if act:
                    self.scope[sid] = act

        elif state == "recovery_capability":
            if not r.RESILIENCE:
                raise ValueError("Mechanism E is off in this world; RESILIENCE is empty")
            self.lo, self.hi = _median_split_levels(
                [r.RESILIENCE[s] for s in ids if s in r.RESILIENCE])
            self.scope = {sid: set(range(len(r.T0S))) for sid in ids
                          if sid in r.RESILIENCE}
        else:
            raise ValueError(state)

    @contextlib.contextmanager
    def world(self, sup_id: str, level: str):
        """Install do(Z=level) for ONE supplier, yield `(st_of, abs_of)`, then restore.

        `level` is `"lo"`/`"hi"` for the two contrasted worlds. The mutation happens in the
        generator's own globals, so `stress()` and `absorption()` -- the real functions, not
        copies -- compute the counterfactual themselves.
        """
        r = self.r
        if self.state == "supply_stress":
            z = self.lo if level == "lo" else self.hi
            yield (lambda sid, sh: z), r.absorption
            return

        if self.state == "supplier_reliability":
            saved = r.IDIO.get(sup_id)
            if level == "lo":
                r.IDIO[sup_id] = None
            try:
                yield ((lambda sid, sh: r.stress(sid, r.sup_by_id[sid]["base_rel"],
                                                 sh["dispatched_at"])), r.absorption)
            finally:
                r.IDIO[sup_id] = saved
            return

        saved = r.RESILIENCE.get(sup_id)
        r.RESILIENCE[sup_id] = self.lo if level == "lo" else self.hi
        try:
            yield ((lambda sid, sh: r.stress(sid, r.sup_by_id[sid]["base_rel"],
                                             sh["dispatched_at"])), r.absorption)
        finally:
            if saved is None:
                r.RESILIENCE.pop(sup_id, None)
            else:
                r.RESILIENCE[sup_id] = saved


# ---------------------------------------------------------------------------
# observation matrix
# ---------------------------------------------------------------------------

def featurise(feat) -> np.ndarray:
    """Feature tuple -> a fixed-width vector with explicit missingness indicators.

    `sup_features` returns `None` whenever a window holds fewer than 3 deliveries (or fewer
    than 5, or fewer than 2 late ones). That missingness is itself observable and is itself
    moved by Z -- an outage that pushes deliveries out of a 30-day window changes whether
    `on_time_rate_30d` exists at all -- so it is encoded rather than imputed away.
    """
    vals, miss = [], []
    for i, v in enumerate(feat):
        if i < N_NULLABLE:
            miss.append(0.0 if v is None else 1.0)
            vals.append(0.0 if v is None else float(v))
        else:
            vals.append(float(v))
    return np.asarray(vals + miss, dtype=float)


def collect(spec: DoSpec, level: str) -> dict:
    """`{(sup_id, t0_index): feature vector}` for one world."""
    out = {}
    for sid, idxs in spec.scope.items():
        with spec.world(sid, level) as (st_of, abs_of):
            rows = spec.r.realise_supplier(sid, st_of, abs_of)
        for i, (_t0, feat) in enumerate(rows):
            if i in idxs:
                out[(sid, i)] = featurise(feat)
    return out


def contrast(A: dict, B: dict, init_seeds: list[int], split_seed: int) -> dict:
    """Divergence rate + classifier AUC for "which world did this row come from".

    Both are needed. The divergence rate is exact and answers the in-principle question; the
    AUC answers whether the difference is large enough and consistent enough that a model
    could act on it.
    """
    keys = sorted(set(A) & set(B))
    if not keys:
        return {"n_rows": 0}
    XA = np.stack([A[k] for k in keys])
    XB = np.stack([B[k] for k in keys])
    diff = ~np.all(np.isclose(XA, XB, rtol=0, atol=1e-12), axis=1)

    X = np.concatenate([XA, XB], axis=0)
    y = np.concatenate([np.zeros(len(XA)), np.ones(len(XB))])
    # Supplier-disjoint split, so the head cannot lean on identity even in principle.
    sups = sorted({k[0] for k in keys})
    rng = random.Random(split_seed)
    rng.shuffle(sups)
    train_sup = set(sups[:max(1, int(0.6 * len(sups)))])
    m = np.array([k[0] in train_sup for k in keys])
    m2 = np.concatenate([m, m])
    if m2.sum() == 0 or (~m2).sum() == 0:
        return {"n_rows": len(keys), "divergence_rate": float(diff.mean())}

    aucs = [train_head(X[m2], y[m2], X[~m2], y[~m2], s) for s in init_seeds]
    aucs = [a for a in aucs if a is not None]
    return {
        "n_rows": int(len(keys)),
        "n_divergent_rows": int(diff.sum()),
        "divergence_rate": float(diff.mean()),
        "mean_abs_feature_delta": float(np.abs(XA - XB).mean()),
        "n_train": int(m2.sum()), "n_test": int((~m2).sum()),
        "auc_mean": statistics.fmean(aucs) if aucs else None,
        "auc_all": aucs,
        "init_seed_spread": (max(aucs) - min(aucs)) if len(aucs) > 1 else None,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def harness_check(r: Realiser) -> dict:
    """Does re-realisation at FACTUAL Z reproduce the factual observable distribution?

    Run before any intervention. The re-realiser consumes its own RNG, so it cannot and should
    not reproduce the factual outcomes shipment-by-shipment -- but if it is a faithful
    reimplementation of `new_shipment`'s tail, the *distribution* of the emitted feature must
    match. If it did not, every do(Z) number below would be measuring the harness rather than
    the world.
    """
    def late_rate(sid):
        """Counted INSIDE the realised world. `realise_supplier` restores the factual dicts
        before it returns, so counting afterwards would silently re-measure the factual world
        -- which is what an earlier version of this function did."""
        d = n = 0
        for sh in r.sup_ship[sid]:
            if sh["delivered_at"]:
                n += 1
                d += int(sh["delivered_at"] > sh["eta"])
        return d, n

    def pool(realise: bool):
        vals, delayed, n = [], 0, 0
        st_fac = (lambda s, sh: r.stress(s, r.sup_by_id[s]["base_rel"], sh["dispatched_at"]))
        for sid in sorted(r.sup_ship):
            if sid not in r.visible:
                continue
            if realise:
                rows, counts = r.realise_supplier(sid, st_fac, r.absorption,
                                                  probe=late_rate)
                d, nn = counts
            else:
                srow = r.sup_by_id[sid]
                rows = [(t0, r.sup_features(srow, t0)) for t0 in r.T0S]
                d, nn = late_rate(sid)
            delayed += d
            n += nn
            for _t0, f in rows:
                if f[2] is not None:
                    vals.append(float(f[2]))
        return vals, (delayed / n if n else float("nan")), n

    fv, fr, fn = pool(False)
    rv, rr, rn = pool(True)
    return {"factual_mean_on_time_180d": float(np.mean(fv)), "factual_late_rate": fr,
            "factual_n_delivered": fn, "factual_n_rows": len(fv),
            "realised_mean_on_time_180d": float(np.mean(rv)), "realised_late_rate": rr,
            "realised_n_delivered": rn, "realised_n_rows": len(rv),
            "abs_diff_mean_on_time_180d": abs(float(np.mean(fv)) - float(np.mean(rv))),
            "abs_diff_late_rate": abs(fr - rr)}


def run_seed(variant: str, dseed: int, config: str, states: list[str],
             init_seeds: list[int], realisation_seeds: tuple) -> dict:
    ns = generator_namespace(variant, dseed, config)
    r_main = Realiser(ns, realisation_seeds[0])
    r_null = Realiser(ns, realisation_seeds[1])

    out = {"_harness_check": harness_check(r_main)}
    h = out["_harness_check"]
    print(f"  harness: factual on_time_180d {h['factual_mean_on_time_180d']:.4f} / late rate "
          f"{h['factual_late_rate']:.4f}   re-realised {h['realised_mean_on_time_180d']:.4f} / "
          f"{h['realised_late_rate']:.4f}", flush=True)
    for state in states:
        try:
            spec_main = DoSpec(r_main, state)
            spec_null = DoSpec(r_null, state)
        except ValueError as exc:
            out[state] = {"unavailable": str(exc)}
            print(f"  {state:<22} unavailable: {exc}", flush=True)
            continue

        lo_main = collect(spec_main, "lo")
        hi_main = collect(spec_main, "hi")
        # The NULL: same do(Z=lo) on both sides, different realisation seed. Every source of
        # apparent separation except Z itself is still present, which is the point.
        lo_null = collect(spec_null, "lo")

        signal = contrast(lo_main, hi_main, init_seeds, split_seed=dseed)
        null = contrast(lo_main, lo_null, init_seeds, split_seed=dseed)
        cell = {"signal": signal, "null": null,
                "z_lo": spec_main.lo, "z_hi": spec_main.hi,
                "n_suppliers_in_scope": len(spec_main.scope)}

        if state == "supplier_reliability":
            # A strictly MORE GENEROUS arm, added because the scoped arm above answers
            # "is the outage visible at the snapshot where the label says it is active",
            # and a null there could equally mean "visible, but later". This one widens the
            # scope to every t0 in the timeline for every outaged supplier, so an outage whose
            # observable consequence lands two snapshots after the window still counts. If
            # this arm is also flat, "not observable at the labelled t0" hardens into
            # "not observable at all".
            wide = DoSpec(r_main, state)
            wide.scope = {sid: set(range(len(r_main.T0S))) for sid in wide.scope}
            w_lo, w_hi = collect(wide, "lo"), collect(wide, "hi")
            cell["signal_all_t0"] = contrast(w_lo, w_hi, init_seeds, split_seed=dseed)
            # How much room the intervention ever had: shipments actually dispatched inside
            # the outage window are the only channel it can act through.
            inwin = [sum(1 for sh in r_main.sup_ship[sid]
                         if r_main.IDIO[sid][0] <= sh["dispatched_at"] <= r_main.IDIO[sid][2])
                     for sid in wide.scope]
            cell["outage_exposure"] = {
                "n_outaged_suppliers_in_scope": len(inwin),
                "mean_shipments_dispatched_in_window": float(np.mean(inwin)) if inwin else 0.0,
                "frac_with_any_shipment_in_window": (
                    float(np.mean([x > 0 for x in inwin])) if inwin else 0.0)}
            w = cell["signal_all_t0"]
            print(f"  {'  (all-t0 arm)':<22} rows {w.get('n_rows',0):>6,}  "
                  f"divergence {w.get('divergence_rate', float('nan')):.3f}  "
                  f"AUC {(w.get('auc_mean') or float('nan')):.4f}   "
                  f"shipments dispatched in outage window: mean "
                  f"{cell['outage_exposure']['mean_shipments_dispatched_in_window']:.2f}, "
                  f"{cell['outage_exposure']['frac_with_any_shipment_in_window']:.0%} of "
                  f"suppliers have >=1", flush=True)
        out[state] = cell
        s, n = signal, null
        print(f"  {state:<22} rows {s.get('n_rows',0):>6,}  "
              f"divergence {s.get('divergence_rate', float('nan')):.3f} "
              f"(null {n.get('divergence_rate', float('nan')):.3f})  "
              f"AUC {(s.get('auc_mean') or float('nan')):.4f} "
              f"(null {(n.get('auc_mean') or float('nan')):.4f})", flush=True)
    return out


def summarise(res: dict) -> dict:
    summary = {}
    for state in res["states"]:
        cells = [res["per_seed"][str(d)][state] for d in res["dataset_seeds"]
                 if state in res["per_seed"].get(str(d), {})
                 and "unavailable" not in res["per_seed"][str(d)][state]]
        if not cells:
            summary[state] = {"unavailable": True}
            continue
        sig = [c["signal"] for c in cells]
        nul = [c["null"] for c in cells]
        sa = [c["auc_mean"] for c in sig if c.get("auc_mean") is not None]
        na = [c["auc_mean"] for c in nul if c.get("auc_mean") is not None]
        if not sa:
            summary[state] = {"unavailable": True}
            continue
        init_floor = max([c["init_seed_spread"] for c in sig
                          if c.get("init_seed_spread") is not None] or [float("nan")])
        dseed_floor = (max(sa) - min(sa)) if len(sa) > 1 else float("nan")
        # The null arm's own excursion from 0.5 is a third floor and the only one that
        # measures the pipeline rather than the seeds. Gate on the largest of the three.
        null_excursion = max(abs(a - 0.5) for a in na) if na else float("nan")
        floor = max(f for f in (init_floor, dseed_floor, null_excursion) if f == f)
        mean = statistics.fmean(sa)
        summary[state] = {
            "n_dataset_seeds": len(sa),
            "divergence_rate": statistics.fmean([c["divergence_rate"] for c in sig]),
            "divergence_rate_null": statistics.fmean([c["divergence_rate"] for c in nul]),
            "mean_abs_feature_delta": statistics.fmean(
                [c["mean_abs_feature_delta"] for c in sig]),
            "rows_per_seed": [c["n_rows"] for c in sig],
            "auc_mean": mean, "auc_per_dataset_seed": sa,
            "null_auc_mean": statistics.fmean(na) if na else None,
            "null_auc_per_dataset_seed": na,
            "init_seed_floor": init_floor, "dataset_seed_floor": dseed_floor,
            "null_excursion_floor": null_excursion, "reproduction_floor": floor,
            "above_chance_by": mean - 0.5,
            "clears_floor": (mean - 0.5) > floor,
            "sign_consistent": all(a > 0.5 for a in sa) or all(a < 0.5 for a in sa),
            "z_lo": cells[0]["z_lo"], "z_hi": cells[0]["z_hi"],
        }
        summary[state]["identifiable"] = bool(summary[state]["clears_floor"]
                                              and summary[state]["sign_consistent"])
        wide = [c["signal_all_t0"] for c in cells if c.get("signal_all_t0")]
        if wide:
            wa = [c["auc_mean"] for c in wide if c.get("auc_mean") is not None]
            summary[state]["all_t0_arm"] = {
                "divergence_rate": statistics.fmean([c["divergence_rate"] for c in wide]),
                "rows_per_seed": [c["n_rows"] for c in wide],
                "auc_mean": statistics.fmean(wa) if wa else None,
                "auc_per_dataset_seed": wa,
            }
        exp = [c["outage_exposure"] for c in cells if c.get("outage_exposure")]
        if exp:
            summary[state]["outage_exposure"] = {
                k: statistics.fmean([e[k] for e in exp]) for k in exp[0]}
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="E")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--config", default="v1")
    ap.add_argument("--states", default=",".join(STATES))
    ap.add_argument("--init-seeds", default="0,1,2,3,4")
    ap.add_argument("--realisation-seeds", default="1,2")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dseeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    iseeds = [int(s) for s in a.init_seeds.split(",") if s.strip()]
    rseeds = tuple(int(s) for s in a.realisation_seeds.split(","))
    states = [s.strip() for s in a.states.split(",") if s.strip()]

    res = {"variant": a.variant, "config": a.config, "dataset_seeds": dseeds,
           "init_seeds": iseeds, "realisation_seeds": list(rseeds), "states": states,
           "per_seed": {}}
    for d in dseeds:
        print(f"\n=== variant {a.variant} seed {d} ===", flush=True)
        t = time.time()
        res["per_seed"][str(d)] = run_seed(a.variant, d, a.config, states, iseeds, rseeds)
        print(f"  ({time.time() - t:.0f}s)", flush=True)
    res["summary"] = summarise(res)

    print("\n" + "=" * 126)
    print(f"STEP 2A — IDENTIFIABILITY under do(Z), variant {a.variant}, "
          f"{len(dseeds)} dataset seeds x {len(iseeds)} init seeds")
    print("=" * 126)
    hdr = (f"{'state':<24}{'rows':>8}{'diverge':>9}{'null dv':>9}{'AUC':>9}{'null AUC':>10}"
           f"{'floor':>9}{'above .5':>10}{'clears':>8}{'sign':>6}{'IDENT':>7}")
    print(hdr); print("-" * len(hdr))
    for st in states:
        s = res["summary"].get(st, {})
        if s.get("unavailable"):
            print(f"{st:<24}{'unavailable on this variant':>60}")
            continue
        print(f"{st:<24}{sum(s['rows_per_seed']):>8,}{s['divergence_rate']:>9.3f}"
              f"{s['divergence_rate_null']:>9.3f}{s['auc_mean']:>9.4f}"
              f"{s['null_auc_mean']:>10.4f}{s['reproduction_floor']:>9.4f}"
              f"{s['above_chance_by']:>+10.4f}"
              f"{('YES' if s['clears_floor'] else 'no'):>8}"
              f"{('yes' if s['sign_consistent'] else 'NO'):>6}"
              f"{('YES' if s['identifiable'] else 'NO'):>7}")
    print("-" * len(hdr))

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
