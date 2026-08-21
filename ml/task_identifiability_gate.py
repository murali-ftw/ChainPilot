#!/usr/bin/env python3
"""
Layer 3 v2, PHASE 0 -- two-tier identifiability gate, one invocation per task.

**One gate per task, never a shared gate feeding a three-way fork.** Identifiability is
per-target: Supplier Reliability failed this exact test (AUC ~0.50, Case 3 / STOP-C) while
Supply Stress and Recovery Capability passed it, and a shared gate would have let one task's
outcome leak into another's.

---

**What is being tested, and what is deliberately NOT being tested.**

The operational test is a controlled intervention, `P(X | do(Z=z1)) != P(X | do(Z=z2))`, judged
against an **empirically measured null** rather than against 0.5, and required to reproduce
across independent dataset seeds.

This phase does **not** ask whether SHARE already contains related information. SHARE may
already encode part or all of the relevant signal; that overlap is exactly what Phase 2's
`P(Y|X,H,Z)` vs `P(Y|X,H)` comparison exists to measure. Folding it in here would close a task
early because SHARE carries *related* information, even where the task's own residual signal is
genuinely identifiable.

---

**Two tiers, because `Z_t` is defined (§2) as a post-hoc learned residual with no required
single interpretable referent -- and there is no lever to set an untrained encoder's output to
`z1` or `z2` before the encoder exists.**

**Tier A -- named-mechanism tier.** Where the task's residual is hypothesised to trace to a
specific named generator mechanism, run the existing CRN protocol on that mechanism directly.
`Realiser`, `draw_table` and `_median_split_levels` are imported **unmodified** from
`ml/identifiability_check.py`; the only extension is that the readout is the task label `Y_t`
rather than the supplier observable row.

**Tier B -- unconstrained-residual tier.** Where no single named mechanism is hypothesised,
perturb the generator's upstream stochastic mechanisms **broadly** -- the relevant joint set for
the task, globally over all suppliers, not one named variable -- and ask whether a classifier can
detect a reproducible, above-null shift in `Y_t` that `(X, H)` alone cannot explain. The residual
construction is: fit `g: (X, X_nbr) -> Y` pooled over both worlds out-of-fold, take `r = Y - g`,
and test whether the world label is recoverable **from the residual**. If `(X, X_nbr)` accounted
for the shift, `r` would carry no world information.

`X_nbr` is the entity's graph-neighbourhood observable block, standing in for what one round of
message passing can see. Conditioning on the emitted observables plus their neighbourhood is the
tightest conditioning available without re-running SHARE inside each counterfactual world, and
the residual direction is stated in the report: conditioning on *less* than the true `H` makes a
Tier B positive **easier**, which is why a Tier B pass is licence to attempt the encoder and
never confirmation that it will succeed. Phase 2 is what confirms that.

---

**Label recomputation is validated exactly, not distributionally.** Mechanism G is off on both
testbeds (`report_delay()` returns 0), so recorded time equals true time and `asof_status`
reduces to the true-time status. The label rule reimplemented here is therefore checked
**row-for-row against the generator's own `label_rows`** at factual Z before any intervention
number is trusted -- a strictly stronger check than the distributional one
`identifiability_check.py::harness_check` uses, and available only because the labels are a
deterministic function of state the namespace still holds.

    python3 ml/task_identifiability_gate.py --variant A --tasks delay,impact \
        --seeds 42,43,44,45,46 --out out/layer3_v2/phase0_gate_A.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import time
from datetime import timedelta

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.confirm_latent_states import namespace                     # noqa: E402
from ml.hypothesis_ranker import roc_auc                           # noqa: E402
from ml.identifiability_check import (                             # noqa: E402
    Realiser, _median_split_levels, featurise,
)
from ml.latent_state_head import train_head                        # noqa: E402
from ml.layer3_sufficiency import fit_probs                        # noqa: E402

TASKS = ("delay", "shortage", "impact")


# ---------------------------------------------------------------------------
# realise many suppliers at once, and read task labels out of the open world
# ---------------------------------------------------------------------------

class TaskRealiser(Realiser):
    """`Realiser`, extended to hold MANY suppliers realised simultaneously.

    `Realiser.realise_supplier` restores the factual dicts before returning, which is right for
    a per-supplier observable readout and wrong here: a task label is computed across every
    supplier's shipments at once, so the world has to stay open while it is read. Subclassed
    rather than edited so `ml/identifiability_check.py` keeps the digest every prior report
    records.
    """

    def __init__(self, ns: dict, realisation_seed: int):
        super().__init__(ns, realisation_seed)
        self.HORIZON = ns["HORIZON"]
        self.label_rows = ns["label_rows"]
        self.DELAY_SAMPLE = ns["DELAY_SAMPLE"]
        self.VISIBLE = self.visible
        # Factual delayed-event times, read from the generator's own transition log. Shipments
        # that no intervention touches keep these, so only re-realised suppliers move.
        self.factual_delay_ev = {}
        for sid, trs in ns["trans_by_sid"].items():
            ev = next((at for (_s, stat, _p, at, _r) in trs if stat == "delayed"), None)
            if ev is not None:
                self.factual_delay_ev[sid] = ev
        self.all_ships = ns["shipments"]

    @contextlib.contextmanager
    def open_world(self, sup_ids, st_of, abs_of):
        """Rewrite every listed supplier's shipments and keep them rewritten inside the block.

        Yields `{shipment_id: delayed_event_time_or_None}` for the rewritten shipments, so the
        label reader can distinguish "delayed" from "delivered late for another reason".
        """
        saved, ev = {}, {}
        try:
            for sid in sup_ids:
                ships = self.sup_ship.get(sid, [])
                if not ships:
                    continue
                ab = abs_of(sid)
                for sh in ships:
                    saved[sh["id"]] = (sh["status"], sh["delivered_at"], sh["delivered_rec"])
                    u, g, e, j = self.draws[self.ship_index[sh["id"]]]
                    st = self._port_bump(sh, st_of(sid, sh))
                    p = min(0.80, 0.025 + 0.38 * st * ab)
                    delayed = u < p
                    if delayed:
                        actual = sh["eta"] + timedelta(days=max(1, int(g * (1 + 2 * st))))
                    else:
                        actual = sh["eta"] - timedelta(hours=e)
                    actual = actual + timedelta(seconds=j)
                    # The generator logs `delayed` at J(eta + 6h) and only when eta <= T_END
                    # (db/generate_dataset.py). Reproduced exactly, including the guard.
                    ev[sh["id"]] = (sh["eta"] + timedelta(hours=6)
                                    if (delayed and sh["eta"] <= self.T_END) else None)
                    if actual <= self.T_END:
                        sh["status"], sh["delivered_at"] = "delivered", actual
                        sh["delivered_rec"] = actual + self.rec_lag[sh["id"]]
                    else:
                        sh["status"] = "delayed" if delayed else "in_transit"
                        sh["delivered_at"] = sh["delivered_rec"] = None
            yield ev
        finally:
            for sh in self.all_ships:
                if sh["id"] in saved:
                    sh["status"], sh["delivered_at"], sh["delivered_rec"] = saved[sh["id"]]

    # -- the label rule -------------------------------------------------

    def _asof_status(self, sh, t0, ev_map: dict | None = None):
        """Status a model may believe at t0. G is off on both testbeds, so recorded == true
        and this is the true-time status of the last transition at or before t0.

        `ev_map` supplies re-realised delayed-event times; a shipment absent from it keeps its
        factual one. Passing it matters: under an intervention the as-of status is itself part
        of what an observer sees move, so reading the factual status inside an intervened world
        would understate the observable shift.
        """
        ev_map = ev_map if ev_map is not None else {}
        st = ""
        if sh["created_at"] <= t0:
            st = "scheduled"
        if sh["dispatched_at"] <= t0:
            st = "in_transit"
        ev = ev_map.get(sh["id"], self.factual_delay_ev.get(sh["id"]))
        if ev is not None and ev <= t0:
            st = "delayed"
        if sh["delivered_at"] and sh["delivered_at"] <= t0:
            st = "delivered"
        return st

    def task_labels_for(self, sup_ids, override_ev: dict | None = None) -> dict:
        """Labels for ONE supplier set's own entities, walking only their shipments.

        Exact rather than approximate, and that is a property of the label definitions rather
        than a convenience: `impact(s,t0)` is "did any of s's OWN shipments log a delay in the
        horizon", and `delay(sh,t0)` depends on `sh` alone. Neither reads another supplier's
        shipments, so scoping the walk to the intervened supplier changes no label -- which is
        what makes Tier A's per-entity isolation computable at all. The global walk is O(all
        shipments x 15 t0s) and running it once per supplier would be ~470M iterations.
        """
        override_ev = override_ev or {}
        delay, impact = {}, {}
        for sid in sup_ids:
            ships = self.sup_ship.get(sid, [])
            if not ships:
                continue
            for i, t0 in enumerate(self.T0S):
                hz = t0 + timedelta(days=self.HORIZON)
                hit = False
                for sh in ships:
                    if sh["created_at"] > t0:
                        continue
                    if self._asof_status(sh, t0, override_ev) not in ("scheduled", "in_transit"):
                        continue
                    ev = (override_ev.get(sh["id"]) if sh["id"] in override_ev
                          else self.factual_delay_ev.get(sh["id"]))
                    lab = ev is not None and t0 < ev <= hz
                    hit = hit or lab
                    if sh["id"] in self.DELAY_SAMPLE:
                        delay[(sh["id"], i)] = float(lab)
                if sid in self.visible:
                    impact[(sid, i)] = float(hit)
        return {"delay": delay, "impact": impact}

    def task_labels(self, override_ev: dict | None = None) -> dict:
        """`{"delay": {(ship_id,t0_idx): y}, "impact": {(sup_id,t0_idx): y}}`.

        `override_ev` supplies re-realised delayed-event times; every other shipment keeps its
        factual one. Impact is read off EVERY eligible shipment, sampled or not -- the
        generator's own comment, and the reason the delay sample must not thin impact with it.
        """
        override_ev = override_ev or {}
        delay, impact = {}, {}
        for i, t0 in enumerate(self.T0S):
            hz = t0 + timedelta(days=self.HORIZON)
            sup_hit = set()
            for sh in self.all_ships:
                if sh["created_at"] > t0:
                    continue
                if self._asof_status(sh, t0, override_ev) not in ("scheduled", "in_transit"):
                    continue
                ev = (override_ev.get(sh["id"]) if sh["id"] in override_ev
                      else self.factual_delay_ev.get(sh["id"]))
                lab = ev is not None and t0 < ev <= hz
                if lab and sh["supplier_id"]:
                    sup_hit.add(sh["supplier_id"])
                if sh["id"] in self.DELAY_SAMPLE:
                    delay[(sh["id"], i)] = float(lab)
            for sid in self.visible:
                impact[(sid, i)] = float(sid in sup_hit)
        return {"delay": delay, "impact": impact}


def label_rule_check(r: TaskRealiser) -> dict:
    """Row-for-row agreement between the reimplemented label rule and the generator's own
    `label_rows`, at factual Z with nothing re-realised.

    This is the strong form of the harness check: not "does the distribution match" but "is
    every single label identical". If it is not, every intervention number below would be
    measuring the reimplementation.
    """
    got = r.task_labels()
    t0_index = {t0: i for i, t0 in enumerate(r.T0S)}
    by_snap = {}
    for t0 in r.T0S:
        by_snap[t0.isoformat()] = t0_index[t0]
    want = {"delay": {}, "impact": {}}
    snap_of = {}
    for t0 in r.T0S:
        snap_of[r.ns["uid"]("snap", t0)] = t0_index[t0]
    for row in r.label_rows:
        task = row[4]
        if task not in ("delay", "impact"):
            continue
        i = snap_of.get(row[1])
        if i is None:
            continue
        want[task][(row[3], i)] = 1.0 if row[5] == "true" else 0.0

    out = {}
    for task in ("delay", "impact"):
        keys = sorted(set(got[task]) & set(want[task]))
        mism = [k for k in keys if got[task][k] != want[task][k]]
        out[task] = {
            "n_generator_rows": len(want[task]), "n_recomputed_rows": len(got[task]),
            "n_compared": len(keys), "n_mismatches": len(mism),
            "exact": len(mism) == 0 and len(keys) == len(want[task]),
            "generator_positive_rate": (statistics.fmean(want[task].values())
                                        if want[task] else float("nan")),
            "recomputed_positive_rate": (statistics.fmean([got[task][k] for k in keys])
                                         if keys else float("nan")),
        }
    return out


# ---------------------------------------------------------------------------
# observables: X for the task entity, and its neighbourhood block
# ---------------------------------------------------------------------------

def observables(r: TaskRealiser, task: str, sup_ids=None, ev_map: dict | None = None) -> dict:
    """`{(entity_id, t0_idx): vector}` -- the emitted observable row plus a neighbourhood block,
    **read from whatever world is currently open**.

    For `delay` the entity is a Shipment and `X` mirrors what `ml/data/loader.py` builds for it
    (days-to-eta, days-since-dispatch, sea flag, as-of status one-hot); the neighbourhood block
    is the origin supplier's own `sup_features` row, which is what one hop of message passing
    would reach. For `impact` the entity is a Supplier and `X` is `sup_features` itself, with the
    neighbourhood block aggregated over its co-parents.

    This MUST be called inside the intervened world, not outside it. `do(Z)` moves the
    observables as well as the labels, and the whole point of the residual test is to ask
    whether `Y` moves by more than `X` does -- which is unanswerable if `X` is held factual.
    """
    sup_ids = sorted(r.visible) if sup_ids is None else list(sup_ids)
    ev_map = ev_map if ev_map is not None else {}
    sup_feat = {}
    for sid in sup_ids:
        srow = r.sup_by_id.get(sid)
        if srow is None:
            continue
        for i, t0 in enumerate(r.T0S):
            sup_feat[(sid, i)] = featurise(r.sup_features(srow, t0))

    out = {}
    if task == "impact":
        coparents = r.ns.get("coparents", {})
        width = len(next(iter(sup_feat.values()))) if sup_feat else 0
        for (sid, i), v in sup_feat.items():
            # Co-parents are only realised in the global (Tier B) world; under Tier A's
            # per-supplier isolation their rows are factual, which is correct -- the
            # intervention deliberately did not touch them.
            partners = [p for p in coparents.get(sid, ()) if (p, i) in sup_feat]
            nbr = (np.mean([sup_feat[(p, i)] for p in partners], axis=0)
                   if partners else np.zeros(width))
            out[(sid, i)] = np.concatenate([v, nbr, [float(len(partners))]])
        return out

    statuses = ("scheduled", "in_transit", "delayed", "delivered", "")
    width = len(next(iter(sup_feat.values()))) if sup_feat else 0
    for sid in sup_ids:
        for sh in r.sup_ship.get(sid, []):
            if sh["id"] not in r.DELAY_SAMPLE:
                continue
            for i, t0 in enumerate(r.T0S):
                if sh["created_at"] > t0:
                    continue
                st = r._asof_status(sh, t0, ev_map)
                if st not in ("scheduled", "in_transit"):
                    continue
                x = [(sh["eta"] - t0).total_seconds() / 86400.0,
                     ((t0 - sh["dispatched_at"]).total_seconds() / 86400.0
                      if sh["dispatched_at"] <= t0 else 0.0),
                     float(sh["carrier"] in r.SEA)]
                x += [float(st == s_) for s_ in statuses]
                nbr = sup_feat.get((sid, i))
                out[(sh["id"], i)] = np.concatenate(
                    [x, nbr if nbr is not None else np.zeros(width)])
    return out


# ---------------------------------------------------------------------------
# Tier A and Tier B
# ---------------------------------------------------------------------------

def _stress_levels(r: TaskRealiser) -> tuple:
    vals = [r.stress(sid, r.sup_by_id[sid]["base_rel"], t0)
            for sid in sorted(r.visible) if r.sup_ship.get(sid) for t0 in r.T0S]
    return _median_split_levels(vals)


def tier_a_worlds(r: TaskRealiser, task: str) -> tuple:
    """do(stress = z_lo) vs do(stress = z_hi) on the ONE named mechanism, **isolated to one
    supplier at a time** -- `ml/identifiability_check.py`'s protocol exactly.

    Isolation is not cosmetic there and is not here: `stress(s,t)` reads its co-parents'
    `own_stress`, so a global edit would move every partner too and the measured effect would be
    contaminated by a second causal path -- which could only ever flatter identifiability.
    Because a supplier's own shipments depend on that supplier's `stress` and `absorption` and
    nothing else, and both task labels read only the entity's own shipments, overriding one
    supplier at a time is exact.

    Levels are the medians of the two halves of a median split -- deliberately the same contrast
    every Model A head in this project is scored on.
    """
    lo, hi = _stress_levels(r)
    sup_ids = [s for s in sorted(r.visible) if r.sup_ship.get(s)]
    out = []
    for z in (lo, hi):
        lab, obs = {}, {}
        for sid in sup_ids:
            with r.open_world([sid], (lambda s_, sh: z), r.absorption) as ev:
                lab.update(r.task_labels_for([sid], ev)[task])
                obs.update(observables(r, task, [sid], ev))
        out.append((lab, obs))
    return out[0], out[1], {"z_lo": lo, "z_hi": hi, "isolation": "one supplier at a time"}


def tier_b_worlds(r: TaskRealiser, task: str, direction: str = "both") -> tuple:
    """Broad JOINT perturbation of the task's upstream stochastic set, globally.

    Not one named variable: the whole joint set that feeds this task's outcome -- every
    supplier's stress (which already aggregates EVENTS, IDIO, co-parent bleed, hidden-parent
    coupling and J's upstream chains), its absorption via RESILIENCE where Mechanism E is on,
    and the PORT_EVENTS carrier bump -- moved together, low against high.
    """
    lo, hi = _stress_levels(r)
    sup_ids = [s for s in sorted(r.visible) if r.sup_ship.get(s)]
    RES = r.RESILIENCE
    res_lo = res_hi = None
    if RES:
        res_lo, res_hi = _median_split_levels([RES[s] for s in RES])

    def build(level):
        z = lo if level == "lo" else hi
        saved = dict(RES) if RES else None
        if RES:
            for s in RES:
                RES[s] = res_lo if level == "lo" else res_hi
        try:
            # The port bump rides along inside `_port_bump`, which `open_world` already applies.
            with r.open_world(sup_ids, (lambda sid, sh: z), r.absorption) as ev:
                return r.task_labels(ev)[task], observables(r, task, sup_ids, ev)
        finally:
            if RES:
                RES.clear()
                RES.update(saved)

    # How many members of the "joint set" are actually live on this variant. With Mechanism E
    # off, RESILIENCE is empty and `absorption()` returns 1.0, so the joint perturbation
    # DEGENERATES to the stress perturbation alone and Tier B coincides with a global Tier A.
    # Recorded rather than left for the reader to infer from equal numbers.
    live = ["stress"] + (["resilience/absorption"] if RES else []) + ["port_events"]
    return build("lo"), build("hi"), {
        "z_lo": lo, "z_hi": hi, "resilience_lo": res_lo, "resilience_hi": res_hi,
        "joint_set_live_members": live, "n_live": len(live),
        "degenerate_to_stress_only": not RES,
        "isolation": "global, all suppliers moved together"}


def residual_world_test(w1: tuple, w2: tuple, init_seeds: list[int], split_seed: int) -> dict:
    """Can the world label be recovered FROM THE RESIDUAL of `Y` given `(X, X_nbr)`?

    Three numbers, and the first two are the controls that make the third readable:

      `label_shift`     -- can the world be told apart from `Y` alone? (the raw effect)
      `observable_shift`-- can it be told apart from `(X, X_nbr)` alone? (what the encoder's
                           inputs already reveal, which is NOT what Phase 0 gates on)
      `residual_shift`  -- can it be told apart from the residual `Y - g(X, X_nbr)`? (the gate)

    **Three entity-disjoint groups**, so no arm is scored on its own training data and no split
    is world-degenerate: `g` is fitted on group 1, the world classifier is trained on group 2's
    residuals and tested on group 3's. Every group carries rows from BOTH worlds -- an earlier
    version split the pooled row order in half, which put world 1 entirely in train and world 2
    entirely in test and made every class degenerate.
    """
    (Y1, X1), (Y2, X2) = w1, w2
    keys = sorted(set(Y1) & set(Y2) & set(X1) & set(X2))
    if len(keys) < 300:
        return {"n_rows": len(keys), "insufficient": True}

    y1 = np.array([Y1[k] for k in keys])
    y2 = np.array([Y2[k] for k in keys])
    Xa = np.stack([X1[k] for k in keys])
    Xb = np.stack([X2[k] for k in keys])

    ents = sorted({k[0] for k in keys})
    rng = np.random.RandomState(split_seed)
    rng.shuffle(ents)
    n = len(ents)
    grp = {e: (0 if i < n // 3 else (1 if i < 2 * n // 3 else 2)) for i, e in enumerate(ents)}
    g_of = np.array([grp[k[0]] for k in keys])

    def stack_both(mask):
        """Rows from BOTH worlds for the selected entities, with the world label."""
        X = np.concatenate([Xa[mask], Xb[mask]], axis=0)
        y = np.concatenate([y1[mask], y2[mask]])
        w = np.concatenate([np.zeros(int(mask.sum())), np.ones(int(mask.sum()))])
        return X, y, w

    m_fit, m_tr, m_te = (g_of == 0), (g_of == 1), (g_of == 2)
    if min(m_fit.sum(), m_tr.sum(), m_te.sum()) < 30:
        return {"n_rows": len(keys), "insufficient": True}

    Xf, yf, _ = stack_both(m_fit)
    Xt, yt, wt = stack_both(m_tr)
    Xe, ye, we = stack_both(m_te)

    out = {"n_rows": int(len(keys)),
           "n_fit": int(len(yf)), "n_train": int(len(yt)), "n_test": int(len(ye)),
           "label_positive_rate_w1": float(y1.mean()),
           "label_positive_rate_w2": float(y2.mean()),
           "n_rows_label_differs": int(np.sum(y1 != y2))}

    def auc_over_seeds(Ftr, wtr, Fte, wte):
        v = [train_head(Ftr, wtr, Fte, wte, s) for s in init_seeds]
        v = [x for x in v if x is not None]
        return (statistics.fmean(v) if v else float("nan")), v

    # (1) raw label shift -- world from Y alone
    out["label_shift_auc"], _ = auc_over_seeds(yt.reshape(-1, 1), wt, ye.reshape(-1, 1), we)
    # (2) observable shift -- world from (X, X_nbr) alone. Reported, NOT gated on.
    out["observable_shift_auc"], _ = auc_over_seeds(Xt, wt, Xe, we)

    # (3) THE GATE: world from the out-of-sample residual of Y given (X, X_nbr).
    res_aucs, resx_aucs = [], []
    for s in init_seeds:
        pr = fit_probs(Xf, yf, [Xt, Xe], s)
        if pr is None:
            continue
        rt = (yt - pr[0]).reshape(-1, 1)
        re = (ye - pr[1]).reshape(-1, 1)
        v = train_head(rt, wt, re, we, s)
        if v is not None:
            res_aucs.append(v)
        vx = train_head(np.concatenate([rt, Xt], axis=1), wt,
                        np.concatenate([re, Xe], axis=1), we, s)
        if vx is not None:
            resx_aucs.append(vx)
    out["residual_shift_auc"] = statistics.fmean(res_aucs) if res_aucs else float("nan")
    out["residual_shift_auc_all"] = res_aucs
    out["residual_plus_X_shift_auc"] = (statistics.fmean(resx_aucs) if resx_aucs
                                        else float("nan"))
    out["init_seed_spread"] = (max(res_aucs) - min(res_aucs)) if len(res_aucs) > 1 else None
    return out


# ---------------------------------------------------------------------------
# shortage: is the CRN protocol applicable at all?
# ---------------------------------------------------------------------------

def shortage_feasibility(variant: str, dseed: int, config: str) -> dict:
    """Measure whether `shortage` can be gated by the CRN protocol at all.

    `delay` and `impact` are read off shipment outcomes, and re-realisation rewrites those
    outcomes in place without changing which shipments exist. `shortage` is produced by the
    weekly inventory walk (`db/generate_dataset.py`), which **calls `new_shipment()` inside its
    own loop**: a replenishment order is created, with a freshly drawn carrier, supplier and
    lead time, whenever stock crosses a trigger that the intervened quantity itself moves. So an
    intervention does not merely change outcomes, it changes *which entities exist* -- and
    common random numbers pin per-entity draws, not an entity set whose size and composition
    move.

    This is measured rather than asserted: the generator is run twice, once factual and once
    with `own_stress` forced to a constant, and the two shipment ledgers are compared
    index-for-index. It is the same structural obstacle `ml/counterfactual_ground_truth.py`
    already measured when it replaced re-simulation with re-realisation.
    """
    import io, contextlib as _c, types as _t, sys as _s
    GEN = os.path.join(REPO, "db", "generate_dataset.py")

    def run(force_stress: float | None):
        src = open(GEN).read()
        src = src.replace("def write(name, header, rows):",
                          "def write(name, header, rows):\n    return", 1)
        src = src.replace(
            'with open(os.path.join(OUT, "resolved_config.json"), "w") as _f:\n'
            '    json.dump(_manifest, _f, indent=2, sort_keys=True)',
            "MANIFEST_OUT = _manifest", 1)
        if force_stress is not None:
            # One-line override at the definition site: every downstream reader -- p_delay, the
            # replenishment trigger, sourcing_stress -- picks it up, which is exactly the broad
            # perturbation Tier B would want and exactly what CRN cannot follow.
            src = src.replace(
                "def own_stress(sup_id, base_rel, t):",
                f"def own_stress(sup_id, base_rel, t):\n    return {force_stress}", 1)
        argv = ["generate_dataset.py", "--variant", variant, "--config", config,
                "--seed", str(dseed)]
        saved, _s.argv = _s.argv, argv
        mod = _t.ModuleType("__shortage_probe__")
        mod.__file__ = GEN
        # `@dataclass` resolves its module through sys.modules, so the shell has to be
        # registered before exec -- the same step `ml/confirm_latent_states.py::namespace` takes.
        _s.modules["__shortage_probe__"] = mod
        ns = mod.__dict__
        ns["__name__"] = "__shortage_probe__"
        buf = io.StringIO()
        try:
            with _c.redirect_stdout(buf):
                try:
                    exec(compile(src, GEN, "exec"), ns)
                except SystemExit:
                    pass
        finally:
            _s.argv = saved
        return ns

    a = run(None)
    b = run(0.40)
    sa, sb = a["shipments"], b["shipments"]
    ida, idb = [x["id"] for x in sa], [x["id"] for x in sb]
    n = min(len(sa), len(sb))
    same_id = sum(1 for i in range(n) if ida[i] == idb[i])
    same_tuple = sum(1 for i in range(n)
                     if (sa[i]["supplier_id"], sa[i]["factory_id"], sa[i]["dispatched_at"],
                         sa[i]["eta"]) == (sb[i]["supplier_id"], sb[i]["factory_id"],
                                           sb[i]["dispatched_at"], sb[i]["eta"]))
    ka, kb = set(a["shortage_events"]), set(b["shortage_events"])
    return {
        "n_shipments_factual": len(sa), "n_shipments_perturbed": len(sb),
        "n_shipments_differs": len(sa) != len(sb),
        "shipment_count_delta": len(sb) - len(sa),
        "frac_same_id_at_same_index": same_id / n if n else float("nan"),
        "frac_identical_identity_tuple_at_same_index": same_tuple / n if n else float("nan"),
        "n_shortage_keys_factual": len(ka), "n_shortage_keys_perturbed": len(kb),
        "jaccard_shortage_keys": (len(ka & kb) / len(ka | kb)) if (ka | kb) else float("nan"),
        "crn_applicable": bool(len(sa) == len(sb) and same_tuple == n),
    }


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
              f"{lc[t]['n_mismatches']} mismatches, exact={lc[t]['exact']}  "
              f"(gen pos rate {lc[t]['generator_positive_rate']:.4f})", flush=True)

    for task in tasks:
        cell = {}
        for tier in tiers.get(task, []):
            builder = tier_a_worlds if tier == "A" else tier_b_worlds
            w1, w2, levels = builder(r_main, task)
            # NULL: same latent configuration on both sides, different realisation seed. Every
            # source of apparent separation except the intervention is still present.
            n1, _n2, _ = builder(r_null, task)
            sig = residual_world_test(w1, w2, init_seeds, split_seed=dseed)
            nul = residual_world_test(w1, n1, init_seeds, split_seed=dseed)
            cell[tier] = {"levels": levels, "signal": sig, "null": nul}
            print(f"  {task:<9} tier {tier}:  rows {sig.get('n_rows',0):,}  "
                  f"labels differ {sig.get('n_rows_label_differs',0):,}  "
                  f"label AUC {sig.get('label_shift_auc', float('nan')):.4f}  "
                  f"obs AUC {sig.get('observable_shift_auc', float('nan')):.4f}  "
                  f"RESIDUAL {sig.get('residual_shift_auc', float('nan')):.4f} "
                  f"(null {nul.get('residual_shift_auc', float('nan')):.4f})", flush=True)
        out[task] = cell
    return out


def summarise(res: dict) -> dict:
    summary = {}
    for task in res["tasks"]:
        summary[task] = {}
        for tier in ("A", "B"):
            cells = [res["per_seed"][str(d)][task][tier] for d in res["dataset_seeds"]
                     if tier in res["per_seed"].get(str(d), {}).get(task, {})]
            if not cells:
                continue
            sa = [c["signal"]["residual_shift_auc"] for c in cells
                  if c["signal"].get("residual_shift_auc") == c["signal"].get("residual_shift_auc")]
            na = [c["null"]["residual_shift_auc"] for c in cells
                  if c["null"].get("residual_shift_auc") == c["null"].get("residual_shift_auc")]
            if not sa:
                continue
            init_floor = max([c["signal"]["init_seed_spread"] for c in cells
                              if c["signal"].get("init_seed_spread") is not None] or [float("nan")])
            dfloor = (max(sa) - min(sa)) if len(sa) > 1 else float("nan")
            null_exc = max(abs(a - 0.5) for a in na) if na else float("nan")
            floor = max(f for f in (init_floor, dfloor, null_exc) if f == f)
            mean = statistics.fmean(sa)
            summary[task][tier] = {
                "n_dataset_seeds": len(sa),
                "residual_auc_mean": mean, "residual_auc_per_dataset_seed": sa,
                "null_auc_mean": statistics.fmean(na) if na else None,
                "null_auc_per_dataset_seed": na,
                "label_shift_auc": statistics.fmean(
                    [c["signal"]["label_shift_auc"] for c in cells]),
                "observable_shift_auc": statistics.fmean(
                    [c["signal"]["observable_shift_auc"] for c in cells]),
                "mean_rows_label_differs": statistics.fmean(
                    [c["signal"]["n_rows_label_differs"] for c in cells]),
                "mean_rows": statistics.fmean([c["signal"]["n_rows"] for c in cells]),
                "init_seed_floor": init_floor, "dataset_seed_floor": dfloor,
                "null_excursion_floor": null_exc, "reproduction_floor": floor,
                "above_chance_by": mean - 0.5,
                "clears_floor": (mean - 0.5) > floor,
                "sign_consistent": all(a > 0.5 for a in sa) or all(a < 0.5 for a in sa),
            }
            summary[task][tier]["identifiable"] = bool(
                summary[task][tier]["clears_floor"] and summary[task][tier]["sign_consistent"])
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="A")
    ap.add_argument("--tasks", default="delay,impact")
    ap.add_argument("--tiers", default="delay:A,B;impact:A,B",
                    help="which tier(s) to run per task, e.g. 'delay:A,B;impact:B'")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--config", default="v1")
    ap.add_argument("--init-seeds", default="0,1,2,3,4")
    ap.add_argument("--realisation-seeds", default="1,2")
    ap.add_argument("--shortage-probe", action="store_true")
    ap.add_argument("--out", default="")
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
           "init_seeds": iseeds, "realisation_seeds": list(rseeds), "tasks": tasks,
           "tiers": tiers, "per_seed": {}}

    # `shortage` cannot be re-realised: its label comes from a walk that CREATES entities as a
    # function of the intervened quantity. Measured once rather than asserted, before any tier
    # is attempted for it.
    if "shortage" in tasks or a.shortage_probe:
        print("\n=== shortage: is the CRN protocol applicable? ===", flush=True)
        t = time.time()
        res["shortage_feasibility"] = shortage_feasibility(a.variant, dseeds[0], a.config)
        f = res["shortage_feasibility"]
        print(f"  shipments {f['n_shipments_factual']:,} -> {f['n_shipments_perturbed']:,} "
              f"({f['shipment_count_delta']:+d})", flush=True)
        print(f"  identical identity tuple at the same ledger index: "
              f"{f['frac_identical_identity_tuple_at_same_index']:.4%}", flush=True)
        print(f"  shortage-event key Jaccard: {f['jaccard_shortage_keys']:.4f}", flush=True)
        print(f"  CRN APPLICABLE: {f['crn_applicable']}   ({time.time() - t:.0f}s)", flush=True)
        tasks = [t_ for t_ in tasks if t_ != "shortage"]
    for d in dseeds:
        print(f"\n=== variant {a.variant} seed {d} ===", flush=True)
        t = time.time()
        res["per_seed"][str(d)] = run_seed(a.variant, d, a.config, tasks, iseeds, rseeds, tiers)
        print(f"  ({time.time() - t:.0f}s)", flush=True)
    res["tasks"] = tasks
    res["summary"] = summarise(res)

    print("\n" + "=" * 130)
    print(f"PHASE 0 — two-tier identifiability gate, variant {a.variant}")
    print("=" * 130)
    hdr = (f"{'task':<10}{'tier':>5}{'rows':>9}{'Y differs':>11}{'label AUC':>11}"
           f"{'obs AUC':>9}{'RESID':>8}{'null':>8}{'floor':>8}{'clears':>8}{'sign':>6}{'IDENT':>7}")
    print(hdr); print("-" * len(hdr))
    for task in tasks:
        for tier in ("A", "B"):
            s = res["summary"].get(task, {}).get(tier)
            if not s:
                continue
            print(f"{task:<10}{tier:>5}{s['mean_rows']:>9,.0f}"
                  f"{s['mean_rows_label_differs']:>11,.0f}{s['label_shift_auc']:>11.4f}"
                  f"{s['observable_shift_auc']:>9.4f}{s['residual_auc_mean']:>8.4f}"
                  f"{s['null_auc_mean']:>8.4f}{s['reproduction_floor']:>8.4f}"
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
