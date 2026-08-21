#!/usr/bin/env python3
"""
Layer 3 v3, PHASE 4 — GATE 3: causal-effect diagnostics, in the mandated order.

`layer3_neurosymbolic_build_prompt.md` §6 fixes the sequence, and the sequence is the design:

    1  identifiability      confirm Gate 0 passed for THIS effect  (read from gate0.json)
    2  ground-truth integrity   re-verify the generator-derived effect calculation against the
                            CURRENT generator before trusting it as ground truth again
    3  failure-source diagnosis  Diagnostic A (representational) vs Diagnostic B (data scale)
    4  estimator evaluation      sign agreement, Spearman, magnitude ratio, false-effect rate
    5  calibration               only for an estimator that already tracks the effect
    6  dual-floor replication    5 dataset seeds x model-init seeds

**Everything upstream is reused, not rebuilt.** `ml/counterfactual_edit.py` supplies the
structural graph edit and the frozen forward pass; `ml/counterfactual_ground_truth.py` supplies
`CausalWorld`, `apply_intervention` and `sample_interventions`; `ml/counterfactual_delta_head.py`
supplies the interventionally-supervised head, its dataset builder and its training loop;
`ml/run_counterfactual_phase1.py` supplies `spearman` and the affected/bystander partition.
None of them is modified.

---

**Diagnostic A — representational limit.** Three arms on the same held-out intervention set:

    A1  frozen SHARE under SCM-valid structural edits            (the prior naive baseline)
    A2  a causal-consistency-trained head on frozen embeddings   (DIAGNOSTIC PROBE ONLY)
    A3  a method operating directly on VERIFIED SCM variables and interventions

A2 is the prompt's "causal-consistency-trained representation", built under this project's
standing constraint that SHARE is not retrained: the causal-consistency objective is carried by a
head on top of the frozen representation, supervised by the generator's own interventional
truth. Per the build's ground rule, **clearing A2 does not satisfy Gate 3** -- it explains a
failure, it does not license a deployment.

A3 is split into three arms, because a single number could not separate three different error
sources and the whole point of Diagnostic A is to locate one:

    A3-oracle   verified SCM equations + TRUE own_stress at the shipment's dispatch time
                -> a ceiling. If this is not ~1.0 the extracted equations are wrong.
    A3-grid     verified SCM equations + TRUE own_stress read off the t0 SNAPSHOT GRID
                -> isolates the cost of time discretisation, which any deployable estimator
                   pays because it only observes snapshots.
    A3-est      verified SCM equations + own_stress ESTIMATED from the emitted observables
                -> the deployable arm, and the only one of the four that is a method.
    A3-const    verified SCM equations + a CONSTANT own_stress (the pooled mean)
                -> the control that decides whether A3-est is a method at all. Adding a
                   co-parent raises stress by `0.35 * own_stress(partner) >= 0` whatever the
                   partner's state is, so an estimator that outputs any positive constant gets
                   the sign right on every `add_dual_source` **by construction**. If A3-const
                   matches A3-est, the apparent tracking is the SCM's structure and not the
                   state estimate, and must be reported as such.

**Two consequences of that control are enforced throughout.** Sign agreement is reported **per
intervention kind** as well as pooled, because `add_dual_source`'s true effect is non-negative by
construction and its sign agreement is therefore uninformative on its own; `remove_supplier` and
`substitute_supplier` are the mixed-sign kinds and are where a sign result means something. And
Spearman, magnitude ratio and the A3-const gap are reported alongside every sign number, since
those are the quantities a constant estimate cannot fake.

**Diagnostic B — data-scale limit.** Four probes, none of which requires retraining SHARE:

    B1  magnitude stratification -- sign agreement by decile of |true delta|. If tracking rises
        with effect size, the limit is effect size, not representation.
    B2  amplified interventions -- the same lever applied k = 1, 2, 4 times, which multiplies the
        true effect without changing its mechanism.
    B3  training-set size curve for A2 -- 25% / 50% / 100% of the available rows, plus a
        nonzero-balanced arm, against held-out sign agreement.
    B4  known-effect sanity check -- couple a supplier to the world's MOST stressed supplier,
        the largest effect this generator can produce through the verified channel.

**Spearman is retained and reported per model seed.** `reports/decision_support_build.md` §1.3.1
found rank correlation near zero and *flipping sign* with the model seed (+0.068 / -0.31 / +0.16),
which is a distinct and more damning failure than sign agreement alone: it means the model cannot
rank who is most affected even when it occasionally gets a direction right. Whether it stabilises
as B's interventions grow the effect is itself evidence about which diagnostic is correct.

    python3 ml/gate3_causal.py --dseeds 42,43,44,45,46 --mseeds 0,1,2 --interventions 20
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import timedelta

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.counterfactual_edit import _Shim, edit_graph, naive_counterfactual  # noqa: E402
from ml.counterfactual_ground_truth import (                           # noqa: E402
    CausalWorld, apply_intervention, generator_namespace, sample_interventions)
from ml.ds_backbone import get_backbone, predict                       # noqa: E402
from ml.identifiability_check import featurise                         # noqa: E402
from ml.run_counterfactual_phase1 import spearman                      # noqa: E402

EPS = 1e-9
PRED_EPS = 1e-4
ARMS = ("naive", "A3_oracle", "A3_grid", "A3_est", "A3_const")
GEN = os.path.join(REPO, "db", "generate_dataset.py")


# ===========================================================================================
# STEP 2 — ground-truth integrity against the CURRENT generator
# ===========================================================================================

GT_TOKENS = {
    "own_stress": "def own_stress(sup_id, base_rel, t):",
    "coparent_bleed": "COPARENT_COUPLING * own_stress(partner",
    "p_delay": "p_delay = min(0.80, 0.025 + 0.38",
    "impact_walk": "if lab and sh[\"supplier_id\"]: sup_hit.add(",
    "delayed_transition": '_tr("delayed", "in_transit", J(eta + timedelta(hours=6)))',
}


def ground_truth_integrity(world: CausalWorld, ns: dict) -> dict:
    """Re-verify the generator-derived effect calculation before trusting it again.

    Four checks, each of which fails differently:

    *provenance* -- the constructs `CausalWorld` transcribes still exist in the generator.
    *stress exactness* -- `CausalWorld.stress` under the FACTUAL structure against the
      generator's own `stress()`, required to be bit-for-bit 0.0.
    *p_delay exactness* -- likewise for the shipment-level conversion, evaluated on real
      shipments at their real dispatch times.
    *label agreement* -- `p_impact` under the factual structure against the generator's OWN
      emitted impact labels: the eligible-shipment walk must select the same suppliers the
      generator's `sup_hit` loop does, and the closed-form probability must discriminate the
      realised label. A ground truth that cannot rank the outcome it is a probability for is
      not a ground truth.
    """
    src = open(GEN).read().split("\n")
    prov = {}
    for k, tok in GT_TOKENS.items():
        hits = [i + 1 for i, ln in enumerate(src) if tok in ln]
        prov[k] = {"found": bool(hits), "lines": hits[:4]}

    gen_stress = ns["stress"]
    sup_by_id = ns["sup_by_id"]
    max_st = 0.0
    n_st = 0
    for t in world.T0S:
        for sid in sorted(world.visible):
            if sid not in sup_by_id:
                continue
            a = world.stress(sid, t, world.base_coparents)
            b = gen_stress(sid, sup_by_id[sid]["base_rel"], t)
            max_st = max(max_st, abs(a - b))
            n_st += 1

    max_pd, n_pd = 0.0, 0
    absorption = ns["absorption"]
    for sh in ns["shipments"]:
        sid = sh.get("supplier_id")
        if not sid or sid not in sup_by_id or not sh.get("dispatched_at"):
            continue
        a = world.p_delay(sh, sid, world.base_coparents)
        st = gen_stress(sid, sup_by_id[sid]["base_rel"], sh["dispatched_at"])
        b = min(0.80, 0.025 + 0.38 * st * absorption(sid))
        max_pd = max(max_pd, abs(a - b))
        n_pd += 1

    # label agreement: p_impact vs the generator's own emitted impact labels
    p0 = world.p_impact(world.base_coparents)
    lab = {}
    snap_of = {ns["uid"]("snap", t0): i for i, t0 in enumerate(world.T0S)}
    for row in ns["label_rows"]:
        if row[4] != "impact":
            continue
        i = snap_of.get(row[1])
        if i is not None:
            lab[(row[3], i)] = 1.0 if row[5] == "true" else 0.0
    keys = sorted(set(p0) & set(lab))
    from ml.hypothesis_ranker import roc_auc
    y = np.array([lab[k] for k in keys])
    p = np.array([p0[k] for k in keys])
    auc = roc_auc(p, y.astype(bool)) if 0 < y.sum() < len(y) else None
    return {
        "provenance": prov,
        "provenance_all_found": all(v["found"] for v in prov.values()),
        "stress_comparisons": n_st, "max_abs_err_stress": max_st,
        "p_delay_comparisons": n_pd, "max_abs_err_p_delay": max_pd,
        "bit_identical": bool(max_st == 0.0 and max_pd == 0.0),
        # `ml/scm.py` matches the generator's summation order exactly and scores 0.0.
        # `CausalWorld` stores each supplier's co-parents as a **set** so the intervention can
        # add and remove members, which reorders the bleed summation -- and float addition is
        # not associative. The residual is therefore one ulp of a value near 0.1, not a
        # transcription error, and is asserted against that bound rather than against zero.
        "exact_to_float_rounding": bool(max_st <= 1e-15 and max_pd <= 1e-15),
        "float_tolerance": 1e-15,
        "label_rows_compared": len(keys),
        "p_impact_auc_vs_emitted_label": auc,
        "p_impact_mean": float(p.mean()) if len(p) else None,
        "emitted_positive_rate": float(y.mean()) if len(y) else None,
        "calibration_gap": (abs(float(p.mean()) - float(y.mean())) if len(y) else None),
    }


# ===========================================================================================
# shared: one pass over interventions producing every arm's prediction
# ===========================================================================================

def naive_edit(model, bundle, chain: list, sup_index: dict, task: str,
               baseline: np.ndarray):
    """The naive counterfactual for a CHAIN of edits: `p_after - p_before`, frozen forward pass.

    A single-element chain goes through `ml/counterfactual_edit.py::naive_counterfactual`
    unmodified, so the k = 1 arm is literally the prior build's baseline. Longer chains compose
    `edit_graph` -- also unmodified -- because each call returns an edited copy, so the edits
    accumulate rather than replacing one another. Applying each edit against the pristine graph
    and keeping the last would silently measure a k = 1 effect and label it k = 4.
    """
    if len(chain) == 1:
        return naive_counterfactual(model, bundle, chain[0]["kind"], chain[0], sup_index,
                                    task=task, baseline=baseline)
    g = bundle.data
    for spec in chain:
        g2 = edit_graph(g, spec["kind"], spec, sup_index)
        if g2 is None:
            return None
        g = g2
    after = predict(model, _Shim(g))[task]
    if after.shape != baseline.shape:
        return None
    return after - baseline


def apply_chain(world: CausalWorld, specs: list) -> tuple:
    """Apply several interventions in sequence, reusing `apply_intervention` unmodified.

    `apply_intervention` always builds from `world.base_coparents`, so chaining is expressed by
    temporarily pointing that attribute at the structure produced so far. This is what B2's
    amplified lever needs and it introduces no new intervention semantics -- k applications of a
    verified edit, not a new kind of edit.
    """
    cop = {k: set(v) for k, v in world.base_coparents.items()}
    owner: dict = {}
    touched: set = set()
    saved = world.base_coparents
    try:
        for spec in specs:
            world.base_coparents = cop
            cop, o2, t2 = apply_intervention(world, spec["kind"], spec)
            owner.update(o2)
            touched |= t2
    finally:
        world.base_coparents = saved
    return cop, owner, touched


class ScmEstimator:
    """Verified SCM equations driven by a supplied `own_stress` oracle or estimate.

    Only equations Gate 2 marked VERIFIED and live on this variant are used:
    `own_stress`, `coparent_bleed` (0.35), `stress`'s 0.95 cap, `p_delay`, and
    `impact_aggregation`. Nothing else is invoked, so a number this arm produces is attributable
    to those equations plus the state it was handed.
    """

    def __init__(self, world: CausalWorld, own_stress_of):
        self.w = world
        self.own = own_stress_of

    def stress(self, sup_id: str, t, coparents: dict) -> float:
        x = self.own(sup_id, t)
        for p in coparents.get(sup_id, ()):
            x += self.w.coparent_coupling * self.own(p, t)
        return min(0.95, x)

    def p_impact(self, coparents: dict, owner: dict | None = None) -> dict:
        out = {}
        for ti, per_sup in self.w._inflight.items():
            buckets: dict = {}
            for sup, ships in per_sup.items():
                for sh in ships:
                    o = (owner or {}).get(sh["id"], sup)
                    if o is None:
                        continue
                    buckets.setdefault(o, []).append(sh)
            for sup, ships in buckets.items():
                if sup not in self.w.visible:
                    continue
                q = 1.0
                for sh in ships:
                    st = self.stress(sup, sh["dispatched_at"], coparents)
                    q *= (1.0 - min(0.80, 0.025 + 0.38 * st))
                out[(sup, ti)] = 1.0 - q
        return out


def grid_own_stress(world: CausalWorld, ns: dict, values: dict):
    """`(sup_id, t) -> own_stress` read off the t0 SNAPSHOT GRID, nearest grid point <= t.

    Both A3-grid and A3-est go through this, so the only difference between them is whether
    `values` holds the true own_stress or an estimate of it. That is what makes the
    discretisation cost separable from the estimation cost.
    """
    t0s = list(world.T0S)

    def of(sup_id, t):
        i = 0
        for j, t0 in enumerate(t0s):
            if t0 <= t:
                i = j
            else:
                break
        return values.get((sup_id, i), 0.0)
    return of


def observable_matrix(ns: dict, world: CausalWorld) -> tuple:
    """`(X, keys, y_own_stress)` -- the generator's OWN emitted supplier feature row per
    `(supplier, t0)`, with true `own_stress` as the regression target.

    `sup_features` is called out of the generator namespace, so `X` is exactly
    `supplier_temporal_features.csv` -- the same channel `ml/identifiability_check.py` uses, and
    the entire time-varying observable a Supplier node carries.
    """
    sup_features, sup_by_id, own_stress = ns["sup_features"], ns["sup_by_id"], ns["own_stress"]
    X, keys, y = [], [], []
    for sid in sorted(world.visible):
        srow = sup_by_id.get(sid)
        if srow is None:
            continue
        for i, t0 in enumerate(world.T0S):
            X.append(featurise(sup_features(srow, t0)))
            keys.append((sid, i))
            y.append(own_stress(sid, srow["base_rel"], t0))
    return np.stack(X), keys, np.asarray(y, dtype=float)


def fit_own_stress(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, seed: int) -> np.ndarray:
    """Ridge regression, fitted on OTHER worlds and applied to this one.

    Ridge rather than a network: the question this arm answers is whether the observable channel
    carries the latent at all, and `reports/phase1_latent_state.md` already measured what a
    trained head gets from the same channel. A closed-form fit removes head-init variance from a
    diagnostic whose job is to locate an error source, not to win a benchmark.

    **`seed` bootstraps the training pool, and that is what makes a floor measurable for this
    arm.** A closed-form fit has no init seed, so the model-init axis -- the one every other
    result in this project is floored against -- does not exist for it. Resampling the fitting
    pool is the analogue: it is the same estimator, refit on an equally valid sample, and the
    spread it produces is the reproduction floor for a number this arm reports. `seed = 0` is
    the full pool and is deterministic.
    """
    if seed:
        rs = np.random.default_rng(seed)
        idx = rs.integers(0, len(ytr), size=len(ytr))
        Xtr, ytr = Xtr[idx], ytr[idx]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    A1 = np.concatenate([A, np.ones((len(A), 1))], axis=1)
    B1 = np.concatenate([B, np.ones((len(B), 1))], axis=1)
    lam = 1.0
    W = np.linalg.solve(A1.T @ A1 + lam * np.eye(A1.shape[1]), A1.T @ ytr)
    return np.clip(B1 @ W, 0.0, 0.95)


def effect_calibration(true: np.ndarray, pred: np.ndarray, n_bins: int = 5) -> dict:
    """STEP 5 — is a tracking estimator's effect SIZE calibrated, and with what uncertainty?

    Sign agreement says the direction is right; it says nothing about whether "+0.02" means
    +0.02. Two quantities, on the affected population:

    *reliability by predicted-effect bin* -- equal-count bins of the predicted delta, mean
    predicted against mean true in each. The headline is the largest absolute gap, which is the
    regression analogue of ECE and is on the same units as the effect itself.

    *calibration slope and intercept* -- the least-squares fit of true on predicted. Slope 1 and
    intercept 0 is calibrated; slope 0.13 (the prior naive baseline's magnitude ratio) is an
    estimator that gets direction right and scale wrong by 8x, which is a different and much
    weaker claim.

    Run only for an arm that already showed valid effect tracking, per §6's ordering: a
    calibration number for an estimator that does not track is a description of noise.
    """
    aff = np.abs(true) > EPS
    if aff.sum() < 20:
        return {"insufficient": True, "n_affected": int(aff.sum())}
    t, q = true[aff], pred[aff]
    order = np.argsort(q, kind="stable")
    bins, gaps = [], []
    for chunk in np.array_split(order, n_bins):
        if len(chunk) < 5:
            continue
        bins.append({"n": int(len(chunk)), "mean_predicted": float(q[chunk].mean()),
                     "mean_true": float(t[chunk].mean())})
        gaps.append(abs(bins[-1]["mean_predicted"] - bins[-1]["mean_true"]))
    A = np.stack([q, np.ones_like(q)], axis=1)
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    resid = t - A @ coef
    return {"n_affected": int(aff.sum()), "bins": bins,
            "max_abs_bin_gap": (max(gaps) if gaps else None),
            "calibration_slope": float(coef[0]), "calibration_intercept": float(coef[1]),
            "r2": float(1 - resid.var() / t.var()) if t.var() > 0 else None,
            "residual_sd": float(resid.std(ddof=1))}


def metrics_by_kind(true: np.ndarray, pred: np.ndarray, kind: np.ndarray) -> dict:
    """The same metric set, split by intervention kind.

    Reported because the three kinds are not equally informative about sign: `add_dual_source`
    only ever raises the true risk, so its sign agreement is bounded below by "does the estimator
    ever predict a decrease". The mixed-sign kinds are the discriminating ones.
    """
    out = {}
    for k in sorted(set(kind.tolist())):
        m = kind == k
        if m.sum():
            d = metrics(true[m], pred[m])
            aff = np.abs(true[m]) > EPS
            if aff.sum():
                d["true_negative_fraction"] = float((true[m][aff] < 0).mean())
            out[k] = d
    return out


def metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    """The full metric set §6 step 4 names, on one (true, predicted) pair."""
    aff = np.abs(true) > EPS
    by = ~aff
    out = {"n_pairs": int(len(true)), "n_affected": int(aff.sum()),
           "n_bystander": int(by.sum())}
    if aff.sum():
        ta, pa = true[aff], pred[aff]
        silent = np.abs(pa) < PRED_EPS
        out.update({
            "sign_agreement": float((np.sign(ta) == np.sign(pa)).mean()),
            "sign_agreement_excl_silent": (float((np.sign(ta[~silent])
                                                  == np.sign(pa[~silent])).mean())
                                           if (~silent).sum() else None),
            "silent_fraction": float(silent.mean()),
            "spearman": spearman(ta, pa),
            "true_abs_mean": float(np.abs(ta).mean()),
            "pred_abs_mean": float(np.abs(pa).mean()),
            "magnitude_ratio": float(np.abs(pa).mean() / max(np.abs(ta).mean(), 1e-12)),
        })
    if by.sum():
        out["false_effect_rate"] = float((np.abs(pred[by]) > PRED_EPS).mean())
    return out


# ===========================================================================================
# the per-cell run: one (dataset seed, model seed)
# ===========================================================================================

def run_cell(dseed: int, mseed: int, variant: str, config: str, csv_root: str,
             n_per_kind: int, sample_seed: int, task: str, amp_ks: list[int],
             other_worlds: dict) -> dict:
    csv_dir = os.path.join(csv_root, f"v{variant}_seed{dseed}")
    model, meta = get_backbone(csv_dir, variant, dseed, mseed, device="cpu", verbose=False)
    test = meta["test_bundles"]
    sup_ids = meta["supplier_ids"]
    sup_index = {s: i for i, s in enumerate(sup_ids)}

    ns = generator_namespace(variant, dseed, config)
    world = CausalWorld(ns)
    n_all = len(world.T0S)
    off = int(round(0.4 * n_all)) + int(round(0.2 * n_all))
    assert len(test) == n_all - off, f"test bundles {len(test)} != {n_all - off}"

    integrity = ground_truth_integrity(world, ns) if mseed == 0 else None

    # --- own_stress oracles / estimates ------------------------------------------------
    own_stress = ns["own_stress"]
    sup_by_id = ns["sup_by_id"]
    true_own = lambda sid, t: own_stress(sid, sup_by_id[sid]["base_rel"], t)  # noqa: E731
    grid_true = {(sid, i): true_own(sid, t0)
                 for sid in sorted(world.visible) if sid in sup_by_id
                 for i, t0 in enumerate(world.T0S)}
    X, keys, y_own = observable_matrix(ns, world)
    Xtr = np.concatenate([other_worlds[d]["X"] for d in other_worlds]) if other_worlds else None
    ytr = np.concatenate([other_worlds[d]["y"] for d in other_worlds]) if other_worlds else None
    est_vals = fit_own_stress(Xtr, ytr, X, mseed) if Xtr is not None else None
    grid_est = ({k: float(v) for k, v in zip(keys, est_vals)} if est_vals is not None else None)
    # The control: the SAME equations driven by a single constant, the pooled mean own_stress of
    # the OTHER worlds. It carries no per-entity information at all, so any metric it matches is
    # a property of the intervention's structure rather than of state estimation.
    const_val = float(np.mean(ytr)) if ytr is not None else None
    grid_const = ({k: const_val for k in keys} if const_val is not None else None)

    scm_oracle = ScmEstimator(world, true_own)
    scm_grid = ScmEstimator(world, grid_own_stress(world, ns, grid_true))
    scm_est = (ScmEstimator(world, grid_own_stress(world, ns, grid_est))
               if grid_est is not None else None)
    scm_const = (ScmEstimator(world, grid_own_stress(world, ns, grid_const))
                 if grid_const is not None else None)

    p0 = world.p_impact(world.base_coparents)
    q0 = {"A3_oracle": scm_oracle.p_impact(world.base_coparents),
          "A3_grid": scm_grid.p_impact(world.base_coparents)}
    if scm_est is not None:
        q0["A3_est"] = scm_est.p_impact(world.base_coparents)
    if scm_const is not None:
        q0["A3_const"] = scm_const.p_impact(world.base_coparents)

    baselines = [predict(model, b)[task] for b in test]
    specs = sample_interventions(world, n_per_kind, sample_seed)

    # --- B2/B4: amplified and maximum-effect levers, built from the same verified edit ---
    active = sorted({sh["supplier_id"] for sh in world.shipments if sh.get("supplier_id")}
                    & world.visible)
    rng = np.random.default_rng(sample_seed + dseed)
    t_mid = world.T0S[off]
    hottest = max(active, key=lambda s: true_own(s, t_mid))
    amp_specs = {}
    for k in amp_ks:
        chains = []
        while len(chains) < max(2, n_per_kind // 2):
            a = active[int(rng.integers(len(active)))]
            pool = [x for x in active if x != a]
            if len(pool) < k:
                break
            partners = [pool[int(i)] for i in rng.choice(len(pool), size=k, replace=False)]
            chain = [{"kind": "add_dual_source", "primary": a, "secondary": b}
                     for b in partners]
            if chain:                     # an empty chain is not an intervention
                chains.append(chain)
        amp_specs[k] = chains
    b4_chains = [[{"kind": "add_dual_source", "primary": a, "secondary": hottest}]
                 for a in active[:max(2, n_per_kind // 2)] if a != hottest]

    def score(chains, tag) -> dict:
        chains = [c for c in chains if c]        # a chain with no edits is not an intervention
        rows = {"true": [], "naive": [], "A3_oracle": [], "A3_grid": [], "A3_est": [],
                "A3_const": [], "kind": []}
        for chain in chains:
            cop, owner, _touched = apply_chain(world, chain)
            p1 = world.p_impact(cop, owner)
            q1 = {"A3_oracle": scm_oracle.p_impact(cop, owner),
                  "A3_grid": scm_grid.p_impact(cop, owner)}
            if scm_est is not None:
                q1["A3_est"] = scm_est.p_impact(cop, owner)
            if scm_const is not None:
                q1["A3_const"] = scm_const.p_impact(cop, owner)
            for j, bundle in enumerate(test):
                ti = off + j
                nv = naive_edit(model, bundle, chain, sup_index, task, baselines[j])
                if nv is None:
                    continue
                for si, s in enumerate(sup_ids):
                    k = (s, ti)
                    if k not in p0:
                        continue
                    v1 = p1.get(k)
                    if v1 is None:
                        continue
                    rows["true"].append(v1 - p0[k])
                    rows["naive"].append(float(nv[si]))
                    for arm in ("A3_oracle", "A3_grid", "A3_est", "A3_const"):
                        if arm in q1 and k in q0.get(arm, {}):
                            rows[arm].append(q1[arm].get(k, q0[arm][k]) - q0[arm][k])
                        else:
                            rows[arm].append(np.nan)
                    rows["kind"].append(chain[0]["kind"])
        true = np.asarray(rows["true"], dtype=float)
        kinds = np.asarray(rows["kind"])
        out = {"tag": tag, "n_chains": len(chains), "n_rows": int(len(true))}
        for arm in ARMS:
            pred = np.asarray(rows[arm], dtype=float)
            if not len(pred) or not np.isfinite(pred).any():
                continue
            ok = np.isfinite(pred) & np.isfinite(true)
            out[arm] = metrics(true[ok], pred[ok])
            out[arm]["by_kind"] = metrics_by_kind(true[ok], pred[ok], kinds[ok])
        out["_true"] = true
        out["_naive"] = np.asarray(rows["naive"], dtype=float)
        out["_preds"] = {arm: np.asarray(rows[arm], dtype=float) for arm in ARMS}
        return out

    main_chains = [[s] for s in specs]
    base_scored = score(main_chains, "main")

    # --- B1: magnitude stratification on the main corpus -------------------------------
    true, naive = base_scored.pop("_true"), base_scored.pop("_naive")
    preds = base_scored.pop("_preds")
    calib = {}
    for arm, pr in preds.items():
        ok = np.isfinite(pr) & np.isfinite(true)
        if ok.sum():
            calib[arm] = effect_calibration(true[ok], pr[ok])
    aff = np.abs(true) > EPS
    b1 = []
    if aff.sum() >= 50:
        mag = np.abs(true[aff])
        qs = np.quantile(mag, np.linspace(0, 1, 6))
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (mag >= lo) & (mag <= hi if hi == qs[-1] else mag < hi)
            if m.sum() < 10:
                continue
            b1.append({"bin": f"[{lo:.2e},{hi:.2e})", "n": int(m.sum()),
                       "mean_abs_true": float(mag[m].mean()),
                       "sign_agreement": float((np.sign(true[aff][m])
                                                == np.sign(naive[aff][m])).mean()),
                       "spearman": spearman(true[aff][m], naive[aff][m])})

    amp = {}
    for k, chains in amp_specs.items():
        s = score(chains, f"amp_k{k}")
        for key in ("_true", "_naive", "_preds"):
            s.pop(key, None)
        amp[str(k)] = s
    b4 = score(b4_chains, "max_effect")
    for key in ("_true", "_naive", "_preds"):
        b4.pop(key, None)

    return {"dseed": dseed, "mseed": mseed, "auc": meta["auc"],
            "ground_truth_integrity": integrity,
            "main": base_scored, "effect_calibration": calib, "B1_magnitude_bins": b1,
            "B2_amplified": amp, "B4_max_effect": b4,
            "own_stress_fit": ({"n_train": int(len(ytr)),
                                "r2": float(1 - np.var(y_own - est_vals) / np.var(y_own))}
                               if est_vals is not None else None),
            "observables": {"X": X, "y": y_own}}


# ===========================================================================================
# B3 — training-set size curve for the causal-consistency head (A2)
# ===========================================================================================

def diagnostic_b3(dseeds, variant, config, csv_root, n_per_kind, mseed, sample_seed, task,
                  fractions, head_seeds) -> dict:
    """A2 at several training-set sizes, plus a nonzero-balanced arm.

    Reuses `ml/counterfactual_delta_head.py`'s `build_dataset` and `train_head` unmodified, in
    the same leave-one-world-out protocol the prior build used, so the 100% arm reproduces that
    build's number and the smaller arms are read against it.
    """
    from ml.counterfactual_delta_head import build_dataset, predict_delta, train_head
    data = {}
    for d in dseeds:
        data[d] = build_dataset(d, variant, config, csv_root, n_per_kind, mseed,
                                sample_seed, task)
        n_aff = int((np.abs(data[d]["y"]) > EPS).sum())
        print(f"    A2 corpus d{d}: {len(data[d]['y']):,} rows, {n_aff:,} affected", flush=True)

    folds = []
    for i, test_d in enumerate(dseeds):
        val_d = dseeds[(i + 1) % len(dseeds)]
        tr_d = [d for d in dseeds if d not in (test_d, val_d)]
        Xtr = np.concatenate([data[d]["X"] for d in tr_d])
        ytr = np.concatenate([data[d]["y"] for d in tr_d])
        yte = data[test_d]["y"]
        aff_te = np.abs(yte) > EPS
        naive_te = data[test_d]["X"][:, 3 * 128]
        for arm in [f"frac_{f}" for f in fractions] + ["balanced"]:
            rng = np.random.default_rng(hash(arm) % (2 ** 31))
            if arm == "balanced":
                pos = np.where(np.abs(ytr) > EPS)[0]
                neg = np.where(np.abs(ytr) <= EPS)[0]
                take = rng.choice(neg, size=min(len(neg), 4 * len(pos)), replace=False)
                sel = np.concatenate([pos, take])
            else:
                f = float(arm.split("_")[1])
                sel = rng.choice(len(ytr), size=max(50, int(f * len(ytr))), replace=False)
            for hs in head_seeds:
                head, mu, sd, _pw = train_head(Xtr[sel], ytr[sel], data[val_d]["X"],
                                               data[val_d]["y"], seed=hs)
                dhat, _g = predict_delta(head, mu, sd, data[test_d]["X"])
                rec = {"test_seed": test_d, "arm": arm, "head_seed": hs,
                       "n_train_rows": int(len(sel)),
                       "n_train_affected": int((np.abs(ytr[sel]) > EPS).sum())}
                if aff_te.sum():
                    rec["sign_agreement"] = float((np.sign(yte[aff_te])
                                                   == np.sign(dhat[aff_te])).mean())
                    rec["naive_sign_agreement"] = float((np.sign(yte[aff_te])
                                                         == np.sign(naive_te[aff_te])).mean())
                    rec["spearman"] = spearman(yte[aff_te], dhat[aff_te])
                    rec["magnitude_ratio"] = float(np.abs(dhat[aff_te]).mean()
                                                   / max(np.abs(yte[aff_te]).mean(), 1e-12))
                folds.append(rec)
                print(f"    A2 {arm:<12} test=d{test_d} h{hs}: "
                      f"sign={rec.get('sign_agreement')} "
                      f"naive={rec.get('naive_sign_agreement')}", flush=True)
    return {"folds": folds}


# ===========================================================================================
# driver
# ===========================================================================================

def dual_floor(vals: dict, dseeds, mseeds) -> dict:
    M = np.array([[vals.get((d, m), np.nan) for m in mseeds] for d in dseeds], dtype=float)
    if not np.isfinite(M).any():
        return {"available": False}
    per_world = [float(np.nanmax(r) - np.nanmin(r)) for r in M if np.isfinite(r).sum() > 1]
    wm = np.array([np.nanmean(r) for r in M], dtype=float)
    init_f = max(per_world) if per_world else float("nan")
    dset_f = float(np.nanmax(wm) - np.nanmin(wm)) if np.isfinite(wm).sum() > 1 else float("nan")
    return {"available": True, "mean": float(np.nanmean(M)),
            "per_world_mean": [float(v) for v in wm],
            "per_cell": {f"{d}|{m}": (None if not np.isfinite(M[i, j]) else float(M[i, j]))
                         for i, d in enumerate(dseeds) for j, m in enumerate(mseeds)},
            "init_seed_floor": init_f, "dataset_seed_floor": dset_f,
            "floor": (max([f for f in (init_f, dset_f) if f == f] or [float("nan")]))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="0")
    ap.add_argument("--config", default="v1")
    ap.add_argument("--csv-root", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--dseeds", default="42,43,44,45,46")
    ap.add_argument("--mseeds", default="0,1,2")
    ap.add_argument("--interventions", type=int, default=20, help="of EACH of the three kinds")
    ap.add_argument("--amp-ks", default="1,2,4")
    ap.add_argument("--sample-seed", type=int, default=20260813)
    ap.add_argument("--task", default="impact")
    ap.add_argument("--gate0", default=os.path.join(REPO, "out", "layer3_v3", "gate0.json"))
    ap.add_argument("--b3-fractions", default="0.25,0.5,1.0")
    ap.add_argument("--b3-head-seeds", default="0,1")
    ap.add_argument("--skip-b3", action="store_true")
    ap.add_argument("--only-dseeds", default="",
                    help="run cells for these worlds only; the leave-one-world-out own_stress "
                         "pool still spans every world in --dseeds, so a sharded run and a "
                         "single run fit the A3-est arm on identical pools")
    ap.add_argument("--only-b3", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "out", "layer3_v3", "gate3.json"))
    a = ap.parse_args()

    dseeds = [int(x) for x in a.dseeds.split(",") if x.strip()]
    mseeds = [int(x) for x in a.mseeds.split(",") if x.strip()]
    amp_ks = [int(x) for x in a.amp_ks.split(",") if x.strip()]

    # ---- STEP 1: identifiability, read from Gate 0 --------------------------------------
    step1 = {"gate0_file": a.gate0, "available": os.path.exists(a.gate0)}
    if step1["available"]:
        g0 = json.load(open(a.gate0))
        cell = g0.get("summary", {}).get(a.task, {}).get("C")
        step1["tier_C"] = cell
        step1["identifiable"] = bool(cell and cell.get("identifiable"))
        print(f"STEP 1 — Gate 0 tier C for `{a.task}`: "
              f"identifiable={step1['identifiable']}"
              + (f"  residual AUC {cell['residual_auc_mean']:.4f} vs null "
                 f"{cell['null_auc_mean']:.4f}, floor {cell['reproduction_floor']:.4f}"
                 if cell and not cell.get("insufficient") else ""))
    else:
        step1["identifiable"] = None
        print(f"STEP 1 — Gate 0 result not found at {a.gate0}; "
              f"Gate 3 proceeds as DIAGNOSTIC ONLY and cannot be passed.")

    # ---- observables for the leave-one-world-out own_stress fit --------------------------
    # Built for EVERY world before any cell runs. The alternative -- filling the cache as the
    # loop goes -- would fit the first world's A3-est arm on an empty pool and every later
    # world's on a bigger one, so the arm's numbers would not be comparable across worlds.
    print("\nSTEP 2 — observable matrices for the leave-one-world-out own_stress fit")
    obs_cache: dict = {}
    cache_dir = os.path.join(REPO, "out", "layer3_v3", "obs_cache")
    os.makedirs(cache_dir, exist_ok=True)
    for d in ([] if a.only_b3 else dseeds):
        # Cached to disk because a sharded run would otherwise re-execute the generator once per
        # (shard x world). The cache key carries the generator's own sha256, so a generator edit
        # invalidates it rather than silently serving a matrix built from different equations.
        import hashlib
        gh = hashlib.sha256(open(GEN, "rb").read()).hexdigest()[:12]
        cpath = os.path.join(cache_dir, f"v{a.variant}_d{d}_{a.config}_{gh}.npz")
        if os.path.exists(cpath):
            z = np.load(cpath)
            obs_cache[d] = {"X": z["X"], "y": z["y"]}
            print(f"  d{d}: {z['X'].shape[0]:,} rows x {z['X'].shape[1]} observables [cached]",
                  flush=True)
            continue
        ns_d = generator_namespace(a.variant, d, a.config)
        w_d = CausalWorld(ns_d)
        X, _keys, y = observable_matrix(ns_d, w_d)
        obs_cache[d] = {"X": X, "y": y}
        np.savez_compressed(cpath, X=X, y=y)
        print(f"  d{d}: {X.shape[0]:,} (supplier, t0) rows x {X.shape[1]} observables",
              flush=True)

    print("\nSTEP 2/3 — per-cell runs")
    cells = []
    cell_dseeds = ([int(x) for x in a.only_dseeds.split(",") if x.strip()]
                   if a.only_dseeds else dseeds)
    for d in ([] if a.only_b3 else cell_dseeds):
        for m in mseeds:
            others = {k: v for k, v in obs_cache.items() if k != d}
            c = run_cell(d, m, a.variant, a.config, a.csv_root, a.interventions,
                         a.sample_seed, a.task, amp_ks, others)
            c.pop("observables", None)
            cells.append(c)
            mn = c["main"]
            print(f"  d{d} m{m}: affected={mn['naive']['n_affected']:,}  "
                  f"naive sign={mn['naive']['sign_agreement']:.4f} rho="
                  f"{mn['naive']['spearman']}  "
                  f"A3-oracle={mn.get('A3_oracle', {}).get('sign_agreement')}  "
                  f"A3-est={mn.get('A3_est', {}).get('sign_agreement')}", flush=True)

    if a.only_b3:
        print("\nSTEP 3 — Diagnostic B3 only")
        b3 = diagnostic_b3(dseeds, a.variant, a.config, a.csv_root, a.interventions,
                           mseeds[0], a.sample_seed, a.task,
                           [float(x) for x in a.b3_fractions.split(",")],
                           [int(x) for x in a.b3_head_seeds.split(",")])
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump({"config": vars(a), "B3": b3}, fh, indent=1, default=float)
        print(f"wrote {a.out}")
        return 0

    b3 = None
    if not a.skip_b3:
        print("\nSTEP 3 — Diagnostic B3 (A2 training-set size curve)")
        b3 = diagnostic_b3(dseeds, a.variant, a.config, a.csv_root, a.interventions,
                           mseeds[0], a.sample_seed, a.task,
                           [float(x) for x in a.b3_fractions.split(",")],
                           [int(x) for x in a.b3_head_seeds.split(",")])

    # ---- STEP 4 + 6: metrics with dual-floor replication ---------------------------------
    arms = ARMS
    step46 = {}
    for arm in arms:
        for metric in ("sign_agreement", "spearman", "magnitude_ratio", "false_effect_rate"):
            vals = {(c["dseed"], c["mseed"]): c["main"].get(arm, {}).get(metric)
                    for c in cells}
            vals = {k: v for k, v in vals.items() if v is not None}
            if vals:
                step46.setdefault(arm, {})[metric] = dual_floor(vals, dseeds, mseeds)

    # per-kind sign agreement, pooled over cells -- `add_dual_source` is non-negative by
    # construction, so its column is a control and not a result.
    per_kind = {}
    for arm in arms:
        per_kind[arm] = {}
        for kind in ("add_dual_source", "remove_supplier", "substitute_supplier"):
            v = [c["main"].get(arm, {}).get("by_kind", {}).get(kind, {}).get("sign_agreement")
                 for c in cells]
            v = [x for x in v if x is not None]
            if v:
                per_kind[arm][kind] = statistics.fmean(v)
        neg = [c["main"].get(arm, {}).get("by_kind", {}).get(k, {})
               .get("true_negative_fraction") for c in cells
               for k in ("remove_supplier", "substitute_supplier")]
        neg = [x for x in neg if x is not None]
        per_kind[arm]["mixed_sign_true_negative_fraction"] = (statistics.fmean(neg)
                                                              if neg else None)

    print("\n" + "=" * 120)
    print(f"GATE 3 — estimator evaluation, {len(dseeds)} worlds x {len(mseeds)} model seeds")
    print("=" * 120)
    hdr = (f"{'arm':<12}{'sign':>9}{'floor':>9}{'>chance':>10}{'clears':>8}{'5/5':>6}"
           f"{'spearman':>10}{'rho floor':>11}{'magratio':>10}{'false eff':>11}")
    print(hdr); print("-" * len(hdr))
    for arm in arms:
        s = step46.get(arm, {}).get("sign_agreement")
        if not s or not s.get("available"):
            continue
        rho = step46[arm].get("spearman", {})
        mr = step46[arm].get("magnitude_ratio", {})
        fe = step46[arm].get("false_effect_rate", {})
        above = s["mean"] - 0.5
        clears = above > s["floor"]
        cons = all(v > 0.5 for v in s["per_world_mean"])
        print(f"{arm:<12}{s['mean']:>9.4f}{s['floor']:>9.4f}{above:>+10.4f}"
              f"{('YES' if clears else 'no'):>8}{('yes' if cons else 'NO'):>6}"
              f"{rho.get('mean', float('nan')):>10.4f}"
              f"{rho.get('floor', float('nan')):>11.4f}"
              f"{mr.get('mean', float('nan')):>10.3f}"
              f"{fe.get('mean', float('nan')):>11.5f}")
    print("-" * len(hdr))

    print(f"\nsign agreement per intervention kind (add_dual_source's true effect is "
          f"NON-NEGATIVE by construction; its column is a control)")
    hdr2 = f"{'arm':<12}{'add_dual':>11}{'remove':>11}{'substitute':>13}{'neg frac':>11}"
    print(hdr2); print("-" * len(hdr2))
    for arm in arms:
        pk = per_kind.get(arm, {})
        if not pk:
            continue
        nf = pk.get("mixed_sign_true_negative_fraction")
        print(f"{arm:<12}{pk.get('add_dual_source', float('nan')):>11.4f}"
              f"{pk.get('remove_supplier', float('nan')):>11.4f}"
              f"{pk.get('substitute_supplier', float('nan')):>13.4f}"
              f"{(nf if nf is not None else float('nan')):>11.3f}")
    print("-" * len(hdr2))

    # ---- STEP 5: calibration, gated on step 4 -------------------------------------------
    tracking = [arm for arm in arms
                if (step46.get(arm, {}).get("sign_agreement", {}).get("available")
                    and (step46[arm]["sign_agreement"]["mean"] - 0.5)
                    > step46[arm]["sign_agreement"]["floor"]
                    and all(v > 0.5 for v in
                            step46[arm]["sign_agreement"]["per_world_mean"]))]
    step5 = {"arms_showing_valid_tracking": tracking,
             "run": bool(tracking),
             "note": ("calibration and uncertainty are measured only for an estimator that "
                      "already shows valid effect tracking (§6 step 5); no arm qualified"
                      if not tracking else "measured for the qualifying arms"),
             "per_arm": {}}
    for arm in tracking:
        rows = [c["effect_calibration"].get(arm) for c in cells
                if c.get("effect_calibration", {}).get(arm)
                and not c["effect_calibration"][arm].get("insufficient")]
        if not rows:
            continue
        step5["per_arm"][arm] = {
            "n_cells": len(rows),
            "calibration_slope_mean": statistics.fmean(r["calibration_slope"] for r in rows),
            "calibration_slope_range": [min(r["calibration_slope"] for r in rows),
                                        max(r["calibration_slope"] for r in rows)],
            "calibration_intercept_mean": statistics.fmean(r["calibration_intercept"]
                                                           for r in rows),
            "max_abs_bin_gap_mean": statistics.fmean(r["max_abs_bin_gap"] for r in rows
                                                     if r["max_abs_bin_gap"] is not None),
            "r2_mean": statistics.fmean(r["r2"] for r in rows if r["r2"] is not None),
            "residual_sd_mean": statistics.fmean(r["residual_sd"] for r in rows),
        }
    print(f"\nSTEP 5 — calibration: {step5['note']}")
    if step5["per_arm"]:
        print(f"  {'arm':<12}{'slope':>9}{'intercept':>12}{'max bin gap':>14}{'R2':>8}"
              f"{'resid sd':>11}")
        for arm, v in step5["per_arm"].items():
            print(f"  {arm:<12}{v['calibration_slope_mean']:>9.3f}"
                  f"{v['calibration_intercept_mean']:>+12.5f}"
                  f"{v['max_abs_bin_gap_mean']:>14.5f}{v['r2_mean']:>8.3f}"
                  f"{v['residual_sd_mean']:>11.5f}")

    blob = {"config": vars(a), "step1_identifiability": step1, "per_kind": per_kind,
            "step2_ground_truth_integrity": [c["ground_truth_integrity"] for c in cells
                                             if c.get("ground_truth_integrity")],
            "cells": cells, "B3": b3, "step4_6_dual_floor": step46, "step5": step5}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(blob, fh, indent=1, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
