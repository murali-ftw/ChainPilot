"""Phase 6 re-verification — the loop rebuilds the data path, so every Phase 5 invariant is measured again.

  F1  folds: the fixed split asserts no leak for every task in both worlds; the eight rolling origins assert theirs;
      the covid snapshots are counted
  F2  determinism on MPS: strict deterministic algorithms, warn-only, repeat-forward spread, same-seed init
  F3  causality end to end on the loop's inference path, for all four shipped models, against the measured
      repeat-forward floor (graph encoders are not bit-deterministic on MPS, so "bitwise" is demanded only where
      the model itself is)
  F4  fill: the 22-cell partition refines the legacy 20 bins exactly, both worlds

Hazard rows-in-loss and survival monotonicity, and quantile crossings, are asserted INSIDE every bundle
(loop.train / loop.invariants) and collected from the bundles for the report.
"""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "models"), os.path.join(HERE, "..", "eval")]
import numpy as np, torch
import phase5_heads as P5
import folds as FO
import loop as L
from heads import fill_cell, fill_to_legacy
from phase5_metrics import legacy_bin
from config import ARTIFACTS

DEV = P5.DEV
R = {}
SHIPPED = json.load(open(os.path.join(os.path.dirname(ARTIFACTS), "configs", "shipped.json")))

# ---------------------------------------------------------------- F1
for w in ("v6", "v7"):
    for task in ("arrival_week", "fill_rate", "capacity_strain", "shortage_qty"):
        lb = P5.labels(w, task)
        tr, va, te = FO.fixed_split(lb.snapshot_date)
        FO.assert_no_leak(lb.snapshot_date, tr, va, te)
        R[f"F1_{w}_{task}_rows_train_val_test"] = [int(tr.sum()), int(va.sum()), int(te.sum())]
    R[f"F1_{w}_rolling_origins"] = FO.assert_rolling_origins(P5.labels(w, "fill_rate").snapshot_date)
    lbd = P5.labels(w, "fill_rate").snapshot_date
    tr, _, _ = FO.fixed_split(lbd)
    cov = FO.covid_mask(w, lbd)
    R[f"F1_{w}_covid_training_snapshots"] = sorted({str(d.date()) for d in lbd[tr & cov].unique()})

# ---------------------------------------------------------------- F2
W = P5.TS.load_world("v6")
X = torch.randn(W["NCH"], 52, 24, device=DEV)
Dfake = dict(X=X, dt=torch.zeros(W["NCH"], 52, device=DEV), obs=torch.ones(W["NCH"], 52, dtype=torch.bool, device=DEV), W=W)
idx = torch.arange(0, 4000, device=DEV)
T = torch.randint(1, 14, (4000,), device=DEV)
for mode in ("strict", "warn_only"):
    try:
        torch.use_deterministic_algorithms(True, warn_only=(mode == "warn_only"))
        L.seed_all(7) if mode == "warn_only" else torch.manual_seed(7)
        torch.use_deterministic_algorithms(True, warn_only=(mode == "warn_only"))
        m = P5.HeadNet(24, "arrival_week", "lite", 4).to(DEV)
        z = P5.forward(m, Dfake, 51, idx); loss, _ = P5.loss_of(m, z, {"T": T, "cen": T > 12}); loss.backward()
        R[f"F2_deterministic_{mode}"] = "runs"
    except Exception as e:
        R[f"F2_deterministic_{mode}"] = f"raises {type(e).__name__}: {str(e)[:110]}"
    finally:
        torch.use_deterministic_algorithms(False)
L.seed_all(7); a = P5.HeadNet(24, "arrival_week", "lite", 4).to(DEV)
L.seed_all(7); b = P5.HeadNet(24, "arrival_week", "lite", 4).to(DEV)
R["F2_same_seed_identical_init"] = all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))

# ---------------------------------------------------------------- F3
floor = {}
for task, spec in SHIPPED["tasks"].items():
    lb = P5.labels("v6", task)
    tr, va, te = FO.fixed_split(lb.snapshot_date)
    D = P5.device_inputs("v6", np.sort(lb.snapshot_date[tr].unique()), False)
    L.seed_all(7)
    m = P5.HeadNet(D["X"].shape[2], task, spec["arch"], spec["depth"]).to(DEV).eval()
    order = P5.ordered(lb, te); s = np.unique(lb.snapshot_date.values[order])[0]; ii = order[lb.snapshot_date.values[order] == s]
    keymap = D["W"]["pp_uniq"] if task == "shortage_qty" else D["W"]["cidx"]
    ix = torch.from_numpy(lb.key.iloc[ii].map(keymap).to_numpy(np.int64)).to(DEV)
    t0 = P5.t0_of(D["W"], s)
    with torch.no_grad():
        reps = [P5.forward(m, D, t0, ix) for _ in range(4)]
        rep_floor = max(float((r - reps[0]).abs().max()) for r in reps[1:])
        saved = [D["X"][:, t0 + 1:].clone(), D["dt"][:, t0 + 1:].clone(), D["obs"][:, t0 + 1:].clone()]
        D["X"][:, t0 + 1:] += 100.0; D["dt"][:, t0 + 1:] += 100.0; D["obs"][:, t0 + 1:] = ~D["obs"][:, t0 + 1:]
        zf = P5.forward(m, D, t0, ix)
        D["X"][:, t0 + 1:], D["dt"][:, t0 + 1:], D["obs"][:, t0 + 1:] = saved
        last = D["X"][:, t0].clone(); D["X"][:, t0] += 1.0
        zl = P5.forward(m, D, t0, ix)
        D["X"][:, t0] = last
    fut = float((zf - reps[0]).abs().max())
    R[f"F3_{task}_{spec['arch']}_h{spec['depth']}"] = dict(
        repeat_forward_floor=rep_floor, future_perturbation_max_abs_diff=fut,
        future_within_floor=bool(fut <= max(rep_floor, 0.0) + 1e-12),
        bitwise_identical=bool(torch.equal(zf, reps[0])),
        last_week_sensitivity=float((zl - reps[0]).abs().max()))
    assert fut <= rep_floor + 1e-12, f"{task}: a change after t0 moved predictions beyond the device's own repeat floor"
    assert R[f"F3_{task}_{spec['arch']}_h{spec['depth']}"]["last_week_sensitivity"] > 1e-4, f"{task}: test is insensitive"
    del D, m
    P5._DEV_CACHE.clear()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()

# ---------------------------------------------------------------- F4
for w in ("v6", "v7"):
    y = P5.labels(w, "fill_rate").label_value.to_numpy(float)
    R[f"F4_{w}_partition_exact_on_{len(y)}_labels"] = bool(np.array_equal(fill_to_legacy(np.eye(22)[fill_cell(y)]).argmax(1), legacy_bin(y)))
    assert R[f"F4_{w}_partition_exact_on_{len(y)}_labels"]

json.dump(R, open(os.path.join(ARTIFACTS, "phase6_verify.json"), "w"), indent=1, default=str)
for k, v in R.items():
    print(f"{k}: {v}")
print("ALL PHASE 6 CHECKS PASSED")
