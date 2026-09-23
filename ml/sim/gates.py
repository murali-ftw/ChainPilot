"""Phase 9's acceptance gates, made runnable on this repo — BEFORE any simulation is built.

Deviation 63: guide 9.2's G10 calls `ground_truth.realised_correlation_matrix()`, and
`db/ground_truth.py` does not exist. Stage 5.1 establishes that it never existed here: the two
`*ground_truth*.py` modules in this repo's history belong to the ABANDONED August 2026 project
built on `db/generate_dataset.py` (itself gone), and the v4 reboot of 2026-09-09 (`ab4ad00`)
brought gen_v6/gen_v7, `schema.sql` and `dataset_structure.md` in fresh without porting them.
G10 has therefore been uncallable for the whole life of this codebase.

Building a simulation whose acceptance gate cannot run is the pattern this project has caught ten
times. So both gates are specified and DEMONSTRATED FAILING here, before `ml/sim/montecarlo.py`
exists.

--------------------------------------------------------------------------------------------
G10 REPLACEMENT — correlation, referenced to DATA rather than to the generator's internals

The original compares simulated correlation against a function inside the generator. A modelling
team never sees that. The replacement takes its reference from the CSVs, out of sample:

    reference = realised within-supplier-group correlation of OBSERVED weekly shortfalls,
                computed from channel_performance_weekly over a window BEFORE t0
    gate      = | simulated within-group correlation - reference | <= 0.05

It fails on the failure it exists to catch: a simulation that samples channels independently
produces r ~ 0 against a reference of ~0.21, so |delta| ~ 0.21 and the gate fires.

**WHAT THIS GATE CANNOT DO, STATED SO NOBODY ASSUMES IT.** On v8 (seed 1001, 2019-2025, 420
suppliers x 365 weeks) the within-group mean correlation is **+0.2101** and the CROSS-group mean
is **+0.2055** — a separation of **+0.0046**, an order of magnitude inside the gate's own +-0.05
tolerance. Every supplier shares a common factor of about r = 0.21 and `supplier_group_id` adds
essentially nothing on top of it. So G10 tests that the simulation reproduces the CORRELATION
LEVEL; it CANNOT test that the copula recovered GROUP STRUCTURE, because a per-group rho and a
single global rho are indistinguishable on this world. That is reported, never gated — gating it
would be a check that cannot fail (guide troubleshooting, instance 11).

This corroborates the guide's own 9.2 warning from a different direction: no result from this
dataset supports the "independence understates exposure" business claim.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TOL_G10 = 0.05


# ----------------------------------------------------------------- reference, from the CSVs
def observed_shortfall_matrix(csv_dir, lo="2019-01-01", hi="2025-12-31"):
    """[supplier x week] count of active channel-weeks that under-delivered. From data only."""
    cpw = pd.read_csv(f"{csv_dir}/channel_performance_weekly.csv",
                      usecols=["channel_id", "week_start", "qty_ordered", "qty_received",
                               "is_active_week"])
    ch = pd.read_csv(f"{csv_dir}/sourcing_channels.csv", usecols=["channel_id", "supplier_id"])
    su = pd.read_csv(f"{csv_dir}/suppliers.csv", usecols=["supplier_id", "supplier_group_id"])
    d = cpw.merge(ch, on="channel_id").merge(su, on="supplier_id")
    d = d[(d.week_start >= lo) & (d.week_start <= hi)]
    act = d.is_active_week.astype(str).str.lower().isin(["true", "1"])
    d["short"] = ((d.qty_received < d.qty_ordered) & act).astype(float)
    m = d.groupby(["supplier_id", "week_start"])["short"].sum().unstack(fill_value=0.0)
    groups = su.set_index("supplier_id").supplier_group_id
    return m, groups.reindex(m.index).to_numpy()


def within_group_correlation(matrix, groups):
    """-> (within mean, cross mean, separation). `matrix` is [entity x time]."""
    X = np.asarray(matrix, float)
    keep = X.std(1) > 0
    X, g = X[keep], np.asarray(groups)[keep]
    if len(X) < 2:
        return float("nan"), float("nan"), float("nan")
    C = np.corrcoef(X)
    iu = np.triu_indices(len(X), 1)
    same = g[iu[0]] == g[iu[1]]
    w = C[iu][same]; b = C[iu][~same]
    wm = float(w.mean()) if len(w) else float("nan")
    bm = float(b.mean()) if len(b) else float("nan")
    return wm, bm, wm - bm


def g10_correlation_gate(sim_matrix, sim_groups, reference_within, tol=TOL_G10):
    """G10: the simulation's realised within-group correlation must match the DATA's.

    Fires when a simulation samples independently (r ~ 0 against a reference of ~0.21).
    """
    w, b, sep = within_group_correlation(sim_matrix, sim_groups)
    delta = abs(w - reference_within)
    return dict(gate="G10_correlation_level", simulated_within=w, simulated_cross=b,
                simulated_separation=sep, reference_within=reference_within,
                abs_delta=delta, tol=tol, PASS=bool(delta <= tol),
                note="tests the correlation LEVEL only; group structure is untestable on v8 "
                     "(data separation +0.0046, inside this tolerance)")


# ----------------------------------------------------------------- 9.1's own gate
def g91_identity_gate(sim_available, stated_available, tol=0):
    """Guide 9.1 verify: with fill fixed at 1.0 and timing at the promise date, the roll-forward
    must reproduce `inventory_position_weekly.qty_available` EXACTLY.

    On v6/v7 that column was entirely zero, so the gate passed on any simulation that also
    produced zero — vacuous, instance 3 on this project's record. On v8 the column is populated
    (B1 cleared it: mean 1,135, max 22,973, reconciling at 100.000000%), so the gate discriminates.
    """
    a = np.asarray(sim_available, float); b = np.asarray(stated_available, float)
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    diff = np.abs(a - b)
    n_bad = int((diff > tol).sum())
    return dict(gate="G9.1_reproduces_qty_available", rows=int(a.size),
                mismatched=n_bad, max_abs_diff=float(diff.max()) if a.size else 0.0,
                pct_exact=round(100.0 * float((diff <= tol).mean()), 6) if a.size else 100.0,
                PASS=bool(n_bad == 0),
                reference_is_all_zero=bool(np.all(b == 0)),
                note=("REFERENCE IS ALL ZERO -- this gate is vacuous on this world (v6/v7 state)"
                      if np.all(b == 0) else "reference is populated; the gate discriminates"))
