"""Phase 21 -- tests of the group-statistics builder (ml/data/grpstats.py). Each test carries its falsification path.

  unit (synthetic, no data):
    test_backoff_monotone      a thin L5 resolves to a coarser level; a rich one to L5   (falsify: k = 0 resolves everything to L5)
    test_shrinkage_limits      n -> inf gives the raw statistic; n = 0 gives the parent
    test_km_censoring          a group with 50% open long-running lines: KM median > completed-only median
  on the world (Stage 1d, run before any feature is used, and again in the isolation audit):
    test_future_poison         every line / receipt / acknowledgement / zero signal recorded after tau is replaced with
                               noise; the statistics at tau do not change. FALSIFICATION: the same test on the constructed
                               offender (receipts filtered on EVENT time) must FAIL, or the test cannot fire.
    test_self_exclusion        each sampled row's OWN line is given a receipt, an acknowledgement and a zero signal recorded
                               BEFORE tau; the row's statistics do not change (its line is not in its own history).
                               FALSIFICATION: admitting the row's own line (line_rec forced <= tau) must change them.

  python ml/tests/test_phase21_grpstats.py [--world v8] [--unit-only]   -> ml/artifacts/phase21/selftest_{world}.json
"""
from __future__ import annotations
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "data"), os.path.join(HERE, ".."), os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd
import grpstats as GS
import phase21_paths as PP


# ================================================================== synthetic
def _Z(n5, n4, n2, raw5=1.0, raw4=2.0, raw2=3.0, raw1=4.0):
    nA = len(GS.ASTATS); iN = GS.ASTATS.index("n")
    def row(v, n):
        x = np.full((1, nA), v, np.float32); x[0, iN] = n; return x
    Z = {"A_L1": row(raw1, 1e6), "A_L2": row(raw2, n2), "A_L4": row(raw4, n4),
         "A_L5c": np.repeat(row(raw5, n5)[:, None, :], 12, 1), "A_L3c": np.repeat(row(raw2, n2)[:, None, :], 12, 1),
         "month": np.zeros(1, np.int8)}
    return Z


def test_backoff_monotone():
    k = 10
    cases = {(0, 50, 500): 4, (3, 50, 500): 4, (20, 50, 500): 5, (0, 2, 500): 2, (0, 0, 5): 1}
    for (n5, n4, n2), want in cases.items():
        _, _, res5, _ = GS.arrival_levels(_Z(n5, n4, n2), k, np.zeros(1, np.int64))
        assert int(res5[0]) == want, f"n=({n5},{n4},{n2}) resolved to L{res5[0]}, expected L{want}"
    # falsification: with k = 0 every non-empty level counts as rich, so the thin L5 no longer backs off
    _, _, res5, _ = GS.arrival_levels(_Z(1, 50, 500), 0, np.zeros(1, np.int64))
    assert int(res5[0]) == 5
    return "PASS"


def test_shrinkage_limits():
    i50 = GS.ASTATS.index("lead_q50")
    sh, _, _, _ = GS.arrival_levels(_Z(1e9, 1e9, 1e9), 10, np.zeros(1, np.int64))
    assert abs(sh["L5"][0, i50] - 1.0) < 1e-6, "n -> inf must give the raw statistic"
    sh, _, _, _ = GS.arrival_levels(_Z(0, 0, 0), 10, np.zeros(1, np.int64))
    assert abs(sh["L5"][0, i50] - 4.0) < 1e-9, "n = 0 at every level must give the global value"
    sh, _, _, _ = GS.arrival_levels(_Z(0, 1e9, 0), 10, np.zeros(1, np.int64))
    assert abs(sh["L5"][0, i50] - 2.0) < 1e-6, "n5 = 0 must give the (rich) parent L4"
    sh, _, _, _ = GS.arrival_levels(_Z(10, 1e9, 0), 10, np.zeros(1, np.int64))
    assert abs(sh["L5"][0, i50] - 1.5) < 1e-6, "n = k must weight raw and parent equally"
    return "PASS"


def test_km_censoring():
    r = np.random.default_rng(0)
    T = np.r_[r.uniform(10, 30, 50), r.uniform(25, 60, 50)]
    E = np.r_[np.ones(50, bool), np.zeros(50, bool)]
    g = np.zeros(100, np.int64)
    km = GS.km_groups(g, T, E, 1)[0.5][0]
    comp = GS.km_median_completed_only(g, T, E, 1)[0]
    assert km > comp, f"KM median {km} must exceed completed-only {comp} with 50% long open lines"
    # no censoring: KM median equals the empirical (lower) median
    km0 = GS.km_groups(g, T, np.ones(100, bool), 1)[0.5][0]
    assert abs(km0 - np.sort(T)[49]) < 1e-12
    # a known small case: times 1..4 all events -> S = .75 .5 .25 0 -> median = 2
    assert GS.km_groups(np.zeros(4, np.int64), np.arange(1., 5.), np.ones(4, bool), 1)[0.5][0] == 2.0
    return dict(verdict="PASS", km_median=float(km), completed_only_median=float(comp))


# ================================================================== on the world
def _stats(B, tau, chan, month):
    A, _, elig = B.arrival(tau)
    F, _ = B.fill(tau)
    K = B.keys(chan, month)
    parts = [A[L][K[L]] for L in GS.LEVELS] + [F[L][K[L]] for L in GS.LEVELS] + \
            [F[L + "_hist"][K[L]] for L in ("L5", "L4", "L2", "L1")] + [F["L4_ack"][chan], F["L2_ack"][B.s.ch_sup[chan]]]
    return np.concatenate([np.nan_to_num(p.reshape(len(chan), -1), nan=-999.0) for p in parts], 1), elig


def sample(world, n_tau=3, n_rows=400, seed=0):
    a, _ = GS.snapshot_rows(world)
    taus = np.sort(pd.unique(a.snapshot_date))
    pick = taus[np.linspace(0, len(taus) - 1, n_tau).astype(int)]
    r = np.random.default_rng(seed)
    out = []
    for t in pick:
        rows = a[a.snapshot_date == t]
        rows = rows.iloc[r.choice(len(rows), min(n_rows, len(rows)), replace=False)]
        out.append((np.datetime64(pd.Timestamp(t), "ns"), rows.entity_id.to_numpy()))
    return out


def test_future_poison(src, samples):
    res = {}
    for leak in (False, True):
        changed = []
        for tau, ents in samples:
            li = src.lidx.reindex(ents).to_numpy().astype(np.int64)
            chan, month = src.chan[li], np.full(len(li), pd.Timestamp(tau).month - 1)
            a, _ = _stats(GS.Builder(src, leak=leak), tau, chan, month)
            b, _ = _stats(GS.Builder(src.poisoned(tau, seed=1), leak=leak), tau, chan, month)
            changed.append(int((np.abs(a - b) > 1e-9).any(1).sum()))
        res["offender (event-time filter)" if leak else "builder"] = changed
    assert sum(res["builder"]) == 0, f"future poison CHANGED the builder's statistics: {res['builder']}"
    assert sum(res["offender (event-time filter)"]) > 0, "the constructed offender was not caught -- the test cannot fire"
    return dict(verdict="PASS", rows_changed=res)


def test_self_exclusion(src, samples):
    import copy
    res = {}
    for admit in (False, True):
        changed = []
        for tau, ents in samples:
            li = src.lidx.reindex(ents).to_numpy().astype(np.int64)
            chan, month = src.chan[li], np.full(len(li), pd.Timestamp(tau).month - 1)
            base = copy.copy(src)
            if admit:                                    # falsification: the row's own line is admitted to the history
                base.line_rec = src.line_rec.copy(); base.line_rec[li] = tau - GS.DAY
                base.created = src.created.copy(); base.created[li] = tau - 30 * GS.DAY
            a, elig = _stats(GS.Builder(base), tau, chan, month)
            if not admit:
                assert not elig[li].any(), "a row's own line is in its own history"
            s = copy.copy(base)
            for nm in ("g_event", "g_rec", "g_qty", "g_final", "k_rec", "k_gap", "k_notfull", "z_rec"):
                setattr(s, nm, getattr(base, nm).copy())
            s.g_rec[li] = tau - GS.DAY; s.g_event[li] = tau - 2 * GS.DAY; s.g_qty[li] = 0.0; s.g_final[li] = True
            s.k_rec[li] = tau - GS.DAY; s.k_gap[li] = 1.0; s.k_notfull[li] = True; s.z_rec[li] = tau - GS.DAY
            b, _ = _stats(GS.Builder(s), tau, chan, month)
            changed.append(int((np.abs(a - b) > 1e-9).any(1).sum()))
        res["own line admitted (offender)" if admit else "builder"] = changed
    assert sum(res["builder"]) == 0, f"a row's own outcome entered its statistics: {res['builder']}"
    assert sum(res["own line admitted (offender)"]) > 0, "admitting the own line changed nothing -- the test cannot fire"
    return dict(verdict="PASS", rows_changed=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="v8")
    ap.add_argument("--unit-only", action="store_true")
    a = ap.parse_args()
    out = {"backoff_monotone": test_backoff_monotone(), "shrinkage_limits": test_shrinkage_limits(),
           "km_censoring": test_km_censoring()}
    print(json.dumps(out, indent=1), flush=True)
    if not a.unit_only:
        PP.register()
        import phase12_common as C
        out["stamp"] = C.stamp()
        src = GS.Source(a.world)
        smp = sample(a.world)
        out["samples"] = [(str(t)[:10], len(e)) for t, e in smp]
        out["future_poison"] = test_future_poison(src, smp)
        print(json.dumps(out["future_poison"]), flush=True)
        out["self_exclusion"] = test_self_exclusion(src, smp)
        print(json.dumps(out["self_exclusion"]), flush=True)
        os.makedirs(GS.OUT, exist_ok=True)
        json.dump(out, open(os.path.join(GS.OUT, f"selftest_{a.world}.json"), "w"), indent=1, default=str)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
