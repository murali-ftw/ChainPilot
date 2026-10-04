"""Phase 22 -- tests of ml/data/clean_panel.py, each with a constructed offender the test MUST flag.

  future poison     at sampled weeks t: every po_lines quantity whose order is visible after t, every receipt (quantity and
                    lead) visible after t, every declared capacity recorded after t is replaced with noise. The nine clean
                    columns at row t must NOT change. OFFENDER: the generator's formula (lead and fill keyed to the ORDER week,
                    load over the whole month) computed from the same poisoned CSVs MUST change.
  self-exclusion    for sampled snapshot rows (lines raised after t0), each row's OWN line is given a receipt visible at t0
                    with arbitrary quantity and lead. The clean columns at row t0 must NOT change (the line's order is not
                    visible by t0). OFFENDER: clean_columns(require_order_visible=False) MUST change.

  HADES_DATA_ROOT=<main checkout> python ml/tests/test_phase22_clean_panel.py [--world v8]
"""
from __future__ import annotations
import os, sys, json, copy, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "data"), os.path.join(HERE, ".."), os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd
import config
import phase21_paths as PP
import clean_panel as CP


def offender_columns(S):
    """The generator's leaking formulas from the CSVs: outcome keyed to the ORDER week; load = whole-month total."""
    NCH, T = S.NCH, S.T
    m = np.isfinite(S.lead) & (S.vw_ord < T)
    leadw = np.full((NCH, T), np.nan); leadw[S.chan[m], S.vw_ord[m]] = S.lead[m]
    k = (S.vw_ord < T)
    fv = np.where(S.recv > 0, np.minimum(S.recv / np.maximum(S.qty, 1), 1.0), 0.0)
    cw = S.chan[k] * T + S.vw_ord[k]
    num = np.bincount(cw, weights=fv[k] * S.qty[k], minlength=NCH * T); den = np.bincount(cw, weights=S.qty[k], minlength=NCH * T)
    fill = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan).reshape(NCH, T)
    return {"lead_time_actual_days": CP._ffill(leadw), "fill_rate": CP._ffill(fill)}


def poisoned(S, t, rng):
    P = copy.copy(S)
    for nm in ("qty", "recv", "lead", "decl_q"):
        setattr(P, nm, getattr(S, nm).copy())
    o = S.vw_ord > t
    P.qty[o] = rng.integers(1, 5000, o.sum())
    g = np.maximum(S.vw_rec, S.vw_ord) > t
    P.recv[g] = rng.integers(0, 5000, g.sum()); P.lead[g] = rng.uniform(3, 200, g.sum())
    d = S.decl_vw > t
    P.decl_q[d] = rng.uniform(60, 5e4, d.sum())
    return P


def at(cols, t):
    return {c: np.nan_to_num(v[:, t], nan=-999.0) for c, v in cols.items()}


def changed(a, b):
    return {c: int((np.abs(a[c] - b[c]) > 1e-9).sum()) for c in a}


def test_future_poison(S, weeks, rng):
    res = []
    for t in weeks:
        P = poisoned(S, t, rng)
        base, pois = at(CP.clean_columns(S), t), at(CP.clean_columns(P), t)
        ob, op = at(offender_columns(S), t), at(offender_columns(P), t)
        res.append(dict(week=int(t), clean_changed=changed(base, pois), offender_changed=changed(ob, op)))
    assert all(sum(r["clean_changed"].values()) == 0 for r in res), f"future poison changed a clean column: {res}"
    assert all(sum(r["offender_changed"].values()) > 0 for r in res), "the offender was not flagged -- the test cannot fire"
    return dict(verdict="PASS", weeks=res)


def test_self_exclusion(S, world, rng, n_rows=400):
    D = config.WORLDS[world]
    lb = pd.read_csv(D + "/training_labels.csv", usecols=["snapshot_date", "entity_id", "task"])
    lb = lb[lb.task == "arrival_week"]
    snaps = sorted(lb.snapshot_date.unique())
    lidx = pd.Series(np.arange(len(S.line_id)), index=S.line_id)
    res = []
    for sd in (snaps[40], snaps[60], snaps[75]):
        t = int((pd.Timestamp(sd) - CP.W0).days // 7)
        ents = lb[lb.snapshot_date == sd].entity_id.to_numpy()[:n_rows]
        li = lidx.reindex(ents).to_numpy().astype(np.int64)
        out = {}
        for kind, req in (("builder", True), ("offender (order visibility dropped)", False)):
            base = at(CP.clean_columns(S, require_order_visible=req), t)
            P = copy.copy(S)
            for nm in ("vw_rec", "recv", "lead"):
                setattr(P, nm, getattr(S, nm).copy())
            P.vw_rec[li] = t; P.recv[li] = rng.integers(0, 5000, len(li)); P.lead[li] = rng.uniform(3, 200, len(li))
            ch = S.chan[li]
            after = at(CP.clean_columns(P, require_order_visible=req), t)
            out[kind] = int(sum((np.abs(base[c][ch] - after[c][ch]) > 1e-9).sum() for c in base))
        res.append(dict(snapshot=str(sd), rows=len(li), own_line_order_visible_by_t0=int((S.vw_ord[li] <= t).sum()), **out))
    assert all(r["builder"] == 0 and r["own_line_order_visible_by_t0"] == 0 for r in res), f"a row's own outcome entered its row: {res}"
    assert all(r["offender (order visibility dropped)"] > 0 for r in res), "the offender was not flagged -- the test cannot fire"
    return dict(verdict="PASS", snapshots=res)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--world", default="v8"); a = ap.parse_args()
    PP.register()
    import phase12_common as C
    meta = json.load(open(os.path.join(config.CACHE, a.world, "meta.json")))
    S = CP.CsvSources(a.world, meta["T"])
    rng = np.random.default_rng(220)
    weeks = [int((pd.Timestamp(d) - CP.W0).days // 7) for d in ("2019-07-01", "2022-03-07", "2025-06-02")]
    out = dict(stamp=C.stamp(), world=a.world, future_poison=test_future_poison(S, weeks, rng))
    print(json.dumps(out["future_poison"]), flush=True)
    out["self_exclusion"] = test_self_exclusion(S, a.world, rng)
    print(json.dumps(out["self_exclusion"]), flush=True)
    os.makedirs(os.path.join(config.ARTIFACTS, "phase22"), exist_ok=True)
    json.dump(out, open(os.path.join(config.ARTIFACTS, "phase22", f"clean_panel_tests_{a.world}.json"), "w"), indent=1, default=str)
    print("ALL PASS")


if __name__ == "__main__":
    main()
