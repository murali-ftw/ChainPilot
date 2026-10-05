"""Phase 23AC T2 -- measurement of the DORMANT fill path (ml/serve/fill_consolidated.py). CPU only, no training.

1  P3 (bit-exactness): serve 200 fixed test rows -- the first 40 rows of each of the test snapshots at sorted indices
   [0, 2, 4, 6, 8] -- and compare with each bundle's stored preds_test.npz "P", rows matched by position in the stored
   order (phase5_heads.ordered over the test mask; label row, entity and Y asserted equal). Records per-seed max |diff| and
   the 5-seed mean's max |diff| against the mean of the stored per-seed P. P3 is right iff every one is exactly 0.0.
   (The stored predictions were made on MPS; this serves on CPU.)
2  FEATURES: the served raw per-row season + cadence matrix against the stored Phase 22 path
   (phase22_rows.RowStore22(["season", "cadence"], world="v8").raw(rows)) -- max |diff| and NaN-pattern equality.
3  SEASON POISON on 3 test snapshots (sorted indices [0, 4, 8]): every plan row with as_of_date + 2 d > t0 gets random
   requirement values; season rebuilt through ml/serve/_plan_reader.py must change 0 of N rows. Constructed offender: a reader
   with no bound (plan rows filtered on week_start only, future-recorded versions included and stamped recorded at t0) must
   change N of N rows. The offender on the UNpoisoned plan is also recorded (the size of the real leak the bound prevents).

Writes ml/artifacts/phase23ac/t2/serve.json, ml/artifacts/phase23ac/t2/serve.log and reports/part2/phase23ac/t2/serve.csv.

    HADES_DEVICE=cpu python ml/eval/phase23ac_t2_serve.py
"""
from __future__ import annotations
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "serve"), os.path.join(HERE, "..", "data"),
                os.path.join(HERE, "..", "train")]
import phase12_common as C
import numpy as np, pandas as pd

OUT = os.path.join(C.ART, "phase23ac", "t2")
CSV = os.path.join(C.REPO, "reports", "part2", "phase23ac", "t2", "serve.csv")
SERVE_IDX = [0, 2, 4, 6, 8]
POISON_IDX = [0, 4, 8]
ROWS_PER_SNAPSHOT = 40
_LOG = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True); _LOG.append(line)


def season_rows(season, n):
    return np.repeat(np.asarray(season, np.float64)[None, :], n, 0)


def rows_changed(A, B):
    same = (A == B) | (np.isnan(A) & np.isnan(B))
    return int((~same.all(1)).sum())


def offender_season(S, t0):
    """CONSTRUCTED OFFENDER (never used to serve): plan rows chosen on week_start only, every version included whatever
    its as_of_date, stamped recorded at t0 so fwd_load's own recorded_ts filter and assertion let them through."""
    import fwd_load as FL
    t0 = pd.Timestamp(t0)
    p = S["pdw"]
    used = p[(p.week_start > t0) & (p.week_start <= t0 + pd.Timedelta(weeks=13))].copy()
    used["recorded_ts"] = t0
    S2 = dict(S); S2["pdw"] = used
    Lb, _ = FL.snapshot_features(S2, t0)
    return np.nanmean(Lb, axis=0).astype(np.float32)


def poison_plan(S, t0, seed):
    """Every plan row with as_of_date + 2 d > t0 gets a random requirement (uniform 0 .. 10 x the table max)."""
    import _plan_reader as PR
    p = S["pdw"].copy()
    fut = ((p.as_of_date + PR.PDW_MAX_LAG) > pd.Timestamp(t0)).to_numpy()
    rng = np.random.default_rng(seed)
    hi = float(p.gross_requirement_p50.max()) * 10.0
    p["gross_requirement_p50"] = p.gross_requirement_p50.astype(float)
    p.loc[fut, "gross_requirement_p50"] = rng.uniform(0.0, hi, int(fut.sum()))
    S2 = dict(S); S2["pdw"] = p
    return S2, int(fut.sum())


def main():
    st = C.require_clean()
    t_start = time.time()
    import torch
    import fill_consolidated as FC
    import _plan_reader as PR
    import phase22_rows as R22
    import loop as L
    assert L.DEV.type == "cpu", f"Phase 23AC serves on CPU only; device is {L.DEV} (set HADES_DEVICE=cpu)"
    os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.dirname(CSV), exist_ok=True)
    log(f"stamp {st['code_version']} device {L.DEV} torch {torch.__version__}")

    svc = FC.FillConsolidatedService()
    snaps = svc.test_snapshots()
    log(f"test snapshots: {len(snaps)}; serving indices {SERVE_IDX}, {ROWS_PER_SNAPSHOT} rows each")
    chosen = [snaps[i] for i in SERVE_IDX]
    out = svc.serve(chosen, ROWS_PER_SNAPSHOT)
    lb = svc.labels()[0]
    order, _ = svc._order
    rows, pos = out["rows"], out["pos"]
    assert len(rows) == ROWS_PER_SNAPSHOT * len(SERVE_IDX), len(rows)
    assert (pos >= 0).all() and np.array_equal(order[pos], rows), "served rows are not test rows in the stored order"

    # ---------------------------------------------------------------- 1  P3
    per_seed, stored_P, Y_ok = {}, [], True
    for M in svc.members:
        z = np.load(os.path.join(M["dir"], "preds_test.npz"))
        assert z["P"].shape[0] == len(order), "stored preds_test length differs from the test mask"
        Ps = z["P"][pos]
        y_served = lb.label_value.to_numpy(float)[rows]
        Y_ok &= bool(np.array_equal(z["Y"][pos], y_served))
        assert np.array_equal(lb.entity_id.to_numpy()[order[pos]], out["entity"]), "entity mismatch"
        d = np.abs(out["per_seed"][M["seed"]].astype(np.float64) - Ps.astype(np.float64))
        per_seed[M["seed"]] = dict(max_abs_diff=float(d.max()), n_cells_nonzero=int((d > 0).sum()),
                                   max_abs_diff_pfull=float(d[:, -1].max()), per_row=d.max(1))
        stored_P.append(Ps)
        log(f"seed {M['seed']}: max |served - stored| = {d.max():.3e} ({int((d > 0).sum())} of {d.size} cells nonzero)")
    assert Y_ok, "stored Y differs from the served rows' labels: rows are not matched"
    stored_mean = np.mean(stored_P, 0)
    dm = np.abs(out["mean"].astype(np.float64) - stored_mean.astype(np.float64))
    p3_exact = all(v["max_abs_diff"] == 0.0 for v in per_seed.values()) and float(dm.max()) == 0.0
    log(f"5-seed mean: max |served - stored| = {dm.max():.3e}; P3 (all exactly 0.0) -> {'RIGHT' if p3_exact else 'WRONG'}")

    # ---------------------------------------------------------------- 2  features vs the stored Phase 22 path
    sc = svc.members[0]["scaler"]
    for M in svc.members[1:]:
        assert M["scaler"]["cols"] == sc["cols"], "members disagree on scaler columns"
    R_srv = svc.raw(lb.iloc[rows], sc["cols"])
    rs = R22.RowStore22(["season", "cadence"], world="v8")
    R_sto = rs.raw(lb.iloc[rows])
    assert R_sto.shape == R_srv.shape, (R_sto.shape, R_srv.shape)
    nan_same = bool(np.array_equal(np.isnan(R_srv), np.isnan(R_sto)))
    fin = ~np.isnan(R_srv) & ~np.isnan(R_sto)
    fd = np.abs(R_srv - R_sto); fd[~fin] = 0.0
    feat = dict(max_abs_diff=float(fd.max()), nan_pattern_equal=nan_same, per_column_max=dict(zip(sc["cols"], fd.max(0).tolist())),
                rows=int(len(rows)), columns=sc["cols"])
    log(f"features: served vs RowStore22 stored max |diff| = {fd.max():.3e}; NaN pattern equal = {nan_same}")

    # ---------------------------------------------------------------- 3  season poison
    S, _ = svc.sources()
    poison = []
    for k, i in enumerate(POISON_IDX):
        t0 = pd.Timestamp(snaps[i])
        n = int((pd.to_datetime(lb.snapshot_date) == t0).sum())      # every fill row of the snapshot
        clean = svc.block(t0)["season"]
        Sp, n_poisoned = poison_plan(S, t0, 2310 + k)
        bounded = PR.snapshot_blocks(Sp, t0)["season"]
        off_p = offender_season(Sp, t0)
        off_c = offender_season(S, t0)
        A = season_rows(clean, n)
        r = dict(t0=str(t0.date()), N=n, plan_rows_poisoned=n_poisoned,
                 bounded_changed=rows_changed(A, season_rows(bounded, n)),
                 offender_changed=rows_changed(A, season_rows(off_p, n)),
                 offender_on_unpoisoned_plan_changed=rows_changed(A, season_rows(off_c, n)),
                 offender_max_abs_diff=float(np.nanmax(np.abs(off_p.astype(float) - clean.astype(float)))),
                 n_plan_rows_asserted=svc.block(t0)["n_plan_rows_asserted"])
        r["pass"] = r["bounded_changed"] == 0 and r["offender_changed"] == n
        poison.append(r)
        log(f"poison {r['t0']}: {n_poisoned} plan rows poisoned; bounded reader changed {r['bounded_changed']} of {n}; "
            f"offender changed {r['offender_changed']} of {n} (on the unpoisoned plan: {r['offender_on_unpoisoned_plan_changed']})")
    poison_pass = all(r["pass"] for r in poison)

    # ---------------------------------------------------------------- write
    res = dict(stamp=st, device=out["device"], seconds=round(time.time() - t_start, 1),
               bundles=out["members"], snapshots_served=[str(pd.Timestamp(s).date()) for s in chosen],
               rows_per_snapshot=ROWS_PER_SNAPSHOT, n_rows=int(len(rows)),
               P3=dict(per_seed={str(s): {k: v for k, v in d.items() if k != "per_row"} for s, d in per_seed.items()},
                       mean_max_abs_diff=float(dm.max()), mean_max_abs_diff_pfull=float(dm[:, -1].max()),
                       exact_zero=p3_exact, verdict="RIGHT" if p3_exact else "WRONG",
                       note="stored preds were written on MPS (Phase 22); served here on CPU"),
               features_vs_rowstore22=feat, season_poison=dict(snapshots=poison, gate_pass=poison_pass),
               plan_rows_asserted_per_snapshot={str(pd.Timestamp(t).date()): b["n_plan_rows_asserted"] for t, b in svc._blocks.items()})
    json.dump(res, open(os.path.join(OUT, "serve.json"), "w"), indent=1, default=str)
    tab = pd.DataFrame(dict(snapshot_date=pd.to_datetime(out["snapshot"]).strftime("%Y-%m-%d"), entity_id=out["entity"],
                            label_row=rows, stored_pos=pos,
                            **{f"max_abs_diff_s{s}": d["per_row"] for s, d in per_seed.items()},
                            max_abs_diff_mean=dm.max(1), served_mean_pfull=out["mean"][:, -1],
                            stored_mean_pfull=stored_mean[:, -1], feature_max_abs_diff=fd.max(1)))
    tab.to_csv(CSV, index=False)
    log(f"wrote {os.path.join(OUT, 'serve.json')} and {CSV}")
    open(os.path.join(OUT, "serve.log"), "w").write("\n".join(_LOG) + "\n")
    assert poison_pass, "season poison gate FAILED (see serve.json)"
    return res


if __name__ == "__main__":
    main()
