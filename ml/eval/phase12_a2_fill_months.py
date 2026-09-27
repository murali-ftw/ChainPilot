"""Phase 12 Wave A2 -- is the fill head's interior-cell error CONCENTRATED in calendar months?

Gate for Test C2 (month-weighted fill). No training: the five shipped-configuration v8 fill heads
(`v8_none_h0_lr0.000125_s{7,17,27,37,47}`) are run in INFERENCE over the training fold, because the
Phase 5 section 6.4 diagnosis lives on the data the head was fitted to and training-fold predictions
were never stored. Validation and test predictions are read from the bundles.

CORRECTION CARRIED IN. The figures the brief quotes -- head 0.0445 against LightGBM 0.0049 -- are
Phase 5 section 6.4's v6 measurement (v7: 0.0537 / 0.0235). This measures v8's own.

DECOMPOSITION. With mean predicted cell mass p_c and observed cell frequency o_c over a fold, the
interior error is E = sum_{c in 1..20} |p_c - o_c|. Writing each as a row-weighted sum over months,
p_c - o_c = sum_m w_m (p_mc - o_mc), the month contribution
        K_m = sum_c sign(p_c - o_c) * w_m * (p_mc - o_mc)
sums EXACTLY to E. Concentration is judged against two references, because per-month interior cells
hold tens of rows and a raw per-month |error| is mostly sampling noise:
  * the ROW SHARE w_m: flat error means K_m / E ~ w_m;
  * a SNAPSHOT-PERMUTATION null: training snapshots (3-4 per calendar month) are reassigned to months
    at random, preserving each snapshot intact, and the concentration statistic recomputed.
Per-month own error S_m = sum_c |p_mc - o_mc| is reported beside its MULTINOMIAL NOISE FLOOR: the
value S_m takes if the head were exactly calibrated in that month and the rows were drawn from it.
"""
from __future__ import annotations
import os, json
import phase12_common as C
import numpy as np, pandas as pd, torch
import phase5_heads as P5, folds
from heads import fill_cell
import loop as L

K = 22
INTERIOR = np.arange(1, 21)


def train_fold_preds(seed, lb, tr):
    path = f"{C.BUND}/fill_rate/v8_none_h0_lr0.000125_s{seed}"
    B = L.load_bundle(path)
    D = P5.device_inputs("v8", np.sort(lb.snapshot_date[tr].unique()), B["cfg"]["wsla"],
                         graph_shuffle=B["cfg"].get("graph_shuffle"))
    assert np.allclose(D["norm_mu"], B["norm"]["mu"]) and np.allclose(D["norm_sd"], B["norm"]["sd"]), \
        "rebuilt normaliser differs from the bundle's -- inputs changed since training"
    m = L._materialise(B, D)
    with torch.no_grad():
        pr = P5.predict(m, D, lb, tr)
    # the bundle's own stored test predictions must be reproduced by this inference path, or the
    # training-fold numbers below are from a different model than the one that shipped
    return pr, D, m


def reproduce_stored(m, D, lb, te, seed):
    with torch.no_grad():
        pt = P5.predict(m, D, lb, te)
    z = np.load(f"{C.BUND}/fill_rate/v8_none_h0_lr0.000125_s{seed}/preds_test.npz")
    return float(np.abs(pt["P"] - z["P"]).max())


def cell_table(P, y):
    obs = np.bincount(fill_cell(y), minlength=K) / len(y)
    return P.mean(0), obs


def interior_err(P, y):
    p, o = cell_table(P, y)
    return float(np.abs(p - o)[INTERIOR].sum()), float(np.abs(p - o).sum())


def month_decomp(P, y, month):
    p, o = cell_table(P, y)
    sgn = np.sign(p - o)
    E = float(np.abs(p - o)[INTERIOR].sum())
    cells = fill_cell(y)
    n = len(y)
    out = {}
    for mo in range(1, 13):
        mk = month == mo
        if not mk.any():
            continue
        w = mk.sum() / n
        pm = P[mk].mean(0)
        om = np.bincount(cells[mk], minlength=K) / mk.sum()
        Km = float((sgn[INTERIOR] * w * (pm - om)[INTERIOR]).sum())
        Sm = float(np.abs(pm - om)[INTERIOR].sum())
        out[mo] = dict(row_share=float(w), contribution=Km, share_of_E=Km / E if E else np.nan,
                       own_interior_err=Sm, n_rows=int(mk.sum()))
    assert abs(sum(v["contribution"] for v in out.values()) - E) < 1e-9, "decomposition does not sum to E"
    return E, out


def noise_floor(P, month, B=200, seed=0):
    """Expected own-month interior error if the head were exactly calibrated in that month."""
    rng = np.random.default_rng(seed)
    res = {}
    for mo in np.unique(month):
        Pm = P[month == mo]
        pm = Pm.mean(0)
        vals = []
        for _ in range(B):
            # draw each row's cell from its own predicted distribution
            u = rng.random(len(Pm))[:, None]
            c = (np.cumsum(Pm, 1) < u).sum(1).clip(0, K - 1)
            om = np.bincount(c, minlength=K) / len(Pm)
            vals.append(np.abs(pm - om)[INTERIOR].sum())
        res[int(mo)] = dict(mean=float(np.mean(vals)), p95=float(np.quantile(vals, 0.95)))
    return res


def concentration(contrib, share):
    """Chi-square-like excess of contributions over row shares: 0 when error is exactly flat."""
    c, s = np.asarray(contrib), np.asarray(share)
    f = c / c.sum()
    return float(((f - s) ** 2 / s).sum())


def perm_null(P, y, snap, B=500, seed=0):
    """Reassign whole snapshots to months at random (same multiset of months), recompute."""
    rng = np.random.default_rng(seed)
    snaps = np.unique(snap)
    true_month = pd.to_datetime(snaps).month.to_numpy()
    E, d = month_decomp(P, y, pd.to_datetime(snap).month.to_numpy())
    obs = concentration([d[m]["contribution"] for m in sorted(d)], [d[m]["row_share"] for m in sorted(d)])
    null = []
    sidx = pd.Index(snaps).get_indexer(snap)
    for _ in range(B):
        pm = rng.permutation(true_month)
        _, dp = month_decomp(P, y, pm[sidx])
        null.append(concentration([dp[m]["contribution"] for m in sorted(dp)], [dp[m]["row_share"] for m in sorted(dp)]))
    null = np.array(null)
    return dict(stat=obs, null_mean=float(null.mean()), null_p95=float(np.quantile(null, 0.95)),
                p_perm=float((null >= obs).mean()), B=B)


def falsify_detector(P, y, snap):
    """The concentration test must be CAPABLE OF FIRING. Construct an input where the error is put
    in one month: in every October snapshot, move 60% of the predicted interior mass onto the
    complete-fill atom. If the permutation test does not flag that, it cannot flag anything."""
    P2 = P.copy()
    mo = pd.to_datetime(snap).month.to_numpy()
    k = mo == 10
    moved = P2[k][:, INTERIOR].sum(1) * 0.6
    P2[np.ix_(k, INTERIOR)] *= 0.4
    P2[k, 21] += moved
    return perm_null(P2, y, snap, B=300, seed=1)


def main():
    st = C.require_clean()
    lb = P5.labels("v8", "fill_rate")
    tr, va, te = folds.fixed_split(lb.snapshot_date)
    folds.assert_no_leak(lb.snapshot_date, tr, va, te)
    res = dict(stamp=st, seeds=list(C.V8_SEEDS))

    # A2.2 occupancy, all three folds, actual counts
    occ = {}
    for name, m in (("train", tr), ("val", va), ("test", te)):
        c = np.bincount(fill_cell(lb.label_value.to_numpy(float)[m]), minlength=K)
        occ[name] = dict(n=int(m.sum()), counts=c.tolist(), atom0=float(c[0] / m.sum()),
                         atom1=float(c[21] / m.sum()), interior_total=float(c[INTERIOR].sum() / m.sum()),
                         interior_min=int(c[INTERIOR].min()), interior_max=int(c[INTERIOR].max()),
                         interior_median=float(np.median(c[INTERIOR])),
                         interior_share_min=float(c[INTERIOR].min() / m.sum()),
                         interior_share_max=float(c[INTERIOR].max() / m.sum()))
    res["occupancy"] = occ

    per_seed, trainP = {}, []
    for s in C.V8_SEEDS:
        pr, D, m = train_fold_preds(s, lb, tr)
        repro = reproduce_stored(m, D, lb, te, s)
        assert repro < 1e-4, f"seed {s}: inference path does not reproduce stored test predictions ({repro})"
        o_tr = P5.ordered(lb, tr)
        assert np.allclose(pr["Y"], lb.label_value.values[o_tr])
        zt = np.load(f"{C.BUND}/fill_rate/v8_none_h0_lr0.000125_s{s}/preds_test.npz")
        zv = np.load(f"{C.BUND}/fill_rate/v8_none_h0_lr0.000125_s{s}/preds_val.npz")
        per_seed[s] = dict(train=interior_err(pr["P"], pr["Y"]), val=interior_err(zv["P"], zv["Y"]),
                           test=interior_err(zt["P"], zt["Y"]), reproduce_max_abs=repro)
        trainP.append(pr["P"])
        ytr = pr["Y"]
        del D, m
    res["per_seed_interior_and_total_err"] = {str(k): v for k, v in per_seed.items()}

    # LightGBM b5flat22 raw, val/test (train-fold predictions for it were never stored and fitting is
    # training, so its training marginal is NOT measured on v8 in this wave)
    lg = {}
    for s in C.V8_SEEDS:
        for f in ("val", "test"):
            z = np.load(f"{C.ART}/phase7_preds/v8_fill_rate_b5flat22_s{s}_{f}.npz")
            lg.setdefault(f, []).append(interior_err(z["P"], z["Y"]))
    res["b5flat22_raw"] = {f: dict(interior=[float(x[0]) for x in v], total=[float(x[1]) for x in v]) for f, v in lg.items()}

    # month decomposition on the SEED-MEAN prediction, train (where the failure lives) and test
    snap_tr = lb.snapshot_date.values[P5.ordered(lb, tr)]
    Ptr = np.mean(trainP, 0)
    out = {}
    for name, P, y, snap in (("train", Ptr, ytr, snap_tr),):
        mo = pd.to_datetime(snap).month.to_numpy()
        E, d = month_decomp(P, y, mo)
        nf = noise_floor(P, mo)
        for k in d:
            d[k].update(noise_floor_mean=nf[k]["mean"], noise_floor_p95=nf[k]["p95"],
                        n_snapshots=int(len(np.unique(snap[mo == k]))))
        out[name] = dict(E=E, months=d, perm=perm_null(P, y, snap))
    zt = [np.load(f"{C.BUND}/fill_rate/v8_none_h0_lr0.000125_s{s}/preds_test.npz") for s in C.V8_SEEDS]
    Pte, yte = np.mean([z["P"] for z in zt], 0), zt[0]["Y"]
    snap_te = lb.snapshot_date.values[P5.ordered(lb, te)]
    mo = pd.to_datetime(snap_te).month.to_numpy()
    E, d = month_decomp(Pte, yte, mo)
    nf = noise_floor(Pte, mo)
    for k in d:
        d[k].update(noise_floor_mean=nf[k]["mean"], noise_floor_p95=nf[k]["p95"], n_snapshots=1)
    out["test"] = dict(E=E, months=d, perm=perm_null(Pte, yte, snap_te))
    # per-seed month profiles on train: is any month's share stable across seeds?
    prof = []
    for P in trainP:
        _, dd = month_decomp(P, ytr, pd.to_datetime(snap_tr).month.to_numpy())
        prof.append([dd[m]["share_of_E"] for m in sorted(dd)])
    prof = np.array(prof)
    out["train"]["share_seed_range"] = {int(m): [float(prof[:, i].min()), float(prof[:, i].max())]
                                        for i, m in enumerate(sorted(dd))}
    out["falsification_one_month_injected"] = falsify_detector(Ptr, ytr, snap_tr)
    res["decomposition"] = out
    print(C.dump(res, "phase12_a2.json"))


if __name__ == "__main__":
    main()
