#!/usr/bin/env python3
"""Phase 2 gate follow-up — coefficient sweep, stratified AUC, per-variant ceiling.

Extends docs/phase2_coverage_recheck.md. Disposable prototype code: Mechanisms E/F
are NOT built, none of this commits to generate_dataset.py's mechanism logic.

1. Sweep the resilience->outcome coupling coefficient (prototype default 0.7) with
   Mechanism H held at spec defaults -- the one lever no earlier run varied.
2. Stratify the continuous estimator's AUC by supplier shipment volume, to tell a
   uniformly weak signal from a strong one diluted by thin suppliers.
3. Recompute the zero-shipper ceiling under Variant F's and Variant K's actual
   compositions, which differ in whether Mechanism A is present.

Usage:  python3 db/phase2_gate_followup.py
"""
import contextlib, io, json, math, os, random, statistics, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "generate_dataset.py")
CFG = {"sup_n": 4000, "snapshots": 40}
LAMBDAS = [0.7, 1.0, 1.3, 1.6, 2.0]


def run_generator(overrides, variant="0"):
    src = open(GEN).read().replace(
        "def write(name, header, rows):", "def write(name, header, rows):\n    return", 1)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(overrides, f)
        cfg_path = f.name
    argv = sys.argv
    sys.argv = ["generate_dataset.py", "--variant", variant, "--config", cfg_path]
    ns = {"__name__": "__followup__", "__file__": GEN}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(compile(src, GEN, "exec"), ns)
            except SystemExit:
                pass
    finally:
        sys.argv = argv
        os.unlink(cfg_path)
    return ns


def auc(sc, lb):
    pr = sorted(zip(sc, lb)); n = len(pr); rk = [0.0]*n; i = 0
    while i < n:
        j = i
        while j+1 < n and pr[j+1][0] == pr[i][0]: j += 1
        for k in range(i, j+1): rk[k] = (i+j)/2.0 + 1.0
        i = j+1
    p = sum(l for _, l in pr); q = n - p
    if not p or not q: return float("nan")
    return (sum(r for r, (_, l) in zip(rk, pr) if l == 1) - p*(p+1)/2.0)/(p*q)


def cv_auc(X, y, folds=5, iters=300, lr=0.3, boots=300):
    if len(X) < 60 or len(set(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    mu = [statistics.fmean(r[k] for r in X) for k in range(len(X[0]))]
    sg = [statistics.pstdev([r[k] for r in X]) or 1.0 for k in range(len(X[0]))]
    Z = [[(r[k]-mu[k])/sg[k] for k in range(len(r))] for r in X]
    idx = list(range(len(Z))); random.Random(31337).shuffle(idx)
    fold = {i: p % folds for p, i in enumerate(idx)}
    sc, yy = [], []
    for f in range(folds):
        tr = [i for i in range(len(Z)) if fold[i] != f]
        te = [i for i in range(len(Z)) if fold[i] == f]
        if not te or len(set(y[i] for i in tr)) < 2: continue
        d = len(Z[0]); w = [0.0]*d; b = 0.0
        for _ in range(iters):
            gw = [0.0]*d; gb = 0.0
            for i in tr:
                z = b + sum(w[k]*Z[i][k] for k in range(d))
                e = 1.0/(1.0+math.exp(-max(-30.0, min(30.0, z)))) - y[i]
                for k in range(d): gw[k] += e*Z[i][k]
                gb += e
            for k in range(d): w[k] -= lr*gw[k]/len(tr)
            b -= lr*gb/len(tr)
        sc += [b + sum(w[k]*Z[i][k] for k in range(d)) for i in te]
        yy += [y[i] for i in te]
    a = auc(sc, yy)
    rng = random.Random(7); bs = []
    for _ in range(boots):
        p = [rng.randrange(len(sc)) for _ in range(len(sc))]
        v = auc([sc[i] for i in p], [yy[i] for i in p])
        if v == v: bs.append(v)
    bs.sort()
    return a, bs[int(0.025*len(bs))], bs[int(0.975*len(bs))-1]


def simulate(ns, res, lam, seed=90210):
    """Re-simulate outcomes over the real shipment set at coupling `lam`.

    Same functional form as the recheck prototype:
        p_delay = 0.025 + 0.38 * stress * (1 - lam*resilience)
    The multiplier is clamped at 0 -- above lam*res = 1 a supplier would otherwise
    get a NEGATIVE delay boost. Clamping means the top tail saturates into perfect
    immunity and becomes mutually indistinguishable, which is itself a finding.
    """
    rng = random.Random(seed)
    stress, sbi = ns["stress"], ns["sup_by_id"]
    obs, sat = {}, 0
    for sh in ns["shipments"]:
        sid = sh["supplier_id"]
        if not sid or not sh["dispatched_at"]:
            continue
        st = stress(sid, sbi[sid]["base_rel"], sh["dispatched_at"])
        mult = max(0.0, 1.0 - lam*res[sid])
        p = min(0.80, 0.025 + 0.38*st*mult)
        obs.setdefault(sid, []).append((st, 1 if rng.random() < p else 0))
    sat = sum(1 for s in res if lam*res[s] >= 1.0)
    return obs, sat


def features(obs, min_n=8):
    """Continuous estimator: per-supplier regression of outcome on dispatch stress."""
    X, sids = [], []
    for sid, rows in obs.items():
        if len(rows) < min_n: continue
        ss = [s for s, _ in rows]; ll = [float(l) for _, l in rows]
        ms, ml = statistics.fmean(ss), statistics.fmean(ll)
        vs = sum((a-ms)**2 for a in ss)
        slope = sum((a-ms)*(b-ml) for a, b in zip(ss, ll))/vs if vs > 0 else 0.0
        X.append([ml, ms, slope, ml - 0.38*ms, float(len(rows))])
        sids.append(sid)
    return X, sids


print("=" * 80)
print("Phase 2 gate follow-up  (SUP_N=4,000, 40 snapshots, H at spec defaults)")
print("=" * 80)

ns = run_generator(CFG, "H")
rng = random.Random(90210)
m, s = 0.50, 0.15
k = m*(1-m)/(s*s) - 1
RES = {sup["id"]: rng.betavariate(m*k, (1-m)*k) for sup in ns["suppliers"]}
MED = statistics.median(RES.values())
N_ALL = len(ns["suppliers"])

# ---------------------------------------------------------------- 1. coefficient sweep
print("\n## 1. Coupling-coefficient sweep (Mechanism H at spec defaults)\n")
print(f"{'lambda':>7} {'coverage':>9} {'AUC':>7} {'95% CI':>18} {'mean stress':>12} "
      f"{'med stress':>11} {'saturated':>10}")
sweep = []
for lam in LAMBDAS:
    obs, sat = simulate(ns, RES, lam)
    X, sids = features(obs)
    y = [1 if RES[sid] > MED else 0 for sid in sids]
    a, lo, hi = cv_auc(X, y)
    per_sup = [statistics.fmean([st for st, _ in obs[sid]]) for sid in sids]
    mean_st, med_st = statistics.fmean(per_sup), statistics.median(per_sup)
    sweep.append((lam, len(X)/N_ALL, a, lo, hi, mean_st, med_st, sat/N_ALL))
    print(f"{lam:>7.1f} {len(X)/N_ALL*100:>8.1f}% {a:>7.3f} "
          f"[{lo:>6.3f}, {hi:>6.3f}] {mean_st:>12.3f} {med_st:>11.3f} "
          f"{sat/N_ALL*100:>9.1f}%")

# ---------------------------------------------------------------- 2. stratified AUC
print("\n## 2. AUC stratified by shipment volume\n")
print("Run at BOTH the current default and the sweep's best non-saturating value, to")
print("tell whether raising the coefficient lifts the thin strata or merely amplifies")
print("the signal already present in the fat one. That distinction decides whether the")
print("fix is a coefficient change or partial-coverage reporting.\n")
for lam in (0.7, 1.3):
    obs, _ = simulate(ns, RES, lam)
    X, sids = features(obs)
    vols = [len(obs[sid]) for sid in sids]
    order = sorted(range(len(sids)), key=lambda i: vols[i])
    qs = 4
    print(f"  lambda = {lam}")
    print(f"  {'quartile':>9} {'n':>6} {'shipments':>18} {'AUC':>7} {'95% CI':>18} {'':>6}")
    for q in range(qs):
        lo_i, hi_i = q*len(order)//qs, (q+1)*len(order)//qs
        sel = order[lo_i:hi_i]
        a, lo, hi = cv_auc([X[i] for i in sel], [1 if RES[sids[i]] > MED else 0 for i in sel])
        vv = [vols[i] for i in sel]
        flag = "" if lo > 0.50 else "spans chance"
        print(f"  {'Q'+str(q+1):>9} {len(sel):>6} {min(vv):>7}-{max(vv):<10} "
              f"{a:>7.3f} [{lo:>6.3f}, {hi:>6.3f}] {flag:>6}")
    a, lo, hi = cv_auc(X, [1 if RES[sid] > MED else 0 for sid in sids])
    print(f"  {'pooled':>9} {len(X):>6} {min(vols):>7}-{max(vols):<10} "
          f"{a:>7.3f} [{lo:>6.3f}, {hi:>6.3f}]\n")

# ---------------------------------------------------------------- 3. ceiling per variant
print("\n## 3. Zero-shipper ceiling by variant composition\n")
print("Variant F is Base+E+F and Variant K is Base+A+B+...+J; neither can be generated")
print("until Phase 2 builds E/F. But E and F change transmission and hidden state only --")
print("WHICH suppliers ship is fixed by BOM structure (prod_bom_sup), which E/F never")
print("touch. So Variant 0 is an exact proxy for Variant F's shipper set, and Variant A")
print("(= J + A, buildable today) is an exact proxy for Variant K's visibility structure.\n")
print(f"{'proxy':>26} {'simulated':>10} {'emitted':>9} {'ship>0':>8} "
      f"{'ceiling/simulated':>18} {'ceiling/emitted':>16}")
for variant, label in (("0", "Variant F  (via Variant 0)"), ("A", "Variant K  (via Variant A)")):
    v = run_generator(CFG, variant)
    shippers = {sh["supplier_id"] for sh in v["shipments"] if sh["supplier_id"]}
    visible = v["VISIBLE_SUP"]
    n_sim = len(v["suppliers"])
    n_emit = len(visible)
    ship_vis = len(shippers & visible)
    print(f"{label:>26} {n_sim:>10,} {n_emit:>9,} {ship_vis:>8,} "
          f"{ship_vis/n_sim*100:>17.1f}% {ship_vis/n_emit*100:>15.1f}%")
