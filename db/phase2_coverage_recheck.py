#!/usr/bin/env python3
"""Phase 2 gate — resilience-coverage recheck with real Mechanism H.

docs/phase0_power_check.md §3 measured, on a disposable prototype, that hidden
resilience is estimable for only 5–9% of suppliers. V2_MASTER_PROMPT.md's
sequencing note makes clearing >50% coverage a precondition for building
Mechanisms E/F, on the grounds that Mechanism F built on an unrecoverable latent
is irreducible noise (Clarification 3).

Mechanism H landed in Phase 5. This rerun uses it for real, driving the generator
through its Phase 1 config path rather than regex-patching the source.

Coverage is a property of the generated world alone -- for each supplier, how many
of its shipments were dispatched while it was under high stress, and how many while
it was not. It does not depend on how resilience is assigned, which is what makes
it a usable gate.

Usage:  python3 db/phase2_coverage_recheck.py
"""
import contextlib, io, json, math, os, random, statistics, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "generate_dataset.py")

HI_STRESS = 0.35          # same threshold as the Phase 0 §3 prototype
MIN_EVIDENCE = 5          # shipments needed on EACH side to estimate resilience
GATE = 0.50               # >50% coverage required before Phase 2


def run_generator(overrides, variant="0"):
    """Execute generate_dataset.py in-process with a config, writes stubbed."""
    src = open(GEN).read()
    src = src.replace("def write(name, header, rows):",
                      "def write(name, header, rows):\n    return", 1)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(overrides, f)
        cfg_path = f.name
    argv = sys.argv
    sys.argv = ["generate_dataset.py", "--variant", variant, "--config", cfg_path]
    ns = {"__name__": "__recheck__", "__file__": GEN}
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


def coverage(ns, shock_events=None, hi=HI_STRESS, min_ev=MIN_EVIDENCE):
    """Fraction of suppliers with >= min_ev shipments on BOTH sides of the stress
    threshold. Optionally swaps in an alternative shock list first, which lets a
    parameter sweep reuse one generated world.

    Returns (coverage_all, coverage_shipping, n_estimable, n_shipping, per_sup)."""
    if shock_events is not None:
        ns["SHOCK_EVENTS"][:] = shock_events
    stress, sup_by_id = ns["stress"], ns["sup_by_id"]
    hist = {}
    for sh in ns["shipments"]:
        sid = sh["supplier_id"]
        if not sid or not sh["dispatched_at"]:
            continue
        st = stress(sid, sup_by_id[sid]["base_rel"], sh["dispatched_at"])
        h, l = hist.setdefault(sid, [0, 0])
        if st >= hi:
            hist[sid][0] += 1
        else:
            hist[sid][1] += 1
    n_all = len(ns["suppliers"])
    n_ship = len(hist)
    est = [s for s, (h, l) in hist.items() if h >= min_ev and l >= min_ev]
    return len(est)/n_all, (len(est)/n_ship if n_ship else 0.0), len(est), n_ship, hist


def build_shocks(ns, rate, radius, seed=42, dur_lo=25, dur_hi=60):
    """Replicate the generator's Mechanism H construction with other parameters."""
    from datetime import timedelta
    rng = random.Random(0x50CC57 + seed)
    years = ns["TIMELINE_DAYS"] / 365.0
    n = int(round(rate * years))
    by_country = {}
    for s in ns["suppliers"]:
        by_country.setdefault(s["country"], []).append(s["id"])
    groups = [g for g in (ns["H_PORT"], ns["H_TRUCK"], ns["H_CUSTOMS"]) if g] + \
             [set(v) for v in by_country.values() if len(v) >= 3]
    out = []
    for _ in range(n):
        grp = sorted(rng.choice(groups))
        hit = set(rng.sample(grp, min(len(grp), radius)))
        s0 = ns["T_START"] + timedelta(days=rng.randint(0, max(1, ns["TIMELINE_DAYS"] - 40)))
        out.append((hit, s0, s0 + timedelta(days=rng.randint(5, 15)),
                    s0 + timedelta(days=rng.randint(dur_lo, dur_hi)),
                    rng.uniform(0.35, 0.75)))
    return out


# ---------------------------------------------------------------- report
CFG = {"sup_n": 4000, "snapshots": 40}
print("=" * 78)
print("Phase 2 gate — resilience-coverage recheck (SUP_N=4,000, 40 snapshots)")
print("=" * 78)

rows = []
for variant, label in (("0", "Variant 0 (baseline, no H)"), ("H", "Variant H (real Mechanism H)")):
    ns = run_generator(CFG, variant)
    ca, cs, ne, nsh, hist = coverage(ns)
    shocks = len(ns["SHOCK_EVENTS"])
    touched = len({s for m, *_ in ns["SHOCK_EVENTS"] for s in m})
    rows.append((label, ca, cs, ne, nsh, shocks, touched))
    print(f"\n{label}")
    print(f"  shipments            {len(ns['shipments']):,}  "
          f"({len(ns['shipments'])/nsh:.0f} per shipping supplier)")
    print(f"  shocks               {shocks}  touching {touched} suppliers "
          f"({touched/CFG['sup_n']*100:.1f}%)")
    print(f"  estimable suppliers  {ne:,} / {CFG['sup_n']:,}  "
          f"= {ca*100:.1f}%  [GATE >{GATE*100:.0f}%: {'PASS' if ca > GATE else 'FAIL'}]")
    print(f"  ... of the {nsh:,} that ship at all: {cs*100:.1f}%")
    hs = sorted(h for h, l in hist.values())
    print(f"  high-stress shipments per supplier: median {statistics.median(hs):.0f}, "
          f"p90 {hs[int(0.9*len(hs))]}, need >= {MIN_EVIDENCE}")
    if variant == "H":
        keep = ns

# ---------------------------------------------------------------- diagnosis
print("\n" + "-" * 78)
print("Bottleneck diagnosis — sweeping H's parameters on the Variant H world")
print("-" * 78)
base = list(keep["SHOCK_EVENTS"])
print(f"{'shock_rate/yr':>14} {'blast_radius':>13} {'duration':>10} "
      f"{'suppliers hit':>14} {'coverage':>10}")
for rate, radius, dlo, dhi in [(12, 5, 25, 60), (12, 50, 25, 60), (52, 50, 25, 60),
                               (52, 200, 25, 60), (52, 400, 60, 180), (104, 400, 60, 180)]:
    ev = build_shocks(keep, rate, radius, dur_lo=dlo, dur_hi=dhi)
    ca, cs, ne, nsh, _ = coverage(keep, ev)
    touched = len({s for m, *_ in ev for s in m})
    print(f"{rate:>14} {radius:>13} {dlo:>4}-{dhi:<5} {touched:>14,} {ca*100:>9.1f}%")

# evidence-threshold sensitivity, on the default-H world
keep["SHOCK_EVENTS"][:] = base
print("\nSensitivity to the evidence threshold (default H, real world):")
for m in (2, 3, 5, 10):
    ca, cs, ne, nsh, _ = coverage(keep, min_ev=m)
    print(f"  >= {m:>2} shipments each side:  coverage {ca*100:5.1f}%  "
          f"({ne:,} suppliers)")

# ---------------------------------------------------------------- estimator design
# The 5-9% figure comes from an estimator that BINS stress into high/low and demands
# >=5 shipments in each bin. That requirement, not the world, may be what limits
# coverage: a continuous estimator uses every shipment a supplier has. Test both on
# the same world, with the same hidden resilience, so the comparison is clean.
def _auc(sc, lb):
    pr = sorted(zip(sc, lb)); n = len(pr); rk = [0.0]*n; i = 0
    while i < n:
        j = i
        while j+1 < n and pr[j+1][0] == pr[i][0]: j += 1
        for k in range(i, j+1): rk[k] = (i+j)/2.0 + 1.0
        i = j+1
    p = sum(l for _, l in pr); q = n - p
    if not p or not q: return float("nan")
    return (sum(r for r, (_, l) in zip(rk, pr) if l == 1) - p*(p+1)/2.0)/(p*q)


def _cv_auc(X, y, folds=5, iters=300, lr=0.3):
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
    a = _auc(sc, yy)
    rng = random.Random(7); bs = []
    for _ in range(300):
        p = [rng.randrange(len(sc)) for _ in range(len(sc))]
        v = _auc([sc[i] for i in p], [yy[i] for i in p])
        if v == v: bs.append(v)
    bs.sort()
    return a, bs[int(0.025*len(bs))], bs[int(0.975*len(bs))-1]


print("\n" + "-" * 78)
print("Is 5-9% a property of the world, or of the binned estimator?")
print("-" * 78)
keep["SHOCK_EVENTS"][:] = base
_rng = random.Random(90210)
# Beta matched to the spec's mean_resilience / resilience_std
_m, _s = 0.50, 0.15
_k = _m*(1-_m)/(_s*_s) - 1
_res = {s["id"]: _rng.betavariate(_m*_k, (1-_m)*_k) for s in keep["suppliers"]}
_stress, _sbi = keep["stress"], keep["sup_by_id"]
_obs = {}
for sh in keep["shipments"]:
    sid = sh["supplier_id"]
    if not sid or not sh["dispatched_at"]:
        continue
    st = _stress(sid, _sbi[sid]["base_rel"], sh["dispatched_at"])
    p = min(0.80, 0.025 + 0.38*st*(1.0 - 0.7*_res[sid]))
    _obs.setdefault(sid, []).append((st, 1 if _rng.random() < p else 0))
_med = statistics.median(_res.values())

for name, need_bins in (("binned (>=5 each side)", True), ("continuous (all shipments)", False)):
    X, y = [], []
    for sid, rows in _obs.items():
        hi = [l for st, l in rows if st >= HI_STRESS]
        lo = [l for st, l in rows if st < HI_STRESS]
        if need_bins:
            if len(hi) < MIN_EVIDENCE or len(lo) < MIN_EVIDENCE: continue
            hr, lr_ = sum(hi)/len(hi), sum(lo)/len(lo)
            X.append([hr, lr_, hr-lr_, float(len(hi)), float(len(rows))])
        else:
            if len(rows) < 8: continue
            ss = [st for st, _l in rows]; ll = [float(l) for _s2, l in rows]
            ms, ml = statistics.fmean(ss), statistics.fmean(ll)
            vs = sum((a-ms)**2 for a in ss)
            slope = sum((a-ms)*(b-ml) for a, b in zip(ss, ll))/vs if vs > 0 else 0.0
            X.append([ml, ms, slope, ml - 0.38*ms, float(len(rows))])
        y.append(1 if _res[sid] > _med else 0)
    cov = len(X)/len(keep["suppliers"])
    if len(X) < 50:
        print(f"  {name:28s} coverage {cov*100:5.1f}%  (too few to score)")
        continue
    a, lo_, hi_ = _cv_auc(X, y)
    print(f"  {name:28s} coverage {cov*100:5.1f}%  ({len(X):,} suppliers)  "
          f"AUC {a:.3f} [{lo_:.3f}, {hi_:.3f}]  "
          f"{'above chance' if lo_ > 0.50 else 'SPANS CHANCE'}")

print("\n" + "-" * 78)
print("Would a heavier shock regime make resilience recoverable? (continuous estimator)")
print("NOTE: shocks are swapped onto the already-generated shipment set, so this")
print("      ignores the feedback from shocks onto shipment cadence. Indicative only.")
print("-" * 78)
for _rate, _radius, _dlo, _dhi in [(12, 5, 25, 60), (52, 200, 25, 60), (52, 400, 60, 180)]:
    keep["SHOCK_EVENTS"][:] = build_shocks(keep, _rate, _radius, dur_lo=_dlo, dur_hi=_dhi)
    _o2 = {}
    for sh in keep["shipments"]:
        sid = sh["supplier_id"]
        if not sid or not sh["dispatched_at"]:
            continue
        st = _stress(sid, _sbi[sid]["base_rel"], sh["dispatched_at"])
        p = min(0.80, 0.025 + 0.38*st*(1.0 - 0.7*_res[sid]))
        _o2.setdefault(sid, []).append((st, 1 if _rng.random() < p else 0))
    X, y = [], []
    for sid, rows in _o2.items():
        if len(rows) < 8: continue
        ss = [st for st, _l in rows]; ll = [float(l) for _s2, l in rows]
        ms, ml = statistics.fmean(ss), statistics.fmean(ll)
        vs = sum((a-ms)**2 for a in ss)
        slope = sum((a-ms)*(b-ml) for a, b in zip(ss, ll))/vs if vs > 0 else 0.0
        X.append([ml, ms, slope, ml - 0.38*ms, float(len(rows))])
        y.append(1 if _res[sid] > _med else 0)
    a, lo_, hi_ = _cv_auc(X, y)
    _mean_stress = statistics.fmean([statistics.fmean([s for s, _ in r]) for r in _o2.values()])
    print(f"  rate={_rate:>3}/yr radius={_radius:>3}: coverage {len(X)/4000*100:5.1f}%  "
          f"AUC {a:.3f} [{lo_:.3f}, {hi_:.3f}]  mean stress {_mean_stress:.3f}  "
          f"{'above chance' if lo_ > 0.50 else 'SPANS CHANCE'}")
keep["SHOCK_EVENTS"][:] = base

print("\nStructural ceiling:")
nsh = len({sh['supplier_id'] for sh in keep['shipments'] if sh['supplier_id']})
print(f"  {CFG['sup_n'] - nsh:,} of {CFG['sup_n']:,} suppliers "
      f"({(CFG['sup_n']-nsh)/CFG['sup_n']*100:.1f}%) ship NOTHING and can never be "
      f"estimable at any shock rate.")
print(f"  Maximum achievable coverage is therefore {nsh/CFG['sup_n']*100:.1f}%.")
