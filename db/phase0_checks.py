#!/usr/bin/env python3
"""Phase 0 continuation — scale-sensitivity checks.

Runs the real db/generate_dataset.py once at a given scale, then evaluates three
structural checks against that generated world:

  1. Co-degradation margin (V1's existing hidden-dependency check, measured
     exactly rather than at the 2-decimal precision the generator prints).
  2. Clarification 1 -- can a simple classifier separate Mechanism B's Type
     A/B/C groups from structural features alone? Prototyped here, because
     Mechanism B does not exist yet; the point is to learn whether the CHECK is
     scale-stable before Phase 3 depends on it.
  3. Clarification 3 -- is hidden resilience recoverable from observed history?
     Also prototyped, for the same reason.

Checks 2 and 3 are prototypes built on the real generated population, stress
trajectories and shipment timings. They are not the mechanisms. See
docs/phase0_power_check.md for what that does and does not license.

Stdlib only, deterministic: every RNG here is seeded from a fixed constant, and
the logistic regressions use fixed initialisation and a fixed iteration count.

Usage:  P0_SUP_N=5000 P0_SNAPS=36 P0_ABSORB=0.533 python3 db/phase0_checks.py
"""
import io, math, os, random, re, sys, contextlib, statistics
from datetime import timedelta

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_dataset.py")
SUP_N  = int(os.environ.get("P0_SUP_N", 800))
SNAPS  = int(os.environ.get("P0_SNAPS", 15))
ABSORB = float(os.environ.get("P0_ABSORB", 1.0))

# ---------------------------------------------------------------- run generator


def sub1(pattern, repl, text, tag):
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"patch '{tag}' matched {n} times, expected 1")
    return new


def run_generator():
    src = open(SRC).read()
    t0s = [(2024 + (6 + i) // 12, (6 + i) % 12 + 1) for i in range(SNAPS)]
    ly, lm = t0s[-1]
    ey, em = (ly + 1, 1) if lm == 12 else (ly, lm + 1)
    src = sub1(r"^SUP_N = 800$", f"SUP_N = {SUP_N}", src, "SUP_N")
    src = sub1(r"^T_END   = ts\(2025, 9, 30, 23\)$", f"T_END   = ts({ey}, {em}, 28, 23)", src, "T_END")
    src = sub1(r"^T0S = \[ts\(2024, m, 1\).*$",
               "T0S = [ts(y, m, 1) for (y, m) in " + repr(t0s) + "]", src, "T0S")
    src = sub1(r"p_delay = min\(0\.80, 0\.025 \+ 0\.38 \* st\)",
               f"p_delay = min(0.80, 0.025 + 0.38 * {ABSORB} * st)", src, "p_delay")
    # Stub the writer: only the in-memory structures are needed here, and writing
    # ~1.5 GB of gzip per sweep point would dominate the runtime. An early return
    # leaves the real body in place as unreachable code, so this patch does not
    # need to track edits to write()'s internals -- only its signature.
    src = sub1(r"^def write\(name, header, rows\):$",
               "def write(name, header, rows):\n    return", src, "stub-write")
    ns = {"__name__": "__p0__", "__file__": SRC}
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            exec(compile(src, SRC, "exec"), ns)
        except SystemExit:
            pass
    return ns


# ---------------------------------------------------------------- tiny ML kit
# Hand-rolled so the analysis carries no dependency the generator doesn't.


def auc(scores, labels):
    """Mann-Whitney rank AUC. labels are 0/1."""
    pairs = sorted(zip(scores, labels))
    ranks, i, n = [0.0] * len(pairs), 0, len(pairs)
    while i < n:                       # average ranks within ties
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    pos = sum(l for _, l in pairs)
    neg = n - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rsum = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (rsum - pos * (pos + 1) / 2.0) / (pos * neg)


def standardise(rows):
    if not rows:
        return rows
    d = len(rows[0])
    mus = [statistics.fmean(r[k] for r in rows) for k in range(d)]
    sds = [statistics.pstdev([r[k] for r in rows]) or 1.0 for k in range(d)]
    return [[(r[k] - mus[k]) / sds[k] for k in range(d)] for r in rows]


def logreg(X, y, iters=400, lr=0.3):
    """Plain full-batch gradient descent. Deterministic: zero init, fixed steps."""
    d = len(X[0])
    w, b = [0.0] * d, 0.0
    n = len(X)
    for _ in range(iters):
        gw, gb = [0.0] * d, 0.0
        for xi, yi in zip(X, y):
            z = b + sum(w[k] * xi[k] for k in range(d))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            e = p - yi
            for k in range(d):
                gw[k] += e * xi[k]
            gb += e
        for k in range(d):
            w[k] -= lr * gw[k] / n
        b -= lr * gb / n
    return w, b


def predict(X, w, b):
    return [b + sum(w[k] * xi[k] for k in range(len(w))) for xi in X]


def cv_scores(X, y, folds=5):
    """Deterministic k-fold held-out scores. In-sample AUC on a fitted model is
    optimistic even on pure noise, which is exactly the failure mode that would
    make an indistinguishability check look broken when it is fine."""
    if sum(y) < folds or len(y) - sum(y) < folds:
        return [], []
    Xs = standardise(X)
    idx = list(range(len(Xs)))
    random.Random(31337).shuffle(idx)          # shuffled, not i % folds -- see build_groups
    fold_of = {i: p % folds for p, i in enumerate(idx)}
    scores, labels = [], []
    for f in range(folds):
        te = [i for i in range(len(Xs)) if fold_of[i] == f]
        tr = [i for i in range(len(Xs)) if fold_of[i] != f]
        if sum(y[i] for i in tr) == 0 or sum(y[i] for i in tr) == len(tr):
            continue
        w, b = logreg([Xs[i] for i in tr], [y[i] for i in tr])
        scores += predict([Xs[i] for i in te], w, b)
        labels += [y[i] for i in te]
    return scores, labels


def auc_ci(scores, labels, boots=400, seed=7):
    """Point estimate plus a percentile bootstrap 95% CI over the held-out
    predictions. Without the CI, 'AUC 0.566' says nothing about whether the
    estimator beat chance at this sample size."""
    if not scores:
        return float("nan"), float("nan"), float("nan")
    a = auc(scores, labels)
    rng, n, out = random.Random(seed), len(scores), []
    for _ in range(boots):
        pick = [rng.randrange(n) for _ in range(n)]
        v = auc([scores[i] for i in pick], [labels[i] for i in pick])
        if v == v:
            out.append(v)
    if len(out) < 20:
        return a, float("nan"), float("nan")
    out.sort()
    return a, out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]


def cv_auc(X, y, folds=5):
    s, l = cv_scores(X, y, folds)
    return auc(s, l) if s else float("nan")


# ---------------------------------------------------------------- check 1


def check_codegradation(ns):
    """V1's hidden-dependency check, measured exactly. Members' 90-day on-time
    rate at 2024-12-01 must sit >0.10 below the fleet mean."""
    stf, poly = ns["stf_rows"], ns["H_POLYMER"]
    members = [float(r[4]) for r in stf if r[2] == "2024-12-01" and r[4] != "" and r[1] in poly]
    fleet = [float(r[4]) for r in stf if r[2] == "2024-12-01" and r[4] != "" and r[1] not in poly]
    if not members or not fleet:
        return None
    m, f = statistics.fmean(members), statistics.fmean(fleet)
    # H_POLYMER has a FIXED 4 members at every SUP_N, so this margin is a
    # 4-sample mean compared against a fleet mean that keeps tightening. Report
    # the per-member values and a leave-one-out range so the reader can see how
    # much one member moves the verdict.
    loo = []
    if len(members) > 1:
        loo = [f - statistics.fmean([x for j, x in enumerate(members) if j != i])
               for i in range(len(members))]
    return {"n_members": len(members), "n_fleet": len(fleet),
            "members": m, "fleet": f, "margin": f - m,
            "member_vals": sorted(members),
            "loo_min": min(loo) if loo else float("nan"),
            "loo_max": max(loo) if loo else float("nan")}


# ---------------------------------------------------------------- check 2


def build_groups(ns, leaky=False):
    """Mechanism B prototype. Hidden Parent Rate 0.20, Type A:B:C = 1:1:1, per
    the spec's config table. All three types are drawn by ONE process; type is
    then assigned round-robin, i.e. independent of every structural property.

    leaky=True is a positive control: type is assigned BY group size instead, so
    structural features genuinely do carry type. If the classifier cannot detect
    that, the check has no power and its pass on the honest process means nothing.
    """
    rng = random.Random(20240701)
    sups = list(ns["suppliers"])
    rng.shuffle(sups)
    n_hidden = int(round(len(sups) * 0.20))
    pool = sups[:n_hidden]
    groups, i = [], 0
    while i + 2 < len(pool):
        size = rng.choice([3, 4, 5, 6])
        g = pool[i:i + size]
        if len(g) < 3:
            break
        groups.append(g)
        i += size
    if leaky:
        order = sorted(range(len(groups)), key=lambda k: (len(groups[k]), k))
        types = [0] * len(groups)
        for rank, k in enumerate(order):
            types[k] = rank * 3 // len(order)
    else:
        # Balanced 1:1:1, assigned by a seeded SHUFFLE rather than k % 3. Modulo
        # assignment interacts with any other modulo in the pipeline (notably
        # fold = i % folds) and produces a systematic below-chance CV AUC on data
        # that carries no signal at all -- which reads as a failed
        # indistinguishability check when nothing is actually wrong.
        types = [k % 3 for k in range(len(groups))]
        rng.shuffle(types)
    return groups, types


def group_features(ns, groups):
    comp_of = {}
    for c in ns["components"]:
        comp_of[c["supplier_id"]] = comp_of.get(c["supplier_id"], 0) + 1
    ship_of = {}
    for sh in ns["shipments"]:
        if sh["supplier_id"]:
            ship_of[sh["supplier_id"]] = ship_of.get(sh["supplier_id"], 0) + 1
    dual = {}
    for cs in ns["component_suppliers"]:
        dual[cs["supplier_id"]] = dual.get(cs["supplier_id"], 0) + 1
    feats = []
    for g in groups:
        deg = [comp_of.get(s["id"], 0) for s in g]
        shp = [ship_of.get(s["id"], 0) for s in g]
        dl = [dual.get(s["id"], 0) for s in g]
        feats.append([
            float(len(g)),
            statistics.fmean(deg), statistics.pstdev(deg) if len(deg) > 1 else 0.0,
            float(min(deg)), float(max(deg)),
            statistics.fmean(shp), statistics.pstdev(shp) if len(shp) > 1 else 0.0,
            statistics.fmean(dl),
            float(len({s["country"] for s in g})),
            float(sum(1 for s in g if s["sea"])) / len(g),
        ])
    return feats


def check_indistinguishability(ns, leaky=False):
    groups, types = build_groups(ns, leaky)
    if len(groups) < 30:
        return {"n_groups": len(groups), "aucs": [], "macro": float("nan")}
    X = group_features(ns, groups)
    aucs, alls, alll = [], [], []
    for t in (0, 1, 2):
        y = [1 if tt == t else 0 for tt in types]
        s, l = cv_scores(X, y)
        aucs.append(auc(s, l) if s else float("nan"))
        alls += s
        alll += l
    good = [a for a in aucs if a == a]
    macro, lo, hi = auc_ci(alls, alll)          # CI over the pooled one-vs-rest scores
    return {"n_groups": len(groups), "aucs": aucs, "lo": lo, "hi": hi,
            "macro": statistics.fmean(good) if good else float("nan"),
            "pooled": macro}


# ---------------------------------------------------------------- check 3


def check_resilience(ns, lam=0.7):
    """Mechanism E/F prototype.

    Hidden resilience r in [0,1] per supplier, never emitted. It attenuates the
    stress->delay conversion: p_delay = 0.025 + 0.38*st*(1 - lam*r). Outcomes are
    re-simulated over the generator's REAL shipment set, using each shipment's
    real dispatch time and its supplier's real stress trajectory, so per-supplier
    evidence volume is whatever the world actually provides.

    Recovery uses only observables a model could compute from
    shipment_status_history: on-time rate in high-stress vs low-stress windows,
    the gap between them, and volume. Reported as held-out AUC for
    'resilience above median'.
    """
    rng = random.Random(90210)
    stress, sup_by_id = ns["stress"], ns["sup_by_id"]
    res = {s["id"]: rng.betavariate(2.0, 2.0) for s in ns["suppliers"]}

    # per-shipment: real dispatch time, real stress, resilience-attenuated outcome
    hist = {}
    for sh in ns["shipments"]:
        sid = sh["supplier_id"]
        if not sid or not sh["dispatched_at"]:
            continue
        st = stress(sid, sup_by_id[sid]["base_rel"], sh["dispatched_at"])
        p = min(0.80, 0.025 + 0.38 * st * (1.0 - lam * res[sid]))
        late = rng.random() < p
        hist.setdefault(sid, []).append((st, late))

    HI = 0.35                                    # stress threshold for "disrupted"
    X, y, kept = [], [], 0
    med = statistics.median(res.values())
    for sid, rows in hist.items():
        hi = [late for st, late in rows if st >= HI]
        lo = [late for st, late in rows if st < HI]
        if len(hi) < 5 or len(lo) < 5:           # too little evidence to estimate
            continue
        hr = sum(hi) / len(hi)
        lr = sum(lo) / len(lo)
        X.append([hr, lr, hr - lr, float(len(hi)), float(len(rows))])
        y.append(1 if res[sid] > med else 0)
        kept += 1
    if kept < 50:
        return {"n_estimable": kept, "auc": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "coverage": kept / len(ns["suppliers"])}
    a, lo, hi = auc_ci(*cv_scores(X, y))
    return {"n_estimable": kept, "auc": a, "lo": lo, "hi": hi,
            "coverage": kept / len(ns["suppliers"])}


# ---------------------------------------------------------------- main

ns = run_generator()
print(f"### SUP_N={SUP_N} SNAPS={SNAPS} ABSORB={ABSORB} "
      f"(suppliers={len(ns['suppliers']):,} shipments={len(ns['shipments']):,})")

c1 = check_codegradation(ns)
if c1:
    verdict = "PASS" if c1["margin"] > 0.10 else "FAIL"
    print(f"  [1] co-degradation   margin={c1['margin']:+.4f} "
          f"(members {c1['members']:.4f} n={c1['n_members']} | "
          f"fleet {c1['fleet']:.4f} n={c1['n_fleet']:,})  need >0.10  {verdict}")
    print(f"      member values     {[round(v, 3) for v in c1['member_vals']]}  "
          f"leave-one-out margin range [{c1['loo_min']:+.4f}, {c1['loo_max']:+.4f}]")
else:
    print("  [1] co-degradation   NO DATA at 2024-12-01")

h = check_indistinguishability(ns, leaky=False)
l = check_indistinguishability(ns, leaky=True)
fmt = lambda a: "  ".join(f"{x:.3f}" if x == x else "  nan" for x in a)
print(f"  [2] indistinguish.   groups={h['n_groups']:,}  macro_AUC={h['macro']:.3f} "
      f"pooled={h['pooled']:.3f} [{h['lo']:.3f}, {h['hi']:.3f}]  (A/B/C: {fmt(h['aucs'])})  "
      f"need CI to span 0.50")
print(f"      positive control  groups={l['n_groups']:,}  macro_AUC={l['macro']:.3f} "
      f"pooled={l['pooled']:.3f} [{l['lo']:.3f}, {l['hi']:.3f}]  (A/B/C: {fmt(l['aucs'])})  "
      f"need CI above 0.50")

r = check_resilience(ns)
print(f"  [3] resilience       AUC={r['auc']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]  "
      f"n_estimable={r['n_estimable']:,} ({r['coverage']*100:.1f}% of suppliers)  "
      f"need CI above 0.50")
