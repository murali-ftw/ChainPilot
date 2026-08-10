#!/usr/bin/env python3
"""Phase 0 power harness.

Runs db/generate_dataset.py in-process with four knobs patched in, and reports
label counts only. No CSVs are written (the real generator's write() is stubbed),
so a sweep point costs generation time and nothing else.

Knobs
  P0_SUP_N   supplier count (drives SCALE, hence components/products/customers/orders)
  P0_SNAPS   number of monthly t0s starting 2024-07-01 (drives T_END too)
  P0_ABSORB  multiplier on the stress->delay-probability slope. Stands in for
             Mechanism E/F absorption: a resilient supplier converts less of its
             latent stress into an actual late shipment. 1.0 == V1.
  P0_VISRET  fraction of suppliers the model can still see under Mechanism A
             visibility truncation. Applied post-hoc to the label tally, not to
             the simulation -- truncation hides entities, it does not change
             physics.
"""
import io, os, re, sys, collections, contextlib

SRC = "/Users/muralik/Documents/Programs/HADES_v2/db/generate_dataset.py"
SUP_N   = int(os.environ.get("P0_SUP_N", 800))
SNAPS   = int(os.environ.get("P0_SNAPS", 15))
ABSORB  = float(os.environ.get("P0_ABSORB", 1.0))
VISRET  = float(os.environ.get("P0_VISRET", 1.0))

src = open(SRC).read()


def sub1(pattern, repl, text, tag):
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"patch '{tag}' matched {n} times, expected 1")
    return new


# --- timeline: SNAPS monthly t0s from 2024-07-01; T_END = last t0 + 29d ---
t0s = [(2024 + (6 + i) // 12, (6 + i) % 12 + 1) for i in range(SNAPS)]
ly, lm = t0s[-1]
ey, em = (ly + 1, 1) if lm == 12 else (ly, lm + 1)

src = sub1(r"^SUP_N = 800$", f"SUP_N = {SUP_N}", src, "SUP_N")
src = sub1(r"^T_END   = ts\(2025, 9, 30, 23\)$",
           f"T_END   = ts({ey}, {em}, 28, 23)", src, "T_END")
src = sub1(r"^T0S = \[ts\(2024, m, 1\).*$",
           "T0S = [ts(y, m, 1) for (y, m) in " + repr(t0s) + "]", src, "T0S")
src = sub1(r"p_delay = min\(0\.80, 0\.025 \+ 0\.38 \* st\)",
           f"p_delay = min(0.80, 0.025 + 0.38 * {ABSORB} * st)", src, "p_delay")
# Stub the writer: we only need the in-memory label_rows. An early return leaves
# the real body in place as unreachable code, so this patch tracks write()'s
# signature only, not its internals.
src = sub1(r"^def write\(name, header, rows\):$",
           "def write(name, header, rows):\n    return", src, "stub-write")

ns = {"__name__": "__p0__", "__file__": SRC}
buf = io.StringIO()
rc = 0
with contextlib.redirect_stdout(buf):
    try:
        exec(compile(src, SRC, "exec"), ns)
    except SystemExit as e:          # generator ends with raise SystemExit(fail_count)
        rc = e.code or 0
gen_out = buf.getvalue()
if rc or os.environ.get("P0_VERBOSE"):
    print(gen_out[gen_out.find("Chapter 15") if "Chapter 15" in gen_out else 0:])
    print(f"--- validation suite exit code: {rc}")

label_rows = ns["label_rows"]
suppliers = ns["suppliers"]

# Mechanism A: retain the first VISRET fraction of suppliers by generation order
# (a stand-in for "tier-1 visible, deeper tiers truncated"); entities outside
# the visible set contribute no usable labels.
keep_n = int(round(len(suppliers) * VISRET))
visible_sup = {s["id"] for s in suppliers[:keep_n]}
sup_of_ship = {sh["id"]: sh["supplier_id"] for sh in ns["shipments"]}

snap_id_to_t0 = {ns["uid"]("snap", t): t for t in ns["T0S"]}
per_snap = collections.defaultdict(collections.Counter)

tot = collections.Counter()
pos = collections.Counter()
for r in label_rows:
    _id, _snap, etype, eid, task, lab, *_ = r
    if lab == "true":
        per_snap[snap_id_to_t0[_snap].date().isoformat()][task] += 1
    if VISRET < 1.0:
        if task == "impact" and eid not in visible_sup:
            continue
        if task == "delay" and sup_of_ship.get(eid) not in visible_sup:
            continue
    tot[task] += 1
    pos[task] += int(lab == "true")

n_ship = len(ns["shipments"])
print(f"SUP_N={SUP_N} SNAPS={SNAPS} ABSORB={ABSORB} VISRET={VISRET} "
      f"| shipments={n_ship:,} inv_pairs={len(ns['inv_pairs']):,}")
for t in ("delay", "shortage", "impact"):
    r = pos[t] / tot[t] * 100 if tot[t] else 0.0
    flag = "OK " if pos[t] >= 2000 else "LOW"
    print(f"  {flag} {t:9s} n={tot[t]:>7,}  pos={pos[t]:>6,}  rate={r:5.2f}%")

if os.environ.get("P0_PERSNAP"):
    print("  per-snapshot positives (delay/shortage/impact):")
    for d in sorted(per_snap):
        c = per_snap[d]
        print(f"    {d}  {c['delay']:>5} {c['shortage']:>5} {c['impact']:>5}")
