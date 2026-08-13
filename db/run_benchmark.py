#!/usr/bin/env python3
"""Phase 6 — benchmark generation, reporting and determinism verification.

Generates the twelve benchmark variants across N seeds, collects each run's
metadata, and produces the per-variant/per-task report table the build's
definition-of-done asks for: positive counts, positive rates, node/edge counts,
and whether the statistical-power target is met.

Two modes:

  --stats-only   run the simulation and collect label/row counts WITHOUT writing
                 CSVs. The full 12x5 matrix at spec scale would otherwise emit
                 ~24 GB (docs/phase0_power_check.md §7); this gives the complete
                 report table at no storage cost.
  (default)      write the datasets to db/csv/v<variant>_seed<seed>/.

Determinism is verified by regenerating one variant twice and diffing the
COMPRESSED bytes -- the stronger check, since matching decompressed contents
would not catch a timestamp leaking into a gzip header.

Usage:
    python3 db/run_benchmark.py --stats-only --config v1
    python3 db/run_benchmark.py --variants 0,K --seeds 42,43 --config v1
    python3 db/run_benchmark.py --verify-determinism --variant K --config v1
"""
import argparse, contextlib, io, json, os, re, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "generate_dataset.py")
ALL_VARIANTS = ["0", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

# docs/00_Benchmark_Specification.md §4 "Statistical Power"
POWER_LO, POWER_HI = 2000, 5000


def run_one(variant, seed, config, stats_only, out_dir=None):
    """Execute the generator once. Returns (manifest, validation_failures, seconds)."""
    src = open(GEN).read()
    if stats_only:
        src = src.replace("def write(name, header, rows):",
                          "def write(name, header, rows):\n    return", 1)
        # the manifest is normally written next to the CSVs; capture it in-memory
        src = src.replace(
            'with open(os.path.join(OUT, "resolved_config.json"), "w") as _f:\n'
            '    json.dump(_manifest, _f, indent=2, sort_keys=True)',
            "MANIFEST_OUT = _manifest", 1)
    cfg = config
    argv = ["generate_dataset.py", "--variant", variant, "--config", cfg,
            "--seed", str(seed)]
    if out_dir:
        argv += ["--out-dir", out_dir]
    saved, sys.argv = sys.argv, argv
    ns = {"__name__": "__bench__", "__file__": GEN}
    buf = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            try:
                exec(compile(src, GEN, "exec"), ns)
            except SystemExit:
                pass
    finally:
        sys.argv = saved
    secs = time.time() - t0
    out = buf.getvalue()
    fails = len(re.findall(r"^  \[FAIL\]", out, re.M))
    man = ns.get("MANIFEST_OUT")
    if man is None:
        path = os.path.join(ns["OUT"], "resolved_config.json")
        man = json.load(open(path))
    return man, fails, secs, out


def power_flag(pos):
    if pos < POWER_LO:
        return "LOW"
    if pos > POWER_HI:
        return "HIGH"
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--config", default="spec", help="preset name or JSON path")
    ap.add_argument("--stats-only", action="store_true",
                    help="collect counts without writing CSVs")
    ap.add_argument("--verify-determinism", action="store_true",
                    help="regenerate one variant twice and diff the compressed bytes")
    ap.add_argument("--variant", default="K", help="variant for --verify-determinism")
    ap.add_argument("--report", default="", help="write the report table to this path")
    ap.add_argument("--manifest-dir", default="",
                    help="also dump each run's full resolved_config manifest here as "
                         "<variant>_<seed>.json. With HADES_RATE_CURVE=1 in the environment "
                         "those manifests carry the sampling rate curve the rates were "
                         "derived from (docs/phase6_spec_scale_report.md §2).")
    args = ap.parse_args()

    if args.verify_determinism:
        import shutil, filecmp
        a = tempfile.mkdtemp(prefix="det_a_"); b = tempfile.mkdtemp(prefix="det_b_")
        for d in (a, b):
            run_one(args.variant, 42, args.config, False, out_dir=d)
        names = sorted(f for f in os.listdir(a) if f.endswith(".gz"))
        same = [n for n in names if filecmp.cmp(os.path.join(a, n), os.path.join(b, n), shallow=False)]
        print(f"determinism: variant {args.variant}, {len(same)}/{len(names)} "
              f".csv.gz files byte-identical across two runs")
        for n in names:
            if n not in same:
                print(f"  DIFFERS: {n}")
        shutil.rmtree(a); shutil.rmtree(b)
        return 0 if len(same) == len(names) else 1

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows, total_fails = [], 0
    if args.manifest_dir:
        os.makedirs(args.manifest_dir, exist_ok=True)

    print(f"generating {len(variants)} variants x {len(seeds)} seeds "
          f"(config={args.config}, stats_only={args.stats_only})\n")
    for v in variants:
        for sd in seeds:
            man, fails, secs, _out = run_one(v, sd, args.config, args.stats_only)
            total_fails += fails
            if args.manifest_dir:
                with open(os.path.join(args.manifest_dir, f"{v}_{sd}.json"), "w") as f:
                    json.dump(man, f, indent=2, sort_keys=True)
            lc, rc = man["label_counts"], man["row_counts"]
            rows.append(dict(variant=v, seed=sd, fails=fails, secs=round(secs, 1),
                             mechanisms=man["mechanisms_enabled"],
                             suppliers_emitted=rc["suppliers_emitted"],
                             suppliers_simulated=rc["suppliers_simulated"],
                             shipments=rc["shipments"], labels=rc["labels"],
                             **{f"{t}_{k}": lc[t][k] for t in lc for k in ("n", "positives")}))
            print(f"  variant {v:>1} seed {sd}  {secs:6.1f}s  "
                  f"fails={fails}  "
                  f"delay {lc['delay']['positives']:>6,}  "
                  f"shortage {lc['shortage']['positives']:>6,}  "
                  f"impact {lc['impact']['positives']:>6,}")

    # ---- report table: mean over seeds, per variant
    print("\n" + "=" * 100)
    print("BENCHMARK REPORT — positives are the mean over seeds; power target "
          f"{POWER_LO:,}–{POWER_HI:,} per task per variant")
    print("=" * 100)
    hdr = (f"{'var':>3} {'mechanisms':<24} {'sup(emit/sim)':>14} {'shipments':>10} "
           f"{'delay':>14} {'shortage':>15} {'impact':>14} {'fails':>6}")
    print(hdr); print("-" * len(hdr))
    summary = []
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        if not rs:
            continue
        mean = lambda k: sum(r[k] for r in rs) / len(rs)
        d, s_, i = mean("delay_positives"), mean("shortage_positives"), mean("impact_positives")
        dn, sn, iN = mean("delay_n"), mean("shortage_n"), mean("impact_n")
        mech = ",".join(rs[0]["mechanisms"]) or "(base)"
        print(f"{v:>3} {mech:<24} "
              f"{rs[0]['suppliers_emitted']:>6,}/{rs[0]['suppliers_simulated']:<7,} "
              f"{mean('shipments'):>10,.0f} "
              f"{d:>8,.0f} {power_flag(d):>5} "
              f"{s_:>9,.0f} {power_flag(s_):>5} "
              f"{i:>8,.0f} {power_flag(i):>5} "
              f"{sum(r['fails'] for r in rs):>6}")
        summary.append(dict(variant=v, mechanisms=rs[0]["mechanisms"], seeds=len(rs),
                            delay=dict(n=dn, positives=d, rate=d/dn if dn else 0,
                                       power=power_flag(d)),
                            shortage=dict(n=sn, positives=s_, rate=s_/sn if sn else 0,
                                          power=power_flag(s_)),
                            impact=dict(n=iN, positives=i, rate=i/iN if iN else 0,
                                        power=power_flag(i)),
                            validation_failures=sum(r["fails"] for r in rs)))
    print("-" * len(hdr))
    print(f"{'':3} {'LOW = below ' + str(POWER_LO) + ' positives; HIGH = above ' + str(POWER_HI)}")
    print(f"\ntotal validation failures across all runs: {total_fails}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump({"config": args.config, "seeds": seeds,
                       "power_target": [POWER_LO, POWER_HI],
                       "per_variant": summary, "runs": rows}, f, indent=2)
        print(f"report written to {args.report}")
    return 0 if total_fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
