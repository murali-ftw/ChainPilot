"""Phase 6 — collect the report's numbers from the bundles.

  §4  reproduction gate: target vs achieved for the five Phase 5 configurations, against each metric's seed band,
      with epochs and best validation score beside Phase 5's cell
  §5  bundle invariants: rows in the hazard loss, survival monotone, quantile crossings, fill partition, the join
  §6  seed bands on the shipped configurations, per task per world, and every Addendum B margin re-read against its
      OWN band rather than a borrowed one
  §7  the drift monitor, run through loop.predict(): arrival with its h0 reference in both worlds (is the fallback
      engaged, and what is the week-ECE of the distribution actually served), fill and capacity

Writes ml/artifacts/phase6_collect.json and phase6_tables.md.  --no-drift skips §7 (it runs inference on MPS).
"""
from __future__ import annotations
import os, sys, json, glob, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "models")]
import numpy as np
from config import ARTIFACTS

B = os.path.join(ARTIFACTS, "bundles")
out, R = [], {}
P = lambda *a: out.append(" ".join(str(x) for x in a))


def bpath(task, world, arch, depth, lr, seed):
    return os.path.join(B, task, f"{world}_{arch}_h{depth}_lr{lr:g}_s{seed}")


def load(path):
    if not os.path.exists(os.path.join(path, "config.json")):
        return None
    j = lambda f: json.load(open(os.path.join(path, f)))
    return dict(cfg=j("config.json"), met=j("metrics.json"), log=j("train_log.json"), rec=j("recalibration.json"),
                base=j("drift_baseline.json"), join=j("join_check.json"))


def m(bundle, part, key):
    try:
        return float(bundle["met"]["test"][part][key][0])
    except (KeyError, TypeError):
        return float("nan")


# ================================================================== §4 reproduction gate
REPRO = [
    ("R1", "arrival h⁰ v6 @ 2e-3", ("arrival_week", "v6", "none", 0, 2e-3), [("raw", "cindex", 0.6602, 0.0004, "phase-5 §7")],
     ("phase5_grid_arrival.json", dict(world="v6", arch="none", depth=0, lr=0.002, seed=7, gate=0, wsla=0))),
    ("R2", "arrival SHARE-lite h⁴ v6 @ 2.5e-4", ("arrival_week", "v6", "lite", 4, 2.5e-4), [("raw", "cindex", 0.6731, 0.0004, "Addendum B")],
     ("phase5_grid_p4_arrival.json", dict(world="v6", arch="lite", depth=4, lr=0.00025, seed=7))),
    ("R3", "fill h⁰ v6 @ 1.25e-4", ("fill_rate", "v6", "none", 0, 1.25e-4),
     [("raw", "crps_exact", 0.0618, 0.0001, "§6.1"), ("raw", "ece20", 0.0448, 0.0073, "§6.1"), ("recal", "ece20", 0.0203, 0.0021, "Addendum A")],
     ("phase5_grid_fill.json", dict(world="v6", arch="none", depth=0, lr=0.000125, seed=7, gate=0, wsla=0))),
    ("R4", "capacity h⁰ v6 @ 2.5e-4", ("capacity_strain", "v6", "none", 0, 2.5e-4), [("raw", "pinball_mean", 0.0464, 0.0004, "§8")],
     ("phase5_grid_capacity.json", dict(world="v6", arch="none", depth=0, lr=0.00025, seed=7, gate=0, wsla=0))),
    ("R5", "capacity HeteroMP h⁴ v7 @ 2.5e-4", ("capacity_strain", "v7", "mp", 4, 2.5e-4), [("raw", "pinball_mean", 0.0707, 0.0004, "Addendum B")],
     ("phase5_grid_p4_capacity.json", dict(world="v7", arch="mp", depth=4, lr=0.00025, seed=7))),
]


def p5row(gridfile, key):
    p = os.path.join(ARTIFACTS, gridfile)
    for r in (json.load(open(p)) if os.path.exists(p) else []):
        if all(r.get(k) == v for k, v in key.items()) and not r.get("fill_loss"):
            return r
    return None


P("\n### §4 Reproduction gate\n")
P("| id | configuration | metric | target | source | band | **achieved** | \\|Δ\\| | Δ / band | verdict | epochs (best) new vs Phase 5 | best val new vs Phase 5 |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
gate = []
for rid, name, (task, w, arch, d, lr), checks, (gfile, key) in REPRO:
    bd = load(bpath(task, w, arch, d, lr, 7))
    old = p5row(gfile, key)
    if bd is None:
        P(f"| {rid} | {name} | — | — | — | — | *not run yet* | | | | | |"); gate.append((rid, None)); continue
    for part, metric, target, band, src in checks:
        got = m(bd, part, metric); delta = abs(got - target)
        ok = delta <= band + 1e-12
        gate.append((rid, ok))
        ep = f"{bd['log']['epochs_run']} ({bd['log']['best_epoch']}) vs {old['epochs_run']} ({old['best_epoch']})" if old else "—"
        bv = f"{bd['log']['best_val']:.5f} vs {old['best_val']:.5f}" if old else "—"
        P(f"| {rid} | {name} | {metric} {'(recalibrated)' if part == 'recal' else ''} | {target:.4f} | {src} | {band:.4f} | **{got:.4f}** | "
          f"{delta:.4f} | {delta / band:.2f} | {'**PASS**' if ok else '**FAIL**'} | {ep} | {bv} |")
R["reproduction"] = gate
P(f"\n**Gate: {sum(1 for _, ok in gate if ok)} of {len(gate)} checks pass"
  f"{'; not all configurations run' if any(ok is None for _, ok in gate) else ''}.**")

# ================================================================== §5 bundle invariants
P("\n### §5 Invariants asserted inside every bundle\n")
P("| bundle | trained by | rows entering the loss per epoch (training population) | survival violations | quantile crossings | fill partition | join: predictions / with actual / one-to-one | nondeterministic ops warned | peak RSS GB |")
P("|---|---|---|---|---|---|---|---|---|")
inv_all = []
for cfgp in sorted(glob.glob(os.path.join(B, "*", "*", "config.json"))):
    d = os.path.dirname(cfgp); bd = load(d)
    if bd is None: continue
    iv = bd["met"]["invariants"]; jn = bd["join"]; lg = bd["log"]
    rows = lg.get("rows_in_loss_each_epoch"); ntr = lg.get("n_train_rows")
    P(f"| {os.path.relpath(d, B)} | {'loop' if lg['trained_by'].startswith('ml/train') else 'converted'} | "
      f"{rows if rows else '— (converted)'}{f' ({ntr:,})' if ntr else ''} | {iv.get('survival_monotone_violations', '—')} | "
      f"{iv.get('quantile_crossings', '—')} | {iv.get('partition_refines_legacy_bins', '—')} | "
      f"{jn['predictions']:,} / {jn['joined_with_actual']:,} / {jn['one_row_per_prediction']} | {', '.join(lg.get('nondeterministic_ops_warned') or []) or '—'} | "
      f"{lg.get('peak_rss_gb', float('nan')):.2f} |")
    inv_all.append(dict(bundle=d, **iv, **jn))
R["invariants"] = inv_all

# ================================================================== §6 seed bands
SHIP = {"arrival_week": ("lite", 4, 2.5e-4, [("raw", "cindex"), ("raw", "roc_auc_late"), ("raw", "ece_week"), ("recal", "ece_week")]),
        "capacity_strain": ("mp", 4, 2.5e-4, [("raw", "pinball_mean"), ("raw", "coverage80"), ("raw", "spearman_p50")]),
        "fill_rate": ("none", 0, 1.25e-4, [("raw", "crps_exact"), ("raw", "ece20"), ("recal", "ece20"), ("recal", "rel_one")])}
bands = {}
P("\n### §6 Seed bands on the shipped configurations (seeds 7 / 17 / 27, test 2025)\n")
P("| task | world | metric | seed 7 | seed 17 | seed 27 | mean | **spread** | sd | borrowed band used in Addendum B |")
P("|---|---|---|---|---|---|---|---|---|---|")
BORROWED = {("arrival_week", "cindex"): 0.0004, ("capacity_strain", "pinball_mean"): 0.0004, ("fill_rate", "crps_exact"): 0.0001,
            ("fill_rate", "ece20"): 0.0073}
for task, (arch, d, lr, mets) in SHIP.items():
    for w in ("v6", "v7"):
        bs = [load(bpath(task, w, arch, d, lr, s)) for s in (7, 17, 27)]
        for part, metric in mets:
            vals = [m(b, part, metric) if b else float("nan") for b in bs]
            have = [v for v in vals if np.isfinite(v)]
            spread = max(have) - min(have) if len(have) == 3 else float("nan")
            sd = float(np.std(have, ddof=1)) if len(have) == 3 else float("nan")
            bands[(task, w, part, metric)] = spread
            lab = f"{metric}{' (recal)' if part == 'recal' else ''}"
            bor = BORROWED.get((task, metric)) if part == "raw" or metric != "ece20" else None
            P(f"| {task} | {w} | {lab} | " + " | ".join("—" if not np.isfinite(v) else f"{v:.4f}" for v in vals) +
              f" | {np.mean(have):.4f} | **{spread:.4f}** | {sd:.4f} | {'—' if bor is None else f'{bor:.4f}'} |"
              if have else f"| {task} | {w} | {lab} | — | — | — | — | — | — | — |")
R["bands"] = {"|".join(map(str, k)): v for k, v in bands.items()}

# Addendum B margins, re-read. (test metric of the shipped config) vs (the alternative Addendum B compared it with)
P("\n### §6 Addendum B verdicts re-read against their own bands\n")
P("| task | world | comparison (Addendum B) | margin | borrowed band → multiple | **own band (shipped config, 3 seeds)** → multiple | verdict moves? |")
P("|---|---|---|---|---|---|---|")
MARGINS = [
    ("capacity_strain", "v6", "HeteroMP h⁴ 0.0477 vs h⁰ 0.0464 (validation chose h⁴; test preferred h⁰)", 0.0477 - 0.0464, ("raw", "pinball_mean"), 0.0004),
    ("capacity_strain", "v7", "HeteroMP h⁴ 0.0707 vs h⁰ 0.0765", 0.0765 - 0.0707, ("raw", "pinball_mean"), 0.0004),
    ("arrival_week", "v6", "SHARE-lite h⁴ 0.6731 vs h⁰ 0.6604", 0.6731 - 0.6604, ("raw", "cindex"), 0.0004),
    ("arrival_week", "v7", "SHARE-lite h⁴ 0.6780 vs h⁰ 0.6626", 0.6780 - 0.6626, ("raw", "cindex"), 0.0004),
    ("fill_rate", "v6", "h⁰ 0.0618 vs validation's HeteroMP h⁴ 0.0622", 0.0622 - 0.0618, ("raw", "crps_exact"), 0.0001),
    ("fill_rate", "v7", "h⁰ 0.1025 vs validation's HeteroMP h⁴ 0.1038", 0.1038 - 0.1025, ("raw", "crps_exact"), 0.0001),
    ("fill_rate", "v6", "recalibrated ECE 0.0203 vs LightGBM same protocol 0.0187 (Addendum A)", 0.0203 - 0.0187, ("recal", "ece20"), 0.0021),
]
verdicts = []
for task, w, desc, margin, (part, metric), borrowed in MARGINS:
    own = bands.get((task, w, part, metric), float("nan"))
    bm = abs(margin) / borrowed
    om = abs(margin) / own if np.isfinite(own) and own > 0 else float("nan")
    moves = "—" if not np.isfinite(om) else ("**yes** — inside its own band" if (bm > 1 and om <= 1) else
                                             ("**yes** — resolves under its own band" if (bm <= 1 and om > 1) else "no"))
    verdicts.append(dict(task=task, world=w, comparison=desc, margin=margin, borrowed=borrowed, own=own, moves=moves))
    P(f"| {task} | {w} | {desc} | {abs(margin):.4f} | {borrowed:.4f} → {bm:.1f}× | "
      f"{'—' if not np.isfinite(own) else f'{own:.4f} → {om:.1f}×'} | {moves} |")
R["verdicts"] = verdicts


# ================================================================== §7 drift monitor
def drift_section():
    import loop as L
    from phase5_metrics import ece_marginal
    P("\n### §7 The drift monitor, through loop.predict() on the 2025 test inputs\n")
    P("| task | world | bundle | h⁰ reference | drift pp | h⁰ drift pp | excess pp | status | Monte Carlo distribution | week-ECE of the distribution served | week-ECE if not switched |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    specs = [("arrival_week", w, bpath("arrival_week", w, "lite", 4, 2.5e-4, 7), bpath("arrival_week", w, "none", 0, 2.5e-4, 7)) for w in ("v6", "v7")] + \
            [("fill_rate", w, bpath("fill_rate", w, "none", 0, 1.25e-4, 7), None) for w in ("v6", "v7")] + \
            [("capacity_strain", "v6", bpath("capacity_strain", "v6", "mp", 4, 2.5e-4, 7), bpath("capacity_strain", "v6", "none", 0, 2.5e-4, 7)),
             ("capacity_strain", "v7", bpath("capacity_strain", "v7", "mp", 4, 2.5e-4, 7), None)]
    for task, w, bp, hp in specs:
        if load(bp) is None or (hp and load(hp) is None):
            P(f"| {task} | {w} | missing | | | | | | | | |"); continue
        r = L.predict(bp, "test", hp)
        o = r["overall"]
        served = unswitched = "—"
        if task == "arrival_week":
            z = np.load(os.path.join(bp, "preds_test.npz"))
            cells = L.cells13(z["Y"], z["EV"])
            served = f"{ece_marginal(r['distribution'], cells)[0]:.4f}"
            mine = L.apply_recalibration(json.load(open(os.path.join(bp, "recalibration.json"))), task, z)["P13"]
            unswitched = f"{ece_marginal(mine, cells)[0]:.4f}"
        P(f"| {task} | {w} | {os.path.relpath(bp, B)} | {os.path.relpath(hp, B) if hp else '—'} | {o['drift_pp']:+.2f} | "
          f"{o.get('h0_drift_pp', float('nan')):+.2f} | {o.get('excess_pp', float('nan')):.2f} | {o['status']} | {o['distribution_source']} | {served} | {unswitched} |")
        rows.append(dict(task=task, world=w, **o, served_week_ece=served, unswitched_week_ece=unswitched,
                         batch_drift_pp=[b["drift_pp"] for b in r["batches"]]))
    R["drift"] = rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-drift", action="store_true")
    a = ap.parse_args()
    if not a.no_drift:
        drift_section()
    json.dump(R, open(os.path.join(ARTIFACTS, "phase6_collect.json"), "w"), indent=1, default=str)
    txt = "\n".join(out)
    open(os.path.join(ARTIFACTS, "phase6_tables.md"), "w").write(txt)
    print(txt)
