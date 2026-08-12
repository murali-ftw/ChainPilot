#!/usr/bin/env python3
"""
Gate diagnostic analysis — the four questions of `STEP3B_GATE_DIAGNOSTIC_PROMPT.md`.

Q1  How often does the gate disagree with the Markov prior?  (match rate, V1's metric)
Q2  When it disagrees, which nodes are affected?
Q3  Do those nodes share properties — degree, chain length, hidden resilience (true
    and observable-proxy, kept strictly separate)?
Q4  Are the depth representations different enough for ANY depth mechanism to matter?

Every test reports an effect size with a bootstrap or permutation reference, never a
bare threshold comparison.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from ml.models.depth import TASKS, TASK_ENTITY_TYPE  # noqa: E402
from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH  # noqa: E402

DEPTH_PAIRS = ["h0-h1", "h1-h2", "h2-h3", "h3-h4", "h0-h4"]
NAME = {"rgcn_attn_rung5_a": "Variant A", "rgcn_attn_rung5": "Rung 5 (base)",
        "rgcn_attn_rung5_ac": "Variant A+C", "rgcn_attn_rung5_c": "Variant C"}


def load(pattern: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(pattern)):
        z = np.load(path, allow_pickle=True)
        meta = z["meta"]
        out.append({"path": path, "arch": str(meta[0]), "variant": str(meta[1]),
                    "seed": int(meta[2]), "lam": str(meta[3]), "anneal": str(meta[4]) == "True",
                    "z": z})
    return out


def match_rates(runs: list[dict]) -> dict:
    """(arch, variant, lam/anneal tag, task) -> list of per-seed match rates."""
    out = defaultdict(list)
    for r in runs:
        tag = "anneal" if r["anneal"] else r["lam"]
        for task in TASKS:
            k = f"{task}__argmax"
            if k not in r["z"]:
                continue
            am = r["z"][k]
            out[(r["arch"], r["variant"], tag, task)].append(
                float((am == MARKOV_READOUT_DEPTH[task]).mean()))
    return out


def q1(runs):
    print("\n" + "=" * 92)
    print("Q1 — DISAGREEMENT RATE.  match = share of scored nodes whose argmax depth still")
    print("     equals the Markov prior.  V1's Variant A: 1.000 (zero deviating nodes).")
    print("=" * 92)
    mr = match_rates(runs)
    print(f"{'arm':<16} {'var':>3} {'lam':>7} {'task':<9} {'mean match':>11} {'min':>7} {'max':>7} "
          f"{'deviating nodes/seed':>21}")
    for (arch, variant, tag, task), vals in sorted(mr.items()):
        n_dev = []
        for r in runs:
            if (r["arch"], r["variant"], ("anneal" if r["anneal"] else r["lam"])) != (arch, variant, tag):
                continue
            am = r["z"][f"{task}__argmax"]
            n_dev.append(int((am != MARKOV_READOUT_DEPTH[task]).sum()))
        print(f"{NAME.get(arch, arch):<16} {variant:>3} {tag:>7} {task:<9} "
              f"{np.mean(vals):>11.4f} {np.min(vals):>7.4f} {np.max(vals):>7.4f} "
              f"{np.mean(n_dev):>13,.0f} of {len(am):,}")
    return mr


def q2(runs):
    print("\n" + "=" * 92)
    print("Q2 — WHICH NODES DISAGREE")
    print("=" * 92)
    any_dev = False
    for r in runs:
        for task in TASKS:
            am = r["z"].get(f"{task}__argmax")
            if am is None:
                continue
            dev = am != MARKOV_READOUT_DEPTH[task]
            if not dev.any():
                continue
            any_dev = True
            ids = r["z"][f"{task}__entity_id"][dev]
            uniq = len(set(ids.tolist()))
            chosen = np.bincount(am[dev], minlength=5)
            print(f"  {NAME.get(r['arch'], r['arch'])} v{r['variant']} seed {r['seed']} {task}: "
                  f"{dev.sum():,} deviating rows / {uniq:,} distinct entities "
                  f"(of {len(set(r['z'][f'{task}__entity_id'].tolist())):,}); "
                  f"chosen depth histogram {chosen.tolist()}")
    if not any_dev:
        print("  No deviating node in any run, any arm, any task, any seed.")
        print("  Q2 and Q3 are therefore vacuous for these arms: there is no disagreement set")
        print("  to characterise. This is an answer, not a missing measurement.")
    return any_dev


def _logistic_auc_perm(x: np.ndarray, y: np.ndarray, n_perm: int = 200, seed: int = 0):
    """Effect size (AUC of a single continuous predictor for a binary outcome) with a
    permutation null. Univariate, so AUC is just the rank statistic -- no fitting."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return None
    if len(y) > 20000:      # the rank statistic is stable well below the full population
        idx = np.random.default_rng(seed).choice(len(y), 20000, replace=False)
        x, y = x[idx], y[idx]
        if len(np.unique(y)) < 2:
            return None
    obs = roc_auc_score(y, x)
    rng = np.random.default_rng(seed)
    null = np.array([roc_auc_score(rng.permutation(y), x) for _ in range(n_perm)])
    return {"auc": float(obs), "null_lo": float(np.percentile(null, 2.5)),
            "null_hi": float(np.percentile(null, 97.5)),
            "outside": bool(obs < np.percentile(null, 2.5) or obs > np.percentile(null, 97.5))}


def q3(runs, hidden_dir):
    print("\n" + "=" * 92)
    print("Q3 — DO DISAGREEING NODES SHARE PROPERTIES")
    print("=" * 92)
    tested = False
    for r in runs:
        hid_path = os.path.join(hidden_dir, f"{r['variant']}_{r['seed']}.json")
        hid = json.load(open(hid_path)) if os.path.exists(hid_path) else {}
        for task in TASKS:
            am = r["z"].get(f"{task}__argmax")
            if am is None:
                continue
            dev = (am != MARKOV_READOUT_DEPTH[task]).astype(int)
            if dev.sum() == 0 or dev.sum() == len(dev):
                continue
            tested = True
            ids = r["z"][f"{task}__entity_id"]
            props = {"degree": r["z"][f"{task}__degree"].astype(float)}
            ent = TASK_ENTITY_TYPE[task]
            sup_of = (hid.get("shipment_supplier", {}) if ent == "Shipment" else None)
            def sup_id(i):
                return sup_of.get(i) if sup_of is not None else (i if ent == "Supplier" else None)
            if hid.get("resilience"):
                props["true_resilience"] = np.array(
                    [hid["resilience"].get(sup_id(i), np.nan) for i in ids], dtype=float)
            if hid.get("chain_len"):
                props["chain_length"] = np.array(
                    [hid["chain_len"].get(sup_id(i), np.nan) for i in ids], dtype=float)
            for pname, x in props.items():
                ok = ~np.isnan(x)
                if ok.sum() < 50 or len(np.unique(dev[ok])) < 2:
                    continue
                res = _logistic_auc_perm(x[ok], dev[ok])
                if res is None:
                    continue
                print(f"  {NAME.get(r['arch'], r['arch'])} v{r['variant']} s{r['seed']} "
                      f"{task:<9} {pname:<16} AUC {res['auc']:.3f} "
                      f"null [{res['null_lo']:.3f}, {res['null_hi']:.3f}] "
                      f"{'OUTSIDE null' if res['outside'] else 'inside null'}")
    if not tested:
        print("  Not testable: no run produced a non-empty disagreement set (see Q2).")
        print("  Property association requires both classes to exist; with zero deviating")
        print("  nodes there is no contrast to measure, for degree, chain length, or either")
        print("  resilience measure.")
    return tested


def q4(runs):
    print("\n" + "=" * 92)
    print("Q4 — ARE THE DEPTH REPRESENTATIONS DIFFERENT ENOUGH TO MATTER")
    print("     cosine similarity between depth pairs, before any gate selects among them.")
    print("     1.000 = identical direction: nothing for any depth mechanism to choose between.")
    print("=" * 92)
    agg = defaultdict(list)
    for r in runs:
        for key in r["z"].files:
            if not key.startswith("cos__"):
                continue
            nt = key.split("__", 1)[1]
            agg[(r["arch"], r["variant"], nt)].append(r["z"][key].mean(axis=0))
    print(f"{'arm':<16} {'var':>3} {'node type':<10} " + " ".join(f"{p:>8}" for p in DEPTH_PAIRS))
    for (arch, variant, nt), vals in sorted(agg.items()):
        m = np.nanmean(np.stack(vals), axis=0)
        print(f"{NAME.get(arch, arch):<16} {variant:>3} {nt:<10} " + " ".join(f"{v:>8.4f}" for v in m))
    return agg


def q4_by_chain(runs, hidden_dir):
    """Does depth separation grow with Mechanism J's chain length? Supplier nodes only,
    since chain length is a supplier property."""
    print("\n  Q4b — does depth separation vary with chain length (Supplier nodes, Variant J/K)?")
    rows = []
    for r in runs:
        if "cos__Supplier" not in r["z"].files:
            continue
        hid_path = os.path.join(hidden_dir, f"{r['variant']}_{r['seed']}.json")
        if not os.path.exists(hid_path):
            continue
        hid = json.load(open(hid_path))
        if not hid.get("chain_len"):
            continue
        cos = r["z"]["cos__Supplier"]
        ids = r["z"]["cosid__Supplier"]
        cl = np.array([hid["chain_len"].get(i, np.nan) for i in ids], dtype=float)
        ok = ~np.isnan(cl)
        if ok.sum() < 50:
            continue
        h04 = cos[ok, 4]
        for lo, hi, label in [(0, 2, "short (<=2)"), (3, 4, "mid (3-4)"), (5, 99, "long (>=5)")]:
            m = (cl[ok] >= lo) & (cl[ok] <= hi)
            if m.sum() >= 20:
                rows.append((r["variant"], label, float(h04[m].mean()), int(m.sum())))
        c = np.corrcoef(cl[ok], h04)[0, 1]
        rows.append((r["variant"], f"corr(chain_len, h0-h4 cos) seed {r['seed']}", float(c), int(ok.sum())))
    agg = defaultdict(list)
    for v, label, val, n in rows:
        agg[(v, label)].append(val)
    for (v, label), vals in sorted(agg.items()):
        print(f"    v{v} {label:<40} {np.mean(vals):+.4f}  (n_seeds={len(vals)})")
    if not rows:
        print("    no chain-length data available for these runs")


def alpha_dispersion(runs):
    """Is the gate PER-NODE or globally shifted? If the gate had learned a per-node
    policy, alpha would vary from node to node; if it merely moved the per-task
    position bias, every node shares almost the same distribution. Reported as the
    across-node std of each depth's weight, averaged over depths."""
    print("\n" + "=" * 92)
    print("Q2b — IS THE GATE ACTUALLY PER-NODE?  across-node std of the depth weights")
    print("      ~0 => every node gets the same distribution (a global depth switch,")
    print("      not per-node adaptivity).")
    print("=" * 92)
    agg = defaultdict(list)
    for r in runs:
        tag = "anneal" if r["anneal"] else r["lam"]
        for task in TASKS:
            k = f"{task}__alpha"
            if k not in r["z"]:
                continue
            a = r["z"][k]
            agg[(r["arch"], r["variant"], tag, task)].append(
                (float(a.std(axis=0).mean()), float(a.max(axis=1).mean())))
    print(f"{'arm':<16} {'var':>3} {'lam':>7} {'task':<9} {'across-node std':>16} {'mean max weight':>16}")
    for (arch, variant, tag, task), vals in sorted(agg.items()):
        s = float(np.mean([v[0] for v in vals])); m = float(np.mean([v[1] for v in vals]))
        print(f"{NAME.get(arch, arch):<16} {variant:>3} {tag:>7} {task:<9} {s:>16.5f} {m:>16.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default=os.path.join(REPO, "out", "gate", "*.npz"))
    ap.add_argument("--hidden-dir", default=os.path.join(REPO, "out", "hidden"))
    args = ap.parse_args()
    runs = load(args.glob)
    print(f"{len(runs)} diagnostic runs: " + ", ".join(
        sorted({f"{NAME.get(r['arch'], r['arch'])}/v{r['variant']}" for r in runs})))
    q1(runs)
    q2(runs)
    alpha_dispersion(runs)
    q3(runs, args.hidden_dir)
    q4(runs)
    q4_by_chain(runs, args.hidden_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

