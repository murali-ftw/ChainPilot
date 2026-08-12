#!/usr/bin/env python3
"""
Layer 3 runner — train a Transformer 2 arm and capture BOTH halves V1 always reported
together: hidden-dependency **discovery quality**, and **downstream AUC**.

V1's `reports/layer3.md` established that these are separate claims and must never be
collapsed into one number: Transformer 2 discovered the planted structure decisively on
every seed while moving no AUC at all. This script measures both on the same trained
model, per run.

Discovery metrics, both against a defined chance baseline, matching `layer3.md` §1.4:

* **percentile rank** — where each ground-truth co-member pair's cosine similarity sits
  among all C(n,2) Supplier pairs. Chance = 0.500.
* **pool-membership discovery rate** — does the pair appear in either endpoint's own
  top-k retrieval pool? Chance is measured directly from the same run's global pool
  occupancy, not assumed.

V2 lets both be reported **per group type**: Type A (causally coupled via Mechanism D),
Type B (correlated but α=0 — V1's H_POLYMER behaviour, the null control) and Type C
(decoy). V1 could only ever measure one undifferentiated scenario.

Ground truth comes from `ml/extract_hidden_state.py` (the generator's internal `HP_GROUPS`),
never from any emitted CSV, and is used only to score the model -- never as an input.

    python3 ml/run_transformer2.py --variants B,D --archs rgcn_attn_t2_confidence \\
        --seeds 42,43,44,45,46
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data.loader import load_bundles, split_bundles  # noqa: E402
from ml.evaluate import collect_predictions  # noqa: E402
from ml.models.depth import TASKS  # noqa: E402
from ml.train import train_model  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO, "ml", ".cache")
CFG = dict(hidden=128, num_bases=10)


@torch.no_grad()
def discovery(model, bundles, groups: list[dict], sup_ids_per_bundle) -> dict:
    """Percentile rank and pool-membership discovery rate, per group type."""
    model.eval()
    per_type = {t: {"pct": [], "found": []} for t in "ABC"}
    chance_rates = []
    for bundle, sup_ids in zip(bundles, sup_ids_per_bundle):
        model(bundle.data.x_dict, bundle.data.edge_index_dict)
        z = getattr(model, "_last_z_impact", None)
        t2 = getattr(model, "_last_t2", None)
        if z is None or t2 is None:
            return {}
        idx = {s: i for i, s in enumerate(sup_ids)}
        zc = torch.nn.functional.normalize(z.float().cpu(), dim=-1)
        sim = zc @ zc.T
        n = sim.size(0)
        sim.fill_diagonal_(-2.0)
        # Rank once: the percentile of a value among all off-diagonal pair similarities.
        flat = sim[torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)].numpy()
        order = np.sort(flat)
        pool = t2[0].cpu().numpy()          # [n, k] retrieved indices per supplier
        pool_sets = [set(row.tolist()) for row in pool]
        # Chance discovery rate, MEASURED not assumed (V1 §1.4 does the same): the share of
        # all C(n,2) pairs that appear in some pool. The analytic 2k/(n-1) overstates it,
        # because retrieval is largely mutual -- if a is in b's pool, b is usually in a's,
        # so the distinct-pair coverage is well below twice the pool size.
        covered = set()
        for i, row in enumerate(pool_sets):
            for j in row:
                covered.add((i, j) if i < j else (j, i))
        chance_rates.append(len(covered) / max(1, n * (n - 1) / 2))
        for g in groups:
            mem = [idx[m] for m in g["members"] if m in idx]
            for a_i in range(len(mem)):
                for b_i in range(a_i + 1, len(mem)):
                    a, b = mem[a_i], mem[b_i]
                    v = float(sim[a, b])
                    per_type[g["type"]]["pct"].append(
                        float(np.searchsorted(order, v) / max(1, len(order))))
                    per_type[g["type"]]["found"].append(
                        float(b in pool_sets[a] or a in pool_sets[b]))
    out = {"chance_discovery_rate": float(np.mean(chance_rates)) if chance_rates else float("nan")}
    for t, d in per_type.items():
        if d["pct"]:
            out[f"pct_{t}"] = float(np.mean(d["pct"]))
            out[f"rate_{t}"] = float(np.mean(d["found"]))
            out[f"n_{t}"] = len(d["pct"])
    return out


@torch.no_grad()
def trust_stats(model, bundles, groups, sup_ids_per_bundle) -> dict:
    """V1 §2.7's trust-gate stability check, extended to compare Type A/B/C members
    against the population -- which V1 could not do, having only one group type."""
    model.eval()
    vals, degs, types = [], [], []
    member_type = {m: g["type"] for g in groups for m in g["members"]}
    for bundle, sup_ids in zip(bundles, sup_ids_per_bundle):
        model(bundle.data.x_dict, bundle.data.edge_index_dict)
        tr = getattr(model, "_last_trust", None)
        if tr is None:
            return {}
        tr = tr.float().cpu().numpy().reshape(-1)
        data = bundle.data
        n = data["Supplier"].num_nodes
        d = torch.zeros(n)
        for et in data.edge_types:
            if et[2] == "Supplier" and data[et].edge_index.numel():
                d += torch.bincount(data[et].edge_index[1].cpu(), minlength=n).float()
        vals.append(tr); degs.append(d.numpy())
        types.append(np.array([member_type.get(s, "-") for s in sup_ids], dtype=object))
    v = np.concatenate(vals); dg = np.concatenate(degs); ty = np.concatenate(types)
    out = {"mean_trust": float(v.mean()), "std_trust": float(v.std()),
           "share_pos": float((v > 0.01).mean()), "share_neg": float((v < -0.01).mean()),
           "share_zero": float((np.abs(v) <= 0.01).mean()),
           "corr_abs_trust_degree": float(np.corrcoef(np.abs(v), dg)[0, 1])}
    for t in "ABC":
        m = ty == t
        if m.sum():
            out[f"mean_trust_type_{t}"] = float(v[m].mean())
    out["mean_trust_non_member"] = float(v[ty == "-"].mean()) if (ty == "-").any() else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--variants", default="B,D")
    ap.add_argument("--archs", default="rgcn_attn_rung5_a")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--hidden-dir", default=os.path.join(REPO, "out", "hidden"))
    ap.add_argument("--out", default=os.path.join(REPO, "out", "t2"))
    ap.add_argument("--save-preds", action="store_true",
                    help="also dump per-row test predictions (y/p/snapshot block) per task, "
                         "which is what the paired block bootstrap against the no-T2 baseline "
                         "needs -- AUC scalars alone cannot produce a CI on a delta.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
            d = os.path.join(args.csv_dir, f"v{variant}_seed{seed}")
            bundles, manifest = load_bundles(d, cache_dir=CACHE_DIR)
            tr, va, te = split_bundles(bundles)
            sup_ids = [b.data["Supplier"].node_id for b in te]
            hid_path = os.path.join(args.hidden_dir, f"{variant}_{seed}.json")
            groups = json.load(open(hid_path))["hp_groups"] if os.path.exists(hid_path) else []
            for arch in [a.strip() for a in args.archs.split(",") if a.strip()]:
                name = f"{arch}_v{variant}_seed{seed}"
                path = os.path.join(args.out, name + ".json")
                if os.path.exists(path):
                    print(f"  {name}: exists, skipping", flush=True)
                    continue
                t0 = time.time()
                r = train_model(arch, tr, va, epochs=args.epochs, seed=args.model_seed, **CFG)
                model = r["model"]
                preds = collect_predictions(model, te)
                from sklearn.metrics import roc_auc_score
                aucs = {}
                for task in TASKS:
                    y, p = preds[task]["y"], preds[task]["p"]
                    aucs[task] = (float(roc_auc_score(y, p))
                                  if len(y) and 0 < y.sum() < len(y) else None)
                    aucs[f"{task}_positives"] = int(y.sum()) if len(y) else 0
                blob = {"arch": arch, "variant": variant, "seed": seed,
                        "params": model.parameter_count(), "seconds": time.time() - t0,
                        "epochs_run": r["epochs_run"], "auc": aucs,
                        "discovery": discovery(model, te, groups, sup_ids),
                        "trust": trust_stats(model, te, groups, sup_ids)}
                with open(path, "w") as fh:
                    json.dump(blob, fh, indent=1)
                if args.save_preds:
                    np.savez_compressed(
                        os.path.join(args.out, name + "_preds.npz"),
                        **{f"{task}_{k}": preds[task][k]
                           for task in TASKS for k in ("y", "p", "block")
                           if len(preds[task]["y"])})
                dsc = blob["discovery"]
                print(f"  {name}: {blob['seconds']:.0f}s  "
                      + " ".join(f"{t}={aucs[t]:.4f}" for t in TASKS if aucs[t] is not None)
                      + (f"  pctA={dsc.get('pct_A', float('nan')):.3f} "
                         f"rateA={dsc.get('rate_A', float('nan')):.3f} "
                         f"chance={dsc.get('chance_discovery_rate', float('nan')):.3f}"
                         if dsc else "  (no T2)"), flush=True)
            del bundles
    return 0


if __name__ == "__main__":
    sys.exit(main())
