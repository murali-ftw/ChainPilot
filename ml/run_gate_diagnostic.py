#!/usr/bin/env python3
"""
Gate diagnostic — capture what the depth gate actually does, per node.

`reports/layer2_testing.md` §7 found no adaptive-depth benefit and §9 recorded why
that could not be explained: gate weights were never persisted, so "the gate found
nothing useful" and "the gate never moved" were indistinguishable. This script
closes that by persisting, for every node scored on the TEST split:

* the full softmax distribution over depths h^0..h^4 (not just the argmax),
* the argmax depth, and the Markov prior depth for that task,
* the entity id and entity type it belongs to,
* that node's total in-degree (the same quantity Variant B feeds its gate),

plus, independent of the gate entirely, the **cosine similarity between consecutive
depth representations** per node type — question 4, which asks whether there is
anything for *any* depth mechanism to choose between in the first place.

The capture happens on the evaluation path only. Training is untouched, so the
models here are the same models §5-§7 measured.

Usage:
    python3 ml/run_gate_diagnostic.py --variants 0,J --archs rgcn_attn_rung5_a \\
        --seeds 42,43,44,45,46 --out out/gate/
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data.loader import load_bundles, split_bundles  # noqa: E402
from ml.models.depth import TASKS, TASK_ENTITY_TYPE  # noqa: E402
from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH  # noqa: E402
from ml.train import train_model  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO, "ml", ".cache")
CFG = dict(hidden=128, num_bases=10)      # SHARE's matched-d arm, as every Layer 2 run uses

# Node types whose depth separation is worth measuring: the three readout entities
# plus Component, which sits between Supplier and Product on the BOM path.
COS_NODE_TYPES = ["Shipment", "Product", "Supplier", "Component"]


@torch.no_grad()
def capture(model, bundles) -> dict:
    """Per-node gate output and depth geometry, pooled over the test snapshots."""
    model.eval()
    out = {t: {"alpha": [], "argmax": [], "entity_id": [], "degree": [], "snapshot": []}
           for t in TASKS}
    cos = {nt: [] for nt in COS_NODE_TYPES}

    for b_i, bundle in enumerate(bundles):
        data = bundle.data
        _logits, layers = model(data.x_dict, data.edge_index_dict)
        gate = getattr(model, "_last_gate_weights", None)

        # In-degree per node type, summed over every relation where it is the
        # destination -- the same definition Variant B's gate input uses.
        deg = {}
        for nt in data.node_types:
            n = data[nt].num_nodes
            d = torch.zeros(n)
            for et in data.edge_types:
                if et[2] == nt and data[et].edge_index.numel():
                    d += torch.bincount(data[et].edge_index[1].cpu(), minlength=n).float()
            deg[nt] = d

        if gate is not None:
            for task in TASKS:
                nt = TASK_ENTITY_TYPE[task]
                a = gate[task].float().cpu().numpy()
                out[task]["alpha"].append(a)
                out[task]["argmax"].append(a.argmax(1))
                out[task]["entity_id"].append(np.asarray(data[nt].node_id, dtype=object))
                out[task]["degree"].append(deg[nt].numpy())
                out[task]["snapshot"].append(np.full(a.shape[0], b_i))

        # Question 4: how different are the depth representations, before any gate?
        for nt in COS_NODE_TYPES:
            if nt not in layers[0]:
                continue
            h = [layers[d][nt] for d in range(len(layers))]
            row = []
            for d in range(len(h) - 1):
                row.append(torch.cosine_similarity(h[d], h[d + 1], dim=-1).cpu().numpy())
            row.append(torch.cosine_similarity(h[0], h[-1], dim=-1).cpu().numpy())
            cos[nt].append(np.stack(row, axis=1))     # [N, 5]: h0h1 h1h2 h2h3 h3h4 h0h4

    blob = {}
    for task in TASKS:
        if not out[task]["alpha"]:
            continue
        for k, v in out[task].items():
            blob[f"{task}__{k}"] = np.concatenate(v)
        blob[f"{task}__markov_depth"] = np.array(MARKOV_READOUT_DEPTH[task])
    for nt, v in cos.items():
        if v:
            blob[f"cos__{nt}"] = np.concatenate(v)
            blob[f"cosid__{nt}"] = np.concatenate(
                [np.asarray(b.data[nt].node_id, dtype=object) for b in bundles])
    return blob


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default=os.path.join(REPO, "db", "csv_v1scale"))
    ap.add_argument("--variants", default="0,J")
    ap.add_argument("--archs", default="rgcn_attn_rung5_a,rgcn_attn_rung5")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--lambda-bound", type=float, default=0.3)
    ap.add_argument("--anneal", action="store_true",
                    help="linearly grow lambda_bound from 0.1 to --lambda-bound over training")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(REPO, "out", "gate"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
            d = os.path.join(args.csv_dir, f"v{variant}_seed{seed}")
            bundles, manifest = load_bundles(d, cache_dir=CACHE_DIR)
            if args.device != "cpu":
                for b in bundles:
                    b.to(args.device)
            tr, va, te = split_bundles(bundles)
            for arch in [a.strip() for a in args.archs.split(",") if a.strip()]:
                tag = args.tag or (f"lam{args.lambda_bound:g}" if "rung5_a" in arch else "")
                name = f"{arch}_v{variant}_seed{seed}{'_' + tag if tag else ''}"
                path = os.path.join(args.out, name + ".npz")
                if os.path.exists(path):
                    print(f"  {name}: exists, skipping", flush=True)
                    continue
                t0 = time.time()
                r = train_model(arch, tr, va, epochs=args.epochs, seed=args.model_seed,
                                device=args.device, rung5a_lambda_bound=args.lambda_bound,
                                anneal_lambda=args.anneal, **CFG)
                blob = capture(r["model"], te)
                # AUC on the same test split, so this run is comparable to §5-§7's table
                from ml.evaluate import collect_predictions
                from sklearn.metrics import roc_auc_score
                preds = collect_predictions(r["model"], te)
                aucs = {}
                for task in TASKS:
                    y, p = preds[task]["y"], preds[task]["p"]
                    aucs[task] = float(roc_auc_score(y, p)) if len(y) and 0 < y.sum() < len(y) else np.nan
                blob["auc"] = np.array([aucs[t] for t in TASKS])
                blob["auc_tasks"] = np.array(list(TASKS), dtype=object)
                blob["meta"] = np.array([arch, variant, str(seed), str(args.lambda_bound),
                                         str(args.anneal), str(r["epochs_run"])], dtype=object)
                np.savez_compressed(path, **blob)
                match = {}
                for task in TASKS:
                    if f"{task}__argmax" in blob:
                        match[task] = float((blob[f"{task}__argmax"] ==
                                             MARKOV_READOUT_DEPTH[task]).mean())
                print(f"  {name}: {time.time() - t0:.0f}s  "
                      + "  ".join(f"{t}={aucs[t]:.4f}/match={match.get(t, float('nan')):.3f}"
                                  for t in TASKS), flush=True)
            del bundles
    return 0


if __name__ == "__main__":
    sys.exit(main())
