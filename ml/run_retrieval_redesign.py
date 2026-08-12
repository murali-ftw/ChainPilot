#!/usr/bin/env python3
"""
Layer 3 retrieval redesign — Stage 1 runner (contrastive ceiling) and control.

`reports/layer3_testing.md` §2 established that Transformer 2's same-type cosine
retrieval cannot separate V2's real hidden-parent groups (Type A/B) from the Type C
decoy at all: percentile rank 0.479-0.514 against a 0.500 chance baseline, discovery
rate 0.77x-1.11x chance, in all four fusion arms at both scales. §7's diagnosis was that
the embeddings being compared were never trained to encode co-membership. This script
tests the direct consequence of that diagnosis by adding an auxiliary contrastive loss
pointed straight at the retrieval geometry (`ml/models/contrastive.py`) and re-measuring
retrieval with the full IR metric set (`ml/retrieval_metrics.py`).

**Two arms, identical in every respect but the objective:**

* `rgcn_attn_variant_a_transformer2` — the plain T2 arm, retrained here as the on-device
  control. §2's numbers were measured on CPU; this session runs on MPS (see `--device`),
  which changes float reduction order, so the control is re-measured rather than quoted
  across devices.
* `rgcn_attn_t2_contrastive` — the same model class, the same 814,627 parameters, the
  same forward pass, plus the InfoNCE term. **Its supervision is privileged**: the
  generator's internal `HP_GROUPS` reaches a gradient for the first time in this project.
  Every number this arm produces is a ceiling under privileged supervision, not evidence
  of emergent discovery, and must be reported as such.

Supervision covers a random 70% of the Type A/B groups; the remaining 30% are held out
of the loss entirely and scored separately. The supervised split answers the brief's
question ("can this representation encode the signal if explicitly taught to"); the
held-out split answers the one that actually matters for a deployment ("did it learn
co-membership, or memorise ninety specific groups").

    python3 ml/run_retrieval_redesign.py --arch rgcn_attn_t2_contrastive \\
        --variants B,D --seeds 42,43,44,45,46 --device mps --out out/t2redesign
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
from ml.models.contrastive import build_supervision  # noqa: E402
from ml.models.depth import TASKS  # noqa: E402
from ml.retrieval_metrics import merge, score_retrieval  # noqa: E402
from ml.train import train_model  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO, "ml", ".cache")
CFG = dict(hidden=128, num_bases=10)


@torch.no_grad()
def retrieval_quality(model, bundles, groups, sup_ids_per_bundle, supervision) -> dict:
    """Score the model's retrieval on the test snapshots, per group type and per
    supervised/held-out split.

    The score matrix is the **cosine similarity over `z_impact`** -- the exact quantity
    `Transformer2GlobalAttention` ranks its candidate pool by -- and the pool is the one
    the model actually attended over, read back from `_last_t2`. Scoring a
    reconstruction of either would risk measuring something the model never used.
    """
    model.eval()
    accs = []
    type_by_group = {i: g["type"] for i, g in enumerate(groups)}
    for bundle, sup_ids in zip(bundles, sup_ids_per_bundle):
        model(bundle.data.x_dict, bundle.data.edge_index_dict)
        z = getattr(model, "_last_z_impact", None)
        t2 = getattr(model, "_last_t2", None)
        if z is None or t2 is None:
            return {}
        idx = {s: i for i, s in enumerate(sup_ids)}
        zc = torch.nn.functional.normalize(z.float().cpu(), dim=-1)
        score = (zc @ zc.T).numpy()
        pool = t2[0].cpu().numpy()
        members = {gi: [idx[m] for m in g["members"] if m in idx]
                   for gi, g in enumerate(groups)}
        subset = None
        if supervision is not None:
            held = supervision.held_out_mask.cpu().numpy()
            subset = {}
            for gi, g in enumerate(groups):
                if g["type"] not in ("A", "B"):
                    continue
                rows = members[gi]
                if rows:
                    subset[gi] = "held_out" if held[rows[0]] else "supervised"
        accs.append(score_retrieval(score, pool, members, type_by_group,
                                    top_k=pool.shape[1], subset_by_group=subset))
    return merge(accs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default=os.path.join(REPO, "db", "csv_mid"))
    ap.add_argument("--variants", default="B,D")
    ap.add_argument("--arch", default="rgcn_attn_t2_contrastive")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--device", default="mps", choices=("cpu", "mps"))
    ap.add_argument("--hidden-dir", default=os.path.join(REPO, "out", "hidden_mid"))
    ap.add_argument("--out", default=os.path.join(REPO, "out", "t2redesign"))
    ap.add_argument("--tag", default="", help="suffix on the result filename, so a second "
                                              "identically-configured replicate can be run "
                                              "for the reproduction floor without collision")
    ap.add_argument("--contrastive-weight", type=float, default=1.0)
    ap.add_argument("--contrastive-temp", type=float, default=0.1)
    ap.add_argument("--contrastive-warmup", type=int, default=3)
    ap.add_argument("--held-out-frac", type=float, default=0.3)
    ap.add_argument("--save-preds", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    use_contrastive = args.arch == "rgcn_attn_t2_contrastive"

    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
            name = f"{args.arch}_v{variant}_seed{seed}" + (f"_{args.tag}" if args.tag else "")
            path = os.path.join(args.out, name + ".json")
            if os.path.exists(path):
                print(f"  {name}: exists, skipping", flush=True)
                continue

            d = os.path.join(args.csv_dir, f"v{variant}_seed{seed}")
            bundles, _ = load_bundles(d, cache_dir=CACHE_DIR)
            tr, va, te = split_bundles(bundles)
            # The supervision object is resolved once per variant-seed against ONE
            # snapshot's Supplier order and reused for every bundle, so that order had
            # better be invariant. `ml/data/loader.py` builds it once from suppliers.csv,
            # which makes it so -- asserted rather than trusted, because a silent
            # mismatch here would scramble the labels and still train happily.
            ref = bundles[0].data["Supplier"].node_id
            for b in bundles:
                assert b.data["Supplier"].node_id == ref, "Supplier node order varies by snapshot"

            hid_path = os.path.join(args.hidden_dir, f"{variant}_{seed}.json")
            groups = json.load(open(hid_path))["hp_groups"] if os.path.exists(hid_path) else []
            sup = build_supervision(groups, ref, held_out_frac=args.held_out_frac, seed=seed)

            t0 = time.time()
            trd = [b.to(args.device) for b in tr]
            vad = [b.to(args.device) for b in va]
            ted = [b.to(args.device) for b in te]
            con = (dict(supervision=sup, weight=args.contrastive_weight,
                        temperature=args.contrastive_temp,
                        warmup_epochs=args.contrastive_warmup) if use_contrastive else None)
            r = train_model(args.arch, trd, vad, epochs=args.epochs, seed=args.model_seed,
                            device=args.device, contrastive=con, **CFG)
            model = r["model"]

            preds = collect_predictions(model, ted)
            from sklearn.metrics import roc_auc_score
            aucs = {}
            for task in TASKS:
                y, p = preds[task]["y"], preds[task]["p"]
                aucs[task] = (float(roc_auc_score(y, p))
                              if len(y) and 0 < y.sum() < len(y) else None)
                aucs[f"{task}_positives"] = int(y.sum()) if len(y) else 0

            sup_ids = [b.data["Supplier"].node_id for b in ted]
            rq = retrieval_quality(model, ted, groups, sup_ids, sup if use_contrastive else None)
            # Retrieval on the FINAL-epoch weights as well as on the best-validation-AUC
            # checkpoint the AUC table rests on. The two selection rules answer different
            # questions and can disagree sharply for this arm, so both are recorded.
            rq_final = {}
            if r.get("final_state") is not None:
                best_sd = {k: v.clone() for k, v in model.state_dict().items()}
                model.load_state_dict(r["final_state"])
                rq_final = retrieval_quality(model, ted, groups, sup_ids, sup)
                model.load_state_dict(best_sd)
            hist = r["history"]
            blob = {"arch": args.arch, "variant": variant, "seed": seed, "tag": args.tag,
                    "params": model.parameter_count(), "seconds": time.time() - t0,
                    "epochs_run": r["epochs_run"], "device": args.device,
                    "auc": aucs, "retrieval": rq, "retrieval_final_epoch": rq_final,
                    "contrastive_curve": [h.get("contrastive_loss") for h in hist],
                    "contrastive": r["hyperparameters"]["contrastive"],
                    "contrastive_loss_first": hist[0].get("contrastive_loss"),
                    "contrastive_loss_last": hist[-1].get("contrastive_loss"),
                    "best_val_auc": r["best_val_auc"], "best_epoch": r["best_epoch"]}
            with open(path, "w") as fh:
                json.dump(blob, fh, indent=1)
            if args.save_preds:
                np.savez_compressed(
                    os.path.join(args.out, name + "_preds.npz"),
                    **{f"{task}_{k}": preds[task][k]
                       for task in TASKS for k in ("y", "p", "block")
                       if len(preds[task]["y"])})

            line = f"  {name}: {blob['seconds']:.0f}s  " + " ".join(
                f"{t}={aucs[t]:.4f}" for t in TASKS if aucs[t] is not None)
            for key in ("A", "B", "C", "A|supervised", "A|held_out"):
                if key in rq:
                    m = rq[key]
                    line += (f"  {key}: rank={m['pct_rank']:.3f} R@k={m['recall_at_k']:.3f}"
                             f" mrr={m['mrr']:.4f}")
            if blob["contrastive_loss_last"] is not None:
                line += (f"  con {blob['contrastive_loss_first']:.2f}"
                         f"->{blob['contrastive_loss_last']:.2f}")
            print(line, flush=True)

            for b in bundles:
                b.to("cpu")
            del bundles, trd, vad, ted, model, r
            if args.device == "mps":
                torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
