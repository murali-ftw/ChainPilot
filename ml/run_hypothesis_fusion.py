#!/usr/bin/env python3
"""
§10.5 — Confidence-Aware Fusion, weighted by §10's own calibrated hypothesis confidence.

**This is the secondary experiment, and it is decoupled from §10's primary claim on
purpose.** Every version of Transformer 2 fusion tested across four prior sessions came back
flat-to-strongly-negative on impact AUC: the original additive fusion (§3), the
attention-entropy confidence variant, the trust gate (§5, unstable), cross-attention, and the
contrastive-retrieval arm (§9, **−0.145 / −0.158**, the largest effect this project has
measured anywhere). Nothing about reframing retrieval as hypothesis ranking changes that
history, and this script does not assume it will. **A null or negative result here does not
undermine §10.3's calibration findings.** Hypothesis quality and downstream AUC are different
claims, and §7 already recorded what happens in this project when the two get conflated.

**What is actually different from `ml/models/transformer2_confidence.py`.** That variant
derived its per-node fusion weight from the attention pool's *own* entropy and mean cosine --
a self-referential quantity, since a confident-looking pool and a correct one are not the
same thing, and §2 had already shown the pools contain no co-members to be confident about.
This arm supplies the weight from **outside** the model: §10's hypothesis head, fit on
observable co-degradation and *validated against ground truth before being used here*. Per
supplier,

    confidence_i = max over i's detected partners j of (1 - P_calibrated(unknown | i, j))

i.e. how confidently any of this supplier's observed co-degradations can be attributed to a
*named* mechanism at all. Suppliers with no usable trajectory get 0 -- no evidence, so no
reason to trust retrieval for them -- which is itself the substantive behavioural difference
from a global scalar.

**The confidence is produced out-of-fold.** The head scoring a dataset seed's suppliers was
trained on the *other* seeds, exactly as in §10.3, so no supplier's fusion weight was fitted
on that supplier. Without that, the arm would be tuned on its own test set and any AUC gain
would be meaningless.

Two arms, paired by dataset seed, both trained here on this session's device so the delta and
the floor are measured under identical conditions (§9.10: the floor is device- and
arm-dependent and must not be quoted across sessions):

* `baseline`  -- plain T2, `hyp_confidence=None`, byte-for-byte the §9.3 control.
* `hypconf`   -- the same model, same parameter count, plus the confidence hook.

    python3 ml/run_hypothesis_fusion.py --variants B --seeds 42,43,44,45,46 --device cpu
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
from ml.hypothesis_features import (  # noqa: E402
    FUTURE, PairContext, assert_no_privileged_features, build_features, detect_patterns)
from ml.hypothesis_labels import CLASSES, label_matrix, load_mechanism_state  # noqa: E402
from ml.hypothesis_ranker import (  # noqa: E402
    isotonic_apply, isotonic_fit, predict, train_head)
from ml.models.depth import TASKS  # noqa: E402
from ml.train import train_model  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO, "ml", ".cache")
CFG = dict(hidden=128, num_bases=10)
ARCH = "rgcn_attn_variant_a_transformer2"


def _build(csv_dir: str, state_path: str, top_k: int) -> dict:
    state = load_mechanism_state(state_path, csv_dir)
    ctx = PairContext(csv_dir, as_of=FUTURE)
    pairs = detect_patterns(ctx, top_k=top_k)
    X, names = build_features(ctx, pairs)
    assert_no_privileged_features(names)
    return {"seed": state["seed"], "sup_ids": state["supplier_ids"], "pairs": pairs,
            "X": X, "y": label_matrix(state, pairs), "n": ctx.n}


def hypothesis_confidence(variant: str, seeds: list[int], csv_dir: str, state_dir: str,
                          top_k: int, epochs: int, model_seed: int) -> dict[int, np.ndarray]:
    """`{seed: [n_suppliers] confidence in [0, 1]}`, each produced out-of-fold.

    The fold structure is §10.3's: for a given seed, the head is trained on the other seeds
    and calibrated on a further held-out one, so nothing about the seed being scored was
    seen during fitting.
    """
    data = {s: _build(os.path.join(csv_dir, f"v{variant}_seed{s}"),
                      os.path.join(state_dir, f"{variant}_{s}.json"), top_k) for s in seeds}
    unknown = CLASSES.index("unknown")
    out = {}
    for i, test_seed in enumerate(seeds):
        calib_seed = seeds[(i + 1) % len(seeds)]
        train_seeds = [s for s in seeds if s not in (test_seed, calib_seed)]
        fit = train_head(np.concatenate([data[s]["X"] for s in train_seeds]),
                         np.concatenate([data[s]["y"] for s in train_seeds]),
                         data[calib_seed]["X"], data[calib_seed]["y"],
                         epochs=epochs, seed=model_seed)
        curve = isotonic_fit(predict(fit, data[calib_seed]["X"])[:, unknown],
                             data[calib_seed]["y"][:, unknown])
        p_unknown = isotonic_apply(curve, predict(fit, data[test_seed]["X"])[:, unknown])
        named = 1.0 - p_unknown

        d = data[test_seed]
        conf = np.zeros(d["n"], dtype=np.float32)
        # A supplier's confidence is the best explanation available for ANY of its detected
        # patterns; a supplier with no detected pattern keeps 0. `np.maximum.at` rather than
        # a groupby because both endpoints of every pair must be updated.
        np.maximum.at(conf, d["pairs"][:, 0], named.astype(np.float32))
        np.maximum.at(conf, d["pairs"][:, 1], named.astype(np.float32))
        out[test_seed] = conf
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default=os.path.join(REPO, "db", "csv_mid"))
    ap.add_argument("--state-dir", default=os.path.join(REPO, "out", "hidden_mid_ext"))
    ap.add_argument("--variants", default="B,D")
    ap.add_argument("--seeds", default="42,43,44,45,46",
                    help="dataset seeds to train Transformer 2 on")
    ap.add_argument("--head-seeds", default="42,43,44,45,46",
                    help="seeds forming the hypothesis head's CV folds. Always the full "
                         "set: a seed's confidence must come from a head fitted on the "
                         "OTHER seeds, so this cannot be narrowed alongside --seeds without "
                         "silently making the weight in-fold")
    ap.add_argument("--arms", default="baseline,hypconf")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--head-epochs", type=int, default=400)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    ap.add_argument("--tag", default="", help="replicate tag, for the reproduction floor")
    ap.add_argument("--out", default=os.path.join(REPO, "out", "hypfusion"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    head_seeds = [int(s) for s in args.head_seeds.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    missing = [s for s in seeds if s not in head_seeds]
    if missing:
        sys.exit(f"--seeds {missing} are not in --head-seeds; their confidence would have "
                 f"no out-of-fold head to come from")

    for variant in variants:
        conf_by_seed = hypothesis_confidence(variant, head_seeds, args.csv_dir,
                                             args.state_dir, args.top_k, args.head_epochs,
                                             args.model_seed)
        for seed in seeds:
            d = os.path.join(args.csv_dir, f"v{variant}_seed{seed}")
            bundles, _ = load_bundles(d, cache_dir=CACHE_DIR)
            tr, va, te = split_bundles(bundles)
            ref = bundles[0].data["Supplier"].node_id
            for b in bundles:
                assert b.data["Supplier"].node_id == ref, "Supplier node order varies"
            # The confidence vector is indexed by `suppliers.csv.gz` row order; the model's
            # Supplier rows are the loader's order. Equal in this codebase, asserted here
            # because a silent mismatch would scramble every weight and still train happily
            # -- the same failure mode `ml/hypothesis_labels.py` guards on the label side.
            sup_ids = load_mechanism_state(
                os.path.join(args.state_dir, f"{variant}_{seed}.json"), d)["supplier_ids"]
            assert list(ref) == list(sup_ids), "Supplier order differs from suppliers.csv.gz"

            trd = [b.to(args.device) for b in tr]
            vad = [b.to(args.device) for b in va]
            ted = [b.to(args.device) for b in te]
            for arm in arms:
                name = f"{arm}_v{variant}_seed{seed}" + (f"_{args.tag}" if args.tag else "")
                path = os.path.join(args.out, name + ".json")
                if os.path.exists(path):
                    print(f"  {name}: exists, skipping", flush=True)
                    continue
                conf = (torch.tensor(conf_by_seed[seed]) if arm == "hypconf" else None)
                t0 = time.time()
                r = train_model(ARCH, trd, vad, epochs=args.epochs, seed=args.model_seed,
                                device=args.device, hyp_confidence=conf, **CFG)
                preds = collect_predictions(r["model"], ted)
                from sklearn.metrics import roc_auc_score
                aucs = {}
                for task in TASKS:
                    y, p = preds[task]["y"], preds[task]["p"]
                    aucs[task] = (float(roc_auc_score(y, p))
                                  if len(y) and 0 < y.sum() < len(y) else None)
                blob = {"arch": ARCH, "arm": arm, "variant": variant, "seed": seed,
                        "tag": args.tag, "device": args.device,
                        "params": r["model"].parameter_count(),
                        "seconds": time.time() - t0, "epochs_run": r["epochs_run"],
                        "auc": aucs, "best_val_auc": r["best_val_auc"],
                        "best_epoch": r["best_epoch"],
                        "hyp_confidence": r["hyperparameters"]["hyp_confidence"]}
                with open(path, "w") as fh:
                    json.dump(blob, fh, indent=1)
                print(f"  {name}: {blob['seconds']:.0f}s  " +
                      " ".join(f"{t}={aucs[t]:.4f}" for t in TASKS
                               if aucs[t] is not None), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
