#!/usr/bin/env python3
"""
Shared frozen-backbone harness for the decision-support build (Phases 1-3).

All three phases of `STEP5B_DECISION_SUPPORT_BUILD_PROMPT.md` need the same thing: a
**trained SHARE + Markov Blanket depth readout**, loaded, frozen, and re-runnable on a graph
that may have been edited. No checkpoints existed in this repository when this session
started (`out/` carries per-run JSON results, never weights), so this module both trains and
caches them, and every phase reads from the same cache rather than retraining.

**Nothing here modifies SHARE or the Markov readout.** `train_model` is called with the
existing architecture id `rgcn_attn_markov` and the existing configuration, the objective is
untouched, and `freeze()` asserts afterwards that no backbone parameter carries a gradient.
The Markov readout is what it has always been -- a zero-parameter, per-task index-select,
`delay -> h^1`, `shortage -> h^3`, `impact -> h^4` -- and no phase in this session changes
which depth a task reads.

Scale note: Phases 1 and 3 run at the **v1 preset** (`sup_n=800`, `db/csv_v1scale/`), the
cheap already-validated anchor the build prompt specifies, not at mid or spec scale.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import torch

from ml.data.loader import load_bundles, split_bundles
from ml.models.depth import TASKS
from ml.train import train_model

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO, "ml", ".cache")
CKPT_DIR = os.path.join(REPO, "out", "ds_ckpt")

ARCH = "rgcn_attn_markov"
CFG = dict(hidden=128, num_bases=10)          # SHARE's matched-parameter budget
EPOCHS = 100                                   # V1/V2 standard for this arm


def ckpt_path(csv_dir: str, variant: str, dseed: int, mseed: int, epochs: int) -> str:
    tag = f"{os.path.basename(csv_dir)}_v{variant}_d{dseed}_m{mseed}_e{epochs}"
    return os.path.join(CKPT_DIR, tag + ".pt")


def load_world(csv_dir: str, device: str = "cpu"):
    """Snapshot bundles for one world, temporally split 40/20/40 -- the existing loader,
    the existing split, no changes."""
    bundles, _ = load_bundles(csv_dir, cache_dir=CACHE_DIR)
    tr, va, te = split_bundles(bundles)
    ref = bundles[0].data["Supplier"].node_id
    for b in bundles:
        assert b.data["Supplier"].node_id == ref, "Supplier node order varies by snapshot"
    return ([b.to(device) for b in tr], [b.to(device) for b in va],
            [b.to(device) for b in te], ref)


def get_backbone(csv_dir: str, variant: str, dseed: int, mseed: int = 0,
                 epochs: int = EPOCHS, device: str = "cpu", verbose: bool = True):
    """Train-or-load one frozen backbone. Returns `(model, meta)`.

    Caching is by `(csv_dir, variant, dataset seed, model seed, epochs)`. Training is
    deterministic given those, so a cache hit is exact rather than merely equivalent -- the
    same argument `db/regenerate_seed.py` makes for regenerating worlds instead of storing
    them.
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = ckpt_path(csv_dir, variant, dseed, mseed, epochs)
    tr, va, te, sup_ids = load_world(csv_dir, device)

    if os.path.exists(path):
        blob = torch.load(path, map_location=device, weights_only=False)
        model = _build_shell(tr, device)
        model.load_state_dict(blob["state_dict"])
        meta = blob["meta"]
        if verbose:
            print(f"  [cached] {os.path.basename(path)}  "
                  + " ".join(f"{t}={meta['auc'][t]:.4f}" for t in TASKS
                             if meta["auc"].get(t) is not None), flush=True)
    else:
        t0 = time.time()
        r = train_model(ARCH, tr, va, epochs=epochs, seed=mseed, device=device, **CFG)
        model = r["model"]
        meta = {"arch": ARCH, "variant": variant, "dataset_seed": dseed,
                "model_seed": mseed, "epochs": epochs, "device": device,
                "seconds": time.time() - t0, "params": model.parameter_count(),
                "best_val_auc": r["best_val_auc"], "best_epoch": r["best_epoch"],
                "auc": _test_auc(model, te)}
        torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
        if verbose:
            print(f"  [trained] {os.path.basename(path)}  {meta['seconds']:.0f}s  "
                  + " ".join(f"{t}={meta['auc'][t]:.4f}" for t in TASKS
                             if meta["auc"].get(t) is not None), flush=True)

    freeze(model)
    return model, {**meta, "test_bundles": te, "supplier_ids": sup_ids}


def _build_shell(train_bundles, device: str):
    """An untrained model of the right shape, for loading a state dict into."""
    from ml.models.rgcn_attn_markov_encoder import MarkovHADESModel
    metadata = train_bundles[0].data.metadata()
    in_dims = {nt: train_bundles[0].data[nt].x.size(-1)
               for nt in train_bundles[0].data.node_types}
    return MarkovHADESModel(metadata, in_dims, **CFG).to(device)


def freeze(model) -> None:
    """Freeze every parameter and assert it took.

    The assertion is the point. A silently-unfrozen backbone would let a downstream head's
    gradients reach SHARE, which would invalidate every result this project has recorded for
    it -- and it would not be visible in any metric until far too late.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    live = [n for n, p in model.named_parameters() if p.requires_grad]
    if live:
        raise RuntimeError(f"backbone not frozen: {live[:5]}")


@torch.no_grad()
def predict(model, bundle, device: str = "cpu") -> dict:
    """`{task: [n_entities] probabilities}` for one snapshot, from the frozen forward pass.

    This is the existing forward pass, called exactly as `ml/evaluate.py` calls it. The whole
    of Phase 1's "naive counterfactual" is this function applied to an edited graph.
    """
    model.eval()
    logits, _ = model(bundle.data.x_dict, bundle.data.edge_index_dict)
    return {t: torch.sigmoid(logits[t]).float().cpu().numpy() for t in TASKS}


@torch.no_grad()
def _test_auc(model, test_bundles) -> dict:
    from sklearn.metrics import roc_auc_score
    from ml.evaluate import collect_predictions
    preds = collect_predictions(model, test_bundles)
    out = {}
    for t in TASKS:
        y, p = preds[t]["y"], preds[t]["p"]
        out[t] = (float(roc_auc_score(y, p)) if len(y) and 0 < y.sum() < len(y) else None)
        out[f"{t}_positives"] = int(y.sum()) if len(y) else 0
    return out


def world_hash(csv_dir: str) -> str:
    """Hash of the emitted supplier + component tables, so two worlds can be proven
    different (or identical) without comparing every file."""
    h = hashlib.sha256()
    for name in ("suppliers.csv.gz", "components.csv.gz", "component_suppliers.csv.gz"):
        p = os.path.join(csv_dir, name)
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="train-or-load one backbone; prints timing")
    ap.add_argument("--csv-dir", default=os.path.join(REPO, "db", "csv_v1scale", "v0_seed42"))
    ap.add_argument("--variant", default="0")
    ap.add_argument("--dseed", type=int, default=42)
    ap.add_argument("--mseed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    _, meta = get_backbone(a.csv_dir, a.variant, a.dseed, a.mseed, a.epochs, a.device)
    print(json.dumps({k: v for k, v in meta.items()
                      if k not in ("test_bundles", "supplier_ids")}, indent=1, default=str))
