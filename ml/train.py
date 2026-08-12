"""
Training loop, ported from `HADES_v1/ml/train.py` with the PostgreSQL
`model_registry` write path removed (V2 has no live database in this harness --
run provenance goes into the run manifest instead, see `ml/run_benchmark_eval.py`).

Everything that affects the fitted model is carried over unchanged: AdamW at
lr=1e-3 / weight_decay=1e-4, focal loss with gamma=2 and alpha from the TRAIN
split's own positive rate, gradient clipping at 1.0, full-batch forward per
snapshot, and best-validation-AUC checkpoint selection rather than last-epoch
weights.

The split is strictly temporal and comes from `ml/data/loader.py::split_bundles`.
"""

from __future__ import annotations

import copy
import time

import torch
from torch.nn.utils import clip_grad_norm_

from ml.models.depth import TASKS
from ml.models.heads import FocalLoss, alpha_from_positive_rate
from ml.models.model import HADESModel


def compute_task_alphas(train_bundles) -> dict[str, float]:
    """Focal-loss alpha per task from the TRAIN split's positive rate only --
    alpha is a training hyperparameter, never tuned against held-out data."""
    alphas = {}
    for task in TASKS:
        ys = [b.labels[task][1] for b in train_bundles if b.labels[task][1].numel() > 0]
        alphas[task] = alpha_from_positive_rate(torch.cat(ys).mean().item()) if ys else 0.5
    return alphas


def _epoch_forward(model, bundle, losses):
    logits, _ = model(bundle.data.x_dict, bundle.data.edge_index_dict)
    total = None
    for task in TASKS:
        idx, y = bundle.labels[task]
        if idx.numel() == 0:
            continue
        term = losses[task](logits[task][idx], y)
        total = term if total is None else total + term
    return total


@torch.no_grad()
def _mean_val_auc(model, val_bundles) -> float | None:
    from sklearn.metrics import roc_auc_score
    model.eval()
    aucs = []
    for bundle in val_bundles:
        logits, _ = model(bundle.data.x_dict, bundle.data.edge_index_dict)
        for task in TASKS:
            idx, y = bundle.labels[task]
            if idx.numel() == 0 or y.unique().numel() < 2:
                continue
            probs = torch.sigmoid(logits[task][idx]).float().cpu().numpy()
            aucs.append(roc_auc_score(y.cpu().numpy(), probs))
    return sum(aucs) / len(aucs) if aucs else None


def train_model(architecture: str, train_bundles, val_bundles, num_layers: int = 4,
                shared_depth: int | None = None, depth_prior: dict[str, int] | None = None,
                hidden: int = 64, epochs: int = 100, lr: float = 1e-3,
                weight_decay: float = 1e-4, seed: int = 0, num_bases: int = 8,
                relation_embed_dim: int = 16, num_bases_attn: int = 8,
                device: str = "cpu", patience: int | None = None,
                progress: bool = False, rung5a_lambda_bound: float = 0.3,
                rung5c_freeze_epochs: int = 15, rung5c_lr_mult: float = 0.1,
                anneal_lambda: bool = False, t2_top_k: int = 64) -> dict:
    """Train one model. `seed` controls model init and dropout masks only -- the
    dataset, split and features are deterministic upstream, so varying it alone
    is a clean model-fit-variance probe.

    `patience`: stop after this many epochs without a new best validation AUC.
    V1 always ran the full 100 epochs; at V2's spec scale an epoch costs orders
    of magnitude more, so early stopping is available and any run that uses it
    records `epochs_run` in its result. `patience=None` reproduces V1 exactly.

    The CALLER must have moved the bundles to `device` already."""
    torch.manual_seed(seed)
    metadata = train_bundles[0].data.metadata()
    in_dims = {nt: train_bundles[0].data[nt].x.size(-1) for nt in train_bundles[0].data.node_types}

    # The depth-selection family (Phase 7c) builds its own model class over the shared
    # h^0..h^L encoder; everything else is plain HADESModel. Ported from V1's dispatch in
    # `HADES_v1/ml/train.py`, trimmed to the arms this project carries.
    if architecture == "rgcn_attn_markov":
        from ml.models.rgcn_attn_markov_encoder import MarkovHADESModel
        model = MarkovHADESModel(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                 num_bases=num_bases)
    elif architecture in ("rgcn_attn_rung5", "rgcn_attn_rung5_c"):
        # Variant C is architecturally IDENTICAL to the base Rung 5 -- it is the freeze
        # schedule below and nothing else. `model.architecture` is overridden so the logged
        # rows say which arm was actually requested.
        from ml.models.rgcn_attn_rung5_encoder import Rung5HADESModel
        model = Rung5HADESModel(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                num_bases=num_bases)
        model.architecture = architecture
    elif architecture in ("rgcn_attn_rung5_a", "rgcn_attn_rung5_ac"):
        # `_ac` is Variant A's architecture with Variant C's freeze schedule applied on top --
        # a composition of one model change and one training-loop change, so it needs no new
        # model file, exactly as Variant C itself needs none.
        from ml.models.rgcn_attn_rung5_variant_a import Rung5VariantAHADESModel
        model = Rung5VariantAHADESModel(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                        num_bases=num_bases, lambda_bound=rung5a_lambda_bound)
        model.architecture = architecture
    elif architecture == "rgcn_attn_rung5_b":
        from ml.models.rgcn_attn_rung5_variant_b import Rung5VariantBHADESModel
        model = Rung5VariantBHADESModel(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                        num_bases=num_bases)
    elif architecture == "rgcn_attn_variant_a_transformer2":
        from ml.models.rgcn_attn_variant_a_transformer2 import Transformer2HADESModel
        model = Transformer2HADESModel(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                       num_bases=num_bases, lambda_bound=rung5a_lambda_bound,
                                       t2_top_k=t2_top_k)
    elif architecture == "rgcn_attn_t2_confidence":
        from ml.models.transformer2_confidence import Transformer2ConfidenceHADESModel
        model = Transformer2ConfidenceHADESModel(metadata, in_dims, hidden=hidden,
                                                 num_layers=num_layers, num_bases=num_bases,
                                                 lambda_bound=rung5a_lambda_bound, t2_top_k=t2_top_k)
    elif architecture == "rgcn_attn_t2_trustgate":
        from ml.models.transformer2_trustgate import Transformer2TrustGateHADESModel
        model = Transformer2TrustGateHADESModel(metadata, in_dims, hidden=hidden,
                                                num_layers=num_layers, num_bases=num_bases,
                                                lambda_bound=rung5a_lambda_bound, t2_top_k=t2_top_k)
    elif architecture == "rgcn_attn_t2_crossattn":
        from ml.models.transformer2_crossattn import Transformer2CrossAttnHADESModel
        model = Transformer2CrossAttnHADESModel(metadata, in_dims, hidden=hidden,
                                                num_layers=num_layers, num_bases=num_bases,
                                                lambda_bound=rung5a_lambda_bound, t2_top_k=t2_top_k)
    else:
        model = HADESModel(architecture, metadata, in_dims, hidden=hidden, num_layers=num_layers,
                           shared_depth=shared_depth, depth_prior=depth_prior, num_bases=num_bases,
                           relation_embed_dim=relation_embed_dim, num_bases_attn=num_bases_attn)
    model = model.to(device)

    if architecture in ("rgcn_attn_rung5_c", "rgcn_attn_rung5_ac"):
        # Variant C, verbatim in effect from V1: freeze the gate for the first
        # `rung5c_freeze_epochs` epochs so the encoder and heads settle on the
        # Markov-prior-initialised readout, then unfreeze it at a reduced LR for the rest.
        # One optimizer with two param groups throughout -- a frozen parameter produces no
        # gradient, so AdamW skips it for free and no second optimizer is needed.
        gate_params = list(model.gates.parameters())
        gate_ids = {id(q) for q in gate_params}
        other_params = [q for q in model.parameters() if id(q) not in gate_ids]
        for q in gate_params:
            q.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            [{"params": other_params, "lr": lr},
             {"params": gate_params, "lr": lr * rung5c_lr_mult}], weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    alphas = compute_task_alphas(train_bundles)
    losses = {task: FocalLoss(gamma=2.0, alpha=alphas[task]) for task in TASKS}

    best_state, best_auc, best_epoch = None, -1.0, -1
    history, since_best = [], 0
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        if architecture in ("rgcn_attn_rung5_c", "rgcn_attn_rung5_ac") \
                and epoch == rung5c_freeze_epochs + 1:
            for q in gate_params:
                q.requires_grad_(True)
        if anneal_lambda and hasattr(model, "gates"):
            # Annealed cap: start tight so the gate inherits the prior's behaviour, widen
            # linearly to the target so it can explore later, once the encoder has settled.
            # Tests whether the gate sits at the prior because the cap is tight or because it
            # has found nothing worth moving for. Set by mutating the gate heads' attribute
            # rather than by editing `rgcn_attn_rung5_variant_a.py`, which is a byte-identical
            # port of V1's file and must stay that way -- the head reads `self.lambda_bound`
            # on every forward, so this is exactly equivalent to a scheduled constructor arg.
            frac = (epoch - 1) / max(1, epochs - 1)
            lam = 0.1 + frac * (rung5a_lambda_bound - 0.1)
            for _g in model.gates.values():
                _g.lambda_bound = lam
        model.train()
        epoch_loss = 0.0
        for bundle in train_bundles:
            optimizer.zero_grad()
            loss = _epoch_forward(model, bundle, losses)
            if loss is None:
                continue
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        val_auc = _mean_val_auc(model, val_bundles)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_auc": val_auc})
        if val_auc is not None and val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = copy.deepcopy(model.state_dict())
            since_best = 0
        else:
            since_best += 1
        if progress:
            print(f"    epoch {epoch:>3}  loss {epoch_loss:10.4f}  val_auc "
                  f"{val_auc if val_auc is not None else float('nan'):.4f}  "
                  f"({time.time() - t_start:.0f}s)", flush=True)
        if patience is not None and since_best >= patience:
            break

    epochs_run = len(history)
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs_run

    return {
        "model": model,
        "alphas": alphas,
        "history": history,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "seconds": time.time() - t_start,
        "best_val_auc": best_auc if best_state is not None else None,
        "hyperparameters": {
            "architecture": architecture, "hidden": hidden, "num_layers": num_layers,
            "shared_depth": shared_depth,
            "depth_prior": depth_prior if depth_prior is not None else "as_documented",
            "epochs": epochs, "epochs_run": epochs_run, "patience": patience, "lr": lr,
            "weight_decay": weight_decay, "focal_gamma": 2.0, "focal_alpha": alphas,
            "optimizer": "AdamW", "seed": seed,
            "num_bases": num_bases if architecture in (
                "rgcn", "rgcn_attn", "rgcn_relemb", "rgcn_battn", "rgcn_attn_markov",
                "rgcn_attn_rung5", "rgcn_attn_rung5_a", "rgcn_attn_rung5_b",
                "rgcn_attn_rung5_c") else None,
            "rung5a_lambda_bound": rung5a_lambda_bound if architecture in (
                "rgcn_attn_rung5_a", "rgcn_attn_rung5_ac") else None,
            "anneal_lambda": anneal_lambda,
            "t2_top_k": t2_top_k if architecture in ('rgcn_attn_variant_a_transformer2', 'rgcn_attn_t2_confidence', 'rgcn_attn_t2_trustgate', 'rgcn_attn_t2_crossattn') else None,
            "rung5c_freeze_epochs": rung5c_freeze_epochs if architecture in (
                "rgcn_attn_rung5_c", "rgcn_attn_rung5_ac") else None,
            "rung5c_lr_mult": rung5c_lr_mult if architecture in (
                "rgcn_attn_rung5_c", "rgcn_attn_rung5_ac") else None,
            "relation_embed_dim": relation_embed_dim if architecture == "rgcn_relemb" else None,
            "num_bases_attn": num_bases_attn if architecture == "rgcn_battn" else None,
            "device": device,
        },
    }
