"""
Unified training loop + model_registry logger — Steps 3-5
(`docs/14_Model_Development_Roadmap.md` §6-8, `docs/11_Implementation_Guide.md`
§11).

Split is strictly by `t0` (`project_HADES.md` §8.1: "NEVER randomly").

v3 (800 suppliers, 15 monthly snapshots Jul 2024 - Sep 2025) redefines the
cutoffs used by v1/v2's 6-snapshot schedule -- reusing "last 2 snapshots as
test" would put Aug/Sep 2025 in test, which is fine, but "first 3 as train"
would leave the Oct/Nov 2024 label spike (delay 248/194 vs a 27-74 baseline
every other month, `reports/step5_result_v3.md` Part I) sitting inside
validation, isolated from both train and test. Chosen instead:

    train      t0 <= 2024-12-31   (Jul-Dec 2024 -- 6 snapshots, includes the
                                    Oct/Nov spike so the model trains on the
                                    disrupted regime rather than being
                                    evaluated against a regime it never saw)
    validation 2025-01-01 <= t0 < 2025-04-01   (Jan-Mar 2025 -- 3 snapshots)
    test       t0 >= 2025-04-01   (Apr-Sep 2025 -- 6 snapshots)

One `model_registry` row is written *before* training starts (status
`training`) and flipped to `active` on completion, so `git_commit` and
`hyperparameters` can never drift from what was actually run
(`docs/11_Implementation_Guide.md` §10, `project_HADES.md` §8.4).
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import subprocess
from dataclasses import dataclass, field

import torch
from torch.nn.utils import clip_grad_norm_

from ml.data.snapshots import (
    _assert_label_window,
    _labels_for_snapshot,
    get_snapshot_schedule,
)
from ml.graph.builder import build_snapshot
from ml.models.depth import TASK_ENTITY_TYPE, TASKS
from ml.models.heads import FocalLoss, alpha_from_positive_rate
from ml.models.model import HADESModel

TASK_DB_ENTITY_TYPE = {"delay": "shipment", "shortage": "product", "impact": "supplier"}

TRAIN_CUTOFF = dt.datetime(2024, 12, 31, tzinfo=dt.timezone.utc)
TEST_START = dt.datetime(2025, 4, 1, tzinfo=dt.timezone.utc)


@dataclass
class SnapshotBundle:
    t0: dt.datetime
    snapshot_id: object
    data: object
    id_maps: dict
    labels: dict = field(default_factory=dict)  # task -> (LongTensor idx, FloatTensor y)


def current_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _task_labels_for_bundle(conn, snapshot_id, t0, horizon_days, task: str, id_map: dict):
    """
    (index, label) tensors for one task/snapshot. For `shortage`, multiple
    label rows legitimately share the same product's `entity_id` (one row
    per stocking warehouse, `ml/models/heads.py`'s module docstring) -- the
    resulting `idx` tensor can and does contain repeated indices, each
    paired with its own (possibly different) label. That's intentional:
    every repeated index is a genuine distinct labelled instance, and the
    loss/AUC computation over them is valid even though they all read the
    same Product node embedding.
    """
    entity_type = TASK_DB_ENTITY_TYPE[task]
    labels_df = _labels_for_snapshot(conn, snapshot_id, entity_type, task)
    _assert_label_window(labels_df, t0, horizon_days)
    if labels_df.empty:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.float)
    idx = labels_df["entity_id"].map(id_map)
    valid = idx.notna()
    idx_t = torch.tensor(idx[valid].astype(int).to_numpy(), dtype=torch.long)
    y_t = torch.tensor(labels_df.loc[valid, "label"].astype(float).to_numpy(), dtype=torch.float)
    return idx_t, y_t


def load_all_snapshot_bundles(conn) -> list[SnapshotBundle]:
    """Build every t0's HeteroData + per-task label tensors, once."""
    schedule = get_snapshot_schedule(conn)
    bundles = []
    for row in schedule.itertuples():
        data, id_maps = build_snapshot(conn, row.t0)
        labels = {
            task: _task_labels_for_bundle(conn, row.snapshot_id, row.t0, row.horizon_days,
                                           task, id_maps[TASK_ENTITY_TYPE[task]])
            for task in TASKS
        }
        bundles.append(SnapshotBundle(t0=row.t0, snapshot_id=row.snapshot_id, data=data,
                                       id_maps=id_maps, labels=labels))
    return bundles


def split_bundles(bundles: list[SnapshotBundle]) -> tuple[list, list, list]:
    train = [b for b in bundles if b.t0 <= TRAIN_CUTOFF]
    val = [b for b in bundles if TRAIN_CUTOFF < b.t0 < TEST_START]
    test = [b for b in bundles if b.t0 >= TEST_START]
    return train, val, test


def move_bundles_to_device(bundles: list[SnapshotBundle], device: str) -> None:
    """In-place: moves each bundle's `.data` (HeteroData, native `.to()` support) and
    `.labels` (task -> (idx, y) tensor pairs) to `device`. Call ONCE per device, before any
    `train_model(..., device=...)` calls that use it -- `train_model` never moves bundle
    tensors itself (see its docstring), so every run reusing these bundles gets the
    transfer for free instead of paying it again per run."""
    for bundle in bundles:
        bundle.data = bundle.data.to(device)
        bundle.labels = {task: (idx.to(device), y.to(device)) for task, (idx, y) in bundle.labels.items()}


def compute_task_alphas(train_bundles: list[SnapshotBundle]) -> dict[str, float]:
    """Focal-loss alpha per task, from the TRAIN split's own positive rate
    (never from val/test -- alpha is a training hyperparameter, not something
    tuned against held-out data)."""
    alphas = {}
    for task in TASKS:
        ys = torch.cat([b.labels[task][1] for b in train_bundles if b.labels[task][1].numel() > 0])
        alphas[task] = alpha_from_positive_rate(ys.mean().item()) if ys.numel() > 0 else 0.5
    return alphas


def _epoch_forward(model, bundle: SnapshotBundle, losses: dict[str, FocalLoss]):
    logits, _layers = model(bundle.data.x_dict, bundle.data.edge_index_dict)
    total = None
    for task in TASKS:
        idx, y = bundle.labels[task]
        if idx.numel() == 0:
            continue
        term = losses[task](logits[task][idx], y)
        total = term if total is None else total + term

    # Step 6 depth-gate side-channel: models that set `_last_kl_total` (only
    # `DepthGateHADESModel`, `ml/models/rgcn_attn_depthgate_encoder.py`) get
    # `lambda_kl * KL` added to the loss here -- a no-op for every other
    # architecture, which never sets this attribute.
    kl = getattr(model, "_last_kl_total", None)
    lambda_kl = getattr(model, "lambda_kl", None)
    if kl is not None and lambda_kl:
        total = (lambda_kl * kl) if total is None else total + lambda_kl * kl
    return total


@torch.no_grad()
def _mean_val_auc(model, val_bundles: list[SnapshotBundle]) -> float | None:
    from sklearn.metrics import roc_auc_score

    model.eval()
    aucs = []
    for bundle in val_bundles:
        logits, _ = model(bundle.data.x_dict, bundle.data.edge_index_dict)
        for task in TASKS:
            idx, y = bundle.labels[task]
            if idx.numel() == 0 or y.unique().numel() < 2:
                continue
            # `.cpu()` first: no-op on CPU tensors, required on MPS/other
            # non-CPU devices -- see `ml/evaluate.py::collect_predictions`'s
            # identical fix for the same reason.
            probs = torch.sigmoid(logits[task][idx]).cpu().numpy()
            aucs.append(roc_auc_score(y.cpu().numpy(), probs))
    return sum(aucs) / len(aucs) if aucs else None


def train_model(architecture: str, train_bundles, val_bundles, num_layers: int = 4,
                 shared_depth: int | None = None, depth_prior: dict[str, int] | None = None,
                 hidden: int = 64, epochs: int = 100,
                 lr: float = 1e-3, weight_decay: float = 1e-4, seed: int = 0,
                 num_bases: int = 8, relation_embed_dim: int = 16, num_bases_attn: int = 8,
                 lambda_kl: float = 0.1, device: str = "cpu",
                 rung5a_lambda_bound: float = 0.3,
                 rung5c_freeze_epochs: int = 15, rung5c_lr_mult: float = 0.1,
                 t2_top_k: int = 64) -> dict:
    """
    Train one model for `epochs` epochs (matching
    `docs/14_Model_Development_Roadmap.md` §6's "train for 100 epochs" --
    every epoch actually runs; the *best-val-AUC* checkpoint is kept for
    evaluation rather than only the final epoch's weights, standard
    practice that doesn't shorten training itself). AdamW is used rather
    than plain Adam so `weight_decay=1e-4` is decoupled, per
    `project_HADES.md` §8.2's fuller optimizer spec, while keeping the
    lr/weight-decay values as specified.

    `depth_prior`: which fixed structural-prior dict to read from (None ->
    `ml.models.depth.STRUCTURAL_DEPTH_PRIOR`, the as-documented prior).
    Ignored when `shared_depth` is set (the L-sweep ablation). Step A's
    corrected-impact-prior comparison passes `STRUCTURAL_DEPTH_PRIOR_V2`.

    `seed`: controls model init (and dropout masks) only -- the dataset,
    split, and feature assembly are already fixed/deterministic upstream of
    this call, so varying `seed` alone is a clean model-fit-variance probe
    (Step B).

    `num_bases`: applies to 'rgcn', 'rgcn_attn', 'rgcn_relemb', 'rgcn_battn',
    'rgcn_attn_depthgate', 'rgcn_attn_markov', 'rgcn_attn_rung4', and
    'rgcn_attn_rung5' (`ml/models/rgcn_encoder.py`'s basis-decomposition
    relation count, used for the message transform in all eight); ignored
    by every other architecture. `relation_embed_dim`: 'rgcn_relemb' only
    (`ml/models/rgcn_relemb_encoder.py`'s per-relation embedding size).
    `num_bases_attn`: 'rgcn_battn' only (`ml/models/rgcn_battn_encoder.py`'s
    separate attention-side basis count). `lambda_kl`: 'rgcn_attn_depthgate'
    only (Step 6 pilot, `ml/models/rgcn_attn_depthgate_encoder.py`'s
    KL-anchor weight on the learned per-task depth gate) -- 'rgcn_attn_markov'
    (Step 6 Phase 1) uses a FIXED, non-learned per-task depth instead, and
    'rgcn_attn_rung4'/'rgcn_attn_rung5' (Step 6 Phase 2, per-node depth-gate
    hybrids) use init-time prior-biasing instead of a training-time KL
    penalty -- all three ignore `lambda_kl` entirely. `rgcn_attn_rung5_a/b/c/d`
    (Step 6 Phase 2, `reports/step6_rung_pilot.md`'s four isolated-variant
    follow-ups to Rung 5) also ignore `lambda_kl`; `rung5a_lambda_bound`
    applies only to 'rgcn_attn_rung5_a' (the tanh-bounded residual's scale,
    `ml/models/rgcn_attn_rung5_variant_a.py`); `rung5c_freeze_epochs` /
    `rung5c_lr_mult` apply only to 'rgcn_attn_rung5_c' (gate parameters are
    frozen for the first `rung5c_freeze_epochs` epochs, then unfrozen at
    `lr * rung5c_lr_mult` for the remainder -- a training-loop-only change,
    same architecture class as 'rgcn_attn_rung5'). All ignored by every
    architecture they don't apply to.

    `device`: every architecture supports it (`model.to(device)` below), but
    the CALLER is responsible for having already moved `train_bundles` /
    `val_bundles` (both `.data` and `.labels`) to the same device before
    calling this function -- `train_model` itself never touches bundle
    tensors' device placement, only the freshly-constructed model's.
    """
    torch.manual_seed(seed)
    metadata = train_bundles[0].data.metadata()
    in_dims = {nt: train_bundles[0].data[nt].x.size(-1) for nt in train_bundles[0].data.node_types}

    if architecture == "rgcn_attn_depthgate":
        from ml.models.rgcn_attn_depthgate_encoder import DepthGateHADESModel
        model = DepthGateHADESModel(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                     num_bases=num_bases, lambda_kl=lambda_kl)
    elif architecture == "rgcn_attn_markov":
        from ml.models.rgcn_attn_markov_encoder import MarkovHADESModel
        model = MarkovHADESModel(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                  num_bases=num_bases)
    elif architecture == "rgcn_attn_rung4":
        from ml.models.rgcn_attn_rung4_encoder import Rung4HADESModel
        model = Rung4HADESModel(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                 num_bases=num_bases)
    elif architecture in ("rgcn_attn_rung5", "rgcn_attn_rung5_c"):
        # Variant C reuses Rung5HADESModel UNCHANGED (architecturally identical
        # to the original) -- its only difference is the optimizer/freeze
        # schedule set up below, not the model class. `model.architecture` is
        # overridden so registry/eval logging reflects which id was actually
        # requested.
        from ml.models.rgcn_attn_rung5_encoder import Rung5HADESModel
        model = Rung5HADESModel(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                 num_bases=num_bases)
        model.architecture = architecture
    elif architecture == "rgcn_attn_rung5_a":
        from ml.models.rgcn_attn_rung5_variant_a import Rung5VariantAHADESModel
        model = Rung5VariantAHADESModel(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                         num_bases=num_bases, lambda_bound=rung5a_lambda_bound)
    elif architecture == "rgcn_attn_rung5_b":
        from ml.models.rgcn_attn_rung5_variant_b import Rung5VariantBHADESModel
        model = Rung5VariantBHADESModel(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                         num_bases=num_bases)
    elif architecture == "rgcn_attn_rung5_d":
        from ml.models.rgcn_attn_rung5_variant_d import Rung5VariantDHADESModel
        model = Rung5VariantDHADESModel(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                         num_bases=num_bases)
    elif architecture == "rgcn_attn_variant_a_transformer2":
        from ml.models.rgcn_attn_variant_a_transformer2 import Transformer2HADESModel
        model = Transformer2HADESModel(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                        num_bases=num_bases, lambda_bound=rung5a_lambda_bound,
                                        t2_top_k=t2_top_k)
    elif architecture == "rgcn_attn_t2_confidence":
        from ml.models.transformer2_confidence import Transformer2ConfidenceHADESModel
        model = Transformer2ConfidenceHADESModel(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                                  num_bases=num_bases, lambda_bound=rung5a_lambda_bound,
                                                  t2_top_k=t2_top_k)
    elif architecture == "rgcn_attn_t2_trustgate":
        from ml.models.transformer2_trustgate import Transformer2TrustGateHADESModel
        model = Transformer2TrustGateHADESModel(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                                 num_bases=num_bases, lambda_bound=rung5a_lambda_bound,
                                                 t2_top_k=t2_top_k)
    elif architecture == "rgcn_attn_t2_crossattn":
        from ml.models.transformer2_crossattn import Transformer2CrossAttnHADESModel
        model = Transformer2CrossAttnHADESModel(metadata, in_dims, hidden=hidden, num_layers=num_layers,
                                                 num_bases=num_bases, lambda_bound=rung5a_lambda_bound,
                                                 t2_top_k=t2_top_k)
    else:
        model = HADESModel(architecture, metadata, in_dims, hidden=hidden, num_layers=num_layers,
                            shared_depth=shared_depth, depth_prior=depth_prior, num_bases=num_bases,
                            relation_embed_dim=relation_embed_dim, num_bases_attn=num_bases_attn)
    model = model.to(device)

    if architecture == "rgcn_attn_rung5_c":
        # Freeze the gate for the first `rung5c_freeze_epochs` epochs (encoder
        # + heads train normally on the Markov-prior-initialized readout,
        # exactly like the `rgcn_attn_markov` arm would), then unfreeze the
        # gate at a reduced LR for the rest of training -- testing whether
        # letting the ENCODER settle first, before the gate is allowed to
        # move the readout away from the validated prior, produces more
        # stable per-node behaviour than training everything jointly from
        # epoch 1 (the original Rung 5's approach,
        # `reports/step6_rung_pilot.md`). A single optimizer with two param
        # groups is used throughout (not rebuilt at the unfreeze point):
        # `requires_grad=False` params produce no gradient at all, so AdamW's
        # `step()` skips them for free -- no separate freeze-phase optimizer
        # needed.
        gate_params = list(model.gates.parameters())
        gate_param_ids = {id(p) for p in gate_params}
        other_params = [p for p in model.parameters() if id(p) not in gate_param_ids]
        for p in gate_params:
            p.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            [{"params": other_params, "lr": lr}, {"params": gate_params, "lr": lr * rung5c_lr_mult}],
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    alphas = compute_task_alphas(train_bundles)
    losses = {task: FocalLoss(gamma=2.0, alpha=alphas[task]) for task in TASKS}

    best_state, best_auc, best_epoch = None, -1.0, -1
    history = []
    for epoch in range(1, epochs + 1):
        if architecture == "rgcn_attn_rung5_c" and epoch == rung5c_freeze_epochs + 1:
            for p in gate_params:
                p.requires_grad_(True)
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

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs  # no val AUC was ever computable -- fall back to final epoch

    # Gate weights are read fresh from the (now best-checkpoint-restored)
    # model's own parameters, not from any stale forward-pass side-channel
    # -- correct regardless of when load_state_dict happened above.
    gate_weights = model.gate_weight_summary() if hasattr(model, "gate_weight_summary") else None

    return {
        "model": model,
        "alphas": alphas,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_auc": best_auc if best_state is not None else None,
        "hyperparameters": {
            "architecture": architecture, "hidden": hidden, "num_layers": num_layers,
            "shared_depth": shared_depth,
            "depth_prior": depth_prior if depth_prior is not None else "as_documented",
            "epochs": epochs, "lr": lr,
            "weight_decay": weight_decay, "focal_gamma": 2.0, "focal_alpha": alphas,
            "optimizer": "AdamW", "seed": seed,
            "lambda_kl": lambda_kl if architecture == "rgcn_attn_depthgate" else None,
            "gate_weights": gate_weights,
            "num_bases": num_bases if architecture in (
                "rgcn", "rgcn_attn", "rgcn_relemb", "rgcn_battn", "rgcn_attn_depthgate",
                "rgcn_attn_markov", "rgcn_attn_rung4", "rgcn_attn_rung5", "rgcn_attn_rung5_a",
                "rgcn_attn_rung5_b", "rgcn_attn_rung5_c", "rgcn_attn_rung5_d",
                "rgcn_attn_variant_a_transformer2", "rgcn_attn_t2_confidence",
                "rgcn_attn_t2_trustgate", "rgcn_attn_t2_crossattn") else None,
            "relation_embed_dim": relation_embed_dim if architecture == "rgcn_relemb" else None,
            "num_bases_attn": num_bases_attn if architecture == "rgcn_battn" else None,
            "device": device,
            "rung5a_lambda_bound": rung5a_lambda_bound if architecture in (
                "rgcn_attn_rung5_a", "rgcn_attn_variant_a_transformer2", "rgcn_attn_t2_confidence",
                "rgcn_attn_t2_trustgate", "rgcn_attn_t2_crossattn") else None,
            "rung5c_freeze_epochs": rung5c_freeze_epochs if architecture == "rgcn_attn_rung5_c" else None,
            "rung5c_lr_mult": rung5c_lr_mult if architecture == "rgcn_attn_rung5_c" else None,
            "t2_top_k": t2_top_k if architecture in (
                "rgcn_attn_variant_a_transformer2", "rgcn_attn_t2_confidence",
                "rgcn_attn_t2_trustgate", "rgcn_attn_t2_crossattn") else None,
        },
    }


def register_model(conn, model_version: str, architecture: str, hyperparameters: dict,
                    parameter_count: int, purpose: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO model_registry
                (model_version, architecture, training_dataset, training_timestamp,
                 git_commit, hyperparameters, parameter_count, purpose, status)
            VALUES (%s, %s, %s, now(), %s, %s, %s, %s, 'training')
            ON CONFLICT (model_version) DO UPDATE SET
                architecture = EXCLUDED.architecture,
                training_timestamp = now(),
                git_commit = EXCLUDED.git_commit,
                hyperparameters = EXCLUDED.hyperparameters,
                parameter_count = EXCLUDED.parameter_count,
                purpose = EXCLUDED.purpose,
                status = 'training'
            """,
            (model_version, architecture, "chainpilot", current_git_commit(),
             json.dumps(hyperparameters), parameter_count, purpose),
        )
    conn.commit()


def activate_model(conn, model_version: str, status: str = "active") -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE model_registry SET status = %s WHERE model_version = %s",
                    (status, model_version))
    conn.commit()


def run_training_job(conn, model_version: str, architecture: str, train_bundles, val_bundles,
                      num_layers: int = 4, shared_depth: int | None = None,
                      depth_prior: dict[str, int] | None = None, hidden: int = 64,
                      epochs: int = 100, seed: int = 0, purpose: str = "",
                      num_bases: int = 8, relation_embed_dim: int = 16,
                      num_bases_attn: int = 8, lambda_kl: float = 0.1, device: str = "cpu",
                      rung5a_lambda_bound: float = 0.3, rung5c_freeze_epochs: int = 15,
                      rung5c_lr_mult: float = 0.1, t2_top_k: int = 64) -> dict:
    """Register -> train -> activate, one call per model_registry row. `device`: see
    `train_model`'s docstring -- `train_bundles`/`val_bundles` must already be on `device`
    if it isn't `"cpu"`."""
    result = train_model(architecture, train_bundles, val_bundles, num_layers=num_layers,
                          shared_depth=shared_depth, depth_prior=depth_prior, hidden=hidden,
                          epochs=epochs, seed=seed, num_bases=num_bases,
                          relation_embed_dim=relation_embed_dim, num_bases_attn=num_bases_attn,
                          lambda_kl=lambda_kl, device=device,
                          rung5a_lambda_bound=rung5a_lambda_bound,
                          rung5c_freeze_epochs=rung5c_freeze_epochs, rung5c_lr_mult=rung5c_lr_mult,
                          t2_top_k=t2_top_k)
    register_model(conn, model_version, architecture, result["hyperparameters"],
                   result["model"].parameter_count(), purpose)
    activate_model(conn, model_version, "active")
    return result
