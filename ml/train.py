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
            probs = torch.sigmoid(logits[task][idx]).numpy()
            aucs.append(roc_auc_score(y.numpy(), probs))
    return sum(aucs) / len(aucs) if aucs else None


def train_model(architecture: str, train_bundles, val_bundles, num_layers: int = 4,
                 shared_depth: int | None = None, depth_prior: dict[str, int] | None = None,
                 hidden: int = 64, epochs: int = 100,
                 lr: float = 1e-3, weight_decay: float = 1e-4, seed: int = 0,
                 num_bases: int = 8, relation_embed_dim: int = 16, num_bases_attn: int = 8) -> dict:
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

    `num_bases`: applies to 'rgcn', 'rgcn_attn', 'rgcn_relemb', and
    'rgcn_battn' (`ml/models/rgcn_encoder.py`'s basis-decomposition relation
    count, used for the message transform in all four); ignored by every
    other architecture. `relation_embed_dim`: 'rgcn_relemb' only
    (`ml/models/rgcn_relemb_encoder.py`'s per-relation embedding size).
    `num_bases_attn`: 'rgcn_battn' only (`ml/models/rgcn_battn_encoder.py`'s
    separate attention-side basis count). Both ignored by every architecture
    they don't apply to.
    """
    torch.manual_seed(seed)
    metadata = train_bundles[0].data.metadata()
    in_dims = {nt: train_bundles[0].data[nt].x.size(-1) for nt in train_bundles[0].data.node_types}

    model = HADESModel(architecture, metadata, in_dims, hidden=hidden, num_layers=num_layers,
                        shared_depth=shared_depth, depth_prior=depth_prior, num_bases=num_bases,
                        relation_embed_dim=relation_embed_dim, num_bases_attn=num_bases_attn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    alphas = compute_task_alphas(train_bundles)
    losses = {task: FocalLoss(gamma=2.0, alpha=alphas[task]) for task in TASKS}

    best_state, best_auc, best_epoch = None, -1.0, -1
    history = []
    for epoch in range(1, epochs + 1):
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
            "num_bases": num_bases if architecture in ("rgcn", "rgcn_attn", "rgcn_relemb", "rgcn_battn") else None,
            "relation_embed_dim": relation_embed_dim if architecture == "rgcn_relemb" else None,
            "num_bases_attn": num_bases_attn if architecture == "rgcn_battn" else None,
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
                      num_bases_attn: int = 8) -> dict:
    """Register -> train -> activate, one call per model_registry row."""
    result = train_model(architecture, train_bundles, val_bundles, num_layers=num_layers,
                          shared_depth=shared_depth, depth_prior=depth_prior, hidden=hidden,
                          epochs=epochs, seed=seed, num_bases=num_bases,
                          relation_embed_dim=relation_embed_dim, num_bases_attn=num_bases_attn)
    register_model(conn, model_version, architecture, result["hyperparameters"],
                   result["model"].parameter_count(), purpose)
    activate_model(conn, model_version, "active")
    return result
