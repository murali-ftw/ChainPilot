#!/usr/bin/env python3
"""
Layer 3 v2, PHASE 1 -- task-specific latent encoder, fixed depth by default.

    Z_t = f_t(X, H^{k_t})

`X` = the task entity's observed feature vector; `H^{k_t}` = the frozen SHARE representation at
the depth this project's existing empirical evidence says carries the most task-relevant signal;
`Z_t` = a small bottleneck activation. `Z_t` is **not** required to correspond to a single
human-interpretable cause -- it may be a distributed combination of operational conditions.

**Fixed depth, and it is not re-derived from nothing.** `k_t` starts from the Markov readout's
existing per-task choices (`ml/models/rgcn_attn_markov_encoder.py::MARKOV_READOUT_DEPTH` --
delay h^1, shortage h^3, impact h^4), which are the depths this benchmark already established
for these targets.

**No multi-depth concatenation, no learned depth weighting, no temporal modelling, no
reconstruction decoder, no causal-loss coupling.** None of these are in scope here. The project's
record against adaptive depth is 8/8, so adaptive depth must re-earn its complexity every time
rather than inherit trust from being more expressive; and `reports/phase_modelB_concat_baseline.md`
already ran the concatenation diagnostic (Model B) for the state-family framing and it did not
clear its floor on any arm. If Phase 2 leaves a task with a marginal-but-inconsistent signal that
suggests other depths might help, that is a separate, explicitly-labelled follow-up run under the
Model A -> B -> C progression, not something folded in here. `assert_no_forbidden_mechanism()`
below makes that a runtime check rather than a promise in a docstring.

---

**One definitional point that governs how Phase 2's number must be read, stated here because it
is a property of this design and not of its implementation.**

`Z_t = f_t(X, H^{k_t})` is a deterministic function of `(X, H)`. Conditional mutual information
of a deterministic function of the conditioning set is exactly zero:

    Z = f(X,H)  =>  I(Y; Z | X, H) = 0

So the quantity Phase 2's brief names as the underlying question -- `I(Y_t; Z_t | X, H) > 0` --
is **identically zero in the population limit for any encoder of this form**, and no amount of
capacity changes that. This is not a reason to stop: the operational test Phase 2 actually runs
is a *downstream performance* comparison at finite sample and finite capacity, and there a
compressed, explicitly-supplied non-linear feature can genuinely help a small head that would
otherwise have to learn the same non-linearity itself from limited data. That is a
**representation/optimisation** benefit, not an information one.

The consequence for the report is specific: a positive `delta_t` here may NOT be described as
"`Z_t` contains information the Layer-2 pathway does not provide", because it provably cannot.
It may only be described as "an explicit `Z_t` feature is easier for the downstream head to use
than the same information implicit in `(X, H)` at this sample size". Phase 2 therefore carries a
width-matched random-feature control, without which any `delta_t` is confounded with input width
-- the same confound `reports/phase_modelB_concat_baseline.md` §"Honest qualifications" 3 flagged
for Model B.

    python3 ml/task_latent_encoder.py --variant A --task delay --seeds 42 --selftest
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch import nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ml.latent_state_head import assert_backbone_frozen              # noqa: E402
from ml.models.rgcn_attn_markov_encoder import MARKOV_READOUT_DEPTH  # noqa: E402

# Fixed per-task depth. Started from the Markov readout's existing choices, not re-derived.
TASK_DEPTH = dict(MARKOV_READOUT_DEPTH)          # delay h^1, shortage h^3, impact h^4
Z_DIM = 8
FORBIDDEN = ("multi_depth_concat", "learned_depth_gate", "temporal", "decoder", "causal_loss")


class TaskLatentEncoder(nn.Module):
    """`[X ; H^k] -> bottleneck(Z_DIM) -> logit`.

    Deliberately small, and deliberately a bottleneck: the question is whether a *compressed*
    task-specific summary helps a downstream head, not how much capacity can be stacked on the
    frozen representation. The prediction head exists only to supply a training signal for the
    bottleneck; Phase 2 consumes `Z`, never this head's logit.

    Trained on `tr`. Phase 2 reads `Z` on `va`/`te`, where it is out of sample.
    """

    def __init__(self, in_dim: int, z_dim: int = Z_DIM, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, z_dim))
        self.head = nn.Linear(z_dim, 1)
        self.config = {"in_dim": in_dim, "z_dim": z_dim, "hidden": hidden, "dropout": dropout,
                       "depth_mode": "fixed", "mechanisms": []}

    def encode(self, x):
        return self.trunk(x)

    def forward(self, x):
        return self.head(self.encode(x)).squeeze(-1)


def assert_no_forbidden_mechanism(enc: TaskLatentEncoder) -> None:
    """Runtime check that Phase 1's constraint held, rather than a promise in a comment.

    "Constrain before you learn" is only a ground rule if something enforces it. This fires if
    an encoder ever declares one of the mechanisms Phase 1 forbids without a preceding gate
    having earned it.
    """
    declared = set(enc.config.get("mechanisms", []))
    bad = declared & set(FORBIDDEN)
    if bad:
        raise RuntimeError(
            f"Phase 1 forbids {sorted(bad)} unless a preceding gate earned it; "
            f"run it as a separately-labelled Model A->B->C follow-up instead")
    if enc.config.get("depth_mode") != "fixed":
        raise RuntimeError(f"Phase 1 is fixed-depth only; got {enc.config.get('depth_mode')!r}")


def train_encoder(Xtr: np.ndarray, ytr: np.ndarray, evals: list, init_seed: int,
                  epochs: int = 120, lr: float = 1e-3, z_dim: int = Z_DIM,
                  model=None) -> list | None:
    """Fit `f_t` on `tr`; return `Z` for each evaluation matrix.

    Every training decision matches `ml/latent_state_head.py::train_head` -- same
    standardisation from train statistics, same Adam(lr=1e-3, weight_decay=1e-4), same 120
    full-batch epochs, same `pos_weight`, same seeding of torch and numpy. The only differences
    are the bottleneck and that the return value is `Z` rather than a probability, so the
    encoder sits on the same axis as every other head in this project.
    """
    if len(np.unique(ytr)) < 2:
        return None
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)

    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    xtr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    ttr = torch.tensor(ytr, dtype=torch.float32)

    enc = TaskLatentEncoder(xtr.shape[1], z_dim=z_dim)
    assert_no_forbidden_mechanism(enc)
    opt = torch.optim.Adam(enc.parameters(), lr=lr, weight_decay=1e-4)
    if model is not None:
        assert_backbone_frozen(model, head=enc, optimizer=opt)

    pos = float(ttr.sum())
    pw = torch.tensor(max(1.0, (len(ttr) - pos) / max(1.0, pos)), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    enc.train()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(enc(xtr), ttr).backward()
        opt.step()
    enc.eval()
    out = []
    with torch.no_grad():
        for Xe in evals:
            xe = torch.tensor((Xe - mu) / sd, dtype=torch.float32)
            out.append(enc.encode(xe).numpy())
    if model is not None:
        assert_backbone_frozen(model, head=enc, optimizer=opt)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        rng = np.random.RandomState(0)
        X = rng.randn(400, 20)
        y = (X[:, 0] * X[:, 1] > 0).astype(float)
        Z = train_encoder(X, y, [X[:50]], 0)
        print(f"encoder selftest: Z shape {Z[0].shape}, depth map {TASK_DEPTH}")
        enc = TaskLatentEncoder(20)
        assert_no_forbidden_mechanism(enc)
        enc.config["mechanisms"] = ["learned_depth_gate"]
        try:
            assert_no_forbidden_mechanism(enc)
            print("FAIL: forbidden-mechanism guard did not fire")
            return 1
        except RuntimeError as exc:
            print(f"forbidden-mechanism guard fires as intended: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
