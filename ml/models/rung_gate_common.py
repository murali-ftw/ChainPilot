"""
Shared helpers for Step 6 Phase 2's two per-node depth-gate hybrids
(`ml/models/rgcn_attn_rung4_encoder.py`, `ml/models/rgcn_attn_rung5_encoder.py`) --
"Markov Scoping and Transformer 1.md" Part 6's "ladder" of shrinking Transformer-1
designs. Both rungs are genuine PER-NODE hybrids -- the upgrade over the task-level
learned gate (`ml/models/rgcn_attn_depthgate_encoder.py`, `reports/step6_depth_gate.md`),
which only ever produced ONE shared distribution per task, never one per node, and so
could never express or test per-node adaptivity.

Both are prior-initialized: at construction, each task's gate is biased so that, BEFORE
any training, its output distribution is a sharp near-one-hot at that task's Markov fixed
depth (`ml/models/rgcn_attn_markov_encoder.py::MARKOV_READOUT_DEPTH`,
`reports/step6_markov_phase1.md`) -- reused directly here, not re-derived, so both rungs
start from the exact same confirmed depths Phase 1 already validated. This bounds the
downside: each rung's own module zero-initializes the part of the network that reads
node content (see each file's docstring for exactly which layer), so pre-training
`logits = position_bias` for EVERY node, identical and content-independent -- epoch 1's
forward pass matches Markov's fixed-depth behaviour to numerical precision, and any
subsequent per-node divergence during training is something the model actually learned.
"""

from __future__ import annotations

import torch

NUM_DEPTHS = 5  # h^0..h^4, num_layers=4


def init_near_onehot_bias(num_depths: int, target_depth: int, sharpness: float = 8.0) -> torch.Tensor:
    """A length-`num_depths` bias vector, zero everywhere except `target_depth`, set to
    `sharpness`. Used as an ADDITIVE term on top of zero-initialized content logits, so
    `softmax(bias)` alone (independent of any node's features) reproduces a sharp
    near-one-hot distribution at `target_depth`: at `sharpness=8.0`, `num_depths=5`,
    `softmax([0,...,8,...,0])` puts ~99.9% of its mass on `target_depth` (~0.03% on each
    of the other 4 depths) -- this is the "initialize to reproduce Markov" requirement,
    done as an init-time bias rather than a training-time KL penalty (contrast with the
    Step 6 depth-gate's anchor, `ml/models/rgcn_attn_depthgate_encoder.py`, which anchors
    continuously during training instead of only at initialization)."""
    bias = torch.zeros(num_depths)
    bias[target_depth] = sharpness
    return bias
