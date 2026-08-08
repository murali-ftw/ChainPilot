"""
Step 6 Phase 1 — Markov-blanket-derived FIXED depth readout on top of SHARE
(RGCN+Attention, `rgcn_attn` in code). Companion to `ml/models/rgcn_attn_depthgate_encoder.py`'s
learned gate (`reports/step6_depth_gate.md`) -- this is Phase 1 of "Markov Scoping and
Transformer 1.md"'s recommended sequence (§7.1): a per-task FIXED readout depth, zero new
trainable parameters, no learned gate, no Transformer component (that's Phase 2, later, per
that doc's own sequencing and this round's explicit scope).

**Depths used (index into h^0..h^4 -- the encoder returns L+1=5 candidates):**

    delay    -> h^1  (index 1) -- the theory doc's own blanket-derived floor was h^2 (§1.5,
                "Delay | Supplier/Shipment | ~2 hops | h^2"), but reports/step6_preflight.md's
                5-seed CONSISTENT empirical result is that h^1 beats baseline (+0.0069, all 5
                seeds positive) while h^2 shows no effect (FLIPS) -- this file uses the
                EMPIRICALLY CONFIRMED depth, not the untested theoretical one, per this round's
                explicit instruction ("use the empirically-confirmed depth, not the theoretical
                floor").
    shortage -> h^3  (index 3) -- matches the theory doc's blanket estimate (§1.5, "Shortage |
                Product/Warehouse | ~3 hops | h^3") AND the learned depth-gate's own convergence
                point (reports/step6_depth_gate.md: argmax_depth=3 at every lambda tested).
    impact   -> h^4  (index 4) -- matches the theory doc's blanket estimate (§1.5, "Impact |
                Order/Customer | ~4 hops | h^4") AND the learned depth-gate's convergence point
                (argmax_depth=4 at every lambda tested); equivalent to reading the final layer,
                i.e. identical readout behaviour to the current `rgcn_attn` baseline for this
                task specifically (baseline already reads h^4 for impact via
                `ml/models/depth.py`'s structural prior).

**Mechanism.** Reuses `RGCNAttnDepthGateEncoder`'s h^0..h^4 exposure
(`ml/models/rgcn_attn_depthgate_encoder.py`) UNCHANGED -- the encoder math is byte-for-byte
identical to the depth-gate pilot's (same basis-decomposed transform, same shared attention
scorer, same h^0 = raw `lin_in` projection prepended to `RGCNAttnEncoder`'s own `h^1..h^L`).
The only difference from that file is what reads the resulting list: instead of
`DepthGateHead`'s learned `softmax(w_task)` weighted sum, `MarkovHADESModel` below does a fixed
Python index-select -- literally `layers[MARKOV_READOUT_DEPTH[task]][entity_type]`. No
`nn.Parameter` anywhere in the readout path, so this is genuinely zero new trainable parameters
relative to plain `rgcn_attn` -- a different, fixed indexing choice into the SAME encoder
output, not a new module.

**Why a separate model class, not reusing `DepthGateHADESModel`.** `DepthGateHADESModel`
always allocates a `DepthGateHead` (5 learned weights/task) and computes a KL term on every
forward pass, neither of which apply here -- reusing it would mean carrying dead parameters and
a permanently-zero KL loss rather than genuinely having none, and would blur the two designs'
very different parameter/compute footprints (this file: +0 params; the depth-gate: +15 params)
in the registry. `MarkovHADESModel` is intentionally the simplest possible model over the same
encoder: `build_encoder` + a fixed index-select + `PredictionHead`, nothing else.

**Why a new architecture identifier (`rgcn_attn_markov`), not reusing `rgcn_attn` or
`rgcn_attn_depthgate`.** `rgcn_attn` reads only `h^L` for every task (via
`ml/models/depth.py`'s structural depth prior, not this file's fixed indices) and carries no
`h^0`. `rgcn_attn_depthgate` reads a learned blend. Neither matches this design's fixed,
per-task, non-learned index-select -- it needs its own logged rows, per this round's explicit
instruction.
"""

from __future__ import annotations

from torch import nn

from ml.models.depth import TASK_ENTITY_TYPE, TASKS
from ml.models.encoder import build_encoder
from ml.models.heads import PredictionHead

# Fixed index into the encoder's returned h^0..h^4 list (5 entries at
# num_layers=4). See module docstring for the rationale behind each value.
MARKOV_READOUT_DEPTH: dict[str, int] = {"delay": 1, "shortage": 3, "impact": 4}


class MarkovHADESModel(nn.Module):
    """Encoder (`RGCNAttnDepthGateEncoder`, via `build_encoder("rgcn_attn_markov", ...)`) + one
    `PredictionHead` per task, with each task reading a FIXED, non-learned depth from
    `MARKOV_READOUT_DEPTH` -- no gate, no attention, no new parameters beyond the encoder and
    the three prediction heads every architecture in this project already has. Returns the same
    `(logits, layers)` 2-tuple every other architecture returns, so `ml/evaluate.py` needs zero
    changes."""

    def __init__(self, metadata, in_dims: dict[str, int], hidden: int = 64,
                 num_layers: int = 4, num_bases: int = 8, dropout: float = 0.2):
        super().__init__()
        self.encoder = build_encoder("rgcn_attn_markov", metadata, in_dims, hidden=hidden,
                                      num_layers=num_layers, num_bases=num_bases, dropout=dropout)
        self.heads = nn.ModuleDict({task: PredictionHead(hidden) for task in TASKS})
        self.num_layers = num_layers
        self.hidden = hidden
        self.architecture = "rgcn_attn_markov"

    def forward(self, x_dict: dict, edge_index_dict: dict) -> tuple[dict, list[dict]]:
        layers = self.encoder(x_dict, edge_index_dict)
        logits = {}
        for task, entity_type in TASK_ENTITY_TYPE.items():
            depth = MARKOV_READOUT_DEPTH[task]
            logits[task] = self.heads[task](layers[depth][entity_type])
        return logits, layers

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
