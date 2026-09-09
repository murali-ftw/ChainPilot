"""Phase 3.1 — dilated causal TCN.

Guide geometry: kernel 2, dilations 1,2,4,8,16,32, 6 layers, receptive field 64 >= the
52-week window, hidden 64.

Causality is enforced by LEFT padding only, then trimming the right overhang. A symmetric
`padding=` argument in nn.Conv1d makes the model non-causal and leaks the future.
"""
import torch, torch.nn as nn, torch.nn.functional as F

DILATIONS = (1, 2, 4, 8, 16, 32)


class TCN(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, kernel: int = 2, dilations=DILATIONS):
        super().__init__()
        self.k, self.dil = kernel, tuple(dilations)
        self.convs = nn.ModuleList()
        self.res = nn.ModuleList()      # 1x1 residual projections
        c = d_in
        for _ in self.dil:
            self.convs.append(nn.Conv1d(c, hidden, kernel))
            self.res.append(nn.Conv1d(c, hidden, 1) if c != hidden else nn.Identity())
            c = hidden
        self.norm = nn.LayerNorm(hidden)

    def all_positions(self, x):                  # x [B, T, d] -> [B, h, T]
        z = x.transpose(1, 2)
        T = z.shape[-1]
        for conv, res, d in zip(self.convs, self.res, self.dil):
            skip = res(z)                        # residual: without it a 6-deep ReLU stack
            y = F.pad(z, ((self.k - 1) * d, 0))  # collapses and the encoder learns nothing
            y = F.relu(conv(y)[:, :, :T])        # LEFT pad only, then trim the overhang
            z = y + skip
        return z

    def forward(self, x):                        # -> [B, hidden], final position only
        return self.norm(self.all_positions(x)[:, :, -1])


class HeteroMP(nn.Module):
    """Learned heterogeneous message passing, channel <-> {supplier, part, plant}.

    One ROUND = channels -> entity means (learned, per relation) -> back to channels
    (learned, gated). A 4-layer encoder is two rounds; h1 is one round; h0 is zero rounds.
    This is the learned counterpart of the fixed mean-pooling used in the run-7 h0 diagnostic.
    """
    def __init__(self, h: int, n_rel: int = 3, rounds: int = 1):
        super().__init__()
        self.rounds = rounds
        self.up = nn.ModuleList([nn.ModuleList([nn.Linear(h, h) for _ in range(n_rel)])
                                 for _ in range(rounds)])
        self.down = nn.ModuleList([nn.ModuleList([nn.Linear(h, h) for _ in range(n_rel)])
                                   for _ in range(rounds)])
        self.gate = nn.ModuleList([nn.Linear(2 * h, h) for _ in range(rounds)])

    def forward(self, hc, rel_index, rel_size):
        """hc [NCH, h]; rel_index list of LongTensor [NCH] mapping channel -> entity id."""
        for r in range(self.rounds):
            msgs = []
            for j, (idx, n) in enumerate(zip(rel_index, rel_size)):
                m = self.up[r][j](hc)
                agg = torch.zeros(n, m.shape[1], device=hc.device, dtype=hc.dtype)
                agg.index_add_(0, idx, m)
                cnt = torch.zeros(n, 1, device=hc.device, dtype=hc.dtype)
                cnt.index_add_(0, idx, torch.ones_like(m[:, :1]))
                agg = agg / cnt.clamp(min=1.0)
                msgs.append(self.down[r][j](agg)[idx])
            nb = torch.stack(msgs, 0).mean(0)
            g = torch.sigmoid(self.gate[r](torch.cat([hc, nb], -1)))
            hc = hc + g * nb                      # residual, so h4 can fall back to h0
        return hc
