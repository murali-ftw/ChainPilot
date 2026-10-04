"""Phase 16 — two sealed, additive encoder variants. NEW CLASSES; the incumbents are untouched.

    ShareTrajLayer / SHARETraj   Candidate A (arrival): trajectory-similarity attention.
        e_ij = LeakyReLU( a^T [ W_r h_i || W_r h_j || Delta_ij ] ),  Delta_ij = [cos(z_i^(l), z_j^(l))]_{l=1..6}
        z^(l) = the TCN's per-dilation block state at the last week, read by FORWARD HOOKS (TCNTaps) -- tcn.py is not
        edited. Entity nodes have no sequence, so their z^(l) is the mean of their member channels' z^(l), exactly as
        SHARE seeds entity nodes from members. Delta is detached: it is an observed similarity, not a new gradient path
        into the TCN. The Delta weights a_traj are ZERO-initialised, so the variant consumes no extra random numbers and
        with Delta zeroed (the A2 control) its forward pass equals SHARE-lite's.
    SharePNALayer / SHAREPNA     Candidate B (arrival): relational PNA (Corso et al., 2020) + low-degree bypass.
    HeteroMPPNA                  Candidate B (capacity): relational PNA on HeteroMP's entity aggregation.

PNA here is an APPLICATION of Principal Neighbourhood Aggregation to a heterogeneous graph, not a new mechanism:
    AGG_r(i) = [mean, max, min, std]_j m_j, each times S(d, a) = (log(d+1)/delta_r)^a, a in {1, 0, -1},
    delta_r = mean of log(d+1) over the destination nodes of relation r. The graph is static (the same channel ->
    entity membership in every split), so the training-set mean equals the whole-graph mean.
    The widened 12d aggregate is projected back to d by one linear map.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from share import SHARELayer

AGGS = ("mean", "max", "min", "std")
N_SCALERS = 3


# ------------------------------------------------------------------ hooks on the TCN (S2)
class TCNTaps:
    """Per-dilation TCN states z_1..z_6 at the last position, captured by forward hooks.

    tcn.all_positions: z_{l+1} = relu(conv_l(pad z_l))[..,:T] + res_l(z_l). res_l's INPUT is z_l, so a hook on res[l]
    (l = 1..5) sees z_1..z_5, and a hook on norm (whose input is z_6[:, :, -1]) sees z_6. Hooks observe; they do not
    alter the computation. Registered by attach(), removed by remove()."""

    def __init__(self, tcn, depths=(1, 2, 3, 4, 5, 6)):
        self.depths = tuple(depths)
        self.z = {}
        self.handles = []
        n = len(tcn.dil)
        for l in range(1, n):
            self.handles.append(tcn.res[l].register_forward_hook(self._grab(l)))
        self.handles.append(tcn.norm.register_forward_hook(self._grab_last(n)))

    def _grab(self, l):
        def f(mod, inp, out):
            self.z[l] = inp[0][:, :, -1]
        return f

    def _grab_last(self, l):
        def f(mod, inp, out):
            self.z[l] = inp[0]
        return f

    def states(self):
        missing = [l for l in self.depths if l not in self.z]
        assert not missing, f"TCN taps missing depths {missing}: the TCN did not run before the encoder"
        return [self.z[l] for l in self.depths]

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def seed_entities(x, ent_of_chan, ent_offset, n_nodes):
    """Node table [n_nodes, d]: channels first, then each entity as the mean of its members (SHARE's seeding)."""
    NCH = x.shape[0]
    h = torch.zeros(n_nodes, x.shape[1], device=x.device, dtype=x.dtype)
    h[:NCH] = x
    for idx, off in zip(ent_of_chan, ent_offset):
        if idx is None:
            continue
        n_ent = int(idx.max()) + 1
        agg = torch.zeros(n_ent, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, idx, x)
        cnt = torch.zeros(n_ent, 1, device=x.device, dtype=x.dtype).index_add_(0, idx, torch.ones_like(x[:, :1]))
        h[off:off + n_ent] = agg / cnt.clamp(min=1.0)
    return h


def edge_softmax(e, dst, n_nodes):
    """The incumbent's softmax, verbatim: over each destination node's whole incoming neighbourhood."""
    m = torch.full((n_nodes,), float("-inf"), device=e.device, dtype=e.dtype)
    m = m.index_reduce(0, dst, e, "amax", include_self=True)
    m = torch.nan_to_num(m, neginf=0.0)
    ex = torch.exp(e - m[dst])
    den = torch.zeros(n_nodes, device=e.device, dtype=e.dtype)
    den.index_add_(0, dst, ex)
    return ex / den[dst].clamp(min=1e-16)


# ------------------------------------------------------------------ Candidate A
class ShareTrajLayer(SHARELayer):
    def __init__(self, dim, n_rel, n_traj=6, negative_slope=0.2):
        super().__init__(dim, n_rel, None, negative_slope)          # SHARE-lite: free W_r, no bases
        self.att_traj = nn.Parameter(torch.zeros(n_traj))          # zero init: no RNG drawn, A2 == A0 at init

    def logits(self, h, src, dst, rel, delta):
        Hr = torch.einsum("nd,rde->rne", h, self.rel_weights())
        hs, hd = Hr[rel, src], Hr[rel, dst]
        e = F.leaky_relu((hd * self.att_dst).sum(-1) + (hs * self.att_src).sum(-1) + delta @ self.att_traj,
                         self.slope)
        return e, hs

    def forward(self, h, src, dst, rel, n_nodes, delta):
        e, hs = self.logits(h, src, dst, rel, delta)
        alpha = edge_softmax(e, dst, n_nodes)
        agg = torch.zeros(n_nodes, self.dim, device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, alpha.unsqueeze(-1) * hs)
        return F.relu(self.self_loop(h) + agg)


class SHARETraj(nn.Module):
    """SHARE-lite with the trajectory term. Same construction order as SHARE, so the random stream is identical."""

    def __init__(self, d_in, dim=128, n_rel=6, n_layers=4, traj_depths=(1, 2, 3, 4, 5, 6), zero_delta=False):
        super().__init__()
        self.dim, self.n_layers = dim, n_layers
        self.traj_depths, self.zero_delta = tuple(traj_depths), bool(zero_delta)
        self.inp = nn.Linear(d_in, dim)
        self.layers = nn.ModuleList([ShareTrajLayer(dim, n_rel, len(self.traj_depths)) for _ in range(n_layers)])
        self.taps = None

    def attach(self, tcn):
        """share_traj's setup: hooks on the TCN. Idempotent."""
        if self.taps is None:
            self.taps = TCNTaps(tcn, self.traj_depths)
        return self

    def detach_taps(self):
        """Teardown: every hook removed."""
        if self.taps is not None:
            self.taps.remove()
            self.taps = None

    def delta(self, ent_of_chan, ent_offset, n_nodes, src, dst):
        cols = []
        for z in self.taps.states():
            zn = seed_entities(z.detach(), ent_of_chan, ent_offset, n_nodes)
            cols.append(F.cosine_similarity(zn[src], zn[dst], dim=-1, eps=1e-8))
        d = torch.stack(cols, -1)
        return torch.zeros_like(d) if self.zero_delta else d

    def forward(self, hc, ent_of_chan, ent_offset, n_nodes, src, dst, rel, depth=None, return_all=False,
                return_delta=False):
        NCH = hc.shape[0]
        h = seed_entities(self.inp(hc), ent_of_chan, ent_offset, n_nodes)
        delta = self.delta(ent_of_chan, ent_offset, n_nodes, src, dst)
        want = self.n_layers if depth is None else depth
        states = [h[:NCH]]
        for l in range(want):
            h = self.layers[l](h, src, dst, rel, n_nodes, delta)
            states.append(h[:NCH])
        out = states if return_all else states[-1]
        return (out, delta) if return_delta else out


# ------------------------------------------------------------------ Candidate B, shared machinery
class SlotIndex:
    """Groups edges by (destination node, relation): the per-relation neighbourhood N_r(i). Static graph -> cached."""

    def __init__(self, dst, rel, n_rel, dtype):
        key = dst * n_rel + rel
        self.uniq, self.slot = torch.unique(key, return_inverse=True)
        self.n = int(self.uniq.shape[0])
        self.slot_dst = self.uniq // n_rel
        self.slot_rel = self.uniq % n_rel
        self.deg = torch.zeros(self.n, device=dst.device, dtype=dtype).index_add_(
            0, self.slot, torch.ones_like(dst, dtype=dtype))
        ld = torch.log(self.deg + 1.0)
        # delta_r: mean of log(d+1) over relation r's destination slots (static graph: train mean = graph mean)
        dl = torch.zeros(n_rel, device=dst.device, dtype=dtype).index_add_(0, self.slot_rel, ld)
        cn = torch.zeros(n_rel, device=dst.device, dtype=dtype).index_add_(0, self.slot_rel, torch.ones_like(ld))
        self.delta_r = dl / cn.clamp(min=1.0)
        r = ld / self.delta_r[self.slot_rel].clamp(min=1e-12)
        self.scalers = torch.stack([r, torch.ones_like(r), 1.0 / r.clamp(min=1e-12)], -1)   # a = 1, 0, -1


def pna_aggregate(m, slot, n_slots, deg, scalers):
    """m [E, d] grouped by slot -> [n_slots, 4 aggregators * 3 scalers * d]."""
    d = m.shape[1]
    s = torch.zeros(n_slots, d, device=m.device, dtype=m.dtype).index_add_(0, slot, m)
    s2 = torch.zeros(n_slots, d, device=m.device, dtype=m.dtype).index_add_(0, slot, m * m)
    mean = s / deg.unsqueeze(-1)
    var = (s2 / deg.unsqueeze(-1) - mean * mean).clamp(min=0.0)
    std = torch.sqrt(var + 1e-5)
    idx = slot.unsqueeze(-1).expand_as(m)
    mx = torch.full((n_slots, d), float("-inf"), device=m.device, dtype=m.dtype).scatter_reduce(
        0, idx, m, "amax", include_self=True)
    mn = torch.full((n_slots, d), float("inf"), device=m.device, dtype=m.dtype).scatter_reduce(
        0, idx, m, "amin", include_self=True)
    A = torch.stack([mean, mx, mn, std], 1)                         # [S, 4, d]
    return (A.unsqueeze(2) * scalers[:, None, :, None]).reshape(n_slots, -1)   # [S, 4*3*d]


class SharePNALayer(SHARELayer):
    """SHARE-lite layer with (B-1) relational PNA in place of the single weighted sum and/or (B-2) a low-degree bypass.

    aggregators None -> the incumbent aggregate (sum_j alpha_ij W_r h_j). low_degree_k None -> the incumbent softmax.
    Bypass: an edge whose per-relation neighbourhood |N_r(i)| <= k gets alpha = 1 (the message passes directly); the
    remaining edges of that destination are softmaxed among themselves."""

    def __init__(self, dim, n_rel, aggregators=None, low_degree_k=None, negative_slope=0.2):
        super().__init__(dim, n_rel, None, negative_slope)
        self.aggregators, self.k = aggregators, low_degree_k
        if aggregators is not None:
            assert tuple(aggregators) == AGGS, f"only the full PNA set {AGGS} is implemented"
            self.proj = nn.Linear(len(AGGS) * N_SCALERS * dim, dim)

    def forward(self, h, src, dst, rel, n_nodes, S, return_parts=False):
        Hr = torch.einsum("nd,rde->rne", h, self.rel_weights())
        hs, hd = Hr[rel, src], Hr[rel, dst]
        e = F.leaky_relu((hd * self.att_dst).sum(-1) + (hs * self.att_src).sum(-1), self.slope)
        if self.k is None:
            alpha = edge_softmax(e, dst, n_nodes)
        else:
            byp = S.deg[S.slot] <= self.k
            alpha_rest = edge_softmax(e.masked_fill(byp, float("-inf")), dst, n_nodes)
            alpha = torch.where(byp, torch.ones_like(e), torch.nan_to_num(alpha_rest, nan=0.0))
        m = alpha.unsqueeze(-1) * hs
        if self.aggregators is None:
            agg = torch.zeros(n_nodes, self.dim, device=h.device, dtype=h.dtype).index_add_(0, dst, m)
            wide = None
        else:
            wide = pna_aggregate(m, S.slot, S.n, S.deg, S.scalers)
            agg = torch.zeros(n_nodes, self.dim, device=h.device, dtype=h.dtype).index_add_(
                0, S.slot_dst, self.proj(wide))
        out = F.relu(self.self_loop(h) + agg)
        return (out, dict(agg=agg, wide=wide, alpha=alpha, m=m)) if return_parts else out


class SHAREPNA(nn.Module):
    def __init__(self, d_in, dim=128, n_rel=6, n_layers=4, aggregators=None, low_degree_k=None):
        super().__init__()
        self.dim, self.n_layers, self.n_rel = dim, n_layers, n_rel
        self.inp = nn.Linear(d_in, dim)
        self.layers = nn.ModuleList([SharePNALayer(dim, n_rel, aggregators, low_degree_k) for _ in range(n_layers)])
        self._S = None

    def slots(self, dst, rel, dtype):
        key = (dst.data_ptr(), int(dst.shape[0]), dtype)
        if self._S is None or self._S[0] != key:
            self._S = (key, SlotIndex(dst, rel, self.n_rel, dtype))
        return self._S[1]

    def forward(self, hc, ent_of_chan, ent_offset, n_nodes, src, dst, rel, depth=None, return_all=False):
        NCH = hc.shape[0]
        h = seed_entities(self.inp(hc), ent_of_chan, ent_offset, n_nodes)
        S = self.slots(dst, rel, h.dtype)
        want = self.n_layers if depth is None else depth
        states = [h[:NCH]]
        for l in range(want):
            h = self.layers[l](h, src, dst, rel, n_nodes, S)
            states.append(h[:NCH])
        return states if return_all else states[-1]


class HeteroMPPNA(nn.Module):
    """HeteroMP with relational PNA on the channel -> entity aggregation. Construction order: HeteroMP's (up, down,
    gate), then the projections. The entity -> channel side is a gather (one message per relation per channel), so a
    low-degree bypass has nothing to bypass there and is not offered."""

    def __init__(self, h, n_rel=3, rounds=1, aggregators=AGGS):
        super().__init__()
        assert tuple(aggregators) == AGGS
        self.rounds = rounds
        self.up = nn.ModuleList([nn.ModuleList([nn.Linear(h, h) for _ in range(n_rel)]) for _ in range(rounds)])
        self.down = nn.ModuleList([nn.ModuleList([nn.Linear(h, h) for _ in range(n_rel)]) for _ in range(rounds)])
        self.gate = nn.ModuleList([nn.Linear(2 * h, h) for _ in range(rounds)])
        self.proj = nn.ModuleList([nn.ModuleList([nn.Linear(len(AGGS) * N_SCALERS * h, h) for _ in range(n_rel)])
                                   for _ in range(rounds)])
        self._scal = {}

    def scalers(self, j, idx, n, dtype):
        key = (j, idx.data_ptr(), n, dtype)
        if key not in self._scal:
            deg = torch.zeros(n, device=idx.device, dtype=dtype).index_add_(0, idx, torch.ones_like(idx, dtype=dtype))
            occ = deg > 0
            ld = torch.log(deg + 1.0)
            delta = ld[occ].mean()
            r = ld / delta
            self._scal[key] = (deg.clamp(min=1.0), torch.stack([r, torch.ones_like(r), 1.0 / r.clamp(min=1e-12)], -1),
                               float(delta))
        return self._scal[key]

    def forward(self, hc, rel_index, rel_size, return_parts=False):
        parts = []
        for r in range(self.rounds):
            msgs = []
            for j, (idx, n) in enumerate(zip(rel_index, rel_size)):
                if idx is None:
                    continue
                m = self.up[r][j](hc)
                deg, sc, _ = self.scalers(j, idx, n, hc.dtype)
                wide = pna_aggregate(m, idx, n, deg, sc)
                agg = self.proj[r][j](wide)
                if return_parts:
                    parts.append(dict(round=r, rel=j, wide=wide, agg=agg))
                msgs.append(self.down[r][j](agg)[idx])
            nb = torch.stack(msgs, 0).mean(0)
            g = torch.sigmoid(self.gate[r](torch.cat([hc, nb], -1)))
            hc = hc + g * nb
        return (hc, parts) if return_parts else hc
