"""Phase 4 / run 8 — SHARE: RGCN basis decomposition + relation-blind shared attention.

Specified in `docs/model_plan.md` "Stage 2" and `benchmark_specification.md` §4, and this
is that specification, not an approximation of it:

    W_r^(l) = sum_{b=1..B} a_rb^(l) V_b^(l)                                    B = 10
    e_ij    = LeakyReLU( a^T [ W_r h_i || W_r h_j ] )
    alpha   = softmax over the WHOLE incoming neighbourhood of i, relation-blind
    h_i^(l+1) = sigma( W_0 h_i + sum_r sum_{j in N_r(i)} alpha_ij W_r h_j )

Two things this graph does to SHARE's economics, both worth saying out loud:

  * **R = 6, not 20.** The graph is channel <-> {supplier, part, plant}: three undirected
    relations, six directed types. Basis decomposition exists to stop 20 relation types
    from each fitting their own 128x128 matrix. With six, it shares far less than it does
    on v3's schema, so most of whatever SHARE gains over a mean aggregator here should
    come from the ATTENTION, not the basis trick.
  * **Entity nodes carry no sequence.** A supplier has no channel_performance row. It
    enters as the mean of its member channels' TCN states -- content-derived, so the model
    stays INDUCTIVE. A learned per-supplier embedding would key on identity and is exactly
    what the run's rules forbid.
"""
import torch, torch.nn as nn, torch.nn.functional as F


class SHARELayer(nn.Module):
    def __init__(self, dim: int, n_rel: int, n_bases: int = 10,
                 negative_slope: float = 0.2):
        """n_bases=None -> SHARE-lite: free per-relation weights, no basis decomposition.

        Everything else is identical, including the relation-blind attention. The arm exists to
        test deviation 6 in the guide: at R = 6 < B = 10 the basis scheme costs 1.67x what free
        weights cost, so if SHARE-lite matches full SHARE the basis machinery is buying nothing.
        """
        super().__init__()
        self.dim, self.n_rel, self.B = dim, n_rel, n_bases
        self.slope = negative_slope
        if n_bases is None:
            self.bases = None
            self.W_r = nn.Parameter(torch.empty(n_rel, dim, dim))   # free per-relation
            nn.init.xavier_uniform_(self.W_r)
        else:
            # V_b: the B shared bases. a_rb: the per-relation mixing coefficients.
            self.bases = nn.Parameter(torch.empty(n_bases, dim, dim))
            self.coef = nn.Parameter(torch.empty(n_rel, n_bases))
            nn.init.xavier_uniform_(self.bases)
            nn.init.xavier_uniform_(self.coef)
        self.self_loop = nn.Linear(dim, dim, bias=True)          # W_0
        # ONE attention vector, shared by every relation -- the "relation-blind" part.
        self.att_dst = nn.Parameter(torch.empty(dim))            # a[:dim]
        self.att_src = nn.Parameter(torch.empty(dim))            # a[dim:]
        nn.init.normal_(self.att_dst, std=0.1)
        nn.init.normal_(self.att_src, std=0.1)

    def rel_weights(self):
        """W_r for every r, as [n_rel, dim, dim]. Never materialised per edge."""
        if self.bases is None:
            return self.W_r
        return torch.einsum("rb,bij->rij", self.coef, self.bases)

    def forward(self, h, src, dst, rel, n_nodes):
        """h [N, dim]; src/dst/rel LongTensor [E]; edges point src -> dst."""
        W = self.rel_weights()                       # [R, d, d]
        Hr = torch.einsum("nd,rde->rne", h, W)       # [R, N, d]  = W_r h for all nodes

        hs = Hr[rel, src]                            # [E, d]  W_r h_j
        hd = Hr[rel, dst]                            # [E, d]  W_r h_i
        e = F.leaky_relu((hd * self.att_dst).sum(-1) + (hs * self.att_src).sum(-1),
                         self.slope)                 # [E]

        # softmax over each destination node's whole neighbourhood, all relations at once
        m = torch.full((n_nodes,), float("-inf"), device=h.device, dtype=e.dtype)
        m = m.index_reduce(0, dst, e, "amax", include_self=True)
        m = torch.nan_to_num(m, neginf=0.0)
        ex = torch.exp(e - m[dst])
        den = torch.zeros(n_nodes, device=h.device, dtype=e.dtype)
        den.index_add_(0, dst, ex)
        alpha = ex / den[dst].clamp(min=1e-16)

        agg = torch.zeros(n_nodes, self.dim, device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, alpha.unsqueeze(-1) * hs)
        return F.relu(self.self_loop(h) + agg)


class SHARE(nn.Module):
    """4 layers, hidden 128, B = 10. Returns [h0, h1, h2, h3, h4] for the channel nodes."""

    def __init__(self, d_in: int, dim: int = 128, n_rel: int = 6,
                 n_layers: int = 4, n_bases: int = 10):
        super().__init__()
        self.dim, self.n_layers = dim, n_layers
        self.inp = nn.Linear(d_in, dim)              # TCN state (64) -> graph width (128)
        self.layers = nn.ModuleList(
            [SHARELayer(dim, n_rel, n_bases) for _ in range(n_layers)])

    def forward(self, hc, ent_of_chan, ent_offset, n_nodes, src, dst, rel, depth=None,
                return_all=False):
        """hc [NCH, d_in] channel states. Entity nodes are seeded from their members.

        ent_of_chan : list of LongTensor [NCH], channel -> local entity id per relation
        ent_offset  : list of int, where each entity block starts in the node table
        """
        NCH = hc.shape[0]
        x = self.inp(hc)
        h = torch.zeros(n_nodes, self.dim, device=hc.device, dtype=x.dtype)
        h[:NCH] = x
        for idx, off in zip(ent_of_chan, ent_offset):
            n_ent = int(idx.max()) + 1
            agg = torch.zeros(n_ent, self.dim, device=hc.device, dtype=x.dtype)
            agg.index_add_(0, idx, x)
            cnt = torch.zeros(n_ent, 1, device=hc.device, dtype=x.dtype)
            cnt.index_add_(0, idx, torch.ones_like(x[:, :1]))
            h[off:off + n_ent] = agg / cnt.clamp(min=1.0)
        want = self.n_layers if depth is None else depth
        # guide 4.2 asks for every intermediate representation, h0..hL, so that each head can
        # select its own readout depth and so that h0 is available as the mandatory ablation.
        # The `depth=` path returns one of them directly, which is what the training loop uses;
        # `return_all=True` returns the whole list, which is what the 4.2 verify step checks.
        states = [h[:NCH]]
        for l in range(want):
            h = self.layers[l](h, src, dst, rel, n_nodes)
            states.append(h[:NCH])
        return states if return_all else states[-1]


def build_graph(rel_index, rel_sizes, device):
    """channel <-> {supplier, part, plant} as one flat directed edge list.

    Relation ids 0,1,2 are channel->entity; 3,4,5 the reverses. Six directed types
    over three undirected relations.
    """
    NCH = rel_index[0].shape[0]
    offsets, off = [], NCH
    for n in rel_sizes:
        offsets.append(off)
        off += n
    n_nodes = off
    ch = torch.arange(NCH, device=device)
    src, dst, rel = [], [], []
    for j, (idx, o) in enumerate(zip(rel_index, offsets)):
        ent = idx + o
        src += [ch, ent]
        dst += [ent, ch]
        rel += [torch.full_like(ch, j), torch.full_like(ch, j + len(rel_sizes))]
    return (torch.cat(src), torch.cat(dst), torch.cat(rel), n_nodes, offsets)
