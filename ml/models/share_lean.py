"""Phase 17 B4b -- the LEAN arrival encoder: SHARE-lite without the part relation and without its dead weights.

NEW classes. `share.SHARELayer` and `share.SHARE` are not touched (the phase asserts an empty diff on them).

What is removed, from the shipped SHARE-lite (4 layers, hidden 128, free per-relation W_r, R = 6 directed types):
  * the part relation, both directions, every layer (Phase 12 C3: no measurable signal) -- 8 x 128^2 = 131,072
  * the last layer's channel -> entity relations: their messages land on entity nodes that are never read
    (deviation 107) -- 3 x 128^2 = 49,152, of which the part slice is already counted above
  => 163,840 parameters fewer (460,288 -> 296,448 in the graph encoder). The last layer also computes channel rows only.

The function is the Phase 12 C3 `drop_relation part` arm's: same edge filter (both directions removed, part nodes never
seeded), same relation-blind attention, same arithmetic order. `assert_equivalent_to_c3` shows it numerically by copying
the incumbent's kept weights into the lean module, and shows the check fails when the wrong relation is kept.
"""
import torch, torch.nn as nn, torch.nn.functional as F

REL_NAMES = ("supplier", "part", "plant")      # temporal_share.load_world's rel_t order; directed ids j and j + 3


class SHARELeanLayer(nn.Module):
    def __init__(self, dim: int, rels, negative_slope: float = 0.2):
        """rels: the ORIGINAL directed relation ids (0..5) this layer carries, in order."""
        super().__init__()
        self.dim, self.rels, self.slope = dim, tuple(int(r) for r in rels), negative_slope
        self.W_r = nn.Parameter(torch.empty(len(self.rels), dim, dim))
        nn.init.xavier_uniform_(self.W_r)
        self.self_loop = nn.Linear(dim, dim, bias=True)
        self.att_dst = nn.Parameter(torch.empty(dim))
        self.att_src = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.att_dst, std=0.1)
        nn.init.normal_(self.att_src, std=0.1)

    def forward(self, h, src, dst, rel_local, n_out):
        """Edges already filtered to this layer's relations; rel_local indexes self.W_r. Returns the first n_out rows."""
        Hr = torch.einsum("nd,rde->rne", h, self.W_r)
        hs = Hr[rel_local, src]
        hd = Hr[rel_local, dst]
        e = F.leaky_relu((hd * self.att_dst).sum(-1) + (hs * self.att_src).sum(-1), self.slope)
        m = torch.full((n_out,), float("-inf"), device=h.device, dtype=e.dtype)
        m = m.index_reduce(0, dst, e, "amax", include_self=True)
        m = torch.nan_to_num(m, neginf=0.0)
        ex = torch.exp(e - m[dst])
        den = torch.zeros(n_out, device=h.device, dtype=e.dtype)
        den.index_add_(0, dst, ex)
        alpha = ex / den[dst].clamp(min=1e-16)
        agg = torch.zeros(n_out, self.dim, device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, alpha.unsqueeze(-1) * hs)
        return F.relu(self.self_loop(h[:n_out]) + agg)


class SHARELean(nn.Module):
    """Drop-in for share.SHARE(n_bases=None) in HeadNet.encode's call: same forward signature, shipped depth only."""
    DROP = REL_NAMES.index("part")

    def __init__(self, d_in: int, dim: int = 128, n_rel: int = 6, n_layers: int = 4):
        super().__init__()
        R = n_rel // 2
        self.dim, self.n_layers, self.R = dim, n_layers, R
        keep = [r for r in range(n_rel) if r % R != self.DROP]             # channel<->supplier, channel<->plant
        last = [r for r in keep if r >= R]                                  # last layer: entity -> channel only
        self.inp = nn.Linear(d_in, dim)
        self.layers = nn.ModuleList([SHARELeanLayer(dim, keep) for _ in range(n_layers - 1)] + [SHARELeanLayer(dim, last)])
        self._edges = {}

    def _layer_edges(self, l, src, dst, rel):
        key = (l, src.device, int(src.shape[0]))
        if key not in self._edges:
            rels = self.layers[l].rels
            lut = torch.full((2 * self.R,), -1, dtype=torch.long, device=rel.device)
            lut[list(rels)] = torch.arange(len(rels), device=rel.device)
            k = lut[rel] >= 0
            self._edges[key] = (src[k], dst[k], lut[rel[k]])
        return self._edges[key]

    def forward(self, hc, ent_of_chan, ent_offset, n_nodes, src, dst, rel, depth=None, return_all=False):
        assert depth in (None, self.n_layers), "the lean encoder is defined at the shipped depth only"
        assert not return_all, "the lean encoder does not compute the entity states of the last layer"
        NCH = hc.shape[0]
        x = self.inp(hc)
        h = torch.zeros(n_nodes, self.dim, device=hc.device, dtype=x.dtype)
        h[:NCH] = x
        for j, (idx, off) in enumerate(zip(ent_of_chan, ent_offset)):
            if j == self.DROP or idx is None:                              # part nodes are never seeded
                continue
            n_ent = int(idx.max()) + 1
            agg = torch.zeros(n_ent, self.dim, device=hc.device, dtype=x.dtype)
            agg.index_add_(0, idx, x)
            cnt = torch.zeros(n_ent, 1, device=hc.device, dtype=x.dtype)
            cnt.index_add_(0, idx, torch.ones_like(x[:, :1]))
            h[off:off + n_ent] = agg / cnt.clamp(min=1.0)
        for l, layer in enumerate(self.layers):
            s, d, r = self._layer_edges(l, src, dst, rel)
            last = l == self.n_layers - 1
            if last:
                assert bool((d < NCH).all()), "last-layer edges must all end at channel nodes"
            h = layer(h, s, d, r, NCH if last else n_nodes)
        return h[:NCH]


def copy_from_incumbent(lean: SHARELean, inc) -> None:
    """Load the incumbent SHARE-lite's KEPT weights into the lean module (used only by the equivalence check)."""
    with torch.no_grad():
        lean.inp.load_state_dict(inc.inp.state_dict())
        for L, I in zip(lean.layers, inc.layers):
            L.W_r.copy_(I.W_r[list(L.rels)])
            L.self_loop.load_state_dict(I.self_loop.state_dict())
            L.att_dst.copy_(I.att_dst); L.att_src.copy_(I.att_src)


def assert_equivalent_to_c3(W: dict, device, seed: int = 0, atol: float = 1e-5):
    """Same weights, same input: lean == incumbent SHARE-lite on the C3 part-dropped world. Then falsify: dropping the
    SUPPLIER relation instead must NOT match. Returns the two max-abs differences."""
    from share import SHARE
    from graph_control import relation_dropped_world
    torch.manual_seed(seed)
    inc = SHARE(64, 128, n_rel=6, n_layers=4, n_bases=None).to(device)
    lean = SHARELean(64, 128).to(device)
    copy_from_incumbent(lean, inc)
    hc = torch.randn(W["NCH"], 64, device=device)
    V = relation_dropped_world(W, "part", device)
    src, dst, rel, n_nodes, offs = V["graph"]
    with torch.no_grad():
        a = inc(hc, V["rel_t"], offs, n_nodes, src, dst, rel, depth=4)
        b = lean(hc, W["rel_t"], W["graph"][4], W["graph"][3], *W["graph"][:3], depth=4)
        diff = float((a - b).abs().max())
        Vs = relation_dropped_world(W, "supplier", device)
        s2, d2, r2, n2, o2 = Vs["graph"]
        c = inc(hc, Vs["rel_t"], o2, n2, s2, d2, r2, depth=4)
        diff_wrong = float((c - b).abs().max())
    assert diff <= atol, f"lean encoder differs from the C3 part-dropped incumbent: max |diff| {diff}"
    assert diff_wrong > 1e-3, f"the check cannot fail: the supplier-dropped incumbent also matches ({diff_wrong})"
    n_inc = sum(p.numel() for p in inc.parameters()); n_lean = sum(p.numel() for p in lean.parameters())
    return dict(max_abs_diff_vs_c3=diff, max_abs_diff_vs_supplier_dropped=diff_wrong, params_incumbent=n_inc,
                params_lean=n_lean, params_removed=n_inc - n_lean)
