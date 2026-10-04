"""Phase 16 — build a HeadNet whose graph stage is a Phase 16 variant, WITHOUT touching HeadNet or the incumbents.

HeadNet.__init__ constructs TCN -> encoder -> head in that order. The variant is swapped in AT the encoder's position
(by temporarily rebinding the names SHARE / HeteroMP that phase5_heads resolves at construction time), so the TCN and
the head draw exactly the random numbers they draw for the incumbent. Nothing is rebound outside the constructor.

With no Phase 16 axis set, loop.py never calls this module: the incumbent path is P5.HeadNet, byte for byte.
"""
from __future__ import annotations
import contextlib
import phase5_heads as P5
import artifact_identity as AI
from phase16_encoders import SHARETraj, SHAREPNA, HeteroMPPNA, AGGS


@contextlib.contextmanager
def _rebind(name, factory):
    old = getattr(P5, name)
    setattr(P5, name, factory)
    try:
        yield
    finally:
        setattr(P5, name, old)


def is_phase16(cfg) -> bool:
    return bool(AI.encoder_axes(cfg))


def parse_aggs(v):
    if v in (None, "sum"):
        return None
    assert v == "mmms", f"pna_aggregators must be 'mmms' (mean,max,min,std) or 'sum', got {v!r}"
    return AGGS


def build(cfg, d_in, **kw):
    """-> a HeadNet with the variant encoder. share_traj attaches its TCN hooks here (setup)."""
    v = cfg.get("encoder_variant")
    arch = cfg["arch"]
    if v == "share_traj":
        assert arch == "lite", "share_traj is a SHARE-lite variant (arrival)"
        depths = tuple(int(c) for c in str(cfg["traj_depths"])) if cfg.get("traj_depths") else (1, 2, 3, 4, 5, 6)
        fac = lambda d_in_, dim, n_rel, n_layers, n_bases: SHARETraj(
            d_in_, dim, n_rel, n_layers, traj_depths=depths, zero_delta=cfg.get("traj_delta") == "zeroed")
        name = "SHARE"
    elif v == "share_pna":
        assert arch == "lite"
        fac = lambda d_in_, dim, n_rel, n_layers, n_bases: SHAREPNA(
            d_in_, dim, n_rel, n_layers, aggregators=parse_aggs(cfg.get("pna_aggregators")),
            low_degree_k=cfg.get("low_degree_k"))
        name = "SHARE"
    elif v == "heteromp_pna":
        assert arch == "mp" and cfg.get("low_degree_k") is None, "HeteroMP has no softmax: no bypass (deviation)"
        assert parse_aggs(cfg.get("pna_aggregators")) is not None
        fac = lambda h, n_rel, rounds: HeteroMPPNA(h, n_rel, rounds)
        name = "HeteroMP"
    else:
        raise ValueError(f"not a Phase 16 variant: {v!r}")
    with _rebind(name, fac):
        m = P5.HeadNet(d_in, cfg["task"], arch, cfg["depth"], **kw)
    assert type(m.enc).__name__ in ("SHARETraj", "SHAREPNA", "HeteroMPPNA"), type(m.enc)
    if v == "share_traj":
        m.enc.attach(m.tcn)
    return m


def teardown(m):
    if hasattr(m, "enc") and hasattr(m.enc, "detach_taps"):
        m.enc.detach_taps()
