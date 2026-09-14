"""Phase 5.0 — staleness gating, with ONE input.

Specification §3.1 has `weeks_since_last_activity` and `reporting_lag_days` jointly replacing
`staleness_days`. In both generated worlds `weeks_since_last_activity` is constant zero across all
8,598,520 `channel_performance_weekly` rows (guide deviation 4), so the gate here is built on
`reporting_lag_days` alone. It is DEGRADED relative to the specification, and says so.

Two further things were measured on the Phase 1.5 cache before this was written (phase-5.md §4):

  * `reporting_lag_days` is FORWARD-FILLED across idle weeks, like the rolling columns: observed on
    94.6% (v6) / 94.2% (v7) of channel-weeks, not the specification's 46.5%, and on every idle
    observed week it equals the previous week's value. On an idle week it is therefore the posting
    lag of the record whose values are being carried -- which is exactly the record the gate is
    deciding whether to trust -- so it is used as-is.
  * Its scale is P50 0.87 / P90 3.5 / P99 11 days, max 284 (v6) and 141 (v7) -- not the
    specification's 1.3 / 5.7 / 14 / 96, which were measured on a different world.

The missing second input is RECONSTRUCTED from `is_active_week` by `weeks_since_last_activity()`
below -- causally, as-of -- and, per specification §5.2, fed as a plain feature, never into the gate.
"""
from __future__ import annotations
import math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

LAG_P90_DAYS = 3.5          # measured, observed positions, both worlds (3.46 v6 / 3.50 v7)


class StalenessGate(nn.Module):
    """x~ = g * x + (1 - g) * prior,   g = exp(-max(0, w * u + b)),   u = log1p(dt) / log1p(P90).

    Per feature, continuous, applied only to the time-varying value channels the caller selects.
    Where dt is missing the gate is forced to 1 -- an absent lag is not a lag of zero.

    Two departures from the guide's snippet, because as written it cannot learn and cannot satisfy
    its own verify step:

      * The guide initialises w = b = 0. That puts every unit exactly on relu's kink, where
        torch's gradient is 0, so w and b receive no gradient for the whole run: the gate stays the
        identity because it is stuck, not because identity is right. Here w starts small and
        positive (g = 0.90 at the P90 lag) and b slightly negative, so gradients flow from step one.
      * b is unconstrained there, so a learned b > 0 decays a record posted with ZERO lag and
        breaks "at dt = 0 the layer is the identity". Here w = softplus(.) >= 0 and
        b = -softplus(.) <= 0, which makes both "dt = 0 -> identity" and "trust never increases
        with lag" structural.
    """

    def __init__(self, d: int, p90_days: float = LAG_P90_DAYS):
        super().__init__()
        self.norm = math.log1p(p90_days)
        self.w_raw = nn.Parameter(torch.full((d,), -2.0))    # softplus(-2) = 0.127
        self.b_raw = nn.Parameter(torch.full((d,), -4.0))    # b = -softplus(-4) = -0.018
        self.prior = nn.Parameter(torch.zeros(d))            # learned fallback, normalised units

    def weights(self):
        return F.softplus(self.w_raw), -F.softplus(self.b_raw)

    def gate(self, dt, observed):
        """dt [B, T] days (raw, not normalised), observed [B, T] bool -> g [B, T, d]."""
        w, b = self.weights()
        u = torch.log1p(dt.clamp(min=0)).unsqueeze(-1) / self.norm
        g = torch.exp(-F.relu(w * u + b))
        return torch.where(observed.unsqueeze(-1), g, torch.ones_like(g))

    def forward(self, x, dt, observed, return_gate=False):
        g = self.gate(dt, observed)
        out = g * x + (1 - g) * self.prior
        return (out, g) if return_gate else out


# ------------------------------------------------------------------ the missing second input
def weeks_since_last_activity(active: np.ndarray) -> np.ndarray:
    """[N, T] 0/1 `is_active_week` -> [N, T] float32 weeks since the last active week, AS-OF.

    Value at week t is t - max{t' <= t : active[t'] = 1}; 0 on an active week. Before a channel's
    first active week it is t + 1, i.e. counted from the start of the panel -- the store has no
    history before week 0 to say otherwise. Computed with a forward running maximum, so week t can
    only ever read weeks 0..t. `assert_asof` checks that claim rather than trusting it.
    """
    a = np.asarray(active) > 0
    T = a.shape[1]
    t = np.arange(T, dtype=np.int64)
    last = np.maximum.accumulate(np.where(a, t[None, :], -1), axis=1)
    return (t[None, :] - last).astype(np.float32)


def assert_asof(active: np.ndarray, n_rows: int = 256, n_cuts: int = 8, seed: int = 0) -> dict:
    """Two independent checks that `weeks_since_last_activity` never reads the future.

    1. brute force: for sampled (channel, t), recompute from active[c, :t+1] ONLY and compare.
    2. perturbation: overwrite every week after a cut with random activity; weeks <= cut must be
       bit-identical to the unperturbed result.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(active[:n_rows]) > 0
    T = a.shape[1]
    full = weeks_since_last_activity(a)
    n_brute = 0
    for c in range(a.shape[0]):
        for t in rng.integers(0, T, 16):
            past = np.flatnonzero(a[c, :t + 1])
            want = t - past[-1] if len(past) else t + 1
            assert full[c, t] == want, f"as-of brute force failed at channel {c}, week {t}"
            n_brute += 1
    for cut in rng.integers(1, T - 1, n_cuts):
        b = a.copy()
        b[:, cut + 1:] = rng.random((a.shape[0], T - cut - 1)) < 0.5
        pert = weeks_since_last_activity(b)
        assert np.array_equal(pert[:, :cut + 1], full[:, :cut + 1]), \
            f"as-of perturbation failed: a change after week {cut} moved a value at or before it"
    return {"brute_force_checks": n_brute, "perturbation_cuts": int(n_cuts), "rows": int(a.shape[0])}
