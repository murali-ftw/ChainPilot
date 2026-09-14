"""Phase 5.1-5.3 — task heads.

Every head is DEPTH-AGNOSTIC: it is constructed with the width of whatever encoder output it will
read (64 for h0, the TCN state; 128 for a SHARE / SHARE-lite readout; 64 for HeteroMP at any depth)
and reads a plain [N, d] tensor. Nothing in a head knows or assumes which depth produced it.

Each head keeps the SAME two-layer trunk the Phase 2-4 heads used, Linear(d, d) -> ReLU ->
Linear(d, out), so an old-versus-new comparison isolates the output parameterisation and the loss.

  HazardHead     arrival   12 conditional hazards, independent sigmoids, censored rows in the NLL
  FillCDFHead    fill      point masses at 0 and 1 as separate cells + 20 interior bins, RPS/CRPS
  QuantileHead   capacity  P10 / P50 / P90, monotone by construction, pinball loss
  BinaryHead     shortage  DIAGNOSTIC only -- the product path is Monte Carlo (Phase 9, blocked)
"""
from __future__ import annotations
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


def _trunk(d_in: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_in), nn.ReLU(), nn.Linear(d_in, out))


# ================================================================== 5.3 arrival — hazard
class HazardHead(nn.Module):
    """lambda_w = P(T = w | T >= w), w = 1..W, through independent sigmoids.

    Label convention, measured against the generator: T is the week index of first receipt counted
    from the snapshot; a row is censored iff it has not arrived by the horizon (week 12). So an
    observed T lies in 1..12, and a censored row has survived all 12 weeks. `label_value` on a
    censored row carries the generator's eventual arrival week (13..92) -- a value from beyond the
    horizon that the model is never shown; only the fact T > 12 enters the likelihood.
    """

    def __init__(self, d_in: int, n_weeks: int = 12):
        super().__init__()
        self.W = n_weeks
        self.net = _trunk(d_in, n_weeks)

    def forward(self, h):
        return self.net(h)                                      # logits [N, W]

    def targets(self, T, censored):
        """-> (event [N, W], at_risk_survived [N, W]) as bool masks."""
        w = torch.arange(1, self.W + 1, device=T.device).unsqueeze(0)
        T = T.unsqueeze(1)
        c = censored.unsqueeze(1)
        event = (w == T) & ~c
        surv = torch.where(c, torch.ones_like(event), w < T)
        return event, surv

    def loss(self, z, T, censored):
        """Mean over rows of -sum_w [1(T=w) log lam_w + 1(T>w) log(1-lam_w)].

        Returns (loss, n_rows_contributing). A censored row contributes W survival terms; an
        observed row T-1 survival terms and one event term. Every row contributes at least one
        term -- the count is returned so the caller can assert it equals the population.
        """
        event, surv = self.targets(T, censored)
        ll = event * F.logsigmoid(z) + surv * F.logsigmoid(-z)
        contributing = (event | surv).any(1)
        return -ll.sum(1).mean(), int(contributing.sum())

    @staticmethod
    def distribution(z):
        """-> lam [N, W], S [N, W] = P(T > w), pT [N, W] = P(T = w)."""
        lam = torch.sigmoid(z)
        S = torch.cumprod(1 - lam, dim=-1)
        S_prev = torch.cat([torch.ones_like(S[:, :1]), S[:, :-1]], dim=-1)
        return lam, S, lam * S_prev

    @staticmethod
    def expected_time(S):
        """E[min(T, W+1)] = 1 + sum_{w=1..W} P(T > w). Monotone in risk, used for ranking."""
        return 1.0 + S.sum(-1)


# ================================================================== 5.2 fill — binned CDF + point masses
FILL_INTERIOR = 20
FILL_K = FILL_INTERIOR + 2          # {0}, 20 interior bins over (0, 1), {1}
FILL_INTERIOR_EDGES = np.linspace(0, 1, FILL_INTERIOR + 1)


def fill_cell(y):
    """Fill value -> cell index. 0: y == 0.  1..20: interior [k/20, (k+1)/20) cut to (0, 1).  21: y == 1.

    The interior edges are the legacy 20-bin edges, so this partition REFINES the legacy one
    exactly: legacy bin 0 = cells {0, 1}, legacy bins 1..18 = cells 2..19, legacy bin 19 = cells
    {20, 21}. That is what lets old and new heads be scored on one partition.
    """
    y = np.asarray(y, float)
    # digitize against the SAME linspace edges the legacy bins use, not floor(y * 20): at exact
    # multiples such as 0.15 the two disagree in the last ulp and the refinement stops being exact
    inner = 1 + np.clip(np.digitize(y, FILL_INTERIOR_EDGES[1:-1]), 0, FILL_INTERIOR - 1)
    return np.where(y <= 0, 0, np.where(y >= 1, FILL_K - 1, inner)).astype(np.int64)


def fill_to_legacy(P22):
    """[N, 22] cell probabilities -> [N, 20] legacy-bin probabilities."""
    P22 = np.asarray(P22)
    return np.concatenate([P22[:, :2].sum(1, keepdims=True), P22[:, 2:20],
                           P22[:, 20:].sum(1, keepdims=True)], 1)


class FillCDFHead(nn.Module):
    """softmax over 22 ordered cells; the endpoints are separate outputs, not bins.

    Loss is the ranked probability score over the 21 cell boundaries -- the CRPS of the ordered
    categorical, proper, and distance-aware -- exactly the guide's `crps_loss` on this partition.
    """

    def __init__(self, d_in: int):
        super().__init__()
        self.net = _trunk(d_in, FILL_K)

    def forward(self, h):
        return self.net(h)                                      # logits [N, 22]

    @staticmethod
    def loss(z, cell, kind="rps"):
        """kind: 'rps' (the specified loss), 'ce' or 'rps+ce' -- the last two are Phase 5 ablations.

        RPS is distance-aware, which is its point, and also why it barely penalises mass moved into
        a NEIGHBOURING cell; cross-entropy is local and penalises exactly that. The ablation asks
        which of the two the marginal calibration depends on.
        """
        out = 0.0
        if "rps" in kind:
            Fhat = torch.cumsum(torch.softmax(z, -1), -1)[:, :-1]  # [N, 21]
            step = (torch.arange(FILL_K - 1, device=z.device).unsqueeze(0) >= cell.unsqueeze(1))
            out = out + ((Fhat - step.to(Fhat.dtype)) ** 2).sum(-1).mean()
        if "ce" in kind:
            out = out + F.cross_entropy(z, cell)
        return out

    @staticmethod
    def probs(z):
        return torch.softmax(z, -1)


# ================================================================== 5.1 capacity — quantiles
class QuantileHead(nn.Module):
    """[N, H, Q] quantiles, lowest predicted directly, the rest as softplus increments.

    H = 1 here: the label exists only at the 90-day horizon, and the 30/60-day targets the
    specification wants are not in `training_labels` (guide 5.1 / specification §11.5). The head
    takes H as an argument so the multi-horizon version needs no structural change.
    """

    def __init__(self, d_in: int, quantiles=(0.1, 0.5, 0.9), n_horizons: int = 1):
        super().__init__()
        self.q = tuple(quantiles)
        self.H = n_horizons
        self.net = _trunk(d_in, n_horizons * len(self.q))

    def forward(self, h):
        z = self.net(h).view(h.shape[0], self.H, len(self.q))
        inc = torch.cat([z[..., :1], F.softplus(z[..., 1:])], -1)
        return torch.cumsum(inc, -1)                            # monotone in q, structurally

    def loss(self, qhat, y):
        """qhat [N, H, Q], y [N, H] -> mean pinball over rows, horizons and quantiles."""
        qs = torch.tensor(self.q, device=qhat.device, dtype=qhat.dtype)
        d = y.unsqueeze(-1) - qhat
        return torch.maximum(qs * d, (qs - 1) * d).mean()


# ================================================================== shortage — diagnostic
class BinaryHead(nn.Module):
    """P(any shortage in the window). A DIAGNOSTIC head: shortage ships from Phase 9's Monte Carlo."""

    def __init__(self, d_in: int):
        super().__init__()
        self.net = _trunk(d_in, 1)

    def forward(self, h):
        return self.net(h).squeeze(-1)

    @staticmethod
    def loss(z, y):
        return F.binary_cross_entropy_with_logits(z, y)
