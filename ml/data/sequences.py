"""Phase 2 — [T x d] tensors, missing-value masking, normalisation.

Guide steps 2.1-2.3. Reads the Phase 1.5 cache; never re-parses a CSV.

Two departures from the guide's text, both because the panel is complete (Phase 1 finding):
  * left-padding is a no-op -- every channel has all 535 weeks -- but the padding mask is
    still produced and returned, because Phase 3's loss must exclude padded positions and
    a window that starts before week 0 does need it.
  * `is_active_week` and the padding mask stay separate, as the guide insists: an idle week
    is an observation, a padded position is not.
"""
from __future__ import annotations
import os, json
import numpy as np

WINDOW = 52


def window_index(t0_week: int, T: int, w: int = WINDOW):
    """Trailing w weeks ending at t0 inclusive. Returns (slice_lo, pad_left)."""
    hi = t0_week + 1
    lo = hi - w
    pad = max(0, -lo)
    return max(0, lo), hi, pad


def build_sequences(panel, miss, active, t0_weeks, chan_ids=None, w: int = WINDOW):
    """-> X [B, w, d+k], value_mask [B, w, k], active [B, w], pad_mask [B, w]

    d value channels + k observed-indicators, per guide 2.2: fill the value with 0.0 AND
    set the indicator to 0, so 'no activity' is distinguishable from 'activity, value zero'.
    """
    NCH, T, d = panel.shape
    k = miss.shape[2]
    chan_ids = np.arange(NCH) if chan_ids is None else np.asarray(chan_ids)
    B = len(chan_ids)
    X = np.zeros((B, w, d + k), np.float32)
    M = np.zeros((B, w, k), np.float32)
    A = np.zeros((B, w), np.float32)
    P = np.zeros((B, w), np.float32)          # 1 = real position, 0 = left padding
    t0_weeks = np.asarray(t0_weeks)
    # Fast path -- every row shares one t0, which is the case for every snapshot-wide
    # build. Same arithmetic as the loop below, one slice instead of B Python iterations.
    if len(t0_weeks) and (t0_weeks == t0_weeks[0]).all():
        lo, hi, pad = window_index(int(t0_weeks[0]), T, w)
        n = hi - lo
        pv = np.asarray(panel[:, lo:hi, :])[chan_ids]
        mv = np.asarray(miss[:, lo:hi, :])[chan_ids]
        av = np.asarray(active[:, lo:hi])[chan_ids]
        X[:, pad:pad + n, :d] = pv
        X[:, pad:pad + n, d:] = mv
        M[:, pad:pad + n] = mv
        A[:, pad:pad + n] = av
        P[:, pad:pad + n] = 1.0
        return X, M, A, P
    for i, (c, t0) in enumerate(zip(chan_ids, t0_weeks)):
        lo, hi, pad = window_index(int(t0), T, w)
        n = hi - lo
        X[i, pad:pad + n, :d] = panel[c, lo:hi]
        X[i, pad:pad + n, d:] = miss[c, lo:hi]
        M[i, pad:pad + n] = miss[c, lo:hi]
        A[i, pad:pad + n] = active[c, lo:hi]
        P[i, pad:pad + n] = 1.0
    return X, M, A, P


class Normaliser:
    """Guide 2.3 -- fit on the TRAINING fold only, over VALID positions only."""
    def __init__(self, log1p_idx=()):
        self.mu = None; self.sd = None; self.log1p_idx = tuple(log1p_idx)

    def _pre(self, X):
        X = X.copy()
        for j in self.log1p_idx:
            X[..., j] = np.log1p(np.clip(X[..., j], 0, None))
        return X

    def fit(self, X, valid):
        """valid is [B, w] -- a position mask, not a per-channel mask."""
        X = self._pre(X)
        v = np.asarray(valid).astype(bool)
        assert v.shape == X.shape[:2], f"valid must be [B, w], got {v.shape}"
        flat = X[v]                                    # [Nvalid, C]
        self.mu = flat.mean(0)
        self.sd = flat.std(0)
        self.sd[self.sd < 1e-6] = 1.0                  # constant channel -> leave unscaled
        return self

    def transform(self, X):
        X = self._pre(X)
        return ((X - self.mu) / self.sd).astype(np.float32)

    def fit_transform(self, X, valid):
        return self.fit(X, valid).transform(X)
