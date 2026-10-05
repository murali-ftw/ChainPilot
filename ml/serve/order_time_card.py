"""Phase 23AC T1 -- a plain-English CARD per PO line from the Phase 22 order-time product (new file; nothing existing edited).

The product itself is ml/serve/order_time.py (OrderTimeService), used unchanged: the bundle, its identity guard, its
features and its numbers. This module only re-presents what that service returns. Every number on a card is traced to
`svc.predict(lines)` or `svc.features_for(lines)` (see the comments in `cards`). The summary line is a fixed English
template filled from those fields -- no language model, no new estimate.

Card fields, one dict per input line:
  po_line                  the caller's po_line_id (only if the input carries that column)
  channel_id, created_date the input line; created_date = the creation timestamp's calendar date
  expected_receipt_date    created + round(7 * expected_lead_weeks) days (= svc.predict "expected_arrival")
  interval_earliest/latest created + round(interval_lo_days / interval_hi_days); the service's lo / hi already equal
                           7 * expected weeks + the month conformal quantile, i.e. the central 80% band
  expected_weeks           expected_lead_weeks, 2 dp; expected_iso_week = ISO week of expected_receipt_date
  late_risk_score          p_late (the strict flag_lag1 5-seed mean, unrounded)
  late_risk_class          "WATCHLIST" -- the Phase 15 class Phase 22 measured for the strict flag (no alert bar)
  late_risk_label          "WATCH (top 5%)" when the service's `watch` is True (score >= the validation top-5% cut),
                           else "WATCHLIST"
  resolved_level/_n        the deepest level of the KM date's L5 chain with >= k receipted lines (grpstats.arrival_levels
                           on the service's own group rows Z) and the receipted-line count n at that level
  summary                  e.g. "Expected around 2025-03-14 (week 11); 8 in 10 such lines arrive between 2025-02-26 and
                           2025-04-23. Late-risk 0.41: watch."
  _expected_lead_weeks, _interval_lo_days, _interval_hi_days   unrounded service values (kept for bit-exact tests)

Usage:
  import order_time_card as OC
  card = OC.OrderTimeCard()                      # raises guard.IdentityMismatch exactly as OrderTimeService does
  card.cards(lines)                              # lines: DataFrame[channel_id, created_ts, qty_ordered (, po_line_id)]
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "data"), os.path.join(HERE, "..", "baselines"),
                os.path.join(HERE, "..", "train"), os.path.join(HERE, "..", "eval")]
import numpy as np, pandas as pd
import order_time as OT
from guard import IdentityMismatch

LEVEL_NAMES = {5: "L5 channel x month", 4: "L4 channel", 2: "L2 supplier", 1: "L1 network"}
LEVEL_KEYS = {5: "L5", 4: "L4", 2: "L2", 1: "L1"}
LATE_CLASS = "WATCHLIST"
WATCH_LABEL = "WATCH (top 5%)"


class OrderTimeCard:
    def __init__(self, svc=None, **service_kwargs):
        """svc: an OT.OrderTimeService (built here if None; service_kwargs pass through, e.g. bundle_dir / expected_name)."""
        self.svc = OT.OrderTimeService(**service_kwargs) if svc is None else svc
        name = getattr(self.svc, "prod", {}).get("name")
        expected = service_kwargs.get("expected_name", OT.NAME)
        if name != expected:
            raise IdentityMismatch(f"card wraps product {name!r}, configured {expected!r}")

    def _served(self, lines):
        """One feature build, shared by predict: the service's own features_for output is memoised on the instance for the
        duration of the predict call (the service's code is not changed; its predict reads self.features_for)."""
        svc = self.svc
        feats = svc.features_for(lines)
        svc.features_for = lambda _lines: feats
        try:
            pred = svc.predict(lines)
        finally:
            del svc.features_for                       # back to the class method
        return feats, pred

    def cards(self, lines):
        lines = lines.reset_index(drop=True)
        (X, Z, tau), pred = self._served(lines)
        GS, k = self.svc.GS, self.svc.prod["k"]
        # resolved level of the KM date's chain: grpstats.arrival_levels on Z (= svc.features_for output), month = Z["month"]
        _, raw, res5, _ = GS.arrival_levels(Z, k, Z["month"].astype(np.int64))
        iN = GS.ASTATS.index("n")
        created = pd.to_datetime(lines.created_ts).dt.normalize()
        out = []
        for i in range(len(lines)):
            weeks = float(pred.expected_lead_weeks.iloc[i])            # svc.predict: KM median (k) + offset a
            lo_d = float(pred.interval_lo_days.iloc[i])                # svc.predict: 7 * weeks + month q10
            hi_d = float(pred.interval_hi_days.iloc[i])                # svc.predict: 7 * weeks + month q90
            p = float(pred.p_late.iloc[i])                             # svc.predict: mean of the 5 flag models
            watch = bool(pred.watch.iloc[i])                           # svc.predict: p >= validation top-5% cut
            c0 = created.iloc[i]
            rec = pd.Timestamp(pred.expected_arrival.iloc[i])          # svc.predict: created + round(7 * weeks) days
            early = c0 + pd.Timedelta(days=int(np.round(lo_d)))
            late = c0 + pd.Timedelta(days=int(np.round(hi_d)))
            lev = int(res5[i])                                         # arrival_levels(Z): deepest level with n >= k
            n_lev = raw[LEVEL_KEYS[lev]][i, iN]                        # raw[L][:, n] from Z
            iso = int(rec.isocalendar()[1])
            label = WATCH_LABEL if watch else LATE_CLASS
            card = {}
            if "po_line_id" in lines.columns:
                card["po_line"] = str(lines.po_line_id.iloc[i])
            card.update(channel_id=str(lines.channel_id.iloc[i]), created_date=str(c0.date()),
                        expected_receipt_date=str(rec.date()), interval_earliest=str(early.date()), interval_latest=str(late.date()),
                        expected_weeks=round(weeks, 2), expected_iso_week=iso,
                        late_risk_score=p, late_risk_class=LATE_CLASS, late_risk_label=label,
                        resolved_level=LEVEL_NAMES[lev], resolved_n=int(n_lev) if np.isfinite(n_lev) else 0,
                        _expected_lead_weeks=weeks, _interval_lo_days=lo_d, _interval_hi_days=hi_d)
            card["summary"] = summary(card, watch)
            out.append(card)
        return out


def summary(card, watch):
    """Deterministic template; every value already on the card."""
    tail = "watch." if watch else "watchlist score, below the watch cut."
    return (f"Expected around {card['expected_receipt_date']} (week {card['expected_iso_week']}); 8 in 10 such lines arrive "
            f"between {card['interval_earliest']} and {card['interval_latest']}. "
            f"Late-risk {card['late_risk_score']:.2f}: {tail}")
