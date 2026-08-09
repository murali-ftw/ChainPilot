"""
Claim B — deterministic dyadic risk reweighting formula — Step 8
(`docs/10_AI_ML_Documentation.md` §8.5, `docs/05_Database_Design.md` §6.26).

**Why this is deliberately not a model component.** The node is the structural encoder's
unit of prediction; there is no architectural slot for a `(supplier, customer)` PAIR. Every
function below is plain, auditable business arithmetic — SQL aggregates and a documented
closed-form combination — computed entirely OUTSIDE the encoder, reweighting an EXISTING,
UNMODIFIED `risk_scores` row rather than replacing it. Nothing here is learned or trained.

===============================================================================================
The three input weights
===============================================================================================

**Join path — corrected after a real, live-data-contradicted assumption.** The original design
(and this round's own initial task framing) assumed `shipments` directly carries both
`supplier_id` and `order_id`, letting a dyadic (customer, supplier) pair be read off one table.
**Checked live before writing the final queries below, and false**: `shipments.order_id` and
`shipments.supplier_id` are NEVER both populated on the same row (confirmed: 0 of 39,184) —
shipments are either supplier→firm (inbound component delivery, `supplier_id` set) or
firm→customer (outbound product delivery, `order_id` set), never both. A customer's order and a
supplier's component are connected only through the BOM chain:
`orders → order_items → product_components → components.supplier_id` — every query below uses
that path instead.

**`order_volume_share`** — this customer's share of a given supplier's total order-attributed
volume over a trailing 90-day window as of `t0` (matching this project's own
`on_time_rate_90d`-style trailing-window convention used throughout `ml/data/features.py`).
Volume is attributed to a supplier via the BOM: `order_items.quantity * product_components
.quantity_required * components.unit_cost`, summed per (customer, supplier), for BOM rows valid
as of the order's `placed_at` (`product_components.created_at <= placed_at <
COALESCE(deactivated_at, infinity)`):

    order_volume_share(C, S) = sum(BOM-attributed cost for C's orders touching S in the window)
                                / sum(BOM-attributed cost for ALL customers' orders touching S)

A value near 1.0 means this customer accounts for nearly all of the DEMAND driving supplier S's
components (a customer suppliers have every incentive to prioritize, transitively); near 0
means a marginal contributor.

**`contract_priority_weight`** — read directly from `customers.priority_tier`
(`customer_priority_tier` ENUM: `strategic`/`standard`/`low`, confirmed live against
`db/schema.sql`), via an explicit, documented numeric mapping (`PRIORITY_TIER_WEIGHTS` below)
— not learned, not buried in a magic number.

**`fulfilment_preference_weight` — THE LEAD SIGNAL.** Derived from historical RATIONING
behavior, now correctly split across the two shipment legs the live-data check above forced:
(1) a supplier S's OWN inbound shipments (`supplier_id` set) determine which weeks S was
capacity-constrained (>= `min_concurrent_delayed` delayed in the same week — a real squeeze,
not one late truck); (2) for each customer C, among C's OWN outbound order shipments
(`order_id` set) whose order touches supplier S via the BOM chain AND whose dispatch week falls
in one of S's constrained weeks, what fraction were STILL delivered on time. High = historically
prioritized when the supplier was stretched thin; low = historically deprioritized. **Why this
is the one signal the encoder structurally cannot already see:** it is derived from
`shipment_status_history`'s DELAY TRANSITIONS on TWO different shipment legs, joined through the
BOM chain and `orders.customer_id` — a customer-level aggregate over other entity types'
(Shipment's) temporal history — which is not, and never has been, a `Customer` node feature
anywhere in `ml/data/features.py::customer_features_asof` (confirmed: that function's ENTIRE
feature set is the `priority_tier` one-hot, nothing else). `priority_tier` itself is visible to
the encoder (`Customer` node feature, reachable via `PLACED_BY`); `fulfilment_preference_weight`
never is.

===============================================================================================
The combination formula f() — auditable, not a black box
===============================================================================================

    combined_protection = weighted_average(order_volume_share, contract_priority_weight,
                                            fulfilment_preference_weight;
                                            weights = [0.25, 0.25, 0.50])
                           -- fulfilment_preference_weight leads at 0.50 of the total weight,
                           -- per docs/10_AI_ML_Documentation.md §8.5's own instruction to
                           -- "lead on fulfilment_preference_weight"; renormalized over
                           -- whichever of the three inputs are actually available (non-NULL)
                           -- for a given pair -- a pair with no fulfilment-preference history
                           -- (no constrained-week shipments observed) falls back to a plain
                           -- 50/50 average of the other two, not a fabricated third value.

    f = 1 + LAMBDA * (0.5 - combined_protection)          LAMBDA = 0.6

    dyadic_risk_score = clip(global_risk * f, 0, 1)

`combined_protection` is in `[0, 1]`, centered at 0.5 = "average treatment, no evidence either
way" -> `f = 1` -> `dyadic_risk_score = global_risk` exactly, unchanged. A maximally-protected
customer (`combined_protection = 1`: large volume share, strategic tier, always fulfilled
during shortages) gets `f = 1 - 0.3 = 0.7` -- risk reduced 30% relative to the supplier's raw
global score. A maximally-exposed customer (`combined_protection = 0`) gets `f = 1.3` -- risk
increased 30%. `LAMBDA = 0.6` was chosen as a moderate, auditable bound: large enough that the
reweighting is visible and meaningful to a reviewer, small enough that `dyadic_risk_score`
never swings wildly from `global_risk` on the strength of business-arithmetic inputs alone
(that would defeat the point of keeping this a light reweighting layer, not a second scorer).

===============================================================================================
The double-counting test's "reordering rate" metric
===============================================================================================

For a single customer with >= 2 suppliers, `reordering_rate` for that customer is the fraction
of that customer's supplier PAIRS whose relative order flips between (a) ranking by
`global_risk` alone and (b) ranking by `dyadic_risk_score` (Kendall-tau discordant-pair
fraction) -- averaged across every customer with >= 2 suppliers gives one number per encoder
arm. See `ml/run_step8_claim_b.py` for how this is used in the with/without-`priority_tier`
ablation itself (the ablation trains the encoder twice; this module's formulas never change
between those two runs -- only `global_risk`, the encoder's own output, does).
"""

from __future__ import annotations

import datetime as dt
from itertools import combinations

import pandas as pd

PRIORITY_TIER_WEIGHTS: dict[str, float] = {"strategic": 1.0, "standard": 0.6, "low": 0.3}

LAMBDA = 0.6
COMBINE_WEIGHTS = {"volume": 0.25, "priority": 0.25, "fulfilment": 0.50}
DEFAULT_WINDOW_DAYS = 90
MIN_CONCURRENT_DELAYED = 2


def contract_priority_weight(priority_tier: str) -> float:
    """`customers.priority_tier` -> the documented numeric mapping. Raises on an unknown
    tier rather than silently defaulting -- a schema/mapping drift should be loud."""
    return PRIORITY_TIER_WEIGHTS[priority_tier]


def compute_order_volume_share(conn, t0: dt.datetime, window_days: int = DEFAULT_WINDOW_DAYS) -> pd.DataFrame:
    """Returns `[customer_id, supplier_id, order_volume_share]` for every (customer, supplier)
    pair with at least one BOM-attributed order in the trailing `window_days` as of `t0`. See
    module docstring for why this goes through `orders -> order_items -> product_components ->
    components.supplier_id` rather than `shipments` directly."""
    window_start = t0 - dt.timedelta(days=window_days)
    df = pd.read_sql(
        """
        SELECT o.customer_id, comp.supplier_id,
               SUM(oi.quantity * pc.quantity_required * COALESCE(comp.unit_cost, 0)) AS volume
        FROM orders o
        JOIN order_items oi ON oi.order_id = o.id
        JOIN product_components pc ON pc.product_id = oi.product_id
          AND pc.created_at <= o.placed_at
          AND (pc.deactivated_at IS NULL OR pc.deactivated_at > o.placed_at)
        JOIN components comp ON comp.id = pc.component_id
        WHERE o.placed_at BETWEEN %(window_start)s AND %(t0)s
        GROUP BY o.customer_id, comp.supplier_id
        """,
        conn,
        params={"window_start": window_start, "t0": t0},
    )
    if df.empty:
        return pd.DataFrame(columns=["customer_id", "supplier_id", "order_volume_share"])
    supplier_totals = df.groupby("supplier_id")["volume"].transform("sum")
    df["order_volume_share"] = (df["volume"] / supplier_totals.replace(0, pd.NA)).fillna(0.0)
    return df[["customer_id", "supplier_id", "order_volume_share"]]


def compute_fulfilment_preference_weight(conn, t0: dt.datetime, window_days: int = DEFAULT_WINDOW_DAYS,
                                          min_concurrent_delayed: int = MIN_CONCURRENT_DELAYED) -> pd.DataFrame:
    """Returns `[customer_id, supplier_id, fulfilment_preference_weight]` -- ONLY for pairs
    with at least one BOM-attributed order shipment in a week where the supplier was
    capacity-constrained on ITS OWN inbound leg (>= `min_concurrent_delayed` shipments delayed
    that week). Pairs with no such history are absent from the result (the caller treats them
    as NULL, per `05_Database_Design.md` §6.26's own nullable column -- not fabricated as a
    neutral default). See module docstring for why this spans two shipment legs joined through
    the BOM chain, not one `shipments` row."""
    window_start = t0 - dt.timedelta(days=window_days)
    df = pd.read_sql(
        """
        WITH supplier_side_delay AS (
            -- Leg 1: supplier -> firm inbound shipments, used only to detect which weeks a
            -- supplier was capacity-constrained.
            SELECT sh.supplier_id, date_trunc('week', sh.dispatched_at) AS wk,
                   EXISTS (
                       SELECT 1 FROM shipment_status_history ssh
                       WHERE ssh.shipment_id = sh.id AND ssh.status = 'delayed'
                         AND ssh.changed_at <= %(t0)s
                   ) AS was_delayed
            FROM shipments sh
            WHERE sh.supplier_id IS NOT NULL AND sh.dispatched_at IS NOT NULL
              AND sh.dispatched_at BETWEEN %(window_start)s AND %(t0)s
        ),
        constrained_weeks AS (
            SELECT supplier_id, wk
            FROM supplier_side_delay
            GROUP BY supplier_id, wk
            HAVING count(*) FILTER (WHERE was_delayed) >= %(min_concurrent)s
        ),
        order_supplier AS (
            -- BOM attribution: which suppliers does a given order touch.
            SELECT DISTINCT oi.order_id, comp.supplier_id
            FROM order_items oi
            JOIN orders o ON o.id = oi.order_id
            JOIN product_components pc ON pc.product_id = oi.product_id
              AND pc.created_at <= o.placed_at
              AND (pc.deactivated_at IS NULL OR pc.deactivated_at > o.placed_at)
            JOIN components comp ON comp.id = pc.component_id
        ),
        customer_order_shipment AS (
            -- Leg 2: firm -> customer outbound shipments, attributed to the supplier(s) whose
            -- components feed that order, restricted to that supplier's own constrained weeks.
            SELECT o.customer_id, os.supplier_id,
                   EXISTS (
                       SELECT 1 FROM shipment_status_history ssh
                       WHERE ssh.shipment_id = sh.id AND ssh.status = 'delayed'
                         AND ssh.changed_at <= %(t0)s
                   ) AS was_delayed
            FROM shipments sh
            JOIN orders o ON sh.order_id = o.id
            JOIN order_supplier os ON os.order_id = o.id
            JOIN constrained_weeks cw ON cw.supplier_id = os.supplier_id
                                      AND cw.wk = date_trunc('week', sh.dispatched_at)
            WHERE sh.order_id IS NOT NULL AND sh.dispatched_at IS NOT NULL
              AND sh.dispatched_at BETWEEN %(window_start)s AND %(t0)s
        )
        SELECT customer_id, supplier_id,
               count(*) AS n_shipments,
               count(*) FILTER (WHERE NOT was_delayed) AS n_on_time
        FROM customer_order_shipment
        GROUP BY customer_id, supplier_id
        """,
        conn,
        params={"window_start": window_start, "t0": t0, "min_concurrent": min_concurrent_delayed},
    )
    if df.empty:
        return pd.DataFrame(columns=["customer_id", "supplier_id", "fulfilment_preference_weight"])
    df["fulfilment_preference_weight"] = df["n_on_time"] / df["n_shipments"]
    return df[["customer_id", "supplier_id", "fulfilment_preference_weight"]]


def combine_dyadic_weights(order_volume_share: float | None, contract_priority: float | None,
                            fulfilment_preference: float | None) -> float:
    """`f()` — the combination multiplier, `combined_protection` re-centered around 1.0. Only
    non-`None` inputs contribute, renormalized over whichever weights are present. Requires at
    least ONE non-`None` input (a (customer, supplier) pair with literally nothing known
    shouldn't be reweighted at all -- callers should skip such pairs, not call this)."""
    present = [
        (COMBINE_WEIGHTS["volume"], order_volume_share),
        (COMBINE_WEIGHTS["priority"], contract_priority),
        (COMBINE_WEIGHTS["fulfilment"], fulfilment_preference),
    ]
    present = [(w, v) for w, v in present if v is not None]
    if not present:
        raise ValueError("combine_dyadic_weights: at least one input must be non-None")
    total_w = sum(w for w, _ in present)
    combined_protection = sum(w * v for w, v in present) / total_w
    return 1.0 + LAMBDA * (0.5 - combined_protection)


def dyadic_risk_score(global_risk: float, order_volume_share: float | None,
                       contract_priority: float | None, fulfilment_preference: float | None) -> float:
    """Full formula: `global_risk * f(...)`, clipped to `[0, 1]` (the schema's own CHECK
    constraint on `supplier_dyadic_risk.dyadic_risk_score`)."""
    f = combine_dyadic_weights(order_volume_share, contract_priority, fulfilment_preference)
    return max(0.0, min(1.0, global_risk * f))


def reordering_rate(customer_supplier_risks: dict[str, dict[str, tuple[float, float]]]) -> float:
    """`customer_supplier_risks`: `{customer_id: {supplier_id: (global_risk, dyadic_risk)}}`.
    For every customer with >= 2 suppliers, computes the Kendall-tau discordant-pair fraction
    between the global-risk ranking and the dyadic-risk ranking of that customer's own
    suppliers, then averages across all such customers. Returns `float('nan')` if no customer
    has >= 2 suppliers (the metric is undefined -- callers should treat that as "no evidence",
    not silently substitute 0)."""
    customer_rates = []
    for supplier_risks in customer_supplier_risks.values():
        if len(supplier_risks) < 2:
            continue
        pairs = list(combinations(supplier_risks.items(), 2))
        discordant = 0
        for (_s_a, (g_a, d_a)), (_s_b, (g_b, d_b)) in pairs:
            global_order = (g_a > g_b) - (g_a < g_b)
            dyadic_order = (d_a > d_b) - (d_a < d_b)
            if global_order != 0 and dyadic_order != 0 and global_order != dyadic_order:
                discordant += 1
        customer_rates.append(discordant / len(pairs))
    if not customer_rates:
        return float("nan")
    return sum(customer_rates) / len(customer_rates)
