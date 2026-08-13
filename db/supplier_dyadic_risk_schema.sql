-- ============================================================================
-- ChainPilot / HADES — Claim B (dyadic risk reweighting) output schema
-- Source of truth: docs/05_Database_Design.md §6.26. Group D ("Persistence for
-- the Components Still Under Validation") -- built now because Claim B is
-- actually being implemented this round (Step 8), per that section's own
-- gating rule ("build only once Claim B is being implemented"), and only
-- after the double-counting test (docs/10_AI_ML_Documentation.md §8.5) has a
-- place to record its result (`double_counting_test_run_id`).
--
-- Deliberately does NOT build the §6.23 suppliers tier/is_frontier amendment
-- or supplier_relationships -- those remain data-gated (no SUB_SUPPLIES-
-- equivalent data exists, confirmed unchanged since every prior round that
-- touched this question) and are unrelated to what supplier_dyadic_risk
-- itself actually needs (supplier_id, risk_score_id, and three business-
-- arithmetic weights -- none of them tier/frontier-dependent).
--
-- DEVIATION FROM §6.26, FLAGGED EXPLICITLY (see reports/step8_claim_b.md for
-- the full discussion): the documented §6.26 column list has NO customer_id
-- column at all, yet Claim B's own formula (§8.5) and this round's own task
-- instructions define order_volume_share/contract_priority_weight/
-- fulfilment_preference_weight as per-(customer, supplier) PAIR quantities,
-- computed from orders.customer_id via shipments -- a table with no way to
-- record WHICH customer a row's weights belong to cannot store what the
-- formula actually produces. A customer_id column is added here as a
-- necessary correction, not a silent one -- every other column matches §6.26
-- exactly.
--
-- Apply: psql -d chainpilot -f db/supplier_dyadic_risk_schema.sql
-- ============================================================================

-- ---------------------------------------------------------------- 6.26 supplier_dyadic_risk
-- Persists Claim B's relationship-specific reweighting of a supplier's global
-- risk score, kept separate from risk_scores so the two are never conflated.
-- Immutable, append-only, mirroring risk_scores' own audit convention --
-- existing risk_scores rows are NEVER modified by this table.
CREATE TABLE supplier_dyadic_risk (
    id                            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id                   UUID NOT NULL REFERENCES suppliers(id),
    customer_id                   UUID NOT NULL REFERENCES customers(id),  -- DEVIATION: not in documented §6.26, see note above
    risk_score_id                 UUID NOT NULL REFERENCES risk_scores(id),
    order_volume_share            NUMERIC(5,4) CHECK (order_volume_share IS NULL OR (order_volume_share >= 0 AND order_volume_share <= 1)),
    contract_priority_weight      NUMERIC(5,4) CHECK (contract_priority_weight IS NULL OR (contract_priority_weight >= 0 AND contract_priority_weight <= 1)),
    fulfilment_preference_weight  NUMERIC(5,4) CHECK (fulfilment_preference_weight IS NULL OR (fulfilment_preference_weight >= 0 AND fulfilment_preference_weight <= 1)),
    dyadic_risk_score             NUMERIC(5,4) NOT NULL CHECK (dyadic_risk_score >= 0 AND dyadic_risk_score <= 1),
    double_counting_test_run_id   VARCHAR(100),
    scored_at                     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_sdr_supplier_scored ON supplier_dyadic_risk (supplier_id, scored_at DESC);
CREATE INDEX idx_sdr_customer ON supplier_dyadic_risk (customer_id);
