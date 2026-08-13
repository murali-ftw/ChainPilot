-- ============================================================================
-- ChainPilot / HADES — Transformer 2 (Claim 3) output schema
-- Source of truth: docs/05_Database_Design.md §6.25. Group D ("Persistence for
-- the Components Still Under Validation") -- built now because Transformer 2
-- is actually being implemented this round, per that section's own gating rule
-- ("built only when that component is actually implemented").
--
-- Apply: psql -d chainpilot -f db/hidden_dependency_links_schema.sql
-- ============================================================================

CREATE TYPE validation_status AS ENUM ('unvalidated', 'confirmed', 'rejected');

-- ---------------------------------------------------------------- 6.25 hidden_dependency_links
-- A row is an investigation lead, never a fact -- see the validation routes in
-- docs/10_AI_ML_Documentation.md §8.3. Canonical ordering (supplier_a_id <
-- supplier_b_id) guarantees one row per pair; both directional attention
-- weights are kept because attention is NOT symmetric (a's top-k pool
-- including b does not imply b's top-k pool includes a).
CREATE TABLE hidden_dependency_links (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_a_id           UUID NOT NULL REFERENCES suppliers(id),
    supplier_b_id           UUID NOT NULL REFERENCES suppliers(id),
    attention_a_to_b        NUMERIC(6,5) NOT NULL,
    attention_b_to_a        NUMERIC(6,5) NOT NULL,
    symmetric_score         NUMERIC(6,5) NOT NULL,
    rank_for_a              SMALLINT,
    rank_for_b               SMALLINT,
    snapshot_t0             TIMESTAMPTZ NOT NULL,
    candidate_pool_version  VARCHAR(50) NOT NULL,
    model_version           VARCHAR(50) NOT NULL,
    validation_status       validation_status NOT NULL DEFAULT 'unvalidated',
    detected_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT hidden_dependency_links_order_check CHECK (supplier_a_id < supplier_b_id)
);
CREATE UNIQUE INDEX idx_hdl_pair_snapshot_model
    ON hidden_dependency_links (supplier_a_id, supplier_b_id, snapshot_t0, model_version);
CREATE INDEX idx_hdl_symmetric_score ON hidden_dependency_links (symmetric_score DESC);
CREATE INDEX idx_hdl_validation_status ON hidden_dependency_links (validation_status);
