-- ============================================================================
-- ChainPilot / HADES — model-governance schema (DDL only)
-- Source of truth: docs/05_Database_Design.md §6.20-6.22. These three tables
-- were "specified, not yet in schema.sql" as of Step 2; Step 3 applies them
-- now, before any training run executes (docs/14_Model_Development_Roadmap.md
-- §6: "a schema with nowhere to store a confidence interval guarantees a
-- point estimate gets reported instead").
--
-- Apply: psql -d chainpilot -f db/governance_schema.sql
-- ============================================================================

CREATE TYPE dataset_split AS ENUM ('train', 'validation', 'test');
CREATE TYPE model_status  AS ENUM ('training', 'evaluating', 'candidate', 'active', 'archived');

-- ---------------------------------------------------------------- 6.22 model_registry
-- Written by the training pipeline directly, never entered by hand -- otherwise
-- this table drifts from what was actually run and stops being trustworthy.
CREATE TABLE model_registry (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version       VARCHAR(50) NOT NULL UNIQUE,
    architecture        VARCHAR(100) NOT NULL,
    training_dataset    VARCHAR(255),
    training_timestamp  TIMESTAMPTZ NOT NULL,
    experiment_id       VARCHAR(100),
    git_commit          VARCHAR(40),
    hyperparameters     JSONB,
    parameter_count     BIGINT,
    purpose             VARCHAR(255),  -- e.g. 'ablation baseline', 'layer-depth sweep L=3'
    status              model_status NOT NULL DEFAULT 'training',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX idx_model_registry_version ON model_registry (model_version);
CREATE INDEX idx_model_registry_status         ON model_registry (status);
CREATE INDEX idx_model_registry_architecture   ON model_registry (architecture);

-- ---------------------------------------------------------------- 6.21 model_evaluation_runs
-- Interval reporting is not optional: repeated snapshots of the same entities
-- are autocorrelated, so i.i.d. methods understate the interval -- ci_method
-- records which estimator was actually used (docs/05_Database_Design.md §6.21).
CREATE TABLE model_evaluation_runs (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- ON UPDATE CASCADE: a model_version is occasionally renamed post-hoc to
    -- match a spec's exact naming (e.g. dropping a d-suffix) without a full
    -- retrain; this lets model_registry's rename propagate instead of being
    -- blocked by dependent rows here.
    model_version  VARCHAR(50) NOT NULL REFERENCES model_registry(model_version) ON UPDATE CASCADE,
    architecture   VARCHAR(100) NOT NULL,
    metric_name    VARCHAR(50) NOT NULL,   -- precision, recall, f1, roc_auc, mae, rmse, mape, calibration_error
    metric_value   NUMERIC(10,6) NOT NULL,
    dataset_split  dataset_split NOT NULL DEFAULT 'test',
    task           VARCHAR(50) NOT NULL,   -- delay, shortage, impact
    depth_l        SMALLINT,               -- populated for layer-depth-sweep rows
    fold_id        SMALLINT,
    train_end_t0   TIMESTAMPTZ,
    test_start_t0  TIMESTAMPTZ,
    test_end_t0    TIMESTAMPTZ,
    ci_lower       NUMERIC(10,6),
    ci_upper       NUMERIC(10,6),
    ci_method      VARCHAR(50),            -- e.g. paired_block_bootstrap, delong
    evaluated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_mer_version_task_metric ON model_evaluation_runs (model_version, task, metric_name);
CREATE INDEX idx_mer_architecture        ON model_evaluation_runs (architecture);
CREATE INDEX idx_mer_fold_test_start     ON model_evaluation_runs (fold_id, test_start_t0);

-- ---------------------------------------------------------------- 6.20 explanation_subgraphs
CREATE TABLE explanation_subgraphs (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    risk_score_id  UUID NOT NULL REFERENCES risk_scores(id),
    nodes          JSONB NOT NULL,  -- [{entity_type, entity_id, contribution_weight}]
    edges          JSONB NOT NULL,  -- [{source, target, edge_type, contribution_weight}]
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX idx_explanation_subgraphs_risk_score ON explanation_subgraphs (risk_score_id);
CREATE INDEX idx_explanation_subgraphs_nodes_gin ON explanation_subgraphs USING GIN (nodes);
