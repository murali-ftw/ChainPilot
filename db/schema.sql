-- ============================================================================
-- ChainPilot / HADES — model-training schema (DDL only)
-- Source of truth: docs/05_Database_Design.md (post temporal-integrity
-- amendment, §§6.1–6.45). This file covers the tables the model pipeline
-- needs; application tables (users, alerts, chat, MCP, audit) are omitted.
-- ============================================================================

-- ---------------------------------------------------------------- Enums
CREATE TYPE shipment_status        AS ENUM ('scheduled', 'in_transit', 'delivered', 'delayed');
CREATE TYPE order_status           AS ENUM ('open', 'fulfilled', 'cancelled', 'at_risk');
CREATE TYPE customer_priority_tier AS ENUM ('strategic', 'standard', 'low');
CREATE TYPE entity_type            AS ENUM ('supplier', 'product', 'order', 'shipment');
CREATE TYPE risk_category          AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE scoring_method         AS ENUM ('gnn_native', 'weighted_formula');

-- ---------------------------------------------------------------- Master tables
CREATE TABLE suppliers (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                VARCHAR(255) NOT NULL,
    country             VARCHAR(100),
    capacity_score      NUMERIC(6,2),
    lead_time_days      INTEGER NOT NULL DEFAULT 0 CHECK (lead_time_days >= 0),
    -- Display only; DEPRECATED for model use. Mutable scalar over all history —
    -- reading it at training time leaks. Pipelines read supplier_temporal_features.
    reliability_history NUMERIC(5,4) CHECK (reliability_history BETWEEN 0 AND 1),
    is_active           BOOLEAN NOT NULL DEFAULT true,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_suppliers_name      ON suppliers (name);
CREATE INDEX idx_suppliers_is_active ON suppliers (is_active);
COMMENT ON COLUMN suppliers.reliability_history IS
    'Display only. Model features come from supplier_temporal_features (leakage contract).';

CREATE TABLE components (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id    UUID NOT NULL REFERENCES suppliers(id),
    name           VARCHAR(255) NOT NULL,
    component_type VARCHAR(100) NOT NULL,
    unit_cost      NUMERIC(12,2),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_components_supplier_id    ON components (supplier_id);
CREATE INDEX idx_components_component_type ON components (component_type);

CREATE TABLE products (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sku        VARCHAR(100) NOT NULL UNIQUE,
    name       VARCHAR(255) NOT NULL,
    category   VARCHAR(100),
    is_active  BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_products_category ON products (category);

CREATE TABLE factories (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                   VARCHAR(255) NOT NULL,
    location               VARCHAR(255),
    capacity_units_per_day INTEGER,          -- site total across all products
    is_active              BOOLEAN NOT NULL DEFAULT true,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_factories_location ON factories (location);

CREATE TABLE warehouses (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name           VARCHAR(255) NOT NULL,
    location       VARCHAR(255),
    capacity_units INTEGER,
    is_active      BOOLEAN NOT NULL DEFAULT true,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_warehouses_location ON warehouses (location);

CREATE TABLE customers (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name           VARCHAR(255) NOT NULL,
    priority_tier  customer_priority_tier NOT NULL DEFAULT 'standard',
    contract_terms JSONB,
    is_active      BOOLEAN NOT NULL DEFAULT true,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_customers_priority_tier ON customers (priority_tier);

-- ---------------------------------------------------------------- Junction tables
CREATE TABLE product_components (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id        UUID NOT NULL REFERENCES products(id),
    component_id      UUID NOT NULL REFERENCES components(id),
    quantity_required INTEGER NOT NULL DEFAULT 1 CHECK (quantity_required > 0),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),  -- BOM validity window start
    deactivated_at    TIMESTAMPTZ,                         -- NULL = still current
    UNIQUE (product_id, component_id, created_at)
);
CREATE INDEX idx_product_components_component_id ON product_components (component_id);
CREATE INDEX idx_product_components_created_at   ON product_components (created_at);

-- Task 3 follow-up experiment (`reports/step5_result_v3.md` addendum):
-- SECONDARY/qualified suppliers for a component, beyond its mandatory
-- `components.supplier_id` primary. Existing for exactly one reason: to give
-- `docs/06_Graph_Database_Design.md` §6.1's co-parent path
-- (Supplier->SUPPLIES->Component->rev_SUPPLIES->Supplier) a real second
-- supplier to reach, which `components.supplier_id` alone (single not-null
-- FK) can never provide. Only ~15-20% of components get a row here — this is
-- a genuinely separate experimental graph, not a retrofit of the base v3
-- dataset (which has zero rows in this table by construction).
CREATE TABLE component_suppliers (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    component_id   UUID NOT NULL REFERENCES components(id),
    supplier_id    UUID NOT NULL REFERENCES suppliers(id),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),  -- qualification validity window start
    deactivated_at TIMESTAMPTZ,                         -- NULL = still qualified
    UNIQUE (component_id, supplier_id, created_at)
);
CREATE INDEX idx_component_suppliers_component_id ON component_suppliers (component_id);
CREATE INDEX idx_component_suppliers_supplier_id  ON component_suppliers (supplier_id);
CREATE INDEX idx_component_suppliers_created_at   ON component_suppliers (created_at);

CREATE TABLE product_factories (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id             UUID NOT NULL REFERENCES products(id),
    factory_id             UUID NOT NULL REFERENCES factories(id),
    is_primary             BOOLEAN NOT NULL DEFAULT false,
    capacity_units_per_day INTEGER CHECK (capacity_units_per_day >= 0),  -- per-product at this site
    qualified_at           TIMESTAMPTZ,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    deactivated_at         TIMESTAMPTZ,
    UNIQUE (product_id, factory_id)
);
CREATE INDEX idx_product_factories_factory_id ON product_factories (factory_id);
CREATE INDEX idx_product_factories_created_at ON product_factories (created_at);
CREATE INDEX idx_product_factories_primary    ON product_factories (is_primary) WHERE is_primary = true;

-- ---------------------------------------------------------------- Operational tables
CREATE TABLE inventory (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id        UUID NOT NULL REFERENCES products(id),
    warehouse_id      UUID NOT NULL REFERENCES warehouses(id),
    stock_level       INTEGER NOT NULL DEFAULT 0 CHECK (stock_level >= 0),
    reorder_threshold INTEGER NOT NULL DEFAULT 0 CHECK (reorder_threshold >= 0),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (product_id, warehouse_id)
);
CREATE INDEX idx_inventory_warehouse_id ON inventory (warehouse_id);
COMMENT ON TABLE inventory IS
    'Current state only — display/operational queries. Model features and the shortage label read inventory_history.';

CREATE TABLE orders (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_number VARCHAR(100) NOT NULL UNIQUE,
    customer_id  UUID NOT NULL REFERENCES customers(id),
    status       order_status NOT NULL DEFAULT 'open',
    placed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    due_at       TIMESTAMPTZ,
    order_value  NUMERIC(14,2),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_orders_customer_id ON orders (customer_id);
CREATE INDEX idx_orders_status      ON orders (status);
CREATE INDEX idx_orders_placed_at   ON orders (placed_at);

CREATE TABLE order_items (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id   UUID NOT NULL REFERENCES orders(id),
    product_id UUID NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()   -- as-of edge filter
);
CREATE INDEX idx_order_items_order_id   ON order_items (order_id);
CREATE INDEX idx_order_items_product_id ON order_items (product_id);
CREATE INDEX idx_order_items_created_at ON order_items (created_at);

CREATE TABLE shipments (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id     UUID REFERENCES suppliers(id),
    factory_id      UUID REFERENCES factories(id),
    warehouse_id    UUID REFERENCES warehouses(id),
    order_id        UUID REFERENCES orders(id),
    carrier         VARCHAR(255),
    -- Current state; display only. As-of status comes from shipment_status_history.
    status          shipment_status NOT NULL DEFAULT 'scheduled',
    eta             TIMESTAMPTZ,
    dispatched_at   TIMESTAMPTZ,       -- source of days_since_dispatch
    delivered_at    TIMESTAMPTZ,       -- LABEL COMPONENT — never a model feature
    origin_location VARCHAR(255),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_shipments_supplier_id   ON shipments (supplier_id);
CREATE INDEX idx_shipments_warehouse_id  ON shipments (warehouse_id);
CREATE INDEX idx_shipments_order_id      ON shipments (order_id);
CREATE INDEX idx_shipments_status        ON shipments (status);
CREATE INDEX idx_shipments_dispatched_at ON shipments (dispatched_at);
COMMENT ON COLUMN shipments.status IS
    'Terminal/current value — display only. The delay label derives from it, so reading it as a feature is leakage.';

-- ---------------------------------------------------------------- History tables (temporal integrity)
CREATE TABLE inventory_history (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    inventory_id      UUID NOT NULL REFERENCES inventory(id),
    product_id        UUID NOT NULL REFERENCES products(id),
    warehouse_id      UUID NOT NULL REFERENCES warehouses(id),
    stock_level       INTEGER NOT NULL CHECK (stock_level >= 0),
    reorder_threshold INTEGER NOT NULL CHECK (reorder_threshold >= 0),
    observed_at       TIMESTAMPTZ NOT NULL,                -- valid time
    recorded_at       TIMESTAMPTZ NOT NULL DEFAULT now(),  -- system time
    source            VARCHAR(50) NOT NULL DEFAULT 'wms_sync'
);
CREATE INDEX idx_inventory_history_asof ON inventory_history (product_id, warehouse_id, observed_at DESC);
CREATE INDEX idx_inventory_history_observed_at ON inventory_history (observed_at);

CREATE TABLE shipment_status_history (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    shipment_id     UUID NOT NULL REFERENCES shipments(id),
    status          shipment_status NOT NULL,
    previous_status shipment_status,                       -- NULL for the initial row
    changed_at      TIMESTAMPTZ NOT NULL,                  -- valid time
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),    -- system time
    source          VARCHAR(50) NOT NULL DEFAULT 'carrier_feed'
);
CREATE INDEX idx_ssh_shipment_changed ON shipment_status_history (shipment_id, changed_at DESC);
CREATE INDEX idx_ssh_status_changed   ON shipment_status_history (status, changed_at);

CREATE TABLE supplier_temporal_features (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id           UUID NOT NULL REFERENCES suppliers(id),
    as_of_date            DATE NOT NULL,
    on_time_rate_30d      NUMERIC(5,4) CHECK (on_time_rate_30d  BETWEEN 0 AND 1),
    on_time_rate_90d      NUMERIC(5,4) CHECK (on_time_rate_90d  BETWEEN 0 AND 1),
    on_time_rate_180d     NUMERIC(5,4) CHECK (on_time_rate_180d BETWEEN 0 AND 1),
    trend_slope           NUMERIC(8,6),
    lateness_variance     NUMERIC(10,4) CHECK (lateness_variance >= 0),
    days_since_last_late  INTEGER CHECK (days_since_last_late >= 0),
    shipment_count_180d   INTEGER NOT NULL DEFAULT 0,
    computed_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    feature_spec_version  VARCHAR(20) NOT NULL,
    UNIQUE (supplier_id, as_of_date, feature_spec_version)
);
CREATE INDEX idx_stf_as_of_date ON supplier_temporal_features (as_of_date);

CREATE TABLE carrier_performance_snapshots (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    carrier              VARCHAR(255) NOT NULL,
    origin_location      VARCHAR(255),
    destination_location VARCHAR(255),
    as_of_date           DATE NOT NULL,
    on_time_rate_90d     NUMERIC(5,4) CHECK (on_time_rate_90d BETWEEN 0 AND 1),
    shipment_count_90d   INTEGER NOT NULL DEFAULT 0,
    computed_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX idx_cps_unique ON carrier_performance_snapshots
    (carrier, COALESCE(origin_location, ''), COALESCE(destination_location, ''), as_of_date);
CREATE INDEX idx_cps_as_of_date ON carrier_performance_snapshots (as_of_date);

-- ---------------------------------------------------------------- Snapshots & labels
CREATE TABLE graph_snapshots (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    t0                   TIMESTAMPTZ NOT NULL,
    horizon_days         SMALLINT NOT NULL DEFAULT 14,
    node_counts          JSONB NOT NULL,
    edge_counts          JSONB NOT NULL,
    label_counts         JSONB NOT NULL,
    feature_spec_version VARCHAR(20) NOT NULL,
    git_commit           VARCHAR(40) NOT NULL,
    construction_seconds NUMERIC(8,2),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (t0, feature_spec_version)
);
CREATE INDEX idx_graph_snapshots_t0 ON graph_snapshots (t0);

CREATE TABLE training_labels (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id  UUID NOT NULL REFERENCES graph_snapshots(id),
    entity_type  entity_type NOT NULL,
    entity_id    UUID NOT NULL,
    task         VARCHAR(50) NOT NULL,       -- 'delay' | 'shortage' | 'impact'
    label        BOOLEAN NOT NULL,
    event_at     TIMESTAMPTZ,                -- positives only; must lie in (t0, t0+horizon]
    label_source VARCHAR(100) NOT NULL,
    -- Populated for task='shortage' only (entity_id there is a product_id, and a
    -- product can be stocked at multiple warehouses with different outcomes in
    -- the same snapshot -- without this column, (snapshot_id, entity_id) alone
    -- could carry conflicting true/false rows for the same product. NULL for
    -- delay/impact, whose entity_id already uniquely identifies one instance.
    warehouse_id UUID REFERENCES warehouses(id)
);
CREATE INDEX idx_training_labels_snapshot ON training_labels (snapshot_id, task, entity_type);
CREATE INDEX idx_training_labels_entity   ON training_labels (entity_type, entity_id);

-- ---------------------------------------------------------------- Model output
CREATE TABLE risk_scores (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type       entity_type NOT NULL,
    entity_id         UUID NOT NULL,
    delay_probability NUMERIC(5,4) CHECK (delay_probability BETWEEN 0 AND 1),
    shortage_risk     NUMERIC(5,4) CHECK (shortage_risk BETWEEN 0 AND 1),
    impact_score      NUMERIC(5,4) NOT NULL CHECK (impact_score BETWEEN 0 AND 1),
    confidence        NUMERIC(5,4) CHECK (confidence BETWEEN 0 AND 1),
    risk_category     risk_category NOT NULL DEFAULT 'low',
    scoring_method    scoring_method NOT NULL DEFAULT 'weighted_formula',
    model_version     VARCHAR(50) NOT NULL,
    snapshot_t0       TIMESTAMPTZ,   -- what moment the prediction is ABOUT
    horizon_days      SMALLINT,
    scored_at         TIMESTAMPTZ NOT NULL DEFAULT now()  -- when inference RAN
);
CREATE INDEX idx_risk_scores_entity        ON risk_scores (entity_type, entity_id, scored_at DESC);
CREATE INDEX idx_risk_scores_impact        ON risk_scores (impact_score);
CREATE INDEX idx_risk_scores_model_version ON risk_scores (model_version);
CREATE INDEX idx_risk_scores_category      ON risk_scores (risk_category);
