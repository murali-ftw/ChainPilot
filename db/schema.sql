-- ============================================================================
-- HADES v4 / ChainPilot -- Rane supply-chain risk dataset
-- PostgreSQL DDL.  Authority: db/dataset_structure.md.
--
-- Every table in dataset_structure.md sections 3-11 is present, with the column
-- names, types and nullability that document specifies.  The CSVs emitted by
-- generate_dataset.py carry exactly these columns in exactly this order.
--
-- ---------------------------------------------------------------------------
-- DEVIATIONS FROM dataset_structure.md, AND WHY
-- ---------------------------------------------------------------------------
-- No column is added to, removed from, or renamed in any CSV.  The deviations
-- below are all constraint-level: the spec declares primary keys that its own
-- effective-dating columns make non-unique.
--
-- D1. Effective-dated tables get the validity start in their key.
--     `part_plant` declares PK (part_id, plant_id) but also carries
--     effective_from/effective_to with the note "Policies change; keep the
--     history".  Those cannot both hold.  PK becomes
--     (part_id, plant_id, effective_from).  Same reasoning applies via UNIQUE
--     indexes to supplier_capacity, supplier_allocation, alternate_sources,
--     part_costs and product_economics -- none of which the spec gives a PK at
--     all.
--
-- D2. Where a natural key contains a nullable column, uniqueness is enforced
--     with a COALESCE expression index rather than a surrogate id.
--     supplier_capacity.part_id is NULL for supplier-level aggregates and
--     supplier_upstream.part_id is NULL for general links; a Postgres PRIMARY
--     KEY cannot contain NULL.  Adding a surrogate id column would change the
--     CSV shape, so the constraint moves to an expression index instead.
--
-- D3. sourcing_channels keeps channel_id as a true PRIMARY KEY: one row per
--     channel, with effective_from/effective_to describing when that channel
--     was live (approval to deactivation).  The alternative reading -- multiple
--     versioned rows per channel_id -- would make po_lines.channel_id
--     unresolvable as a foreign key, and dataset_structure.md section 4 calls
--     the channel "the entity that has a multi-year track record and that the
--     temporal encoder runs over", which requires a stable identity.
--
-- D4. supplier_performance_weekly is specified only as "Same shape, aggregated
--     to supplier level".  Its columns are channel_performance_weekly's with
--     channel_id replaced by supplier_id, plus active_channel_count -- a count
--     of observations, not a prediction.
--
-- D5. calendar."date" is double-quoted throughout: DATE is a Postgres type
--     name.  The CSV header is the unquoted string `date`.
--
-- ---------------------------------------------------------------------------
-- LEAKAGE CONSTRAINTS THE SCHEMA ITSELF ENFORCES
-- ---------------------------------------------------------------------------
-- - recorded_ts >= event_ts as a CHECK on every event table (Rule 2).
-- - effective_from < effective_to as a CHECK wherever both are present.
-- - training_labels CHECK (label_window_start > snapshot_date): the label
--   window is strictly future.  dataset_structure.md section 11.
-- - No table outside model_outputs carries a predicted column.  The columns
--   named in the section 11 warning -- arrival_probability, capacity_strain,
--   network_risk_score, inventory_risk -- appear nowhere in this file.
-- - po_lines has no qty_received and no status; suppliers has no
--   reliability_score.  dataset_structure.md section 15.
-- ============================================================================

BEGIN;

DROP TABLE IF EXISTS
    model_outputs, training_labels, snapshots,
    revealed_capacity_monthly, inventory_position_weekly,
    supplier_performance_weekly, channel_performance_weekly,
    plan_drift_features, part_demand_weekly,
    supplier_financials, logistics_lanes, supplier_audits, supplier_quality_ppm,
    supplier_contracts, product_economics, part_costs,
    expedite_events, line_stop_events, shortage_events,
    inventory_transactions, inventory_snapshots,
    quality_inspections, grn_lines, goods_receipts, asn,
    supplier_acknowledgements, po_line_revisions, po_line_schedules,
    po_lines, purchase_orders,
    production_actual, production_plan,
    tooling, supplier_upstream, alternate_sources, supplier_allocation,
    supplier_capacity, sourcing_channels, part_plant, bom,
    calendar, customers, products, parts, supplier_sites, suppliers,
    plants, business_units,
    dataset_coverage
CASCADE;


-- ============================================================================
-- GROUP A -- Master data (slowly changing).  dataset_structure.md section 3.
-- ============================================================================

CREATE TABLE business_units (
    bu_id                   VARCHAR(20)  PRIMARY KEY,
    bu_name                 VARCHAR(100) NOT NULL,
    bu_code                 VARCHAR(10)  NOT NULL
);

CREATE TABLE plants (
    plant_id                VARCHAR(20)  PRIMARY KEY,
    plant_name              VARCHAR(100) NOT NULL,
    bu_id                   VARCHAR(20)  NOT NULL REFERENCES business_units(bu_id),
    city                    VARCHAR(50),
    state                   VARCHAR(50),
    country                 VARCHAR(50),
    latitude                DECIMAL(9,6),
    longitude               DECIMAL(9,6),
    capacity_units_per_day  INTEGER,
    commissioned_date       DATE,
    is_active               BOOLEAN      NOT NULL
);

-- No reliability_score: dataset_structure.md section 15 rejects any
-- pre-computed whole-history supplier score as outcome leakage.
CREATE TABLE suppliers (
    supplier_id             VARCHAR(20)  PRIMARY KEY,
    supplier_name           VARCHAR(200) NOT NULL,
    supplier_group_id       VARCHAR(20),
    country                 VARCHAR(50),
    state                   VARCHAR(50),
    city                    VARCHAR(50),
    supplier_tier           VARCHAR(10)  NOT NULL,
    supplier_type           VARCHAR(30),
    business_class          VARCHAR(30),
    onboarded_date          DATE,
    is_active               BOOLEAN      NOT NULL,
    msme_flag               BOOLEAN      NOT NULL,
    payment_terms_days      INTEGER,
    CONSTRAINT suppliers_tier_ck  CHECK (supplier_tier IN ('tier1','tier2'))
);
CREATE INDEX suppliers_group_ix ON suppliers(supplier_group_id);
CREATE INDEX suppliers_state_ix ON suppliers(state);

CREATE TABLE supplier_sites (
    site_id                 VARCHAR(20)  PRIMARY KEY,
    supplier_id             VARCHAR(20)  NOT NULL REFERENCES suppliers(supplier_id),
    site_name               VARCHAR(200),
    city                    VARCHAR(50),
    state                   VARCHAR(50),
    country                 VARCHAR(50),
    latitude                DECIMAL(9,6),
    longitude               DECIMAL(9,6),
    is_active               BOOLEAN      NOT NULL
);
CREATE INDEX supplier_sites_supplier_ix ON supplier_sites(supplier_id);

CREATE TABLE parts (
    part_id                 VARCHAR(30)  PRIMARY KEY,
    part_number             VARCHAR(50)  NOT NULL,
    part_name               VARCHAR(200),
    part_category           VARCHAR(50),
    material_type           VARCHAR(50),
    uom                     VARCHAR(10),
    is_critical             BOOLEAN      NOT NULL,
    criticality_reason      VARCHAR(50),
    standard_lead_time_days INTEGER,
    shelf_life_days         INTEGER,
    is_active               BOOLEAN      NOT NULL,
    introduced_date         DATE,
    discontinued_date       DATE
);
CREATE INDEX parts_category_ix ON parts(part_category);
CREATE INDEX parts_critical_ix ON parts(is_critical);

CREATE TABLE customers (
    customer_id             VARCHAR(20)  PRIMARY KEY,
    customer_name           VARCHAR(200) NOT NULL,
    customer_tier           VARCHAR(20),
    penalty_per_unit_inr    DECIMAL(12,2),
    country                 VARCHAR(50),
    is_active               BOOLEAN      NOT NULL
);

CREATE TABLE products (
    product_id              VARCHAR(30)  PRIMARY KEY,
    product_number          VARCHAR(50)  NOT NULL,
    product_name            VARCHAR(200),
    product_family          VARCHAR(50),
    bu_id                   VARCHAR(20)  REFERENCES business_units(bu_id),
    customer_id             VARCHAR(20)  REFERENCES customers(customer_id),
    is_active               BOOLEAN      NOT NULL,
    introduced_date         DATE,
    discontinued_date       DATE
);
CREATE INDEX products_family_ix ON products(product_family);

-- One row per date per plant.  regime_flag is global in incidence but stored
-- per plant because the grain is (date, plant).  chip_shortage is flagged for
-- its full window on every plant; the generator applies its effect only to
-- parts with part_category = 'electronic'.  See generate_dataset.py.
CREATE TABLE calendar (
    "date"                  DATE         NOT NULL,
    plant_id                VARCHAR(20)  NOT NULL REFERENCES plants(plant_id),
    is_working_day          BOOLEAN      NOT NULL,
    shift_count             SMALLINT     NOT NULL,
    is_holiday              BOOLEAN      NOT NULL,
    holiday_name            VARCHAR(100),
    is_shutdown             BOOLEAN      NOT NULL,
    fiscal_year             VARCHAR(10)  NOT NULL,
    fiscal_quarter          SMALLINT     NOT NULL,
    week_of_year            SMALLINT     NOT NULL,
    month_of_year           SMALLINT     NOT NULL,
    regime_flag             VARCHAR(30)  NOT NULL,
    regime_note             VARCHAR(200),
    PRIMARY KEY ("date", plant_id),
    CONSTRAINT calendar_regime_ck CHECK (regime_flag IN
        ('normal','covid','demonetisation','chip_shortage','strike','other')),
    CONSTRAINT calendar_shift_ck  CHECK (shift_count BETWEEN 0 AND 3)
);
CREATE INDEX calendar_regime_ix ON calendar(regime_flag);


-- ============================================================================
-- GROUP B -- Structure and relationships.  dataset_structure.md section 4.
-- ============================================================================

CREATE TABLE bom (
    bom_id                  VARCHAR(30)  PRIMARY KEY,
    product_id              VARCHAR(30)  NOT NULL REFERENCES products(product_id),
    part_id                 VARCHAR(30)  NOT NULL REFERENCES parts(part_id),
    qty_per_unit            DECIMAL(12,4) NOT NULL,
    scrap_factor            DECIMAL(6,4) NOT NULL,
    bom_level               SMALLINT     NOT NULL,
    parent_part_id          VARCHAR(30)  REFERENCES parts(part_id),
    effective_from          DATE         NOT NULL,
    effective_to            DATE,
    is_phantom              BOOLEAN      NOT NULL,
    CONSTRAINT bom_window_ck CHECK (effective_to IS NULL OR effective_from < effective_to),
    CONSTRAINT bom_level_ck  CHECK (bom_level >= 1),
    CONSTRAINT bom_parent_ck CHECK ((bom_level = 1) = (parent_part_id IS NULL))
);
CREATE INDEX bom_product_ix ON bom(product_id, effective_from);
CREATE INDEX bom_part_ix    ON bom(part_id);

-- D1: effective_from joins the key.  The spec's PK (part_id, plant_id) cannot
-- coexist with the policy history its own effective_from/to columns require.
CREATE TABLE part_plant (
    part_id                 VARCHAR(30)  NOT NULL REFERENCES parts(part_id),
    plant_id                VARCHAR(20)  NOT NULL REFERENCES plants(plant_id),
    safety_stock_qty        INTEGER      NOT NULL,
    reorder_point_qty       INTEGER      NOT NULL,
    min_order_qty           INTEGER      NOT NULL,
    lot_size                INTEGER      NOT NULL,
    planning_lead_time_days INTEGER      NOT NULL,
    abc_class               CHAR(1),
    xyz_class               CHAR(1),
    effective_from          DATE         NOT NULL,
    effective_to            DATE,
    PRIMARY KEY (part_id, plant_id, effective_from),
    CONSTRAINT part_plant_window_ck CHECK (effective_to IS NULL OR effective_from < effective_to),
    CONSTRAINT part_plant_abc_ck    CHECK (abc_class IN ('A','B','C')),
    CONSTRAINT part_plant_xyz_ck    CHECK (xyz_class IN ('X','Y','Z'))
);

-- D3: one row per channel; the window is the channel's own lifetime.
CREATE TABLE sourcing_channels (
    channel_id              VARCHAR(40)  PRIMARY KEY,
    supplier_id             VARCHAR(20)  NOT NULL REFERENCES suppliers(supplier_id),
    site_id                 VARCHAR(20)  NOT NULL REFERENCES supplier_sites(site_id),
    part_id                 VARCHAR(30)  NOT NULL REFERENCES parts(part_id),
    plant_id                VARCHAR(20)  NOT NULL REFERENCES plants(plant_id),
    is_approved             BOOLEAN      NOT NULL,
    ppap_date               DATE,
    approval_status         VARCHAR(20)  NOT NULL,
    contracted_lead_time_days INTEGER    NOT NULL,
    transport_mode          VARCHAR(20),
    transport_distance_km   INTEGER,
    effective_from          DATE         NOT NULL,
    effective_to            DATE,
    CONSTRAINT sourcing_channels_status_ck CHECK (approval_status IN
        ('approved','conditional','development','blocked')),
    CONSTRAINT sourcing_channels_window_ck CHECK (effective_to IS NULL OR effective_from < effective_to)
);
CREATE UNIQUE INDEX sourcing_channels_natural_ux
    ON sourcing_channels(supplier_id, part_id, plant_id);
CREATE INDEX sourcing_channels_part_plant_ix ON sourcing_channels(part_id, plant_id);
CREATE INDEX sourcing_channels_supplier_ix   ON sourcing_channels(supplier_id);

-- D2: part_id is NULL for supplier-level aggregates, so uniqueness is an
-- expression index rather than a PK.
CREATE TABLE supplier_capacity (
    supplier_id             VARCHAR(20)  NOT NULL REFERENCES suppliers(supplier_id),
    part_id                 VARCHAR(30)  REFERENCES parts(part_id),
    capacity_qty_per_month  INTEGER      NOT NULL,
    capacity_basis          VARCHAR(30)  NOT NULL,
    shift_basis             SMALLINT,
    effective_from          DATE         NOT NULL,
    effective_to            DATE,
    recorded_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT supplier_capacity_basis_ck  CHECK (capacity_basis IN
        ('contracted','declared','audited','estimated')),
    CONSTRAINT supplier_capacity_window_ck CHECK (effective_to IS NULL OR effective_from < effective_to)
);
CREATE UNIQUE INDEX supplier_capacity_ux
    ON supplier_capacity(supplier_id, COALESCE(part_id, ''), effective_from);

CREATE TABLE supplier_allocation (
    part_id                 VARCHAR(30)  NOT NULL REFERENCES parts(part_id),
    plant_id                VARCHAR(20)  NOT NULL REFERENCES plants(plant_id),
    supplier_id             VARCHAR(20)  NOT NULL REFERENCES suppliers(supplier_id),
    allocation_pct          DECIMAL(5,2) NOT NULL,
    effective_from          DATE         NOT NULL,
    effective_to            DATE,
    changed_by              VARCHAR(50),
    change_reason           VARCHAR(100),
    recorded_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT supplier_allocation_pct_ck    CHECK (allocation_pct BETWEEN 0 AND 100),
    CONSTRAINT supplier_allocation_window_ck CHECK (effective_to IS NULL OR effective_from < effective_to)
);
CREATE UNIQUE INDEX supplier_allocation_ux
    ON supplier_allocation(part_id, plant_id, supplier_id, effective_from);

CREATE TABLE alternate_sources (
    part_id                 VARCHAR(30)  NOT NULL REFERENCES parts(part_id),
    supplier_id             VARCHAR(20)  NOT NULL REFERENCES suppliers(supplier_id),
    plant_id                VARCHAR(20)  NOT NULL REFERENCES plants(plant_id),
    qualification_status    VARCHAR(20)  NOT NULL,
    qualification_lead_days INTEGER,
    ramp_rate_pct_per_month DECIMAL(5,2),
    cost_delta_pct          DECIMAL(6,2),
    effective_from          DATE         NOT NULL,
    effective_to            DATE,
    CONSTRAINT alternate_sources_status_ck CHECK (qualification_status IN
        ('qualified','in_progress','potential')),
    CONSTRAINT alternate_sources_window_ck CHECK (effective_to IS NULL OR effective_from < effective_to)
);
CREATE UNIQUE INDEX alternate_sources_ux
    ON alternate_sources(part_id, supplier_id, plant_id, effective_from);

-- D2: part_id NULL means the link is general rather than part-specific.
CREATE TABLE supplier_upstream (
    supplier_id             VARCHAR(20)  NOT NULL REFERENCES suppliers(supplier_id),
    upstream_supplier_id    VARCHAR(20)  NOT NULL REFERENCES suppliers(supplier_id),
    part_id                 VARCHAR(30)  REFERENCES parts(part_id),
    dependency_type         VARCHAR(30)  NOT NULL,
    criticality             VARCHAR(10)  NOT NULL,
    is_sole_source          BOOLEAN      NOT NULL,
    confidence              VARCHAR(10)  NOT NULL,
    known_since             DATE,
    CONSTRAINT supplier_upstream_type_ck  CHECK (dependency_type IN
        ('raw_material','sub_assembly','process','logistics')),
    CONSTRAINT supplier_upstream_crit_ck  CHECK (criticality IN ('high','medium','low')),
    CONSTRAINT supplier_upstream_conf_ck  CHECK (confidence  IN ('confirmed','reported','inferred')),
    CONSTRAINT supplier_upstream_self_ck  CHECK (supplier_id <> upstream_supplier_id)
);
CREATE UNIQUE INDEX supplier_upstream_ux
    ON supplier_upstream(supplier_id, upstream_supplier_id, COALESCE(part_id, ''));
CREATE INDEX supplier_upstream_parent_ix ON supplier_upstream(upstream_supplier_id);

CREATE TABLE tooling (
    tool_id                 VARCHAR(30)  PRIMARY KEY,
    part_id                 VARCHAR(30)  NOT NULL REFERENCES parts(part_id),
    supplier_id             VARCHAR(20)  NOT NULL REFERENCES suppliers(supplier_id),
    owned_by                VARCHAR(20)  NOT NULL,
    is_transferable         BOOLEAN      NOT NULL,
    transfer_lead_days      INTEGER,
    duplicate_exists        BOOLEAN      NOT NULL,
    CONSTRAINT tooling_owner_ck CHECK (owned_by IN ('rane','supplier'))
);
CREATE INDEX tooling_part_ix ON tooling(part_id);


-- ============================================================================
-- GROUP C -- Planning.  dataset_structure.md section 5.
-- horizon_days = target_period - plan_date is derived, never stored.
-- ============================================================================

CREATE TABLE production_plan (
    plan_id                 VARCHAR(30)  PRIMARY KEY,
    plan_version            INTEGER      NOT NULL,
    plan_date               DATE         NOT NULL,
    product_id              VARCHAR(30)  NOT NULL REFERENCES products(product_id),
    plant_id                VARCHAR(20)  NOT NULL REFERENCES plants(plant_id),
    target_period           DATE         NOT NULL,
    period_grain            VARCHAR(10)  NOT NULL,
    planned_qty             INTEGER      NOT NULL,
    plan_type               VARCHAR(20)  NOT NULL,
    is_firm                 BOOLEAN      NOT NULL,
    recorded_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT production_plan_grain_ck CHECK (period_grain IN ('month','week')),
    CONSTRAINT production_plan_type_ck  CHECK (plan_type IN ('annual','quarterly','rolling','firm')),
    CONSTRAINT production_plan_qty_ck   CHECK (planned_qty >= 0)
);
CREATE UNIQUE INDEX production_plan_version_ux
    ON production_plan(product_id, plant_id, target_period, plan_version);
CREATE INDEX production_plan_target_ix ON production_plan(target_period, plan_date);

CREATE TABLE production_actual (
    product_id              VARCHAR(30)  NOT NULL REFERENCES products(product_id),
    plant_id                VARCHAR(20)  NOT NULL REFERENCES plants(plant_id),
    period                  DATE         NOT NULL,
    period_grain            VARCHAR(10)  NOT NULL,
    actual_qty              INTEGER      NOT NULL,
    scrapped_qty            INTEGER      NOT NULL,
    rework_qty              INTEGER      NOT NULL,
    event_ts                TIMESTAMP    NOT NULL,
    recorded_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT production_actual_grain_ck CHECK (period_grain IN ('month','week')),
    CONSTRAINT production_actual_lag_ck   CHECK (recorded_ts >= event_ts)
);
CREATE UNIQUE INDEX production_actual_ux
    ON production_actual(product_id, plant_id, period, period_grain);


-- ============================================================================
-- GROUP D -- Procurement transactions.  dataset_structure.md section 6.
-- ============================================================================

CREATE TABLE purchase_orders (
    po_id                   VARCHAR(30)  PRIMARY KEY,
    po_number               VARCHAR(50)  NOT NULL,
    supplier_id             VARCHAR(20)  NOT NULL REFERENCES suppliers(supplier_id),
    site_id                 VARCHAR(20)  NOT NULL REFERENCES supplier_sites(site_id),
    plant_id                VARCHAR(20)  NOT NULL REFERENCES plants(plant_id),
    po_type                 VARCHAR(20)  NOT NULL,
    currency                CHAR(3)      NOT NULL,
    incoterm                VARCHAR(10),
    created_ts              TIMESTAMP    NOT NULL,
    recorded_ts             TIMESTAMP    NOT NULL,
    status                  VARCHAR(20),
    CONSTRAINT purchase_orders_type_ck CHECK (po_type IN
        ('standard','schedule_agreement','blanket','emergency')),
    CONSTRAINT purchase_orders_lag_ck  CHECK (recorded_ts >= created_ts)
);
CREATE INDEX purchase_orders_supplier_ix ON purchase_orders(supplier_id, created_ts);
CREATE INDEX purchase_orders_plant_ix    ON purchase_orders(plant_id, created_ts);

-- No qty_received and no status: dataset_structure.md section 15 -- the first
-- is the fill-rate label, the second is terminal-state leakage reconstructed
-- from po_line_revisions.
CREATE TABLE po_lines (
    po_line_id              VARCHAR(30)  PRIMARY KEY,
    po_id                   VARCHAR(30)  NOT NULL REFERENCES purchase_orders(po_id),
    line_number             SMALLINT     NOT NULL,
    part_id                 VARCHAR(30)  NOT NULL REFERENCES parts(part_id),
    channel_id              VARCHAR(40)  NOT NULL REFERENCES sourcing_channels(channel_id),
    qty_ordered             INTEGER      NOT NULL,
    unit_price              DECIMAL(12,4) NOT NULL,
    original_promise_date   DATE         NOT NULL,
    current_promise_date    DATE         NOT NULL,
    requested_date          DATE         NOT NULL,
    created_ts              TIMESTAMP    NOT NULL,
    recorded_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT po_lines_qty_ck CHECK (qty_ordered > 0),
    CONSTRAINT po_lines_lag_ck CHECK (recorded_ts >= created_ts)
);
CREATE UNIQUE INDEX po_lines_po_line_ux ON po_lines(po_id, line_number);
CREATE INDEX po_lines_channel_ix ON po_lines(channel_id, created_ts);
CREATE INDEX po_lines_part_ix    ON po_lines(part_id, created_ts);
CREATE INDEX po_lines_promise_ix ON po_lines(original_promise_date);

CREATE TABLE po_line_schedules (
    schedule_id             VARCHAR(30)  PRIMARY KEY,
    po_line_id              VARCHAR(30)  NOT NULL REFERENCES po_lines(po_line_id),
    schedule_date           DATE         NOT NULL,
    scheduled_qty           INTEGER      NOT NULL,
    schedule_version        INTEGER      NOT NULL,
    released_ts             TIMESTAMP    NOT NULL,
    recorded_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT po_line_schedules_lag_ck CHECK (recorded_ts >= released_ts)
);
CREATE INDEX po_line_schedules_line_ix ON po_line_schedules(po_line_id, schedule_version);

CREATE TABLE po_line_revisions (
    revision_id             VARCHAR(30)  PRIMARY KEY,
    po_line_id              VARCHAR(30)  NOT NULL REFERENCES po_lines(po_line_id),
    revision_number         SMALLINT     NOT NULL,
    field_changed           VARCHAR(30)  NOT NULL,
    old_value               VARCHAR(50),
    new_value               VARCHAR(50),
    initiated_by            VARCHAR(20)  NOT NULL,
    reason_code             VARCHAR(50),
    event_ts                TIMESTAMP    NOT NULL,
    recorded_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT po_line_revisions_field_ck CHECK (field_changed IN
        ('promise_date','qty','price','status')),
    CONSTRAINT po_line_revisions_by_ck    CHECK (initiated_by IN ('supplier','buyer','system')),
    CONSTRAINT po_line_revisions_lag_ck   CHECK (recorded_ts >= event_ts)
);
CREATE INDEX po_line_revisions_line_ix ON po_line_revisions(po_line_id, recorded_ts);

CREATE TABLE supplier_acknowledgements (
    ack_id                  VARCHAR(30)  PRIMARY KEY,
    po_line_id              VARCHAR(30)  NOT NULL REFERENCES po_lines(po_line_id),
    ack_qty                 INTEGER      NOT NULL,
    ack_date                DATE         NOT NULL,
    ack_status              VARCHAR(20)  NOT NULL,
    event_ts                TIMESTAMP    NOT NULL,
    recorded_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT supplier_ack_status_ck CHECK (ack_status IN
        ('full','partial','rejected','date_change')),
    CONSTRAINT supplier_ack_lag_ck    CHECK (recorded_ts >= event_ts)
);
CREATE INDEX supplier_ack_line_ix ON supplier_acknowledgements(po_line_id);

CREATE TABLE asn (
    asn_id                  VARCHAR(30)  PRIMARY KEY,
    po_line_id              VARCHAR(30)  NOT NULL REFERENCES po_lines(po_line_id),
    dispatched_qty          INTEGER      NOT NULL,
    dispatch_ts             TIMESTAMP    NOT NULL,
    expected_arrival_date   DATE         NOT NULL,
    transport_mode          VARCHAR(20),
    vehicle_id              VARCHAR(30),
    recorded_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT asn_lag_ck CHECK (recorded_ts >= dispatch_ts)
);
CREATE INDEX asn_line_ix ON asn(po_line_id);

CREATE TABLE goods_receipts (
    grn_id                  VARCHAR(30)  PRIMARY KEY,
    grn_number              VARCHAR(50)  NOT NULL,
    supplier_id             VARCHAR(20)  NOT NULL REFERENCES suppliers(supplier_id),
    plant_id                VARCHAR(20)  NOT NULL REFERENCES plants(plant_id),
    receipt_ts              TIMESTAMP    NOT NULL,
    recorded_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT goods_receipts_lag_ck CHECK (recorded_ts >= receipt_ts)
);
CREATE INDEX goods_receipts_plant_ix ON goods_receipts(plant_id, receipt_ts);

-- Partial receipts are separate rows.  qty_received is THIS receipt only.
-- dataset_structure.md section 6: summing into one row makes arrival timing
-- unrecoverable and collapses use case 4.
CREATE TABLE grn_lines (
    grn_line_id             VARCHAR(30)  PRIMARY KEY,
    grn_id                  VARCHAR(30)  NOT NULL REFERENCES goods_receipts(grn_id),
    po_line_id              VARCHAR(30)  NOT NULL REFERENCES po_lines(po_line_id),
    part_id                 VARCHAR(30)  NOT NULL REFERENCES parts(part_id),
    qty_received            INTEGER      NOT NULL,
    receipt_sequence        SMALLINT     NOT NULL,
    is_final_receipt        BOOLEAN      NOT NULL,
    event_ts                TIMESTAMP    NOT NULL,
    recorded_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT grn_lines_qty_ck CHECK (qty_received > 0),
    CONSTRAINT grn_lines_seq_ck CHECK (receipt_sequence >= 1),
    CONSTRAINT grn_lines_lag_ck CHECK (recorded_ts >= event_ts)
);
CREATE UNIQUE INDEX grn_lines_seq_ux ON grn_lines(po_line_id, receipt_sequence);
CREATE INDEX grn_lines_event_ix ON grn_lines(event_ts);

CREATE TABLE quality_inspections (
    inspection_id           VARCHAR(30)  PRIMARY KEY,
    grn_line_id             VARCHAR(30)  NOT NULL REFERENCES grn_lines(grn_line_id),
    qty_inspected           INTEGER      NOT NULL,
    qty_accepted            INTEGER      NOT NULL,
    qty_rejected            INTEGER      NOT NULL,
    qty_deviation_accepted  INTEGER      NOT NULL,
    rejection_reason        VARCHAR(100),
    event_ts                TIMESTAMP    NOT NULL,
    recorded_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT quality_inspections_bal_ck CHECK
        (qty_accepted + qty_rejected = qty_inspected),
    CONSTRAINT quality_inspections_lag_ck CHECK (recorded_ts >= event_ts)
);
CREATE UNIQUE INDEX quality_inspections_line_ux ON quality_inspections(grn_line_id);


-- ============================================================================
-- GROUP E -- Inventory.  dataset_structure.md section 7.
-- ============================================================================

CREATE TABLE inventory_snapshots (
    part_id                 VARCHAR(30)  NOT NULL REFERENCES parts(part_id),
    plant_id                VARCHAR(20)  NOT NULL REFERENCES plants(plant_id),
    snapshot_date           DATE         NOT NULL,
    qty_on_hand             INTEGER      NOT NULL,
    qty_blocked             INTEGER      NOT NULL,
    qty_in_transit          INTEGER      NOT NULL,
    qty_reserved            INTEGER      NOT NULL,
    qty_available           INTEGER      NOT NULL,
    event_ts                TIMESTAMP    NOT NULL,
    recorded_ts             TIMESTAMP    NOT NULL,
    PRIMARY KEY (part_id, plant_id, snapshot_date),
    CONSTRAINT inventory_snapshots_avail_ck CHECK
        (qty_available = qty_on_hand - qty_blocked - qty_reserved),
    CONSTRAINT inventory_snapshots_lag_ck   CHECK (recorded_ts >= event_ts)
);
CREATE INDEX inventory_snapshots_date_ix ON inventory_snapshots(snapshot_date);

CREATE TABLE inventory_transactions (
    txn_id                  VARCHAR(30)  PRIMARY KEY,
    part_id                 VARCHAR(30)  NOT NULL REFERENCES parts(part_id),
    plant_id                VARCHAR(20)  NOT NULL REFERENCES plants(plant_id),
    txn_type                VARCHAR(30)  NOT NULL,
    qty                     INTEGER      NOT NULL,
    reference_id            VARCHAR(30),
    from_plant_id           VARCHAR(20)  REFERENCES plants(plant_id),
    event_ts                TIMESTAMP    NOT NULL,
    recorded_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT inventory_transactions_type_ck CHECK (txn_type IN
        ('receipt','issue_to_production','scrap','return',
         'transfer_in','transfer_out','adjustment')),
    CONSTRAINT inventory_transactions_lag_ck  CHECK (recorded_ts >= event_ts)
);
CREATE INDEX inventory_transactions_part_ix ON inventory_transactions(part_id, plant_id, event_ts);
CREATE INDEX inventory_transactions_type_ix ON inventory_transactions(txn_type);


-- ============================================================================
-- GROUP F -- Outcomes (labels).  dataset_structure.md section 8.
-- ============================================================================

CREATE TABLE shortage_events (
    shortage_id             VARCHAR(30)  PRIMARY KEY,
    part_id                 VARCHAR(30)  NOT NULL REFERENCES parts(part_id),
    plant_id                VARCHAR(20)  NOT NULL REFERENCES plants(plant_id),
    shortage_qty            INTEGER      NOT NULL,
    shortage_start_ts       TIMESTAMP    NOT NULL,
    shortage_end_ts         TIMESTAMP,
    root_cause              VARCHAR(50)  NOT NULL,
    responsible_supplier_id VARCHAR(20)  REFERENCES suppliers(supplier_id),
    resolution_action       VARCHAR(50),
    event_ts                TIMESTAMP    NOT NULL,
    recorded_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT shortage_events_cause_ck CHECK (root_cause IN
        ('supplier_delay','supplier_short','quality_reject','demand_spike',
         'plan_change','logistics')),
    CONSTRAINT shortage_events_action_ck CHECK (resolution_action IS NULL OR resolution_action IN
        ('expedite','alternate_source','substitute','reschedule')),
    CONSTRAINT shortage_events_span_ck  CHECK (shortage_end_ts IS NULL OR shortage_end_ts >= shortage_start_ts),
    CONSTRAINT shortage_events_lag_ck   CHECK (recorded_ts >= event_ts)
);
CREATE INDEX shortage_events_part_ix ON shortage_events(part_id, plant_id, event_ts);

CREATE TABLE line_stop_events (
    stop_id                 VARCHAR(30)  PRIMARY KEY,
    plant_id                VARCHAR(20)  NOT NULL REFERENCES plants(plant_id),
    line_id                 VARCHAR(30),
    product_id              VARCHAR(30)  REFERENCES products(product_id),
    part_id                 VARCHAR(30)  REFERENCES parts(part_id),
    stop_start_ts           TIMESTAMP    NOT NULL,
    stop_end_ts             TIMESTAMP    NOT NULL,
    duration_minutes        INTEGER      NOT NULL,
    units_lost              INTEGER      NOT NULL,
    cause_category          VARCHAR(50)  NOT NULL,
    responsible_supplier_id VARCHAR(20)  REFERENCES suppliers(supplier_id),
    event_ts                TIMESTAMP    NOT NULL,
    recorded_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT line_stop_cause_ck CHECK (cause_category IN
        ('material','machine','manpower','quality','power')),
    CONSTRAINT line_stop_span_ck  CHECK (stop_end_ts >= stop_start_ts),
    CONSTRAINT line_stop_lag_ck   CHECK (recorded_ts >= event_ts)
);
CREATE INDEX line_stop_plant_ix ON line_stop_events(plant_id, event_ts);

CREATE TABLE expedite_events (
    expedite_id             VARCHAR(30)  PRIMARY KEY,
    part_id                 VARCHAR(30)  NOT NULL REFERENCES parts(part_id),
    plant_id                VARCHAR(20)  NOT NULL REFERENCES plants(plant_id),
    supplier_id             VARCHAR(20)  REFERENCES suppliers(supplier_id),
    expedite_type           VARCHAR(30)  NOT NULL,
    qty_expedited           INTEGER      NOT NULL,
    premium_cost_inr        DECIMAL(12,2) NOT NULL,
    triggered_by            VARCHAR(50),
    event_ts                TIMESTAMP    NOT NULL,
    recorded_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT expedite_type_ck CHECK (expedite_type IN
        ('air_freight','dedicated_truck','supplier_overtime','inter_plant_transfer')),
    CONSTRAINT expedite_lag_ck  CHECK (recorded_ts >= event_ts)
);
CREATE INDEX expedite_part_ix ON expedite_events(part_id, plant_id, event_ts);


-- ============================================================================
-- GROUP G -- Cost and commercial.  dataset_structure.md section 9.
-- ============================================================================

CREATE TABLE part_costs (
    part_id                 VARCHAR(30)  NOT NULL REFERENCES parts(part_id),
    supplier_id             VARCHAR(20)  NOT NULL REFERENCES suppliers(supplier_id),
    unit_cost_inr           DECIMAL(12,4) NOT NULL,
    freight_cost_inr        DECIMAL(12,4) NOT NULL,
    effective_from          DATE         NOT NULL,
    effective_to            DATE,
    CONSTRAINT part_costs_window_ck CHECK (effective_to IS NULL OR effective_from < effective_to)
);
CREATE UNIQUE INDEX part_costs_ux ON part_costs(part_id, supplier_id, effective_from);

CREATE TABLE product_economics (
    product_id              VARCHAR(30)  NOT NULL REFERENCES products(product_id),
    selling_price_inr       DECIMAL(12,2) NOT NULL,
    contribution_margin_inr DECIMAL(12,2) NOT NULL,
    effective_from          DATE         NOT NULL,
    effective_to            DATE,
    CONSTRAINT product_economics_window_ck CHECK (effective_to IS NULL OR effective_from < effective_to)
);
CREATE UNIQUE INDEX product_economics_ux ON product_economics(product_id, effective_from);

CREATE TABLE supplier_contracts (
    contract_id             VARCHAR(30)  PRIMARY KEY,
    supplier_id             VARCHAR(20)  NOT NULL REFERENCES suppliers(supplier_id),
    part_id                 VARCHAR(30)  NOT NULL REFERENCES parts(part_id),
    min_volume_commitment   INTEGER,
    max_volume_cap          INTEGER,
    moq                     INTEGER      NOT NULL,
    lot_size                INTEGER      NOT NULL,
    price_break_qty         INTEGER,
    penalty_clause_inr      DECIMAL(12,2),
    valid_from              DATE         NOT NULL,
    valid_to                DATE,
    CONSTRAINT supplier_contracts_window_ck CHECK (valid_to IS NULL OR valid_from < valid_to)
);
CREATE INDEX supplier_contracts_sp_ix ON supplier_contracts(supplier_id, part_id);


-- ============================================================================
-- GROUP H -- Phase 2 / optional.  dataset_structure.md section 10.
-- ============================================================================

CREATE TABLE supplier_quality_ppm (
    supplier_id             VARCHAR(20)  NOT NULL REFERENCES suppliers(supplier_id),
    part_id                 VARCHAR(30)  NOT NULL REFERENCES parts(part_id),
    period                  DATE         NOT NULL,
    parts_supplied          INTEGER      NOT NULL,
    parts_rejected          INTEGER      NOT NULL,
    ppm                     INTEGER      NOT NULL,
    recorded_ts             TIMESTAMP    NOT NULL
);
CREATE UNIQUE INDEX supplier_quality_ppm_ux ON supplier_quality_ppm(supplier_id, part_id, period);

CREATE TABLE supplier_audits (
    audit_id                VARCHAR(30)  PRIMARY KEY,
    supplier_id             VARCHAR(20)  NOT NULL REFERENCES suppliers(supplier_id),
    audit_date              DATE         NOT NULL,
    audit_type              VARCHAR(30)  NOT NULL,
    score                   DECIMAL(5,2) NOT NULL,
    major_ncs               SMALLINT     NOT NULL,
    minor_ncs               SMALLINT     NOT NULL,
    recorded_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT supplier_audits_type_ck CHECK (audit_type IN ('process','system','product'))
);
CREATE INDEX supplier_audits_supplier_ix ON supplier_audits(supplier_id, audit_date);

-- via_checkpoint is what makes lanes a correlated-failure channel: two
-- suppliers routing through one port fail in the same week.
CREATE TABLE logistics_lanes (
    lane_id                 VARCHAR(30)  PRIMARY KEY,
    origin_site_id          VARCHAR(20)  NOT NULL REFERENCES supplier_sites(site_id),
    dest_plant_id           VARCHAR(20)  NOT NULL REFERENCES plants(plant_id),
    transport_mode          VARCHAR(20)  NOT NULL,
    distance_km             INTEGER,
    standard_transit_days   DECIMAL(5,2) NOT NULL,
    carrier_id              VARCHAR(20),
    via_checkpoint          VARCHAR(50)
);
CREATE UNIQUE INDEX logistics_lanes_ux ON logistics_lanes(origin_site_id, dest_plant_id, transport_mode);
CREATE INDEX logistics_lanes_checkpoint_ix ON logistics_lanes(via_checkpoint);

CREATE TABLE supplier_financials (
    supplier_id             VARCHAR(20)  NOT NULL REFERENCES suppliers(supplier_id),
    period                  DATE         NOT NULL,
    revenue_inr             DECIMAL(15,2),
    rane_share_of_revenue_pct DECIMAL(5,2),
    credit_rating           VARCHAR(10),
    days_payable_outstanding INTEGER
);
CREATE UNIQUE INDEX supplier_financials_ux ON supplier_financials(supplier_id, period);


-- ============================================================================
-- GROUP I -- [DERIVED] feature stores.  dataset_structure.md section 11.
--
-- Built by derive/build_features.py from the raw CSVs above, never generated
-- directly.  Every column is an observation or a deterministic function of
-- observations.  No column here is a model output: arrival_probability,
-- capacity_strain, network_risk_score and inventory_risk live in
-- model_outputs, which the generator does not populate.
-- ============================================================================

CREATE TABLE part_demand_weekly (
    part_id                 VARCHAR(30)  NOT NULL,
    plant_id                VARCHAR(20)  NOT NULL,
    week_start              DATE         NOT NULL,
    as_of_date              DATE         NOT NULL,
    gross_requirement_p50   INTEGER      NOT NULL,
    gross_requirement_p90   INTEGER      NOT NULL,
    horizon_days            INTEGER      NOT NULL,
    driving_products        TEXT
);
CREATE UNIQUE INDEX part_demand_weekly_ux ON part_demand_weekly(part_id, plant_id, week_start, as_of_date);

CREATE TABLE plan_drift_features (
    product_id              VARCHAR(30)  NOT NULL,
    plant_id                VARCHAR(20)  NOT NULL,
    target_period           DATE         NOT NULL,
    plan_date               DATE         NOT NULL,
    horizon_days            INTEGER      NOT NULL,
    plan_version            INTEGER      NOT NULL,
    original_plan_qty       INTEGER,
    latest_plan_qty         INTEGER,
    actual_qty              INTEGER,
    drift_ratio             DECIMAL(8,4),
    plan_revision_count     SMALLINT,
    plan_volatility         DECIMAL(8,4),
    is_firm                 BOOLEAN,
    regime_flag             VARCHAR(30),
    recorded_ts             TIMESTAMP    NOT NULL
);
CREATE UNIQUE INDEX plan_drift_features_ux
    ON plan_drift_features(product_id, plant_id, target_period, plan_date, plan_version);

-- D6. The rolling columns are named `_lastN`, not `_Nw`.  They average the
--     last N weeks in which the channel ORDERED, not the last N calendar
--     weeks.  With ~90% of channel-weeks empty a calendar window would hold
--     one or two observations and be mostly noise, so the window counts
--     active weeks -- but at this cadence `fill_rate_last13` spans a median
--     of 85 calendar weeks, and the old `_13w` name overstated its recency
--     by about 6.5x.  active_weeks_in_52 carries the calendar scale.
--
-- D7. staleness_days is GONE, replaced by reporting_lag_days +
--     weeks_since_last_activity.  It meant posting lag on active weeks and
--     inactivity on empty ones -- opposite meanings in one column, which no
--     learned gate can separate.  Pooled it read P50 12 / P90 838 days; the
--     genuine posting lag is P50 3 / P90 5, and the tail was entirely
--     inactivity.  Renaming rather than replacing would have let analysis
--     written against the old meaning survive the fix silently.
CREATE TABLE channel_performance_weekly (
    channel_id                VARCHAR(40)  NOT NULL,
    week_start                DATE         NOT NULL,
    qty_ordered               INTEGER      NOT NULL,
    qty_received              INTEGER      NOT NULL,
    is_active_week            BOOLEAN      NOT NULL,
    fill_rate                 DECIMAL(6,4),
    fill_rate_last4           DECIMAL(6,4),
    fill_rate_last13          DECIMAL(6,4),
    fill_rate_last52          DECIMAL(6,4),
    lead_time_actual_days     DECIMAL(6,2),
    lead_time_ratio           DECIMAL(6,4),
    otd_rate_last13           DECIMAL(6,4),
    ack_gap_ratio             DECIMAL(6,4),
    revision_count            SMALLINT     NOT NULL,
    load_ratio                DECIMAL(6,4),
    active_weeks_in_52        SMALLINT     NOT NULL,
    days_since_last_short     INTEGER,
    reporting_lag_days        DECIMAL(6,4),
    weeks_since_last_activity INTEGER,
    weeks_since_last_receipt  INTEGER,
    PRIMARY KEY (channel_id, week_start)
);

-- D4: channel_performance_weekly's shape at supplier grain, plus a count of
-- contributing channels.
CREATE TABLE supplier_performance_weekly (
    supplier_id               VARCHAR(20)  NOT NULL,
    week_start                DATE         NOT NULL,
    qty_ordered               INTEGER      NOT NULL,
    qty_received              INTEGER      NOT NULL,
    is_active_week            BOOLEAN      NOT NULL,
    fill_rate                 DECIMAL(6,4),
    fill_rate_last4           DECIMAL(6,4),
    fill_rate_last13          DECIMAL(6,4),
    fill_rate_last52          DECIMAL(6,4),
    lead_time_actual_days     DECIMAL(6,2),
    lead_time_ratio           DECIMAL(6,4),
    otd_rate_last13           DECIMAL(6,4),
    ack_gap_ratio             DECIMAL(6,4),
    revision_count            SMALLINT     NOT NULL,
    load_ratio                DECIMAL(6,4),
    active_weeks_in_52        SMALLINT     NOT NULL,
    days_since_last_short     INTEGER,
    -- Always NULL at supplier grain: averaging posting lag across plants
    -- would hide the one plant that posts late, which is the whole point of
    -- the plant_data_discipline mechanism.  Read it from the channel table.
    reporting_lag_days        DECIMAL(6,4),
    weeks_since_last_activity INTEGER,
    weeks_since_last_receipt  INTEGER,
    active_channel_count      SMALLINT     NOT NULL,
    PRIMARY KEY (supplier_id, week_start)
);

CREATE TABLE inventory_position_weekly (
    part_id                 VARCHAR(30)  NOT NULL,
    plant_id                VARCHAR(20)  NOT NULL,
    week_start              DATE         NOT NULL,
    as_of_date              DATE         NOT NULL,
    qty_on_hand             INTEGER      NOT NULL,
    qty_blocked             INTEGER      NOT NULL,
    qty_reserved            INTEGER      NOT NULL,
    qty_in_transit          INTEGER      NOT NULL,
    qty_available           INTEGER      NOT NULL,
    open_po_qty             INTEGER      NOT NULL,
    safety_stock_qty        INTEGER      NOT NULL,
    days_of_supply          DECIMAL(6,2),
    consumption_4w          INTEGER      NOT NULL,
    consumption_13w         INTEGER      NOT NULL,
    inter_plant_in_4w       INTEGER      NOT NULL,
    inter_plant_out_4w      INTEGER      NOT NULL,
    staleness_days          INTEGER,
    recorded_ts             TIMESTAMP    NOT NULL
);
CREATE UNIQUE INDEX inventory_position_weekly_ux
    ON inventory_position_weekly(part_id, plant_id, week_start, as_of_date);

-- No capacity_strain column: strain at a future horizon is the UC2 target and
-- lives in training_labels and model_outputs.  dataset_structure.md section 11.
CREATE TABLE revealed_capacity_monthly (
    supplier_id             VARCHAR(20)  NOT NULL,
    part_id                 VARCHAR(30)  NOT NULL,
    month                   DATE         NOT NULL,
    max_delivered_12m       INTEGER,
    p95_delivered_12m       INTEGER,
    constrained_month_flag  BOOLEAN      NOT NULL,
    declared_capacity_qty   INTEGER,
    capacity_basis          VARCHAR(30),
    revealed_capacity_est   INTEGER,
    capacity_utilisation_observed DECIMAL(6,4),
    evidence_strength       DECIMAL(4,2),
    PRIMARY KEY (supplier_id, part_id, month)
);

CREATE TABLE snapshots (
    snapshot_id             VARCHAR(30)  PRIMARY KEY,
    as_of_ts                TIMESTAMP    NOT NULL,
    data_cutoff_ts          TIMESTAMP    NOT NULL,
    horizon_days            INTEGER      NOT NULL,
    node_counts             TEXT,
    edge_counts             TEXT,
    feature_spec_version    VARCHAR(10)  NOT NULL,
    dataset_version         VARCHAR(20)  NOT NULL,
    label_version           VARCHAR(10)  NOT NULL,
    code_commit             VARCHAR(40),
    created_ts              TIMESTAMP    NOT NULL
);

-- label_value is continuous everywhere -- a ratio or a quantity, never a flag.
-- dataset_structure.md section 11: "no binary labels anywhere".
CREATE TABLE training_labels (
    label_id                VARCHAR(40)  PRIMARY KEY,
    snapshot_id             VARCHAR(30)  NOT NULL REFERENCES snapshots(snapshot_id),
    snapshot_date           DATE         NOT NULL,
    entity_type             VARCHAR(20)  NOT NULL,
    entity_id               VARCHAR(40)  NOT NULL,
    task                    VARCHAR(30)  NOT NULL,
    horizon_days            INTEGER      NOT NULL,
    label_window_start      DATE         NOT NULL,
    label_window_end        DATE         NOT NULL,
    label_value             DECIMAL(12,4),
    label_censored          BOOLEAN      NOT NULL,
    censor_time             INTEGER,
    label_definition        VARCHAR(200) NOT NULL,
    label_version           VARCHAR(10)  NOT NULL,
    created_ts              TIMESTAMP    NOT NULL,
    CONSTRAINT training_labels_entity_ck CHECK (entity_type IN
        ('po_line','channel','part_plant','supplier','product_plant')),
    CONSTRAINT training_labels_task_ck   CHECK (task IN
        ('fill_rate','arrival_week','capacity_strain','demand_drift','shortage_qty')),
    -- The label window is strictly future.  This is the leakage check that
    -- data_plan.md's checklist runs before every training job, asserted here so
    -- it cannot be violated by construction.
    CONSTRAINT training_labels_future_ck CHECK (label_window_start > snapshot_date),
    CONSTRAINT training_labels_span_ck   CHECK (label_window_end >= label_window_start),
    CONSTRAINT training_labels_censor_ck CHECK (NOT label_censored OR censor_time IS NOT NULL)
);
CREATE INDEX training_labels_snap_ix   ON training_labels(snapshot_id, task);
CREATE INDEX training_labels_entity_ix ON training_labels(entity_type, entity_id);

-- Every prediction lands here and nowhere else.  The generator writes this
-- table empty (header only): it is populated by the models, not by simulation.
CREATE TABLE model_outputs (
    output_id               VARCHAR(40)  PRIMARY KEY,
    snapshot_id             VARCHAR(30)  NOT NULL REFERENCES snapshots(snapshot_id),
    model_name              VARCHAR(50)  NOT NULL,
    model_version           VARCHAR(20)  NOT NULL,
    entity_type             VARCHAR(20)  NOT NULL,
    entity_id               VARCHAR(40)  NOT NULL,
    horizon_days            INTEGER      NOT NULL,
    point_estimate          DECIMAL(14,4),
    p10                     DECIMAL(14,4),
    p90                     DECIMAL(14,4),
    distribution            TEXT,
    confidence_basis        VARCHAR(30),
    evidence_strength       DECIMAL(4,2),
    created_ts              TIMESTAMP    NOT NULL,
    CONSTRAINT model_outputs_basis_ck CHECK (confidence_basis IS NULL OR confidence_basis IN
        ('observed','interpolated','extrapolated'))
);
CREATE INDEX model_outputs_join_ix ON model_outputs(snapshot_id, entity_id, model_name);


-- ============================================================================
-- GROUP J -- Pipeline metadata.  dataset_structure.md section 1.
--
-- Distinguishes "no data" from "no history".  A channel whose acknowledgement
-- history starts in 2020 must have ack_gap_ratio masked before that date, not
-- zero-filled: a zero reads as "the supplier acknowledged nothing", which is
-- the opposite of "we did not record it".
-- ============================================================================

CREATE TABLE dataset_coverage (
    dataset_name            VARCHAR(50)  PRIMARY KEY,
    earliest_available_date DATE,
    latest_available_date   DATE,
    available_years         DECIMAL(4,1),
    coverage_status         VARCHAR(20)  NOT NULL,
    known_gaps              TEXT,
    grain                   VARCHAR(20)  NOT NULL,
    row_count               BIGINT       NOT NULL,
    notes                   TEXT,
    assessed_ts             TIMESTAMP    NOT NULL,
    CONSTRAINT dataset_coverage_status_ck CHECK (coverage_status IN
        ('complete','partial','sparse','missing')),
    CONSTRAINT dataset_coverage_grain_ck  CHECK (grain IN ('daily','weekly','monthly','event'))
);

COMMIT;
