#!/usr/bin/env python3
"""
Load the HADES synthetic dataset into PostgreSQL.

Usage:
    python load_data.py --dsn "postgresql://user:pass@localhost:5432/chainpilot"
    python load_data.py --dsn "..." --drop        # drop & recreate schema first
    python load_data.py --dsn "..." --verify-only # run checks against loaded DB

Requires: psycopg2-binary  (pip install psycopg2-binary)

The load order satisfies FK dependencies: master → junction → operational →
history → snapshots → labels → scores. Empty CSV fields become NULL.
"""

import argparse, csv, io, os, sys

try:
    import psycopg2
except ImportError:
    sys.exit("psycopg2 is required:  pip install psycopg2-binary")

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(HERE, "csv")
SCHEMA = os.path.join(HERE, "schema.sql")

# (table, csv file) in FK-safe order — Dataset.md Chapter 14
LOAD_ORDER = [
    ("suppliers",                     "suppliers.csv"),
    ("components",                    "components.csv"),
    ("component_suppliers",           "component_suppliers.csv"),  # Task 3 follow-up experiment; absent in base v3
    ("products",                      "products.csv"),
    ("factories",                     "factories.csv"),
    ("warehouses",                    "warehouses.csv"),
    ("customers",                     "customers.csv"),
    ("product_components",            "product_components.csv"),
    ("product_factories",             "product_factories.csv"),
    ("inventory",                     "inventory.csv"),
    ("orders",                        "orders.csv"),
    ("order_items",                   "order_items.csv"),
    ("shipments",                     "shipments.csv"),
    ("inventory_history",             "inventory_history.csv"),
    ("shipment_status_history",       "shipment_status_history.csv"),
    ("supplier_temporal_features",    "supplier_temporal_features.csv"),
    ("carrier_performance_snapshots", "carrier_performance_snapshots.csv"),
    ("graph_snapshots",               "graph_snapshots.csv"),
    ("training_labels",               "training_labels.csv"),
    ("risk_scores",                   "risk_scores.csv"),
]

def copy_csv(cur, table, path):
    """COPY a CSV with header; empty strings load as NULL."""
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    cols = ", ".join(header)
    with open(path) as f:
        cur.copy_expert(
            f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')", f)
    cur.execute(f"SELECT count(*) FROM {table}")
    return cur.fetchone()[0]

VERIFY = [
    ("FK: components -> suppliers",
     "SELECT count(*) FROM components c LEFT JOIN suppliers s ON s.id=c.supplier_id WHERE s.id IS NULL", 0),
    ("FK: training_labels -> graph_snapshots",
     "SELECT count(*) FROM training_labels l LEFT JOIN graph_snapshots g ON g.id=l.snapshot_id WHERE g.id IS NULL", 0),
    ("chronology: dispatch before eta",
     "SELECT count(*) FROM shipments WHERE dispatched_at IS NOT NULL AND eta IS NOT NULL AND dispatched_at >= eta", 0),
    ("history: no negative stock",
     "SELECT count(*) FROM inventory_history WHERE stock_level < 0", 0),
    ("leakage: label events inside (t0, t0+horizon]",
     """SELECT count(*) FROM training_labels l JOIN graph_snapshots g ON g.id=l.snapshot_id
        WHERE l.event_at IS NOT NULL
          AND NOT (l.event_at > g.t0 AND l.event_at <= g.t0 + (g.horizon_days || ' days')::interval)""", 0),
    ("leakage: as-of feature dates precede any labelled event",
     """SELECT count(*) FROM supplier_temporal_features f
        WHERE f.computed_at < (f.as_of_date::timestamptz)""", 0),
    ("status history covers every shipment",
     "SELECT count(*) FROM shipments s LEFT JOIN shipment_status_history h ON h.shipment_id=s.id WHERE h.id IS NULL", 0),
]

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", required=True, help="PostgreSQL DSN")
    ap.add_argument("--drop", action="store_true", help="DROP SCHEMA public CASCADE and recreate before loading")
    ap.add_argument("--verify-only", action="store_true", help="Skip loading; run verification queries only")
    args = ap.parse_args()

    conn = psycopg2.connect(args.dsn)
    conn.autocommit = False
    cur = conn.cursor()
    try:
        if not args.verify_only:
            if args.drop:
                print("dropping schema public ...")
                cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
            print("applying schema.sql ...")
            with open(SCHEMA) as f:
                cur.execute(f.read())
            print("loading CSVs ...")
            for table, fname in LOAD_ORDER:
                path = os.path.join(CSV_DIR, fname)
                if not os.path.exists(path):
                    print(f"  {table:34s} SKIPPED (no {fname})")
                    continue
                n = copy_csv(cur, table, path)
                print(f"  {table:34s} {n:>8,} rows")
        print("verifying ...")
        failed = 0
        for name, sql, expect in VERIFY:
            cur.execute(sql)
            got = cur.fetchone()[0]
            ok = got == expect
            failed += 0 if ok else 1
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"  (got {got}, expected {expect})"))
        if failed:
            conn.rollback()
            sys.exit(f"{failed} verification(s) failed — transaction rolled back, nothing committed.")
        conn.commit()
        print("done — committed.")
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close(); conn.close()

if __name__ == "__main__":
    main()
