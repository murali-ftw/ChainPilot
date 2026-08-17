"""
Load the ChainPilot / HADES CSVs into typed pandas DataFrames.

No leakage-sensitive column is dropped here -- that filtering happens in
graph_builder.py, deliberately, so it's auditable in one place instead of
silently missing at load time.
"""
import pandas as pd
from pathlib import Path

CSV_DIR = Path("/home/claude/work/db/csv")

TIME_COLS = {
    "suppliers": ["created_at", "updated_at"],
    "components": ["created_at", "updated_at"],
    "products": ["created_at", "updated_at"],
    "factories": ["created_at", "updated_at"],
    "warehouses": ["created_at", "updated_at"],
    "customers": ["created_at", "updated_at"],
    "product_components": ["created_at", "deactivated_at"],
    "component_suppliers": ["created_at", "deactivated_at"],
    "product_factories": ["qualified_at", "created_at", "deactivated_at"],
    "inventory": ["updated_at"],
    "orders": ["placed_at", "due_at", "created_at", "updated_at"],
    "order_items": ["created_at"],
    "shipments": ["eta", "dispatched_at", "delivered_at", "created_at", "updated_at"],
    "inventory_history": ["observed_at", "recorded_at"],
    "shipment_status_history": ["changed_at", "recorded_at"],
    "supplier_temporal_features": ["as_of_date", "computed_at"],
    "carrier_performance_snapshots": ["as_of_date", "computed_at"],
    "graph_snapshots": ["t0", "created_at"],
    "training_labels": ["event_at"],
}

TABLES = list(TIME_COLS.keys())


def load_all(csv_dir: Path = CSV_DIR) -> dict[str, pd.DataFrame]:
    dfs = {}
    for name in TABLES:
        df = pd.read_csv(csv_dir / f"{name}.csv")
        for col in TIME_COLS[name]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        dfs[name] = df
    return dfs


if __name__ == "__main__":
    dfs = load_all()
    for name, df in dfs.items():
        print(f"{name:32s} {len(df):>8,} rows  {list(df.columns)}")
