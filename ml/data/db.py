"""
PostgreSQL connection helper for the `chainpilot` model-training database.

Single source of the DSN default so `ml/data/features.py`, `ml/data/snapshots.py`,
and `ml/tests/test_leakage.py` don't each hardcode it. Override via the
CHAINPILOT_DSN environment variable (matches `db/load_data.py --dsn`).
"""

import os

import psycopg2

DEFAULT_DSN = "postgresql://muralik@localhost:5432/chainpilot"


def get_connection(dsn: str | None = None):
    """Open a new psycopg2 connection to the chainpilot database."""
    return psycopg2.connect(dsn or os.environ.get("CHAINPILOT_DSN", DEFAULT_DSN))
