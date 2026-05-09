"""
setup_database.py
─────────────────
Reads the SQL schema and all CSV files from ./data/, then populates a local
SQLite database (portfolio_database.db) that the agent will query at runtime.

Usage:
    python setup_database.py
"""

import sqlite3
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SCHEMA_SQL = os.path.join(BASE_DIR, "database_schema.sql")
DATA_DIR   = os.path.join(BASE_DIR, "data")
DB_PATH    = os.path.join(BASE_DIR, "portfolio_database.db")

# ── CSV → table mapping (load order respects FK dependencies) ─────────────────
CSV_TABLE_ORDER = [
    ("sectors.csv",             "sectors"),
    ("benchmarks.csv",          "benchmarks"),
    ("securities.csv",          "securities"),
    ("portfolios.csv",          "portfolios"),
    ("holdings.csv",            "holdings"),
    ("transactions.csv",        "transactions"),
    ("historical_prices.csv",   "historical_prices"),
    ("portfolio_performance.csv","portfolio_performance"),
    ("risk_metrics.csv",        "risk_metrics"),
]


def create_schema(conn: sqlite3.Connection) -> None:
    """Execute the DDL from database_schema.sql."""
    with open(SCHEMA_SQL, "r") as fh:
        schema_sql = fh.read()
    conn.executescript(schema_sql)
    conn.commit()
    logger.info("Schema created successfully.")


def load_csv(conn: sqlite3.Connection, csv_filename: str, table_name: str) -> None:
    """Load a single CSV file into the given SQLite table."""
    csv_path = os.path.join(DATA_DIR, csv_filename)
    if not os.path.exists(csv_path):
        logger.warning("CSV not found, skipping: %s", csv_path)
        return

    df = pd.read_csv(csv_path)

    # Replace NaN with None so SQLite stores NULL rather than the string 'nan'
    df = df.where(pd.notnull(df), None)

    df.to_sql(table_name, conn, if_exists="replace", index=False)
    logger.info("Loaded %d rows into table '%s'.", len(df), table_name)


def verify_database(conn: sqlite3.Connection) -> None:
    """Run quick sanity checks after loading."""
    cursor = conn.cursor()
    tables = ["sectors", "securities", "portfolios", "holdings",
              "transactions", "historical_prices", "portfolio_performance",
              "risk_metrics", "benchmarks"]
    print("\n── Database verification ──────────────────────────")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table:<28} {count:>5} rows")
    print("───────────────────────────────────────────────────\n")


def main() -> None:
    logger.info("Setting up database at: %s", DB_PATH)

    # Remove stale DB so we start fresh
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        logger.info("Removed existing database.")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        create_schema(conn)

        for csv_file, table in CSV_TABLE_ORDER:
            load_csv(conn, csv_file, table)

        conn.commit()
        verify_database(conn)
        logger.info("Database setup complete: %s", DB_PATH)

    except Exception as exc:
        logger.error("Database setup failed: %s", exc)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
