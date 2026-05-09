"""
db/database.py
──────────────
Provides a thin wrapper around sqlite3 for the portfolio agent.

Key responsibilities:
- Manage a single shared connection to portfolio_database.db
- Expose safe query execution with parameter binding
- Return results as lists-of-dicts (easy to JSON-serialise / pass to LLM)
- Provide schema introspection so the SQL tool can embed table info in prompts
"""

import os
import sqlite3
import logging
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "portfolio_database.db")


class DatabaseManager:
    """Thread-safe SQLite wrapper for the portfolio database."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    # ── Connection management ──────────────────────────────────────────────

    def connect(self) -> None:
        if self._conn is None:
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(
                    f"Database not found at '{self.db_path}'. "
                    "Run `python setup_database.py` first."
                )
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row          # column-name access
            self._conn.execute("PRAGMA foreign_keys = ON")
            logger.debug("Connected to database: %s", self.db_path)

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ── Query execution ────────────────────────────────────────────────────

    def execute_query(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> list[dict]:
        """
        Execute a SELECT query and return results as a list of dicts.

        Args:
            sql:    A SQL SELECT statement (no DDL / DML).
            params: Optional tuple of bind parameters (prevents SQL injection).

        Returns:
            List of row-dicts, e.g. [{"portfolio_name": "Growth Equity Fund", …}, …]
        """
        self.connect()
        sql = sql.strip()

        # Safety: only allow read-only statements
        first_word = sql.split()[0].upper() if sql else ""
        if first_word not in ("SELECT", "WITH", "EXPLAIN"):
            raise ValueError(
                f"Only SELECT queries are permitted. Got: '{first_word}'"
            )

        try:
            cursor = self._conn.execute(sql, params)
            columns = [desc[0] for desc in cursor.description]
            rows    = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except sqlite3.Error as exc:
            logger.error("SQL execution error: %s\nQuery: %s", exc, sql)
            raise

    # ── Schema introspection ───────────────────────────────────────────────

    def get_schema_summary(self) -> str:
        """
        Return a concise, human-readable summary of all tables and their columns.
        Injected into the SQL-generation prompt so Gemini knows the schema.
        """
        self.connect()
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        lines: list[str] = ["Database Schema:"]
        for table in tables:
            col_cursor = self._conn.execute(f"PRAGMA table_info({table})")
            cols = col_cursor.fetchall()
            col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
            lines.append(f"  {table}({col_defs})")

        return "\n".join(lines)

    def get_table_sample(self, table: str, limit: int = 3) -> list[dict]:
        """Return a small sample of rows for a given table (for debugging)."""
        return self.execute_query(f"SELECT * FROM {table} LIMIT {limit}")


# ── Module-level singleton ─────────────────────────────────────────────────────
_db_manager: DatabaseManager | None = None


def get_db() -> DatabaseManager:
    """Return the module-level DatabaseManager singleton."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
