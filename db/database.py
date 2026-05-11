import os
import sqlite3
import logging
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "portfolio_database.db")


class DatabaseManager:

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        if self._conn is None:
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(
                    f"Database not found at '{self.db_path}'. "
                    "Run `python setup_database.py` first."
                )
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row         
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

    def execute_query(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> list[dict]:
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

    def get_schema_summary(self) -> str:
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
        return self.execute_query(f"SELECT * FROM {table} LIMIT {limit}")

_db_manager: DatabaseManager | None = None


def get_db() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
