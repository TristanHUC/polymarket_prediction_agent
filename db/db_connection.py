"""PostgreSQL database connection utilities.

Reads connection parameters from environment variables with defaults matching
the locally published PostgreSQL container:

    DB_HOST       — default "localhost"
    DB_PORT       — default "5432"
    DB_NAME       — default "postgres"
    DB_USER       — default "postgres"
    DB_PASSWORD   — default "postgres"

Usage:

    from RL_realtime_strategies.db.db_connection import get_connection

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT NOW()")
            print(cur.fetchone())
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import psycopg2
from psycopg2 import sql as psycopg2_sql
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from the project root (RL_realtime_strategies)
# ---------------------------------------------------------------------------

_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# ---------------------------------------------------------------------------
# Configuration from environment (with dev-container defaults)
# ---------------------------------------------------------------------------

DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
DB_NAME: str = os.getenv("DB_NAME", "postgres")
DB_USER: str = os.getenv("DB_USER", "postgres")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")
DB_MIN_CONN: int = int(os.getenv("DB_MIN_CONN", "1"))
DB_MAX_CONN: int = int(os.getenv("DB_MAX_CONN", "10"))


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def conn_params() -> dict[str, str | int]:
    """Return the connection parameter dict for ``psycopg2.connect``."""
    return {
        "host": DB_HOST,
        "port": DB_PORT,
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD,
    }


def get_connection():
    """Create and return a new PostgreSQL connection.

    Callers **must** close the connection when done. Prefer the
    ``with get_connection() as conn:`` context-manager form below.
    """
    return psycopg2.connect(**conn_params())  # type: ignore[arg-type]


@contextmanager
def connection_scope(
    commit: bool = True,
) -> Generator[psycopg2.extensions.connection, None, None]:
    """Context manager that yields a connection and cleans up on exit.

    Parameters
    ----------
    commit:
        If True (default), commit on success; roll back on exception.
        If False, the caller is responsible for explicit commit/rollback.
    """
    conn = psycopg2.connect(**conn_params())  # type: ignore[arg-type]
    try:
        yield conn
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def cursor_scope(
    commit: bool = True,
    *,
    dict_cursor: bool = False,
) -> Generator[psycopg2.extensions.cursor, None, None]:
    """Context manager yielding a cursor with automatic connection lifecycle.

    Parameters
    ----------
    commit:
        Whether to commit the transaction when the scope exits normally.
    dict_cursor:
        If True, use ``RealDictCursor`` so rows are returned as dicts
        keyed by column name (like ``psql`` expanded mode).
    """
    with connection_scope(commit=commit) as conn:
        cursor_factory = RealDictCursor if dict_cursor else None
        with conn.cursor(cursor_factory=cursor_factory) as cur:
            yield cur


# ---------------------------------------------------------------------------
# Convenience query helpers
# ---------------------------------------------------------------------------


def fetch_all(sql: str, params: tuple | None = None) -> list[dict]:
    """Execute a query and return all rows as dicts (RealDictCursor)."""
    with cursor_scope(dict_cursor=True) as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def fetch_one(sql: str, params: tuple | None = None) -> dict | None:
    """Execute a query and return the first row as a dict, or None."""
    with cursor_scope(dict_cursor=True) as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None


def execute(sql: str, params: tuple | None = None) -> int:
    """Execute a statement and return the number of rows affected."""
    with cursor_scope(commit=True) as cur:
        cur.execute(sql, params)
        return cur.rowcount
