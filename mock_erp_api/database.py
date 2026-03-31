# mock_erp_api/database.py
"""
Shared SQLite connection layer.
Resolves erp.db relative to the project root (one level above this file).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Generator

# ── Path to the existing database ────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # mock_erp_api/
PROJECT_ROOT = _HERE.parent                       # Thesis/
DB_PATH = PROJECT_ROOT / "erp.db"


def get_connection() -> sqlite3.Connection:
    """Return a raw sqlite3 connection with row_factory set to dict-like rows."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row          # rows behave like dicts
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")   # safe for concurrent reads
    return conn


def get_db() -> Generator[sqlite3.Connection, None, None]:
    """FastAPI dependency — yields a connection, closes it after the request."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()
