"""
SQLite storage for user accounts, login sessions, and saved chat history.

Kept deliberately dependency-free (stdlib sqlite3) - the vector index lives in
Chroma, and this file only holds the small relational bits around it.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    text            TEXT NOT NULL,
    sources         TEXT,
    provider        TEXT,
    created_at      TEXT NOT NULL
);

-- What the vector index currently holds, so a rescan can chunk only the files
-- that are new or changed instead of the whole folder. Keyed on the resolved
-- absolute path, which is also what chunk ids are derived from.
CREATE TABLE IF NOT EXISTS indexed_documents (
    doc_path      TEXT PRIMARY KEY,
    doc_name      TEXT NOT NULL,
    size          INTEGER NOT NULL,
    mtime         REAL NOT NULL,
    chunk_size    INTEGER NOT NULL,
    chunk_overlap INTEGER NOT NULL,
    chunk_count   INTEGER NOT NULL,
    indexed_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_indexed_documents_name ON indexed_documents(doc_name);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
"""

_initialized = False


def utcnow() -> str:
    """Timestamps are stored as ISO-8601 UTC strings so they sort lexically."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Short-lived connection per unit of work, committed on clean exit.

    A connection per call (rather than one shared handle) keeps this safe when
    FastAPI dispatches handlers across threadpool workers.
    """
    global _initialized
    path = Path(settings.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        if not _initialized:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(_SCHEMA)
            _migrate(conn)
            conn.commit()
            _initialized = True
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an older database up to the current schema.

    CREATE TABLE IF NOT EXISTS won't alter a table that already exists, so
    columns added after a database was first created need patching in here.
    Each step is idempotent.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
    if "provider" not in columns:
        conn.execute("ALTER TABLE messages ADD COLUMN provider TEXT")


def init() -> None:
    """Create the schema up front so the first request isn't the one to do it."""
    with connect():
        pass
