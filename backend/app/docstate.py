"""
Bookkeeping for what is already in the vector index.

Lets a rescan chunk only files that are new or changed. Without this, every
rescan re-extracted and re-embedded the whole folder - cheap for a few text
files, but painfully slow once images are in the mix, since each one costs a
full OCR pass.

Change detection is size + mtime (the rsync heuristic): fast, no file reads, and
accurate for ordinary editing. It can miss an edit that preserves both, so
/api/ingest takes force=true to rebuild everything.
"""
from __future__ import annotations

from pathlib import Path

from . import db


def signature(path: Path) -> tuple[int, float]:
    """(size, mtime) for a file - what we compare to decide if it changed."""
    stat = path.stat()
    return stat.st_size, stat.st_mtime


def load(user_id: int) -> dict[str, dict]:
    """Everything currently indexed for one user, keyed by resolved absolute path.

    Per-user workspaces mean paths never collide across users anyway, but
    filtering by user_id keeps this from ever depending on that.
    """
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT doc_path, doc_name, size, mtime, chunk_size, chunk_overlap,
                      chunk_count, indexed_at
                 FROM indexed_documents
                WHERE user_id = ?""",
            (user_id,),
        ).fetchall()
    return {row["doc_path"]: dict(row) for row in rows}


def record(
    user_id: int,
    doc_path: str,
    doc_name: str,
    size: int,
    mtime: float,
    chunk_size: int,
    chunk_overlap: int,
    chunk_count: int,
) -> None:
    with db.connect() as conn:
        conn.execute(
            """INSERT INTO indexed_documents
                   (doc_path, user_id, doc_name, size, mtime, chunk_size,
                    chunk_overlap, chunk_count, indexed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(doc_path) DO UPDATE SET
                   user_id       = excluded.user_id,
                   doc_name      = excluded.doc_name,
                   size          = excluded.size,
                   mtime         = excluded.mtime,
                   chunk_size    = excluded.chunk_size,
                   chunk_overlap = excluded.chunk_overlap,
                   chunk_count   = excluded.chunk_count,
                   indexed_at    = excluded.indexed_at""",
            (
                doc_path,
                user_id,
                doc_name,
                size,
                mtime,
                chunk_size,
                chunk_overlap,
                chunk_count,
                db.utcnow(),
            ),
        )


def forget(user_id: int, doc_path: str) -> None:
    with db.connect() as conn:
        conn.execute(
            "DELETE FROM indexed_documents WHERE user_id = ? AND doc_path = ?",
            (user_id, doc_path),
        )


def forget_name(user_id: int, doc_name: str) -> None:
    """Used when a document is dropped by name, matching the vector store."""
    with db.connect() as conn:
        conn.execute(
            "DELETE FROM indexed_documents WHERE user_id = ? AND doc_name = ?",
            (user_id, doc_name),
        )


def clear(user_id: int) -> None:
    with db.connect() as conn:
        conn.execute("DELETE FROM indexed_documents WHERE user_id = ?", (user_id,))
