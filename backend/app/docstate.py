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


def load() -> dict[str, dict]:
    """Everything currently indexed, keyed by resolved absolute path."""
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT doc_path, doc_name, size, mtime, chunk_size, chunk_overlap,
                      chunk_count, indexed_at
                 FROM indexed_documents"""
        ).fetchall()
    return {row["doc_path"]: dict(row) for row in rows}


def record(
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
                   (doc_path, doc_name, size, mtime, chunk_size, chunk_overlap,
                    chunk_count, indexed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(doc_path) DO UPDATE SET
                   doc_name      = excluded.doc_name,
                   size          = excluded.size,
                   mtime         = excluded.mtime,
                   chunk_size    = excluded.chunk_size,
                   chunk_overlap = excluded.chunk_overlap,
                   chunk_count   = excluded.chunk_count,
                   indexed_at    = excluded.indexed_at""",
            (
                doc_path,
                doc_name,
                size,
                mtime,
                chunk_size,
                chunk_overlap,
                chunk_count,
                db.utcnow(),
            ),
        )


def forget(doc_path: str) -> None:
    with db.connect() as conn:
        conn.execute("DELETE FROM indexed_documents WHERE doc_path = ?", (doc_path,))


def forget_name(doc_name: str) -> None:
    """Used when a document is dropped by name, matching the vector store."""
    with db.connect() as conn:
        conn.execute("DELETE FROM indexed_documents WHERE doc_name = ?", (doc_name,))


def clear() -> None:
    with db.connect() as conn:
        conn.execute("DELETE FROM indexed_documents")
