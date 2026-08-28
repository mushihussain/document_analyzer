"""
Admin-only user management: listing accounts with basic stats, disabling or
re-enabling logins, and deleting an account outright.

Sessions, conversations, messages, and indexed_documents rows all cascade
from the users table's foreign keys (see db.py), so a plain DELETE FROM users
already cleans up everything relational. What the database can't reach is
the user's document folder on disk and their Chroma collection - this module
is what cleans those up too, so a deleted user doesn't leave orphaned files
or an orphaned vector collection behind.
"""
from __future__ import annotations

import shutil

from . import db, ingest, vectorstore
from .auth import is_admin_username
from .config import settings


def list_users() -> list[dict]:
    """Every account, with the counts an admin actually cares about."""
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT u.id, u.username, u.created_at, u.is_disabled,
                      (SELECT COUNT(*) FROM conversations c
                        WHERE c.user_id = u.id)          AS conversation_count,
                      (SELECT COUNT(*) FROM indexed_documents d
                        WHERE d.user_id = u.id)           AS document_count
                 FROM users u
                ORDER BY u.id"""
        ).fetchall()
    return [
        {
            "id": row["id"],
            "username": row["username"],
            "created_at": row["created_at"],
            "is_disabled": bool(row["is_disabled"]),
            "is_admin": is_admin_username(row["username"]),
            "document_count": row["document_count"],
            "conversation_count": row["conversation_count"],
        }
        for row in rows
    ]


def get_user(user_id: int) -> dict | None:
    return next((u for u in list_users() if u["id"] == user_id), None)


def set_disabled(user_id: int, disabled: bool) -> None:
    with db.connect() as conn:
        cur = conn.execute(
            "UPDATE users SET is_disabled = ? WHERE id = ?", (int(disabled), user_id)
        )
        if cur.rowcount == 0:
            raise ValueError("User not found")


def delete_user(user_id: int) -> None:
    """Remove a user entirely: account row (cascading sessions/conversations/
    messages), their indexed_documents rows, their workspace folder, and
    their vector collection. Irreversible.

    indexed_documents.user_id is deleted explicitly rather than left to the
    column's ON DELETE CASCADE: that constraint only exists on databases
    created after per-user workspaces shipped. A database that reached this
    schema via the ALTER TABLE migration in db.py has the column but no FK on
    it, so cascade silently does nothing there - explicit cleanup works
    either way.
    """
    with db.connect() as conn:
        cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        if cur.rowcount == 0:
            raise ValueError("User not found")
        conn.execute("DELETE FROM indexed_documents WHERE user_id = ?", (user_id,))

    folder = ingest.user_folder(settings.documents_folder, user_id)
    shutil.rmtree(folder, ignore_errors=True)
    vectorstore.reset_collection(user_id)
