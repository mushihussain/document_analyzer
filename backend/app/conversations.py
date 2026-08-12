"""
Per-user chat history: conversations and the messages inside them.

Every function takes a user_id and filters on it, so one user can never read or
mutate another's threads even if they guess a conversation id.
"""
from __future__ import annotations

import json

from . import db

TITLE_MAX_LENGTH = 60
# How many past turns are replayed to the model as conversational context.
HISTORY_TURNS = 8


def make_title(question: str) -> str:
    text = " ".join(question.split())
    if len(text) <= TITLE_MAX_LENGTH:
        return text or "New chat"
    return text[: TITLE_MAX_LENGTH - 1].rstrip() + "…"


def create(user_id: int, title: str) -> int:
    now = db.utcnow()
    with db.connect() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (user_id, title, now, now),
        )
        return int(cur.lastrowid)


def owned_by(conversation_id: int, user_id: int) -> bool:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()
    return row is not None


def list_for_user(user_id: int) -> list[dict]:
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT c.id, c.title, c.created_at, c.updated_at,
                      (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id)
                          AS message_count
                 FROM conversations c
                WHERE c.user_id = ?
                ORDER BY c.updated_at DESC, c.id DESC""",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def messages(conversation_id: int, user_id: int, limit: int | None = None) -> list[dict]:
    """Oldest-first messages, or the last `limit` turns when limit is given."""
    with db.connect() as conn:
        if limit is None:
            rows = conn.execute(
                """SELECT m.role, m.text, m.sources, m.provider
                     FROM messages m JOIN conversations c ON c.id = m.conversation_id
                    WHERE m.conversation_id = ? AND c.user_id = ?
                    ORDER BY m.id""",
                (conversation_id, user_id),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM (
                       SELECT m.id, m.role, m.text, m.sources, m.provider
                         FROM messages m JOIN conversations c ON c.id = m.conversation_id
                        WHERE m.conversation_id = ? AND c.user_id = ?
                        ORDER BY m.id DESC
                        LIMIT ?
                   ) ORDER BY id""",
                (conversation_id, user_id, limit),
            ).fetchall()

    return [
        {
            "role": r["role"],
            "text": r["text"],
            "sources": json.loads(r["sources"]) if r["sources"] else [],
            "provider": r["provider"],
        }
        for r in rows
    ]


def add_message(
    conversation_id: int,
    role: str,
    text: str,
    sources: list[dict] | None = None,
    provider: str | None = None,
) -> None:
    now = db.utcnow()
    with db.connect() as conn:
        conn.execute(
            """INSERT INTO messages (conversation_id, role, text, sources, provider, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                conversation_id,
                role,
                text,
                json.dumps(sources) if sources else None,
                provider,
                now,
            ),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
        )


def rename(conversation_id: int, user_id: int, title: str) -> bool:
    with db.connect() as conn:
        cur = conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ? AND user_id = ?",
            (title, conversation_id, user_id),
        )
        return cur.rowcount > 0


def delete(conversation_id: int, user_id: int) -> bool:
    with db.connect() as conn:
        # Messages go too - the schema declares ON DELETE CASCADE and
        # db.connect() enables foreign_keys for every connection.
        cur = conn.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        )
        return cur.rowcount > 0
