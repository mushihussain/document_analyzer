"""
Username/password accounts and opaque session tokens.

Passwords are stretched with PBKDF2-HMAC-SHA256 (stdlib - no extra dependency)
and tokens are random 256-bit strings kept server-side, so there is nothing
sensitive in the value handed to the browser.

Scope note: this is app-level auth suitable for an internal/self-hosted
deployment. It has no rate limiting, lockout, or password reset flow, and it
assumes the API is served over HTTPS in anything but local development.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import db
from .config import settings

_ITERATIONS = 390_000
_SALT_BYTES = 16
# Hashing a throwaway password when the username is unknown keeps the failed
# login timing similar to a wrong-password attempt, so responses don't leak
# which usernames exist.
_DUMMY_HASH = hashlib.pbkdf2_hmac("sha256", b"dummy", b"dummy-salt", _ITERATIONS).hex()

MIN_USERNAME_LENGTH = 3
MIN_PASSWORD_LENGTH = 8

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class User:
    id: int
    username: str


class AuthError(Exception):
    """Raised for anything the caller is allowed to see a reason for."""


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _ITERATIONS
    ).hex()


def normalize_username(username: str) -> str:
    return username.strip()


def validate_credentials(username: str, password: str) -> None:
    if len(username) < MIN_USERNAME_LENGTH:
        raise AuthError(f"Username must be at least {MIN_USERNAME_LENGTH} characters")
    if not username.replace("_", "").replace("-", "").replace(".", "").isalnum():
        raise AuthError("Username may only contain letters, digits, dot, dash, underscore")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")


def register(username: str, password: str) -> User:
    username = normalize_username(username)
    validate_credentials(username, password)

    salt = secrets.token_hex(_SALT_BYTES)
    with db.connect() as conn:
        taken = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone()
        if taken:
            raise AuthError("That username is already taken")
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, _hash(password, salt), salt, db.utcnow()),
        )
        return User(id=int(cur.lastrowid), username=username)


def authenticate(username: str, password: str) -> User:
    username = normalize_username(username)
    with db.connect() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, salt FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if row is None:
        hmac.compare_digest(_DUMMY_HASH, _DUMMY_HASH)  # equalize timing
        raise AuthError("Incorrect username or password")

    if not hmac.compare_digest(row["password_hash"], _hash(password, row["salt"])):
        raise AuthError("Incorrect username or password")

    return User(id=row["id"], username=row["username"])


def create_session(user: User) -> tuple[str, str]:
    """Returns (token, expires_at). Also prunes this user's expired rows."""
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(hours=settings.session_ttl_hours)).isoformat(
        timespec="seconds"
    )
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user.id, now.isoformat(timespec="seconds"), expires_at),
        )
        conn.execute(
            "DELETE FROM sessions WHERE user_id = ? AND expires_at <= ?",
            (user.id, now.isoformat(timespec="seconds")),
        )
    return token, expires_at


def end_session(token: str) -> None:
    with db.connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def user_for_token(token: str) -> User | None:
    with db.connect() as conn:
        row = conn.execute(
            """SELECT u.id, u.username, s.expires_at
                 FROM sessions s JOIN users u ON u.id = s.user_id
                WHERE s.token = ?""",
            (token,),
        ).fetchone()
        if row is None:
            return None
        if row["expires_at"] <= db.utcnow():
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            return None
    return User(id=row["id"], username=row["username"])


def current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> User:
    """FastAPI dependency - 401s unless a live session token was presented.

    Declared sync on purpose: FastAPI runs it in the threadpool, keeping the
    SQLite lookup off the event loop.
    """
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Sign in to continue",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if creds is None or not creds.credentials:
        raise unauthorized
    user = user_for_token(creds.credentials)
    if user is None:
        raise unauthorized
    return user
