import asyncio
import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from functools import lru_cache

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from app.core.exceptions import AuthenticationError

ALGORITHM = "HS256"  # pinned: decode() only ever accepts this, so "alg": "none" tokens fail
ACCESS_TOKEN_TYPE = "access"

_hasher = PasswordHasher()  # argon2id with the library's recommended cost parameters


# --- Passwords ------------------------------------------------------------------------------
# Argon2 is deliberately slow and CPU-bound. Called directly inside `async def`, it would freeze
# the event loop (and every other request) while it runs, so it goes to a worker thread.


async def hash_password(password: str) -> str:
    return await asyncio.to_thread(_hasher.hash, password)


def _verify(password: str, hashed: str) -> bool:
    try:
        return _hasher.verify(hashed, password)
    except (VerificationError, InvalidHashError):  # wrong password, or not an argon2 hash
        return False


async def verify_password(password: str, hashed: str) -> bool:
    return await asyncio.to_thread(_verify, password, hashed)


@lru_cache
def _dummy_hash() -> str:
    return _hasher.hash("not-a-real-password")


async def verify_against_dummy(password: str) -> None:
    """Spend the same time as a real check, for logins with an unknown email.

    Without this, "no such user" would answer much faster than "wrong password", and the timing
    difference would reveal which emails are registered.
    """
    await asyncio.to_thread(_verify, password, _dummy_hash())


# --- Access tokens (JWT) ---------------------------------------------------------------------


def create_access_token(
    user_id: uuid.UUID, secret_key: str, expires_minutes: int, now: datetime | None = None
) -> str:
    now = now or datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "type": ACCESS_TOKEN_TYPE,
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str, secret_key: str) -> uuid.UUID:
    """Return the user id in a valid access token, or raise AuthenticationError."""
    try:
        payload = jwt.decode(
            token,
            secret_key,
            algorithms=[ALGORITHM],
            options={"require": ["exp", "iat", "sub"]},  # a token without an expiry is invalid
        )
        if payload.get("type") != ACCESS_TOKEN_TYPE:
            raise AuthenticationError("Invalid or expired token")
        return uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, ValueError) as error:  # bad signature, expired, malformed, bad uuid
        raise AuthenticationError("Invalid or expired token") from error


# --- Refresh tokens --------------------------------------------------------------------------
# A refresh token is not a JWT: it is 256 random bits. Only its SHA-256 hash is stored, so a
# leaked database cannot be used to refresh sessions. A fast hash is fine here (unlike for
# passwords) because the input is already unguessable.


def generate_refresh_token() -> str:
    return secrets.token_urlsafe(32)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
