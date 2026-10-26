import base64
import json
import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.exceptions import AuthenticationError
from app.core.security import (
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_against_dummy,
    verify_password,
)

SECRET = "unit-test-secret-" + "x" * 64  # long enough that PyJWT accepts it for HS384/HS512 too


def b64(data: dict[str, object]) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()


async def test_password_hash_verifies_and_is_not_the_password() -> None:
    hashed = await hash_password("s3cret-password")

    assert hashed.startswith("$argon2id$")
    assert "s3cret-password" not in hashed
    assert await verify_password("s3cret-password", hashed) is True


async def test_wrong_password_does_not_verify() -> None:
    hashed = await hash_password("s3cret-password")

    assert await verify_password("another-password", hashed) is False


async def test_same_password_gets_a_different_hash_each_time() -> None:
    assert await hash_password("same") != await hash_password("same")  # random salt


async def test_verifying_against_a_non_argon2_value_is_just_false() -> None:
    assert await verify_password("anything", "not-a-hash") is False


async def test_dummy_verification_runs_without_error() -> None:
    await verify_against_dummy("whatever")


def test_access_token_round_trip() -> None:
    user_id = uuid.uuid4()

    token = create_access_token(user_id, SECRET, expires_minutes=5)

    assert decode_access_token(token, SECRET) == user_id


def test_expired_access_token_is_rejected() -> None:
    long_ago = datetime.now(UTC) - timedelta(hours=2)
    token = create_access_token(uuid.uuid4(), SECRET, expires_minutes=5, now=long_ago)

    with pytest.raises(AuthenticationError):
        decode_access_token(token, SECRET)


def test_token_signed_with_another_secret_is_rejected() -> None:
    token = create_access_token(uuid.uuid4(), "another-secret-at-least-32-characters-xx", 5)

    with pytest.raises(AuthenticationError):
        decode_access_token(token, SECRET)


def test_tampered_token_is_rejected() -> None:
    header, payload, signature = create_access_token(uuid.uuid4(), SECRET, 5).split(".")
    forged_payload = b64({"sub": str(uuid.uuid4()), "type": "access", "iat": 0, "exp": 9999999999})

    with pytest.raises(AuthenticationError):
        decode_access_token(f"{header}.{forged_payload}.{signature}", SECRET)


def test_token_with_alg_none_is_rejected() -> None:
    now = int(datetime.now(UTC).timestamp())
    payload = {"sub": str(uuid.uuid4()), "type": "access", "iat": now, "exp": now + 300}
    unsigned = f"{b64({'alg': 'none', 'typ': 'JWT'})}.{b64(payload)}."

    with pytest.raises(AuthenticationError):
        decode_access_token(unsigned, SECRET)


@pytest.mark.parametrize("algorithm", ["HS384", "HS512"])
def test_token_signed_with_a_different_algorithm_is_rejected(algorithm: str) -> None:
    # Same key, valid claims, different algorithm: only the pinned HS256 is accepted.
    now = datetime.now(UTC)
    payload = {
        "sub": str(uuid.uuid4()),
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }

    with pytest.raises(AuthenticationError):
        decode_access_token(jwt.encode(payload, SECRET, algorithm=algorithm), SECRET)


def test_token_without_a_subject_is_rejected() -> None:
    now = datetime.now(UTC)
    payload = {"type": "access", "iat": now, "exp": now + timedelta(minutes=5)}

    with pytest.raises(AuthenticationError):
        decode_access_token(jwt.encode(payload, SECRET, algorithm="HS256"), SECRET)


def test_token_without_expiry_is_rejected() -> None:
    token = jwt.encode({"sub": str(uuid.uuid4()), "type": "access"}, SECRET, algorithm="HS256")

    with pytest.raises(AuthenticationError):
        decode_access_token(token, SECRET)


def test_token_of_the_wrong_type_is_rejected() -> None:
    now = datetime.now(UTC)
    payload = {
        "sub": str(uuid.uuid4()),
        "type": "refresh",
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }

    with pytest.raises(AuthenticationError):
        decode_access_token(jwt.encode(payload, SECRET, algorithm="HS256"), SECRET)


def test_token_with_a_non_uuid_subject_is_rejected() -> None:
    now = datetime.now(UTC)
    payload = {"sub": "not-a-uuid", "type": "access", "iat": now, "exp": now + timedelta(hours=1)}

    with pytest.raises(AuthenticationError):
        decode_access_token(jwt.encode(payload, SECRET, algorithm="HS256"), SECRET)


def test_garbage_is_rejected() -> None:
    with pytest.raises(AuthenticationError):
        decode_access_token("not.a.token", SECRET)


def test_refresh_tokens_are_unique_and_hash_deterministically() -> None:
    first, second = generate_refresh_token(), generate_refresh_token()

    assert first != second
    assert len(first) >= 43  # 32 random bytes, url-safe base64
    assert hash_refresh_token(first) == hash_refresh_token(first)
    assert hash_refresh_token(first) != hash_refresh_token(second)
    assert len(hash_refresh_token(first)) == 64
