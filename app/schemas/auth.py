import uuid
from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, EmailStr, Field


def _normalize_email(value: Any) -> Any:
    """Strip and lowercase before validation, so "  Ann@Example.COM " is "ann@example.com".

    Every email is stored lowercase, so the unique constraint on `users.email` is effectively
    case-insensitive and no extra database index is needed.
    """
    return value.strip().lower() if isinstance(value, str) else value


# EmailStr checks the format; the BeforeValidator runs first and normalizes the raw input.
NormalizedEmail = Annotated[EmailStr, BeforeValidator(_normalize_email)]
# The upper bound matters: argon2 hashes whatever it is given, so unbounded input is a DoS risk.
Password = Annotated[str, Field(min_length=8, max_length=128)]


class UserCreate(BaseModel):
    email: NormalizedEmail
    password: Password


class LoginCreate(BaseModel):
    email: NormalizedEmail
    # No length rules on login: a wrong password should be a 401, not a 422 that hints at rules.
    password: Annotated[str, Field(max_length=128)]


class RefreshCreate(BaseModel):
    refresh_token: Annotated[str, Field(max_length=256)]


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until the access token expires


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # allows UserRead.model_validate(orm_user)

    id: uuid.UUID
    email: str
    created_at: datetime
