import pytest
from pydantic import ValidationError

from app.schemas.auth import LoginCreate, UserCreate


def test_email_is_stripped_and_lowercased() -> None:
    user = UserCreate(email="  Ann@Example.COM ", password="long-enough-pw")

    assert user.email == "ann@example.com"


def test_login_email_is_normalized_too() -> None:
    assert LoginCreate(email="ANN@example.com", password="x").email == "ann@example.com"


@pytest.mark.parametrize("email", ["not-an-email", "a@", "@b.com", "", "a b@c.com"])
def test_invalid_emails_are_rejected(email: str) -> None:
    with pytest.raises(ValidationError):
        UserCreate(email=email, password="long-enough-pw")


@pytest.mark.parametrize("password", ["short", "1234567", "x" * 129])
def test_password_length_is_bounded(password: str) -> None:
    with pytest.raises(ValidationError):
        UserCreate(email="ann@example.com", password=password)


def test_password_at_the_limits_is_accepted() -> None:
    UserCreate(email="ann@example.com", password="x" * 8)
    UserCreate(email="ann@example.com", password="x" * 128)
