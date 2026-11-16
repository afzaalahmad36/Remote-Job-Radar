import pytest
from httpx import AsyncClient
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import JobPreferences, User
from tests.auth_helpers import auth_headers

PREFS = "/api/v1/preferences"

FULL = {
    "seniority_levels": ["senior", "lead"],
    "remote_policies": ["remote"],
    "min_salary": 90000,
    "salary_currency": "usd",
    "salary_period": "year",
    "notes": "No fintech, please. Prefer small teams.",
}


async def count_rows(session: AsyncSession) -> int:
    return await session.scalar(select(func.count()).select_from(JobPreferences)) or 0


# --- reading ---------------------------------------------------------------------------------


async def test_before_anything_is_saved_the_defaults_are_returned_and_no_row_is_created(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    ann = await auth_headers(db_client, "ann@example.com")

    response = await db_client.get(PREFS, headers=ann)

    assert response.status_code == 200
    assert response.json() == {
        "seniority_levels": [],
        "remote_policies": [],
        "min_salary": None,
        "salary_currency": None,
        "salary_period": None,
        "notes": None,
        "id": None,
        "updated_at": None,
    }
    assert await count_rows(db_session) == 0  # reading never saves for the user


async def test_both_endpoints_need_a_login(db_client: AsyncClient) -> None:
    assert (await db_client.get(PREFS)).status_code == 401
    assert (await db_client.put(PREFS, json={})).status_code == 401


# --- saving ----------------------------------------------------------------------------------


async def test_saving_stores_the_preferences_and_returns_them(db_client: AsyncClient) -> None:
    ann = await auth_headers(db_client, "ann@example.com")

    saved = await db_client.put(PREFS, json=FULL, headers=ann)

    assert saved.status_code == 200
    body = saved.json()
    assert body["seniority_levels"] == ["senior", "lead"]
    assert body["remote_policies"] == ["remote"]
    assert (body["min_salary"], body["salary_currency"], body["salary_period"]) == (
        90000,
        "USD",  # upper-cased
        "year",
    )
    assert body["notes"] == "No fintech, please. Prefer small teams."
    assert body["id"] and body["updated_at"]
    assert (await db_client.get(PREFS, headers=ann)).json() == body


async def test_saving_again_replaces_everything_and_keeps_one_row(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    ann = await auth_headers(db_client, "ann@example.com")
    first = (await db_client.put(PREFS, json=FULL, headers=ann)).json()

    second = (await db_client.put(PREFS, json={"remote_policies": ["hybrid"]}, headers=ann)).json()

    assert second["id"] == first["id"]
    assert second["remote_policies"] == ["hybrid"]
    assert second["seniority_levels"] == [] and second["notes"] is None  # left out = cleared
    assert second["min_salary"] is None and second["salary_currency"] is None
    assert second["updated_at"] >= first["updated_at"]
    assert await count_rows(db_session) == 1


async def test_an_empty_body_saves_empty_preferences(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    ann = await auth_headers(db_client, "ann@example.com")

    response = await db_client.put(PREFS, json={}, headers=ann)

    assert response.status_code == 200 and response.json()["id"] is not None
    assert await count_rows(db_session) == 1  # an explicit save of "no preferences"


async def test_duplicates_are_dropped_and_the_order_is_kept(db_client: AsyncClient) -> None:
    ann = await auth_headers(db_client, "ann@example.com")

    body = (
        await db_client.put(
            PREFS,
            json={
                "seniority_levels": ["lead", "senior", "lead"],
                "remote_policies": ["remote"] * 3,
            },
            headers=ann,
        )
    ).json()

    assert body["seniority_levels"] == ["lead", "senior"]
    assert body["remote_policies"] == ["remote"]


async def test_blank_notes_are_stored_as_none(db_client: AsyncClient) -> None:
    ann = await auth_headers(db_client, "ann@example.com")

    body = (await db_client.put(PREFS, json={"notes": "   \n "}, headers=ann)).json()

    assert body["notes"] is None


async def test_contact_details_in_the_notes_are_masked_like_in_a_cv(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    ann = await auth_headers(db_client, "ann@example.com")
    notes = "Call +1 555 010 0199 or mail me@example.com, see https://github.com/me"

    body = (await db_client.put(PREFS, json={"notes": notes}, headers=ann)).json()

    for leaked in ("555 010", "me@example.com", "github.com"):
        assert leaked not in body["notes"]
    stored = await db_session.scalar(select(JobPreferences.notes))
    assert stored == body["notes"]


async def test_each_user_has_their_own_preferences(db_client: AsyncClient) -> None:
    ann = await auth_headers(db_client, "ann@example.com")
    bob = await auth_headers(db_client, "bob@example.com")

    await db_client.put(PREFS, json=FULL, headers=ann)

    assert (await db_client.get(PREFS, headers=bob)).json()["min_salary"] is None
    assert (await db_client.get(PREFS, headers=ann)).json()["min_salary"] == 90000


# --- bad input -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"seniority_levels": ["wizard"]},
        {"remote_policies": ["moon"]},
        {"min_salary": 0, "salary_currency": "USD", "salary_period": "year"},
        {"min_salary": -5, "salary_currency": "USD", "salary_period": "year"},
        {"min_salary": 2_000_000_000, "salary_currency": "USD", "salary_period": "year"},
        {"min_salary": 100, "salary_currency": "USD"},  # no period
        {"min_salary": 100, "salary_period": "year"},  # no currency
        {"salary_currency": "USD", "salary_period": "year"},  # no amount
        {"min_salary": 100, "salary_currency": "US", "salary_period": "year"},
        {"min_salary": 100, "salary_currency": "U$D", "salary_period": "year"},
        {"min_salary": 100, "salary_currency": "USD", "salary_period": "fortnight"},
        {"min_salary": "lots", "salary_currency": "USD", "salary_period": "year"},
        {"notes": "x" * 2_001},
        {"unknown": 1},
        {"years_of_experience": 6},  # a CV suggestion is not a preference
        {"seniority_suggestion": "senior"},
    ],
)
async def test_invalid_preferences_are_422_and_leave_the_saved_ones_alone(
    db_client: AsyncClient, body: dict[str, object]
) -> None:
    ann = await auth_headers(db_client, "ann@example.com")
    saved = (await db_client.put(PREFS, json=FULL, headers=ann)).json()

    response = await db_client.put(PREFS, json=body, headers=ann)

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert (await db_client.get(PREFS, headers=ann)).json() == saved


# --- the database refuses what the API would never send ----------------------------------------


async def test_the_database_refuses_a_salary_without_its_currency_and_period(
    db_session: AsyncSession, db_client: AsyncClient
) -> None:
    await auth_headers(db_client, "ann@example.com")
    user_id = await db_session.scalar(select(User.id))

    with pytest.raises(IntegrityError, match="salary_all_or_nothing"):
        await db_session.execute(insert(JobPreferences).values(user_id=user_id, min_salary=100))


async def test_deleting_the_user_deletes_their_preferences(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    ann = await auth_headers(db_client, "ann@example.com")
    await db_client.put(PREFS, json=FULL, headers=ann)

    await db_session.execute(text("DELETE FROM users"))

    assert await count_rows(db_session) == 0
