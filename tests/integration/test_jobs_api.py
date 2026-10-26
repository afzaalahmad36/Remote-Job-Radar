import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Job
from tests.auth_helpers import auth_headers

JOBS = "/api/v1/jobs"
SOURCES = "/api/v1/sources"
T0 = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)


@pytest.fixture
async def ann(db_client: AsyncClient) -> dict[str, str]:
    return await auth_headers(db_client, "ann@example.com")


@pytest.fixture
async def bob(db_client: AsyncClient) -> dict[str, str]:
    return await auth_headers(db_client, "bob@example.com")


async def new_source(client: AsyncClient, headers: dict[str, str], name: str) -> uuid.UUID:
    url = f"https://example.com/{name}.xml"
    response = await client.post(
        SOURCES, json={"type": "rss", "name": name, "config": {"url": url}}, headers=headers
    )
    assert response.status_code == 201
    return uuid.UUID(response.json()["id"])


async def add_jobs(
    session: AsyncSession, source_id: uuid.UUID, count: int, prefix: str = "job", **fields: Any
) -> list[Job]:
    jobs = [
        Job(source_id=source_id, external_id=f"{prefix}-{i}", raw_text=f"{prefix} {i}", **fields)
        for i in range(count)
    ]
    session.add_all(jobs)
    await session.flush()
    return jobs


# --- authentication and shape --------------------------------------------------------------------


async def test_listing_jobs_requires_authentication(db_client: AsyncClient) -> None:
    response = await db_client.get(JOBS)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


async def test_a_user_with_no_sources_sees_no_jobs(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    response = await db_client.get(JOBS, headers=ann)

    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


async def test_a_job_is_returned_with_its_fields(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str]
) -> None:
    source_id = await new_source(db_client, ann, "feed")
    await add_jobs(db_session, source_id, 1, posted_at=T0)

    item = (await db_client.get(JOBS, headers=ann)).json()["items"][0]

    assert item["source_id"] == str(source_id)
    assert item["external_id"] == "job-0" and item["raw_text"] == "job 0"
    assert item["extraction_status"] == "pending"
    assert item["posted_at"].startswith("2026-10-26T12:00:00")
    assert item["title"] is None and item["company"] is None and item["stack"] == []
    assert "user_id" not in item  # ownership is by source, and is not exposed


# --- ownership: other users' jobs never appear ---------------------------------------------------


async def test_other_users_jobs_never_appear_in_the_list_or_the_total(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str], bob: dict[str, str]
) -> None:
    ann_source = await new_source(db_client, ann, "ann-feed")
    bob_source = await new_source(db_client, bob, "bob-feed")
    await add_jobs(db_session, ann_source, 3, prefix="ann")
    await add_jobs(db_session, bob_source, 5, prefix="bob")

    ann_page = (await db_client.get(JOBS, headers=ann)).json()
    bob_page = (await db_client.get(JOBS, headers=bob)).json()

    assert ann_page["total"] == 3
    assert {j["external_id"] for j in ann_page["items"]} == {"ann-0", "ann-1", "ann-2"}
    assert bob_page["total"] == 5
    assert all(j["external_id"].startswith("bob") for j in bob_page["items"])
    assert {j["source_id"] for j in ann_page["items"]} == {str(ann_source)}


async def test_the_total_stays_scoped_when_paginating(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str], bob: dict[str, str]
) -> None:
    ann_source = await new_source(db_client, ann, "ann-feed")
    await add_jobs(db_session, await new_source(db_client, bob, "bob-feed"), 30, prefix="bob")
    await add_jobs(db_session, ann_source, 4, prefix="ann")

    page = (await db_client.get(JOBS, params={"limit": 2, "offset": 2}, headers=ann)).json()

    assert page["total"] == 4  # bob's 30 are not counted
    assert len(page["items"]) == 2


async def test_a_user_who_deleted_a_source_no_longer_sees_its_jobs(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str]
) -> None:
    source_id = await new_source(db_client, ann, "feed")
    await add_jobs(db_session, source_id, 2)
    await db_client.delete(f"{SOURCES}/{source_id}", headers=ann)

    assert (await db_client.get(JOBS, headers=ann)).json()["total"] == 0


# --- source_id filter ----------------------------------------------------------------------------


async def test_the_source_filter_narrows_to_one_of_the_users_sources(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str]
) -> None:
    first = await new_source(db_client, ann, "first")
    second = await new_source(db_client, ann, "second")
    await add_jobs(db_session, first, 2, prefix="one")
    await add_jobs(db_session, second, 3, prefix="two")

    page = (await db_client.get(JOBS, params={"source_id": str(second)}, headers=ann)).json()

    assert page["total"] == 3
    assert all(j["source_id"] == str(second) for j in page["items"])


async def test_filtering_by_another_users_source_returns_nothing(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str], bob: dict[str, str]
) -> None:
    await new_source(db_client, ann, "ann-feed")
    bob_source = await new_source(db_client, bob, "bob-feed")
    await add_jobs(db_session, bob_source, 3, prefix="bob")

    page = (await db_client.get(JOBS, params={"source_id": str(bob_source)}, headers=ann)).json()

    assert page["items"] == [] and page["total"] == 0  # same as an empty source: nothing leaks


async def test_filtering_by_an_unknown_source_returns_nothing(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    page = (await db_client.get(JOBS, params={"source_id": str(uuid.uuid4())}, headers=ann)).json()

    assert page["items"] == [] and page["total"] == 0


async def test_a_malformed_source_id_is_422(db_client: AsyncClient, ann: dict[str, str]) -> None:
    response = await db_client.get(JOBS, params={"source_id": "nope"}, headers=ann)

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


# --- order and pagination ------------------------------------------------------------------------


async def test_jobs_are_newest_first(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str]
) -> None:
    source_id = await new_source(db_client, ann, "feed")
    for i, name in enumerate(["oldest", "middle", "newest"]):
        db_session.add(
            Job(
                source_id=source_id,
                external_id=name,
                raw_text=name,
                created_at=T0 + timedelta(hours=i),
            )
        )
    await db_session.flush()

    items = (await db_client.get(JOBS, headers=ann)).json()["items"]

    assert [j["external_id"] for j in items] == ["newest", "middle", "oldest"]


async def test_jobs_with_the_same_timestamp_are_ordered_by_id_descending(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str]
) -> None:
    source_id = await new_source(db_client, ann, "feed")
    jobs = [
        Job(source_id=source_id, external_id=f"j{i}", raw_text="x", created_at=T0) for i in range(6)
    ]  # one sync inserts all its jobs with the same created_at
    db_session.add_all(jobs)
    await db_session.flush()

    items = (await db_client.get(JOBS, headers=ann)).json()["items"]

    assert [j["id"] for j in items] == sorted((str(j.id) for j in jobs), reverse=True)


async def test_pages_cover_every_job_exactly_once(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str]
) -> None:
    source_id = await new_source(db_client, ann, "feed")
    for i in range(7):  # all tied on created_at: only the id tiebreaker keeps pages consistent
        db_session.add(Job(source_id=source_id, external_id=f"j{i}", raw_text="x", created_at=T0))
    await db_session.flush()

    seen: list[str] = []
    for offset in (0, 3, 6):
        page = (
            await db_client.get(JOBS, params={"limit": 3, "offset": offset}, headers=ann)
        ).json()
        assert page["total"] == 7
        seen += [j["external_id"] for j in page["items"]]

    assert sorted(seen) == [f"j{i}" for i in range(7)]


@pytest.mark.parametrize(
    "params", [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"offset": 100_001}, {"limit": "x"}]
)
async def test_pagination_bounds_are_enforced(
    db_client: AsyncClient, ann: dict[str, str], params: dict[str, Any]
) -> None:
    response = await db_client.get(JOBS, params=params, headers=ann)

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


async def test_an_offset_past_the_end_gives_an_empty_page_with_the_real_total(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str]
) -> None:
    await add_jobs(db_session, await new_source(db_client, ann, "feed"), 2)

    page = (await db_client.get(JOBS, params={"offset": 50}, headers=ann)).json()

    assert page["items"] == [] and page["total"] == 2
