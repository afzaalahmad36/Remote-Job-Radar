import uuid

import httpx
import pytest
from fastapi import FastAPI
from httpx import AsyncClient, Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api_deps import get_fetchers
from app.core.rate_limit import RateLimiter
from app.models import Job, SourceType
from tests.auth_helpers import auth_headers
from tests.fakes import FakeFeedServer, feed_bytes

URL = "https://example.com/feed.xml"
SOURCES = "/api/v1/sources"


def test_the_default_fetchers_cover_every_creatable_source_type() -> None:
    # Tests replace this dependency with fakes; production uses the real fetchers.
    assert set(get_fetchers()) == {SourceType.RSS}


@pytest.fixture
async def ann(db_client: AsyncClient) -> dict[str, str]:
    return await auth_headers(db_client, "ann@example.com")


@pytest.fixture
async def bob(db_client: AsyncClient) -> dict[str, str]:
    return await auth_headers(db_client, "bob@example.com")


async def new_source(client: AsyncClient, headers: dict[str, str], url: str = URL) -> str:
    response = await client.post(
        SOURCES, json={"type": "rss", "name": "feed", "config": {"url": url}}, headers=headers
    )
    assert response.status_code == 201
    return str(response.json()["id"])


async def sync(client: AsyncClient, headers: dict[str, str], source_id: str) -> Response:
    return await client.post(f"{SOURCES}/{source_id}/sync", headers=headers)


async def test_sync_stores_the_posts_and_returns_a_summary(
    db_client: AsyncClient, ann: dict[str, str], feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, feed_bytes("rss2_basic.xml"))
    source_id = await new_source(db_client, ann)

    response = await sync(db_client, ann, source_id)

    assert response.status_code == 200
    assert response.json() == {"fetched": 3, "inserted": 3, "skipped": 0, "error": None}
    jobs = (await db_client.get("/api/v1/jobs", headers=ann)).json()
    assert jobs["total"] == 3


async def test_syncing_again_inserts_nothing(
    db_client: AsyncClient, ann: dict[str, str], feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, feed_bytes("rss2_basic.xml"))
    source_id = await new_source(db_client, ann)
    await sync(db_client, ann, source_id)

    response = await sync(db_client, ann, source_id)

    assert response.json() == {"fetched": 3, "inserted": 0, "skipped": 3, "error": None}
    assert (await db_client.get("/api/v1/jobs", headers=ann)).json()["total"] == 3


async def test_the_sync_updates_the_source(
    db_client: AsyncClient, ann: dict[str, str], feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, feed_bytes("rss2_basic.xml"))
    source_id = await new_source(db_client, ann)

    await sync(db_client, ann, source_id)
    source = (await db_client.get(f"{SOURCES}/{source_id}", headers=ann)).json()

    assert source["last_synced_at"] is not None
    assert source["last_error"] is None
    assert "etag" not in source  # internal bookkeeping is not exposed


async def test_a_failing_feed_is_a_200_with_the_error_and_it_shows_on_the_source(
    db_client: AsyncClient, ann: dict[str, str], feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, status=503)
    source_id = await new_source(db_client, ann)

    response = await sync(db_client, ann, source_id)
    source = (await db_client.get(f"{SOURCES}/{source_id}", headers=ann)).json()

    assert response.status_code == 200  # the request worked; the feed did not
    assert response.json() == {"fetched": 0, "inserted": 0, "skipped": 0, "error": "HTTP 503"}
    assert source["last_error"] == "HTTP 503"
    assert source["last_synced_at"] is None


async def test_a_later_good_sync_clears_the_error(
    db_client: AsyncClient, ann: dict[str, str], feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, status=500)
    source_id = await new_source(db_client, ann)
    await sync(db_client, ann, source_id)
    feed_server.add(URL, feed_bytes("rss2_basic.xml"))

    await sync(db_client, ann, source_id)

    assert (await db_client.get(f"{SOURCES}/{source_id}", headers=ann)).json()["last_error"] is None


async def test_the_error_returned_to_the_user_does_not_leak_internals(
    db_client: AsyncClient, ann: dict[str, str], feed_server: FakeFeedServer
) -> None:
    feed_server.redirect(URL, "http://internal.example.com/admin")  # resolves to 10.0.0.5
    source_id = await new_source(db_client, ann)

    body = (await sync(db_client, ann, source_id)).json()

    assert body["error"].startswith("Redirect not allowed")
    for leak in ("10.0.0.5", "internal.example.com", "admin"):
        assert leak not in body["error"]
    assert feed_server.urls == [URL]


async def test_an_inactive_source_can_still_be_synced_by_hand(
    db_client: AsyncClient, ann: dict[str, str], feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, feed_bytes("rss2_basic.xml"))
    source_id = await new_source(db_client, ann)
    await db_client.patch(f"{SOURCES}/{source_id}", json={"is_active": False}, headers=ann)

    response = await sync(db_client, ann, source_id)

    assert response.json()["inserted"] == 3  # "inactive" will only exempt it from scheduled syncs


# --- ownership and authentication ----------------------------------------------------------------


async def test_syncing_another_users_source_is_a_404_and_fetches_nothing(
    db_client: AsyncClient, ann: dict[str, str], bob: dict[str, str], feed_server: FakeFeedServer
) -> None:
    feed_server.add(URL, feed_bytes("rss2_basic.xml"))
    source_id = await new_source(db_client, ann)

    response = await sync(db_client, bob, source_id)

    assert response.status_code == 404
    assert response.json() == {"detail": "Source not found", "code": "not_found"}
    assert feed_server.requests == []  # nothing was fetched on bob's behalf


async def test_a_missing_source_looks_the_same_as_someone_elses(
    db_client: AsyncClient, ann: dict[str, str], bob: dict[str, str]
) -> None:
    source_id = await new_source(db_client, ann)

    others = await sync(db_client, bob, source_id)
    missing = await sync(db_client, bob, str(uuid.uuid4()))

    assert others.status_code == missing.status_code == 404
    assert others.json() == missing.json()


async def test_sync_requires_authentication(db_client: AsyncClient) -> None:
    response = await db_client.post(f"{SOURCES}/{uuid.uuid4()}/sync")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


async def test_a_malformed_id_is_422(db_client: AsyncClient, ann: dict[str, str]) -> None:
    assert (await sync(db_client, ann, "not-a-uuid")).status_code == 422


async def test_synced_jobs_belong_to_the_source(
    db_client: AsyncClient,
    db_session: AsyncSession,
    ann: dict[str, str],
    feed_server: FakeFeedServer,
) -> None:
    feed_server.add(URL, feed_bytes("rss2_basic.xml"))
    source_id = await new_source(db_client, ann)
    await sync(db_client, ann, source_id)

    owners = set((await db_session.scalars(select(Job.source_id))).all())

    assert owners == {uuid.UUID(source_id)}
    assert await db_session.scalar(select(func.count()).select_from(Job)) == 3


# --- rate limiting -------------------------------------------------------------------------------


async def test_syncs_are_rate_limited_per_user(
    app: FastAPI,
    db_client: AsyncClient,
    ann: dict[str, str],
    bob: dict[str, str],
    feed_server: FakeFeedServer,
) -> None:
    feed_server.add(URL, feed_bytes("rss2_basic.xml"))
    app.state.sync_rate_limiter = RateLimiter(max_attempts=2, window_seconds=60)
    ann_source = await new_source(db_client, ann)
    bob_source = await new_source(db_client, bob)

    statuses = [(await sync(db_client, ann, ann_source)).status_code for _ in range(3)]
    blocked = await sync(db_client, ann, ann_source)
    bob_response = await sync(db_client, bob, bob_source)

    assert statuses == [200, 200, 429]
    assert blocked.json()["code"] == "rate_limited"
    assert 1 <= int(blocked.headers["retry-after"]) <= 60
    assert bob_response.status_code == 200  # the budget is per user, not shared
    assert len(feed_server.requests) == 3  # the blocked attempt made no outbound request


async def test_a_blocked_sync_does_not_reach_the_feed_server(
    app: FastAPI, db_client: AsyncClient, ann: dict[str, str], feed_server: FakeFeedServer
) -> None:
    app.state.sync_rate_limiter = RateLimiter(max_attempts=1, window_seconds=60)
    source_id = await new_source(db_client, ann)
    feed_server.add_handler(
        URL, lambda request: httpx.Response(200, content=feed_bytes("empty.xml"))
    )

    await sync(db_client, ann, source_id)
    await sync(db_client, ann, source_id)

    assert len(feed_server.requests) == 1
