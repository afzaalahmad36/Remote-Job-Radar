import uuid

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, Response

from app.core.exceptions import QueueUnavailableError
from app.core.rate_limit import RateLimiter
from tests.auth_helpers import auth_headers
from tests.fakes import FakeTaskQueue

URL = "https://example.com/feed.xml"
SOURCES = "/api/v1/sources"


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


# --- accepting the request -----------------------------------------------------------------------


async def test_sync_is_accepted_and_queued(
    db_client: AsyncClient, ann: dict[str, str], task_queue: FakeTaskQueue
) -> None:
    source_id = await new_source(db_client, ann)

    response = await sync(db_client, ann, source_id)

    assert response.status_code == 202
    assert response.json() == {"source_id": source_id, "status": "queued"}
    assert task_queue.enqueued == [uuid.UUID(source_id)]


async def test_the_endpoint_does_not_fetch_anything_itself(
    db_client: AsyncClient, ann: dict[str, str], feed_server: object
) -> None:
    source_id = await new_source(db_client, ann)

    await sync(db_client, ann, source_id)

    assert feed_server.requests == []  # type: ignore[attr-defined]  # the worker does the fetching


async def test_a_sync_already_in_progress_is_accepted_but_not_queued_again(
    db_client: AsyncClient, ann: dict[str, str], task_queue: FakeTaskQueue
) -> None:
    source_id = await new_source(db_client, ann)
    task_queue.syncing.add(uuid.UUID(source_id))

    response = await sync(db_client, ann, source_id)

    assert response.status_code == 202  # not a 409: the data is already being refreshed
    assert response.json() == {"source_id": source_id, "status": "already_running"}
    assert task_queue.enqueued == []


async def test_it_can_be_queued_again_once_the_running_sync_is_done(
    db_client: AsyncClient, ann: dict[str, str], task_queue: FakeTaskQueue
) -> None:
    source_id = await new_source(db_client, ann)
    task_queue.syncing.add(uuid.UUID(source_id))
    await sync(db_client, ann, source_id)

    task_queue.syncing.clear()
    response = await sync(db_client, ann, source_id)

    assert response.json()["status"] == "queued"
    assert task_queue.enqueued == [uuid.UUID(source_id)]


async def test_an_inactive_source_can_still_be_synced_by_hand(
    db_client: AsyncClient, ann: dict[str, str], task_queue: FakeTaskQueue
) -> None:
    source_id = await new_source(db_client, ann)
    await db_client.patch(f"{SOURCES}/{source_id}", json={"is_active": False}, headers=ann)

    response = await sync(db_client, ann, source_id)

    assert response.json()["status"] == "queued"  # inactive only exempts it from scheduled syncs


async def test_the_queue_being_down_is_a_503_with_no_details(
    db_client: AsyncClient, ann: dict[str, str], task_queue: FakeTaskQueue
) -> None:
    source_id = await new_source(db_client, ann)
    task_queue.fail_with = QueueUnavailableError("The sync queue is unavailable")

    response = await sync(db_client, ann, source_id)

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The sync queue is unavailable",
        "code": "queue_unavailable",
    }


# --- ownership and authentication ----------------------------------------------------------------


async def test_syncing_another_users_source_is_a_404_and_queues_nothing(
    db_client: AsyncClient, ann: dict[str, str], bob: dict[str, str], task_queue: FakeTaskQueue
) -> None:
    source_id = await new_source(db_client, ann)

    response = await sync(db_client, bob, source_id)

    assert response.status_code == 404
    assert response.json() == {"detail": "Source not found", "code": "not_found"}
    assert task_queue.enqueued == []
    assert task_queue.is_syncing_calls == []  # not even asked: no way to probe someone's syncs


async def test_a_missing_source_looks_the_same_as_someone_elses(
    db_client: AsyncClient, ann: dict[str, str], bob: dict[str, str]
) -> None:
    source_id = await new_source(db_client, ann)

    others = await sync(db_client, bob, source_id)
    missing = await sync(db_client, bob, str(uuid.uuid4()))

    assert others.status_code == missing.status_code == 404
    assert others.json() == missing.json()


async def test_sync_requires_authentication(
    db_client: AsyncClient, task_queue: FakeTaskQueue
) -> None:
    response = await db_client.post(f"{SOURCES}/{uuid.uuid4()}/sync")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert task_queue.enqueued == []


async def test_a_malformed_id_is_422(db_client: AsyncClient, ann: dict[str, str]) -> None:
    assert (await sync(db_client, ann, "not-a-uuid")).status_code == 422


# --- rate limiting -------------------------------------------------------------------------------


async def test_syncs_are_rate_limited_per_user(
    app: FastAPI,
    db_client: AsyncClient,
    ann: dict[str, str],
    bob: dict[str, str],
    task_queue: FakeTaskQueue,
) -> None:
    app.state.sync_rate_limiter = RateLimiter(max_attempts=2, window_seconds=60)
    ann_source = await new_source(db_client, ann)
    bob_source = await new_source(db_client, bob)

    statuses = [(await sync(db_client, ann, ann_source)).status_code for _ in range(3)]
    blocked = await sync(db_client, ann, ann_source)
    bob_response = await sync(db_client, bob, bob_source)

    assert statuses == [202, 202, 429]
    assert blocked.json()["code"] == "rate_limited"
    assert 1 <= int(blocked.headers["retry-after"]) <= 60
    assert bob_response.status_code == 202  # the budget is per user, not shared
    assert len(task_queue.enqueued) == 3  # the blocked attempts queued nothing
