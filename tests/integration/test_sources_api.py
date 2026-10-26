import uuid
from typing import Any

import pytest
from httpx import AsyncClient, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api_deps import get_resolver
from app.core.url_safety import system_resolver
from app.models import Job, Source, User
from tests.auth_helpers import auth_headers
from tests.fakes import FakeResolver

SOURCES = "/api/v1/sources"
RSS = {"type": "rss", "name": "Remote OK", "config": {"url": "https://example.com/feed.xml"}}


async def create(client: AsyncClient, headers: dict[str, str], **overrides: Any) -> Response:
    return await client.post(SOURCES, json=RSS | overrides, headers=headers)


@pytest.fixture
async def ann(db_client: AsyncClient) -> dict[str, str]:
    return await auth_headers(db_client, "ann@example.com")


@pytest.fixture
async def bob(db_client: AsyncClient) -> dict[str, str]:
    return await auth_headers(db_client, "bob@example.com")


# --- authentication ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", SOURCES),
        ("GET", SOURCES),
        ("GET", f"{SOURCES}/{uuid.uuid4()}"),
        ("PATCH", f"{SOURCES}/{uuid.uuid4()}"),
        ("DELETE", f"{SOURCES}/{uuid.uuid4()}"),
    ],
)
async def test_every_endpoint_requires_authentication(
    db_client: AsyncClient, method: str, path: str
) -> None:
    response = await db_client.request(method, path, json=RSS if method != "GET" else None)

    assert response.status_code == 401
    assert response.json()["code"] == "authentication_failed"
    assert response.headers["www-authenticate"] == "Bearer"


# --- create and read ---------------------------------------------------------------------------


async def test_create_returns_the_new_source(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str]
) -> None:
    response = await create(db_client, ann)

    assert response.status_code == 201
    body = response.json()
    assert body["type"] == "rss"
    assert body["name"] == "Remote OK"
    assert body["config"] == {"url": "https://example.com/feed.xml"}
    assert body["is_active"] is True
    assert body["last_synced_at"] is None
    assert {"id", "created_at", "updated_at"} <= set(body)
    assert "user_id" not in body
    # The owner is the authenticated user, not something the client can choose.
    stored = await db_session.scalar(select(Source))
    user = await db_session.scalar(select(User).where(User.email == "ann@example.com"))
    assert stored is not None and user is not None
    assert stored.user_id == user.id


async def test_create_trims_the_name_and_the_url(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    response = await create(
        db_client, ann, name="  Remote OK  ", config={"url": "  https://example.com/feed.xml  "}
    )

    assert response.json()["name"] == "Remote OK"
    assert response.json()["config"]["url"] == "https://example.com/feed.xml"


async def test_get_returns_own_source(db_client: AsyncClient, ann: dict[str, str]) -> None:
    created = (await create(db_client, ann)).json()

    response = await db_client.get(f"{SOURCES}/{created['id']}", headers=ann)

    assert response.status_code == 200
    assert response.json() == created


async def test_the_client_cannot_choose_the_owner(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    response = await create(db_client, ann, user_id=str(uuid.uuid4()))

    assert response.status_code == 422  # unknown fields are rejected, not silently ignored


# --- validation --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"config": {}}, "config.url: Field required"),
        ({"config": {"url": "https://example.com/feed", "extra": 1}}, "config.extra: Extra inputs"),
        ({"config": {"url": ""}}, "config.url"),
        ({"config": {"url": 123}}, "config.url"),
        ({"config": {"link": "https://example.com/feed"}}, "config.url: Field required"),
        (
            {"type": "telegram", "config": {"channel": "jobs"}},
            "Source type 'telegram' is not supported yet",
        ),
    ],
    ids=["missing-url", "unknown-key", "empty-url", "non-string-url", "wrong-key", "telegram"],
)
async def test_invalid_config_is_422_with_a_clear_message(
    db_client: AsyncClient, ann: dict[str, str], overrides: dict[str, Any], message: str
) -> None:
    response = await create(db_client, ann, **overrides)

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_config"
    assert message in response.json()["detail"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"type": "mastodon"},
        {"type": None},
        {"name": ""},
        {"name": "   "},
        {"name": "x" * 256},
        {"config": "https://example.com/feed"},
        {"config": None},
    ],
    ids=[
        "unknown-type",
        "null-type",
        "empty-name",
        "blank-name",
        "long-name",
        "string-config",
        "null-config",
    ],
)
async def test_malformed_requests_are_422(
    db_client: AsyncClient, ann: dict[str, str], overrides: dict[str, Any]
) -> None:
    response = await create(db_client, ann, **overrides)

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/feed",
        "http://127.0.0.1/feed",
        "http://db/",
        "http://redis/",
        "http://169.254.169.254/latest/meta-data/",
        "http://2130706433/",
        "http://[::ffff:127.0.0.1]/",
        "http://user:pass@example.com/feed",
        "http://example.com:8080/feed",
        "ftp://example.com/feed",
        "http://internal.example.com/feed",  # public-looking name that resolves to 10.0.0.5
    ],
)
async def test_unsafe_urls_are_rejected_on_create(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str], url: str
) -> None:
    response = await create(db_client, ann, config={"url": url})

    assert response.status_code == 422
    assert response.json()["code"] == "unsafe_url"
    assert (await db_session.scalars(select(Source))).all() == []  # nothing was stored


def test_the_default_resolver_is_the_system_one() -> None:
    # Tests replace it with a fake; production uses the operating system's resolver.
    assert get_resolver() is system_resolver


async def test_url_checks_use_the_injected_resolver_and_not_the_network(
    db_client: AsyncClient, ann: dict[str, str], resolver: FakeResolver
) -> None:
    await create(db_client, ann)

    assert resolver.calls == ["example.com"]


# --- list and pagination -----------------------------------------------------------------------


async def test_list_is_empty_for_a_new_user(db_client: AsyncClient, ann: dict[str, str]) -> None:
    response = await db_client.get(SOURCES, headers=ann)

    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


async def test_list_only_shows_the_users_own_sources(
    db_client: AsyncClient, ann: dict[str, str], bob: dict[str, str]
) -> None:
    await create(db_client, ann, name="ann-1", config={"url": "https://example.com/ann-1.xml"})
    await create(db_client, ann, name="ann-2", config={"url": "https://example.com/ann-2.xml"})
    await create(db_client, bob, name="bob-1")

    ann_page = (await db_client.get(SOURCES, headers=ann)).json()
    bob_page = (await db_client.get(SOURCES, headers=bob)).json()

    assert sorted(s["name"] for s in ann_page["items"]) == ["ann-1", "ann-2"]
    assert ann_page["total"] == 2
    assert [s["name"] for s in bob_page["items"]] == ["bob-1"]
    assert bob_page["total"] == 1


async def test_pages_cover_every_source_exactly_once(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    for i in range(5):
        await create(
            db_client, ann, name=f"feed-{i}", config={"url": f"https://example.com/{i}.xml"}
        )

    seen: list[str] = []
    for offset in (0, 2, 4):
        page = (
            await db_client.get(SOURCES, params={"limit": 2, "offset": offset}, headers=ann)
        ).json()
        assert page["total"] == 5
        assert (page["limit"], page["offset"]) == (2, offset)
        seen += [s["name"] for s in page["items"]]

    assert sorted(seen) == [f"feed-{i}" for i in range(5)]  # no repeats, none missing


async def test_an_offset_past_the_end_gives_an_empty_page(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    await create(db_client, ann)

    page = (await db_client.get(SOURCES, params={"offset": 50}, headers=ann)).json()

    assert page["items"] == []
    assert page["total"] == 1


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 101},
        {"limit": -1},
        {"offset": -1},
        {"offset": 100_001},
        {"limit": "x"},
    ],
)
async def test_pagination_bounds_are_enforced(
    db_client: AsyncClient, ann: dict[str, str], params: dict[str, Any]
) -> None:
    response = await db_client.get(SOURCES, params=params, headers=ann)

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


async def test_the_maximum_limit_is_allowed(db_client: AsyncClient, ann: dict[str, str]) -> None:
    response = await db_client.get(SOURCES, params={"limit": 100}, headers=ann)

    assert response.status_code == 200
    assert response.json()["limit"] == 100


# --- update ------------------------------------------------------------------------------------


async def test_update_changes_only_the_fields_sent(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    created = (await create(db_client, ann)).json()

    response = await db_client.patch(
        f"{SOURCES}/{created['id']}", json={"name": "Renamed"}, headers=ann
    )

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Renamed"
    assert body["config"] == created["config"]
    assert body["is_active"] is True


async def test_a_source_can_be_deactivated_and_reactivated(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    source_id = (await create(db_client, ann)).json()["id"]

    off = await db_client.patch(f"{SOURCES}/{source_id}", json={"is_active": False}, headers=ann)
    on = await db_client.patch(f"{SOURCES}/{source_id}", json={"is_active": True}, headers=ann)

    assert off.json()["is_active"] is False
    assert on.json()["is_active"] is True


async def test_update_replaces_the_config_and_rechecks_the_url(
    db_client: AsyncClient, ann: dict[str, str], resolver: FakeResolver
) -> None:
    source_id = (await create(db_client, ann)).json()["id"]

    good = await db_client.patch(
        f"{SOURCES}/{source_id}",
        json={"config": {"url": "https://feeds.example.org/rss"}},
        headers=ann,
    )
    bad = await db_client.patch(
        f"{SOURCES}/{source_id}", json={"config": {"url": "http://169.254.169.254/"}}, headers=ann
    )
    again = await db_client.get(f"{SOURCES}/{source_id}", headers=ann)

    assert good.json()["config"] == {"url": "https://feeds.example.org/rss"}
    assert bad.status_code == 422
    assert bad.json()["code"] == "unsafe_url"
    assert again.json()["config"] == {
        "url": "https://feeds.example.org/rss"
    }  # unchanged by the failure
    assert resolver.calls == ["example.com", "feeds.example.org"]  # the unsafe IP never hit DNS


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"config": {}}, "invalid_config"),
        ({"config": {"url": "https://example.com/x", "extra": 1}}, "invalid_config"),
        ({"name": None}, "validation_error"),
        ({"is_active": None}, "validation_error"),
        ({"config": None}, "validation_error"),
        ({"type": "telegram"}, "validation_error"),  # the type cannot be changed
        ({"user_id": str(uuid.uuid4())}, "validation_error"),
        ({"name": ""}, "validation_error"),
    ],
)
async def test_invalid_updates_are_422(
    db_client: AsyncClient, ann: dict[str, str], body: dict[str, Any], code: str
) -> None:
    source_id = (await create(db_client, ann)).json()["id"]

    response = await db_client.patch(f"{SOURCES}/{source_id}", json=body, headers=ann)

    assert response.status_code == 422
    assert response.json()["code"] == code


async def test_an_empty_update_changes_nothing(db_client: AsyncClient, ann: dict[str, str]) -> None:
    created = (await create(db_client, ann)).json()

    response = await db_client.patch(f"{SOURCES}/{created['id']}", json={}, headers=ann)

    assert response.status_code == 200
    assert response.json()["name"] == created["name"]


# --- update: the full URL check runs again, and a rejected update changes nothing ---------------


async def test_update_with_a_config_looks_the_host_up_again_even_if_the_url_is_unchanged(
    db_client: AsyncClient, ann: dict[str, str], resolver: FakeResolver
) -> None:
    created = (await create(db_client, ann)).json()

    await db_client.patch(
        f"{SOURCES}/{created['id']}", json={"config": created["config"]}, headers=ann
    )

    assert resolver.calls == ["example.com", "example.com"]  # DNS may have changed since creation


async def test_updating_only_the_name_or_active_flag_does_no_dns_lookup(
    db_client: AsyncClient, ann: dict[str, str], resolver: FakeResolver
) -> None:
    source_id = (await create(db_client, ann)).json()["id"]

    await db_client.patch(
        f"{SOURCES}/{source_id}", json={"name": "New", "is_active": False}, headers=ann
    )

    assert resolver.calls == ["example.com"]  # only the lookup from creating it


async def test_update_rejects_a_hostname_that_now_resolves_to_a_private_address(
    db_client: AsyncClient, ann: dict[str, str], resolver: FakeResolver
) -> None:
    source_id = (await create(db_client, ann)).json()["id"]

    response = await db_client.patch(
        f"{SOURCES}/{source_id}",
        json={"config": {"url": "https://internal.example.com/feed"}},  # resolves to 10.0.0.5
        headers=ann,
    )

    assert response.status_code == 422
    assert response.json()["code"] == "unsafe_url"
    assert resolver.calls[-1] == "internal.example.com"  # the DNS part of the check really ran


async def test_a_rejected_update_changes_nothing_at_all(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    created = (await create(db_client, ann)).json()

    response = await db_client.patch(
        f"{SOURCES}/{created['id']}",
        json={"name": "Renamed", "is_active": False, "config": {"url": "http://localhost/"}},
        headers=ann,
    )
    after = (await db_client.get(f"{SOURCES}/{created['id']}", headers=ann)).json()

    assert response.status_code == 422
    assert (after["name"], after["is_active"], after["config"]) == (
        created["name"],
        True,
        created["config"],
    )


# --- duplicates: one source per user per normalized URL ----------------------------------------


@pytest.mark.parametrize(
    "variant",
    [
        "https://example.com/feed.xml",
        "https://EXAMPLE.com/feed.xml",
        "https://example.com/feed.xml/",
        "https://example.com/feed.xml#latest",
        "https://example.com:443/feed.xml",
        "https://example.com./feed.xml",
        "  https://example.com/feed.xml  ",
    ],
)
async def test_the_same_url_twice_is_a_409_for_the_same_user(
    db_client: AsyncClient,
    db_session: AsyncSession,
    ann: dict[str, str],
    resolver: FakeResolver,
    variant: str,
) -> None:
    await create(db_client, ann)

    response = await create(db_client, ann, name="Again", config={"url": variant})

    assert response.status_code == 409
    assert response.json() == {
        "detail": "You already have a source for this URL",
        "code": "duplicate_source",
    }
    assert len((await db_session.scalars(select(Source))).all()) == 1
    assert resolver.calls == ["example.com"]  # the duplicate was caught before any DNS lookup


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/feed.xml",
        "https://example.com/Feed.xml",
        "https://example.com/other.xml",
        "https://example.com/feed.xml?tag=python",
        "https://feeds.example.org/feed.xml",
    ],
)
async def test_different_urls_are_not_duplicates(
    db_client: AsyncClient, ann: dict[str, str], url: str
) -> None:
    await create(db_client, ann)

    response = await create(db_client, ann, name="Other", config={"url": url})

    assert response.status_code == 201


async def test_another_user_can_add_the_same_url(
    db_client: AsyncClient, ann: dict[str, str], bob: dict[str, str]
) -> None:
    assert (await create(db_client, ann)).status_code == 201

    response = await create(db_client, bob)

    assert response.status_code == 201


async def test_a_deactivated_source_still_blocks_its_url(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    source_id = (await create(db_client, ann)).json()["id"]
    await db_client.patch(f"{SOURCES}/{source_id}", json={"is_active": False}, headers=ann)

    assert (await create(db_client, ann)).status_code == 409  # reactivate it instead


async def test_deleting_a_source_frees_its_url(db_client: AsyncClient, ann: dict[str, str]) -> None:
    source_id = (await create(db_client, ann)).json()["id"]
    await db_client.delete(f"{SOURCES}/{source_id}", headers=ann)

    assert (await create(db_client, ann)).status_code == 201


async def test_update_to_another_sources_url_is_a_409_and_changes_nothing(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    await create(db_client, ann)
    other = (
        await create(db_client, ann, name="Other", config={"url": "https://feeds.example.org/rss"})
    ).json()

    response = await db_client.patch(
        f"{SOURCES}/{other['id']}",
        json={"name": "Renamed", "config": {"url": "https://EXAMPLE.com/feed.xml/"}},
        headers=ann,
    )
    after = (await db_client.get(f"{SOURCES}/{other['id']}", headers=ann)).json()

    assert response.status_code == 409
    assert response.json()["code"] == "duplicate_source"
    assert (after["name"], after["config"]) == ("Other", {"url": "https://feeds.example.org/rss"})


async def test_a_source_can_be_updated_to_a_variant_of_its_own_url(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    source_id = (await create(db_client, ann)).json()["id"]

    response = await db_client.patch(
        f"{SOURCES}/{source_id}",
        json={"config": {"url": "https://example.com/feed.xml/"}},
        headers=ann,
    )

    assert response.status_code == 200  # it does not clash with itself
    assert response.json()["config"]["url"] == "https://example.com/feed.xml/"  # stored as typed


async def test_update_to_a_unique_new_url_works(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    await create(db_client, ann)
    other_id = (
        await create(db_client, ann, name="Other", config={"url": "https://feeds.example.org/rss"})
    ).json()["id"]

    response = await db_client.patch(
        f"{SOURCES}/{other_id}",
        json={"config": {"url": "https://example.com/new.xml"}},
        headers=ann,
    )

    assert response.status_code == 200


async def test_another_user_can_update_to_a_url_the_first_user_has(
    db_client: AsyncClient, ann: dict[str, str], bob: dict[str, str]
) -> None:
    await create(db_client, ann)
    bobs = (await create(db_client, bob, config={"url": "https://feeds.example.org/rss"})).json()

    response = await db_client.patch(
        f"{SOURCES}/{bobs['id']}",
        json={"config": {"url": "https://example.com/feed.xml"}},
        headers=bob,
    )

    assert response.status_code == 200


# --- delete ------------------------------------------------------------------------------------


async def test_delete_removes_the_source(db_client: AsyncClient, ann: dict[str, str]) -> None:
    source_id = (await create(db_client, ann)).json()["id"]

    response = await db_client.delete(f"{SOURCES}/{source_id}", headers=ann)

    assert response.status_code == 204
    assert response.content == b""
    assert (await db_client.get(f"{SOURCES}/{source_id}", headers=ann)).status_code == 404


async def test_deleting_a_source_deletes_its_jobs(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str]
) -> None:
    source_id = uuid.UUID((await create(db_client, ann)).json()["id"])
    db_session.add_all(
        [Job(source_id=source_id, external_id=f"post-{i}", raw_text="Hiring") for i in range(3)]
    )
    await db_session.flush()

    await db_client.delete(f"{SOURCES}/{source_id}", headers=ann)

    assert (await db_session.scalars(select(Job))).all() == []


async def test_deactivating_keeps_the_jobs(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str]
) -> None:
    source_id = uuid.UUID((await create(db_client, ann)).json()["id"])
    db_session.add(Job(source_id=source_id, external_id="post-1", raw_text="Hiring"))
    await db_session.flush()

    await db_client.patch(f"{SOURCES}/{source_id}", json={"is_active": False}, headers=ann)

    assert len((await db_session.scalars(select(Job))).all()) == 1


# --- ownership: other users' sources are "not found", never "forbidden" ----------------------


async def test_another_users_source_is_404_on_get_update_and_delete(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str], bob: dict[str, str]
) -> None:
    source_id = (await create(db_client, ann)).json()["id"]
    url = f"{SOURCES}/{source_id}"

    got = await db_client.get(url, headers=bob)
    patched = await db_client.patch(url, json={"name": "hijacked", "is_active": False}, headers=bob)
    deleted = await db_client.delete(url, headers=bob)

    for response in (got, patched, deleted):
        assert response.status_code == 404
        assert response.json() == {"detail": "Source not found", "code": "not_found"}
    # ...and the owner's source is untouched.
    mine = (await db_client.get(url, headers=ann)).json()
    assert (mine["name"], mine["is_active"]) == ("Remote OK", True)


async def test_a_missing_source_looks_exactly_like_someone_elses(
    db_client: AsyncClient, ann: dict[str, str], bob: dict[str, str]
) -> None:
    source_id = (await create(db_client, ann)).json()["id"]

    someone_elses = await db_client.get(f"{SOURCES}/{source_id}", headers=bob)
    missing = await db_client.get(f"{SOURCES}/{uuid.uuid4()}", headers=bob)

    assert someone_elses.status_code == missing.status_code == 404
    assert someone_elses.json() == missing.json()


async def test_a_malformed_id_is_422(db_client: AsyncClient, ann: dict[str, str]) -> None:
    response = await db_client.get(f"{SOURCES}/not-a-uuid", headers=ann)

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
