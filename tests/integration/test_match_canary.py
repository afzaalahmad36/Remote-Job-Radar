"""The canary test for matching: a unique marker planted in the CV text, a CV skill and the
preference notes is followed through the new tasks and endpoints.

It must never appear in a response of the match endpoints, a log record, a task argument, a stored
`match_status` or `job_matches` row, a recorded SQL statement of the scoring run, or an LLM call.
(It legitimately lives in the owner's own CV and preferences, which these tests do not read.)
"""

import logging
import uuid
from typing import Any

import pytest
from httpx import AsyncClient, Response
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.client import GuardedClient
from app.llm.providers.fake import FakeLLMClient
from app.llm.providers.gemini import GeminiClient
from app.services import match_service
from app.services.match_runner import RunOutcome, run_user_scoring
from app.tasks import matching
from tests.fakes import FakeTaskQueue
from tests.integration.test_match_service import SETTINGS, two_jobs
from tests.integration.test_match_tasks import FakeDedup
from tests.integration.test_matches_api import MATCHES, signed_in

MARKER = "CANARY-MARKER-5d7a19c3"
EMAIL = "canary.person@example.com"
PHONE = "+1 555 010 7788"
LINK = "https://github.com/canary-person"
SKILL = f"{MARKER}-skill"
SECRETS = (MARKER, EMAIL, PHONE, LINK, SKILL)

CV_TEXT = f"""{MARKER} Person
{EMAIL} | {PHONE} | {LINK}
Backend engineer. Python, Django, PostgreSQL.
"""


def leaks(value: str) -> list[str]:
    return [secret for secret in SECRETS if secret in value]


def log_text(record: logging.LogRecord) -> str:
    parts = [record.getMessage(), str(record.args), record.exc_text or ""]
    if record.exc_info:
        parts.append(logging.Formatter().formatException(record.exc_info))
    parts.extend(str(v) for v in vars(record).values())
    return "\n".join(parts)


class SharedSession:
    """A session factory that hands out the test's one session (so the run sees its data)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def __call__(self) -> "SharedSession":
        return self

    async def __aenter__(self) -> AsyncSession:
        return self._session

    async def __aexit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def llm_calls(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    calls: list[object] = []

    async def recorder(self: object, request: object) -> object:
        calls.append(request)
        raise AssertionError("scoring must never call an LLM client")

    for client_class in (GuardedClient, FakeLLMClient, GeminiClient):
        monkeypatch.setattr(client_class, "complete", recorder)
    return calls


async def plant_the_canary(
    client: AsyncClient, session: AsyncSession
) -> tuple[dict[str, str], uuid.UUID, list[uuid.UUID]]:
    """Upload the CV, edit the skills, save the notes (the owner's own data), add two jobs."""
    headers, user_id = await signed_in(client, session, "canary@example.com")
    uploaded = await client.put(
        "/api/v1/cv", content=CV_TEXT, headers=headers | {"content-type": "text/plain"}
    )
    assert uploaded.status_code == 200
    skills = await client.put(
        "/api/v1/cv/skills", json={"skills": [SKILL, "Python", "Django"]}, headers=headers
    )
    assert skills.status_code == 200
    notes = f"{MARKER} only remote. Reach me: {EMAIL}, {PHONE}, {LINK}"
    prefs = await client.put(
        "/api/v1/preferences", json={"notes": notes, "remote_policies": ["remote"]}, headers=headers
    )
    assert prefs.status_code == 200
    near, far = await two_jobs(session, user_id)
    return headers, user_id, [near.id, far.id]


async def test_the_marker_never_leaves_through_scoring_the_tasks_or_the_match_endpoints(
    db_client: AsyncClient,
    db_session: AsyncSession,
    task_queue: FakeTaskQueue,
    caplog: pytest.LogCaptureFixture,
    llm_calls: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.DEBUG)  # every logger, lowest level: nothing may slip through
    headers, user_id, (near, _) = await plant_the_canary(db_client, db_session)
    statements: list[str] = []

    def record(conn: object, cursor: object, statement: str, *rest: object) -> None:
        statements.append(statement)

    engine = db_session.bind.sync_engine  # type: ignore[union-attr]
    event.listen(engine, "before_cursor_execute", record)
    try:
        result = await run_user_scoring(SharedSession(db_session), user_id, SETTINGS)  # type: ignore[arg-type]
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert result.outcome is RunOutcome.COMPLETED

    # what a task would carry: publish, with the dedup, and capture the arguments
    published: list[tuple[list[str], float]] = []
    monkeypatch.setattr(
        matching.score_user_matches,
        "apply_async",
        lambda args, countdown: published.append((args, countdown)),
    )
    matching.enqueue_match_scoring(user_id, dedup=FakeDedup())

    responses: list[Response] = [
        await db_client.get(MATCHES, headers=headers),
        await db_client.get(MATCHES, params={"include_filtered": "true"}, headers=headers),
        await db_client.get(f"{MATCHES}/{near}", headers=headers),
        await db_client.get(f"{MATCHES}/status", headers=headers),
        await db_client.post(f"{MATCHES}/rescore", headers=headers),
        await db_client.get(f"{MATCHES}/{uuid.uuid4()}", headers=headers),  # an error body
        await db_client.get(MATCHES, params={"min_score": 500}, headers=headers),  # a 422
    ]

    assert [r.status_code for r in responses] == [200, 200, 200, 200, 202, 404, 422]
    for response in responses:
        assert leaks(response.text) == [], response.url
    # the explanation names only the job's own stack entries
    reasons = responses[2].json()["reasons"]
    assert "Python" in str(reasons) and SKILL not in str(reasons)
    # task arguments and queued requests are user ids and nothing else
    assert published == [([str(user_id)], matching.SCORE_DELAY_SECONDS)]
    assert all(isinstance(queued, uuid.UUID) for queued in task_queue.match_enqueued)
    assert task_queue.match_enqueued  # (the endpoints did ask)
    # what is stored by scoring holds no secret
    for table in ("match_status", "job_matches"):
        dump = (
            (await db_session.execute(text(f"SELECT {table}::text FROM {table}"))).scalars().all()
        )
        assert dump and all(leaks(row) == [] for row in dump), table
    # the run never selected the CV text or the notes
    assert not any("cvs.text" in s or "job_preferences.notes" in s for s in statements)
    # logs and LLM clients
    assert [log_text(r) for r in caplog.records if leaks(log_text(r))] == []
    assert any("Scoring run" in r.getMessage() for r in caplog.records)  # we did log: counts only
    assert llm_calls == []


async def test_a_failure_that_quotes_the_marker_is_counted_but_never_logged_or_exposed(
    db_client: AsyncClient,
    db_session: AsyncSession,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.DEBUG)
    headers, user_id, _ = await plant_the_canary(db_client, db_session)

    async def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(f"cannot score {MARKER} {SKILL} {EMAIL}")

    monkeypatch.setattr(match_service, "score_jobs_for_user", broken)

    result = await run_user_scoring(SharedSession(db_session), user_id, SETTINGS)  # type: ignore[arg-type]

    assert result.outcome is RunOutcome.FAILED
    status_body = (await db_client.get(f"{MATCHES}/status", headers=headers)).json()
    assert status_body["error"] == "unexpected_error" and status_body["failure_count"] == 1
    assert leaks(str(status_body)) == []
    dump = (
        await db_session.execute(text("SELECT match_status::text FROM match_status"))
    ).scalar_one()
    assert leaks(dump) == []
    assert [log_text(r) for r in caplog.records if leaks(log_text(r))] == []
    assert any("RuntimeError" in r.getMessage() for r in caplog.records)  # the type, not the text


async def test_the_dedup_key_and_the_lock_name_hold_a_user_id_and_a_flag_only() -> None:
    from redis import Redis

    from app.tasks.locking import RedisMatchLocks

    user_id = uuid.uuid4()
    sent: list[tuple[str, str, dict[str, Any]]] = []

    class RecordingRedis:
        def set(self, key: str, value: str, **options: Any) -> bool:
            sent.append((key, value, options))
            return True

    assert matching.RedisPublishDedup(RecordingRedis()).claim(user_id, 10)  # type: ignore[arg-type]

    assert sent == [(f"match:queued:{user_id}", "1", {"nx": True, "ex": 10})]
    assert RedisMatchLocks(Redis())._key(user_id) == f"lock:match:{user_id}"
