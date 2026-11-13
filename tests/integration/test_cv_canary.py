"""The canary test: a unique marker is planted in a CV and followed through the whole flow.

It must never appear in an LLM request, a log record or an error response. (It legitimately
appears in the owner's own `GET /cv` and in the database row; those are the user's data.)
"""

import logging

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, Response
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.rate_limit import RateLimiter
from app.llm.client import GuardedClient
from app.llm.providers.fake import FakeLLMClient
from app.llm.providers.gemini import GeminiClient
from app.models import Cv
from tests.auth_helpers import auth_headers
from tests.cv_helpers import make_pdf

CV = "/api/v1/cv"
MARKER = "CANARY-MARKER-8c41f0d2"
EMAIL = "canary.person@example.com"
PHONE = "+1 555 010 4242"
LINK = "https://github.com/canary-person"
SECRETS = (MARKER, EMAIL, PHONE, LINK)

CV_TEXT = f"""{MARKER} Person
{EMAIL} | {PHONE} | {LINK}
Backend engineer. Python, FastAPI, PostgreSQL, Docker.
"""


def leaks(value: str) -> list[str]:
    return [secret for secret in SECRETS if secret in value]


def log_text(record: logging.LogRecord) -> str:
    """Everything a log record can carry: message, arguments, traceback and extra attributes."""
    parts = [record.getMessage(), str(record.args), record.exc_text or ""]
    if record.exc_info:
        parts.append(logging.Formatter().formatException(record.exc_info))
    parts.extend(str(v) for v in vars(record).values())
    return "\n".join(parts)


@pytest.fixture
def llm_calls(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record every call that reaches an LLM client: the CV flow must make none."""
    calls: list[object] = []

    async def recorder(self: object, request: object) -> object:
        calls.append(request)
        raise AssertionError("the CV flow must never call an LLM client")

    for client_class in (GuardedClient, FakeLLMClient, GeminiClient):
        monkeypatch.setattr(client_class, "complete", recorder)
    return calls


async def test_a_cv_marker_never_reaches_an_llm_a_log_line_or_an_error(
    app: FastAPI,
    settings: Settings,
    db_client: AsyncClient,
    db_session: AsyncSession,
    caplog: pytest.LogCaptureFixture,
    llm_calls: list[object],
) -> None:
    app.state.cv_rate_limiter = RateLimiter(max_attempts=100, window_seconds=60)
    caplog.set_level(logging.DEBUG)  # every logger, lowest level: nothing may slip through
    headers = await auth_headers(db_client, "canary@example.com")
    small = settings.model_copy(update={"cv_max_chars": 1_000, "cv_max_file_bytes": 4_096})
    errors: list[Response] = []

    async def put(body: bytes | str, kind: str) -> Response:
        return await db_client.put(CV, content=body, headers=headers | {"content-type": kind})

    # the happy paths: text, then a PDF that replaces it, then text again
    assert (await put(CV_TEXT, "text/plain")).status_code == 200
    assert (await put(make_pdf([[MARKER, EMAIL, PHONE]]), "application/pdf")).status_code == 200
    assert (await put(CV_TEXT, "text/plain")).status_code == 200
    assert (await db_client.get(CV, headers=headers)).status_code == 200

    # every kind of refusal, each carrying the marker in the rejected content
    app.dependency_overrides[get_settings] = lambda: small
    errors.append(await put(f"{MARKER} " * 100, "text/plain"))  # too long
    errors.append(await put(b"%PDF-1.4 " + MARKER.encode() + b" damaged", "application/pdf"))
    errors.append(await put(f"{MARKER} ".encode() * 1_000, "text/plain"))  # too large
    errors.append(await put(MARKER.encode() + b"\xff\xfe", "text/plain"))  # not UTF-8
    errors.append(await put(CV_TEXT, "application/json"))  # wrong type
    errors.append(await put(make_pdf([[MARKER]] * 12), "application/pdf"))  # too many pages
    app.dependency_overrides[get_settings] = lambda: settings

    assert (await db_client.delete(CV, headers=headers)).status_code == 204
    errors.append(await db_client.get(CV, headers=headers))  # 404 after the delete

    assert [r.status_code for r in errors] == [422, 422, 413, 415, 415, 422, 404]
    for response in errors:
        assert leaks(response.text) == [], response.status_code
    assert llm_calls == []
    leaking_logs = [log_text(r) for r in caplog.records if leaks(log_text(r))]
    assert leaking_logs == []
    assert any("CV stored" in r.getMessage() for r in caplog.records)  # we did log, just not text


async def test_the_database_holds_the_masked_text_and_never_the_contact_details(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    headers = await auth_headers(db_client, "canary@example.com")

    await db_client.put(CV, content=CV_TEXT, headers=headers | {"content-type": "text/plain"})

    stored = await db_session.scalar(select(Cv.text))
    assert stored is not None
    assert MARKER in stored  # it is the user's own text
    assert leaks(stored.replace(MARKER, "")) == []  # but email, phone and link were masked
    # and nowhere else in the table either (the hash and the other columns are not the text)
    row = (await db_session.execute(text("SELECT cvs::text FROM cvs"))).scalar_one()
    assert EMAIL not in row and PHONE not in row and LINK not in row


async def test_deleting_the_cv_leaves_no_trace_of_it_in_the_database(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    headers = await auth_headers(db_client, "canary@example.com")
    await db_client.put(CV, content=CV_TEXT, headers=headers | {"content-type": "text/plain"})

    await db_client.delete(CV, headers=headers)

    dump = (await db_session.execute(text("SELECT cvs::text FROM cvs"))).scalars().all()
    assert dump == []


async def test_skills_and_preferences_text_never_reaches_an_llm_a_log_line_or_an_error(
    db_client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
    llm_calls: list[object],
) -> None:
    caplog.set_level(logging.DEBUG)
    headers = await auth_headers(db_client, "canary@example.com")
    skills, prefs = f"{CV}/skills", "/api/v1/preferences"
    notes = f"{MARKER} only remote. Reach me: {EMAIL}, {PHONE}, {LINK}"

    # the happy paths: upload, read and edit skills, save and read preferences
    uploaded = await db_client.put(
        CV, content=CV_TEXT, headers=headers | {"content-type": "text/plain"}
    )
    assert uploaded.status_code == 200
    assert (await db_client.get(skills, headers=headers)).status_code == 200
    edited = await db_client.put(
        skills, json={"skills": [f"{MARKER}-skill", "Python"]}, headers=headers
    )
    assert edited.status_code == 200
    saved = await db_client.put(prefs, json={"notes": notes}, headers=headers)
    assert saved.status_code == 200
    assert (await db_client.get(prefs, headers=headers)).status_code == 200
    assert leaks(saved.text.replace(MARKER, "")) == []  # even the owner's copy has no contact data

    # every kind of refusal, each carrying the marker in the rejected value
    bad_skills: list[object] = [
        [f"{MARKER}-" + "x" * 60],  # too long
        [f"{MARKER}-{n}" for n in range(101)],  # too many
        [MARKER, 7],  # not all strings
    ]
    bad_preferences: list[dict[str, object]] = [
        {"notes": (MARKER + " ") * 200},  # too long
        {"seniority_levels": [MARKER]},  # not a level
        {"remote_policies": [EMAIL]},  # not a policy
        {"min_salary": MARKER, "salary_currency": "USD", "salary_period": "year"},
        {"min_salary": 5, "salary_currency": MARKER, "salary_period": "year"},
        {"min_salary": 5, "salary_currency": "USD", "salary_period": PHONE},
    ]
    errors = [await db_client.put(skills, json={"skills": s}, headers=headers) for s in bad_skills]
    errors += [await db_client.put(prefs, json=body, headers=headers) for body in bad_preferences]

    assert [r.status_code for r in errors] == [422] * (len(bad_skills) + len(bad_preferences))
    for response in errors:
        assert leaks(response.text) == [], response.json()
    assert llm_calls == []
    assert [log_text(r) for r in caplog.records if leaks(log_text(r))] == []
    logged = [r.getMessage() for r in caplog.records]
    assert any("CV skills saved" in line for line in logged)
    assert any("Preferences saved" in line for line in logged)  # we did log, just not the text


async def test_the_database_holds_masked_notes_and_deleting_the_user_removes_them(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    headers = await auth_headers(db_client, "canary@example.com")
    notes = f"{MARKER} contact {EMAIL} {PHONE} {LINK}"

    await db_client.put("/api/v1/preferences", json={"notes": notes}, headers=headers)

    row = (
        await db_session.execute(text("SELECT job_preferences::text FROM job_preferences"))
    ).scalar_one()
    assert MARKER in row  # the user's own words
    assert EMAIL not in row and PHONE not in row and LINK not in row
    await db_session.execute(text("DELETE FROM users"))
    assert (
        await db_session.execute(text("SELECT job_preferences::text FROM job_preferences"))
    ).all() == []
