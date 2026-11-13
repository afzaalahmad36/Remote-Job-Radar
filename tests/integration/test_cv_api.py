import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.rate_limit import RateLimiter
from app.models import Cv, CvFormat, User
from tests.auth_helpers import auth_headers
from tests.cv_helpers import make_pdf, synthetic_cv_text

CV = "/api/v1/cv"
TEXT = {"content-type": "text/plain"}
PDF = {"content-type": "application/pdf"}


@pytest.fixture
async def ann(db_client: AsyncClient) -> dict[str, str]:
    return await auth_headers(db_client, "ann@example.com")


@pytest.fixture
async def bob(db_client: AsyncClient) -> dict[str, str]:
    return await auth_headers(db_client, "bob@example.com")


def limits(app: FastAPI, settings: Settings, **overrides: object) -> None:
    """Run this test's app with smaller caps (a copy of the settings; the fixture is untouched)."""
    app.dependency_overrides[get_settings] = lambda: settings.model_copy(update=overrides)


async def upload(
    client: AsyncClient, headers: dict[str, str], body: bytes | str, kind: dict[str, str] = TEXT
) -> object:
    return await client.put(CV, content=body, headers=headers | kind)


async def count_cvs(session: AsyncSession) -> int:
    return await session.scalar(select(func.count()).select_from(Cv)) or 0


# --- uploading text ----------------------------------------------------------------------------


async def test_a_text_cv_is_stored_with_contact_details_masked(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    response = await db_client.put(CV, content=synthetic_cv_text(), headers=ann | TEXT)

    assert response.status_code == 200
    body = response.json()
    assert body["source_format"] == "text"
    assert body["page_count"] is None
    assert body["char_count"] == len(body["text"])
    assert "Python" in body["text"] and "FastAPI" in body["text"]
    for leaked in ("@example.com", "555 010", "github.com", "linkedin.com"):
        assert leaked not in body["text"]
    assert "[email]" in body["text"] and "[phone]" in body["text"] and "[link]" in body["text"]


async def test_the_stored_cv_can_be_read_back(db_client: AsyncClient, ann: dict[str, str]) -> None:
    uploaded = (await db_client.put(CV, content="Python developer", headers=ann | TEXT)).json()

    fetched = await db_client.get(CV, headers=ann)

    assert fetched.status_code == 200
    assert fetched.json() == uploaded


async def test_uploading_again_replaces_the_cv_instead_of_adding_one(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str]
) -> None:
    first = (await db_client.put(CV, content="Python developer", headers=ann | TEXT)).json()
    second = (await db_client.put(CV, content="Go developer", headers=ann | TEXT)).json()

    assert second["id"] == first["id"]
    assert second["text"] == "Go developer"
    assert second["updated_at"] >= first["updated_at"]
    assert second["created_at"] == first["created_at"]
    assert await count_cvs(db_session) == 1


async def test_a_text_file_with_a_byte_order_mark_is_accepted(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    response = await db_client.put(CV, content=b"\xef\xbb\xbfPython", headers=ann | TEXT)

    assert response.json()["text"] == "Python"


async def test_octet_stream_is_accepted_and_the_content_decides_the_format(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    octet = {"content-type": "application/octet-stream"}

    as_text = await db_client.put(CV, content="Python", headers=ann | octet)
    as_pdf = await db_client.put(CV, content=make_pdf([["Rust developer"]]), headers=ann | octet)

    assert as_text.json()["source_format"] == "text"
    assert as_pdf.json()["source_format"] == "pdf"


# --- uploading a PDF ---------------------------------------------------------------------------


async def test_a_pdf_cv_is_read_and_masked(db_client: AsyncClient, ann: dict[str, str]) -> None:
    pdf = make_pdf([["Alex Example", "alex@example.com", "Python and FastAPI"], ["Page two"]])

    response = await db_client.put(CV, content=pdf, headers=ann | PDF)

    body = response.json()
    assert response.status_code == 200
    assert body["source_format"] == CvFormat.PDF
    assert body["page_count"] == 2
    assert "Python and FastAPI" in body["text"] and "Page two" in body["text"]
    assert "alex@example.com" not in body["text"]


async def test_a_pdf_with_too_many_pages_is_refused(
    app: FastAPI, settings: Settings, db_client: AsyncClient, ann: dict[str, str]
) -> None:
    limits(app, settings, cv_max_pages=2)

    response = await db_client.put(CV, content=make_pdf([["x"]] * 3), headers=ann | PDF)

    assert response.status_code == 422
    assert response.json() == {
        "detail": "The PDF has too many pages (limit 2).",
        "code": "cv_unreadable",
    }


async def test_a_pdf_with_no_text_layer_is_refused_with_a_clear_message(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    response = await db_client.put(CV, content=make_pdf([[]]), headers=ann | PDF)

    assert response.status_code == 422
    assert "Scanned PDFs" in response.json()["detail"]


async def test_a_damaged_pdf_is_refused(db_client: AsyncClient, ann: dict[str, str]) -> None:
    response = await db_client.put(CV, content=b"%PDF-1.4 garbage", headers=ann | PDF)

    assert response.status_code == 422
    assert response.json() == {"detail": "The PDF could not be read.", "code": "cv_unreadable"}


# --- types and sizes ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (b'{"text": "hi"}', "application/json"),
        (b"Python", "image/png"),
        (b"Python", ""),
        (b"not a pdf", "application/pdf"),  # the content is not a PDF
        (b"\xff\xfe\x00\x00 binary", "text/plain"),  # not UTF-8
        (b"Python\x00developer", "text/plain"),  # a NUL byte
    ],
)
async def test_other_types_are_refused(
    db_client: AsyncClient, ann: dict[str, str], body: bytes, content_type: str
) -> None:
    response = await db_client.put(CV, content=body, headers=ann | {"content-type": content_type})

    assert response.status_code == 415
    assert response.json()["code"] == "cv_unsupported_type"


async def test_an_empty_upload_is_refused(db_client: AsyncClient, ann: dict[str, str]) -> None:
    response = await db_client.put(CV, content=b" \n ", headers=ann | TEXT)

    assert response.status_code == 422
    assert "No text" in response.json()["detail"]


async def test_a_file_over_the_byte_limit_is_refused_by_its_declared_size(
    app: FastAPI, settings: Settings, db_client: AsyncClient, ann: dict[str, str]
) -> None:
    limits(app, settings, cv_max_file_bytes=2_048)

    response = await db_client.put(CV, content=b"a" * 3_000, headers=ann | TEXT)

    assert response.status_code == 413
    assert response.json()["code"] == "cv_too_large"


async def test_a_body_that_lies_about_its_size_is_stopped_while_it_streams(
    app: FastAPI, settings: Settings, db_client: AsyncClient, ann: dict[str, str]
) -> None:
    limits(app, settings, cv_max_file_bytes=2_048)

    async def chunks() -> AsyncIterator[bytes]:  # chunked upload: there is no Content-Length
        for _ in range(10):
            yield b"a" * 1_000

    response = await db_client.put(CV, content=chunks(), headers=ann | TEXT)

    assert response.status_code == 413


async def test_text_over_the_character_limit_is_refused_not_cut(
    app: FastAPI, settings: Settings, db_client: AsyncClient, db_session: AsyncSession,
    ann: dict[str, str],
) -> None:  # fmt: skip
    limits(app, settings, cv_max_chars=1_000)

    response = await db_client.put(CV, content="word " * 400, headers=ann | TEXT)

    assert response.status_code == 422
    assert "too long" in response.json()["detail"]
    assert await count_cvs(db_session) == 0


async def test_a_refused_upload_leaves_the_existing_cv_alone(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    await db_client.put(CV, content="Python developer", headers=ann | TEXT)

    refused = await db_client.put(CV, content=b"%PDF-1.4 garbage", headers=ann | PDF)

    assert refused.status_code == 422
    assert (await db_client.get(CV, headers=ann)).json()["text"] == "Python developer"


# --- nothing is written to disk (Starlette would spool a multipart file over 1 MB) ----------------


async def test_no_temporary_file_is_created_by_large_text_or_pdf_uploads(
    app: FastAPI,
    settings: Settings,
    db_client: AsyncClient,
    ann: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    # Every temp-file API in this process, and in the PDF child process, now writes into `scratch`.
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, str(scratch))
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    with tempfile.NamedTemporaryFile() as probe:  # the redirect works, so an empty folder means it
        assert Path(probe.name).parent == scratch
    limits(app, settings, cv_max_file_bytes=2 * 1024 * 1024)
    marker = "DISK-MARKER-7731"
    big = (
        marker + " "
    ) * 100_000  # about 1.7 MB: over Starlette's 1 MB spool threshold, under ours

    too_long = await db_client.put(CV, content=big, headers=ann | TEXT)
    pdf = await db_client.put(CV, content=make_pdf([[marker]]), headers=ann | PDF)
    cut_off = await db_client.put(CV, content=big.encode() * 2, headers=ann | TEXT)

    assert too_long.status_code == 422  # read in memory, then refused for its length
    assert pdf.status_code == 200
    assert cut_off.status_code == 413
    assert list(scratch.rglob("*")) == []  # not a single file or folder was created
    assert os.listdir(scratch) == []


# --- ownership and deletion --------------------------------------------------------------------


@pytest.mark.parametrize("method", ["put", "get", "delete"])
async def test_every_endpoint_needs_a_login(db_client: AsyncClient, method: str) -> None:
    response = await db_client.request(method, CV, content="Python", headers=TEXT)

    assert response.status_code == 401


async def test_a_user_never_sees_or_deletes_someone_elses_cv(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str], bob: dict[str, str]
) -> None:
    await db_client.put(CV, content="Ann's Python CV", headers=ann | TEXT)

    assert (await db_client.get(CV, headers=bob)).status_code == 404
    assert (await db_client.delete(CV, headers=bob)).status_code == 404
    assert (await db_client.get(CV, headers=ann)).json()["text"] == "Ann's Python CV"
    assert await count_cvs(db_session) == 1


async def test_each_user_has_their_own_cv(db_client: AsyncClient, ann: dict[str, str],
                                          bob: dict[str, str]) -> None:  # fmt: skip
    await db_client.put(CV, content="Ann's CV", headers=ann | TEXT)
    await db_client.put(CV, content="Bob's CV", headers=bob | TEXT)

    assert (await db_client.get(CV, headers=ann)).json()["text"] == "Ann's CV"
    assert (await db_client.get(CV, headers=bob)).json()["text"] == "Bob's CV"


async def test_the_cv_does_not_exist_before_an_upload(
    db_client: AsyncClient, ann: dict[str, str]
) -> None:
    response = await db_client.get(CV, headers=ann)

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


async def test_deleting_the_cv_removes_the_row(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str]
) -> None:
    await db_client.put(CV, content="Python developer", headers=ann | TEXT)

    deleted = await db_client.delete(CV, headers=ann)

    assert deleted.status_code == 204
    assert deleted.content == b""
    assert (await db_client.get(CV, headers=ann)).status_code == 404
    assert await count_cvs(db_session) == 0
    assert (await db_client.delete(CV, headers=ann)).status_code == 404  # nothing left to delete


async def test_deleting_the_account_deletes_the_cv(
    db_client: AsyncClient, db_session: AsyncSession, ann: dict[str, str]
) -> None:
    await db_client.put(CV, content="Python developer", headers=ann | TEXT)
    user = await db_session.scalar(select(User).where(User.email == "ann@example.com"))
    assert user is not None

    await db_session.delete(user)
    await db_session.flush()

    assert await count_cvs(db_session) == 0


# --- rate limit --------------------------------------------------------------------------------


async def test_uploads_are_rate_limited_per_user(
    app: FastAPI, db_client: AsyncClient, ann: dict[str, str], bob: dict[str, str]
) -> None:
    app.state.cv_rate_limiter = RateLimiter(max_attempts=2, window_seconds=60)

    statuses = [(await upload(db_client, ann, "Python")).status_code for _ in range(3)]  # type: ignore[attr-defined]
    bob_response = await upload(db_client, bob, "Go")

    assert statuses == [200, 200, 429]
    assert bob_response.status_code == 200  # type: ignore[attr-defined]
    assert (await db_client.get(CV, headers=ann)).status_code == 200  # reading is not limited
