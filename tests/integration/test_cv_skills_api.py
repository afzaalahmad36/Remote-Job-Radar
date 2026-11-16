from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Cv
from tests.auth_helpers import auth_headers
from tests.cv_helpers import synthetic_cv_text

CV = "/api/v1/cv"
SKILLS = "/api/v1/cv/skills"
TEXT = {"content-type": "text/plain"}


async def upload(client: AsyncClient, headers: dict[str, str], body: str) -> None:
    response = await client.put(CV, content=body, headers=headers | TEXT)
    assert response.status_code == 200


# --- reading ---------------------------------------------------------------------------------


async def test_uploading_a_cv_extracts_its_skills_and_suggestions(db_client: AsyncClient) -> None:
    ann = await auth_headers(db_client, "ann@example.com")
    await upload(db_client, ann, synthetic_cv_text())

    response = await db_client.get(SKILLS, headers=ann)

    body = response.json()
    assert response.status_code == 200
    assert body["skills"][:3] == ["Python", "FastAPI", "PostgreSQL"]
    assert "Kubernetes" in body["skills"]
    assert body["edited"] is False
    assert body["suggestions"] == {
        "years_of_experience": {"value": 6, "match": "years phrase"},
        "seniority": {"value": "senior", "match": "title keyword in the current role"},
    }
    assert body["updated_at"]


async def test_a_cv_without_matches_gives_empty_skills_and_no_suggestions(
    db_client: AsyncClient,
) -> None:
    ann = await auth_headers(db_client, "ann@example.com")
    await upload(db_client, ann, "I like hiking.")

    body = (await db_client.get(SKILLS, headers=ann)).json()

    assert body["skills"] == []
    assert body["suggestions"] == {"years_of_experience": None, "seniority": None}


async def test_without_a_cv_both_endpoints_are_404(db_client: AsyncClient) -> None:
    ann = await auth_headers(db_client, "ann@example.com")

    got = await db_client.get(SKILLS, headers=ann)
    put = await db_client.put(SKILLS, json={"skills": ["Python"]}, headers=ann)

    assert (got.status_code, put.status_code) == (404, 404)
    assert got.json()["code"] == "not_found"


async def test_both_endpoints_need_a_login(db_client: AsyncClient) -> None:
    assert (await db_client.get(SKILLS)).status_code == 401
    assert (await db_client.put(SKILLS, json={"skills": []})).status_code == 401


# --- editing ---------------------------------------------------------------------------------


async def test_the_user_can_replace_the_skills_and_the_list_is_marked_edited(
    db_client: AsyncClient,
) -> None:
    ann = await auth_headers(db_client, "ann@example.com")
    await upload(db_client, ann, synthetic_cv_text())

    response = await db_client.put(
        SKILLS, json={"skills": ["postgres", "Team leadership", "DOCKER", "docker"]}, headers=ann
    )

    assert response.status_code == 200
    assert response.json()["skills"] == ["PostgreSQL", "Team leadership", "Docker"]
    assert response.json()["edited"] is True
    assert response.json()["suggestions"]["years_of_experience"]["value"] == 6  # still the text's
    assert (await db_client.get(SKILLS, headers=ann)).json() == response.json()


async def test_an_empty_list_is_allowed_and_clears_the_skills(db_client: AsyncClient) -> None:
    ann = await auth_headers(db_client, "ann@example.com")
    await upload(db_client, ann, synthetic_cv_text())

    response = await db_client.put(SKILLS, json={"skills": []}, headers=ann)

    assert response.json()["skills"] == [] and response.json()["edited"] is True


async def test_editing_skills_leaves_the_cv_text_alone(
    db_client: AsyncClient, db_session: AsyncSession
) -> None:
    ann = await auth_headers(db_client, "ann@example.com")
    await upload(db_client, ann, synthetic_cv_text())
    before = (await db_client.get(CV, headers=ann)).json()

    await db_client.put(SKILLS, json={"skills": ["Go"]}, headers=ann)

    after = (await db_client.get(CV, headers=ann)).json()
    assert after["text"] == before["text"]
    assert after["updated_at"] >= before["updated_at"]
    stored = await db_session.scalar(select(Cv.content_hash))
    assert stored is not None


async def test_uploading_a_new_cv_reads_its_skills_again_and_drops_the_edit(
    db_client: AsyncClient,
) -> None:
    ann = await auth_headers(db_client, "ann@example.com")
    await upload(db_client, ann, synthetic_cv_text())
    await db_client.put(SKILLS, json={"skills": ["Cobol"]}, headers=ann)

    await upload(db_client, ann, "Go developer. Rust and Kotlin.")

    body = (await db_client.get(SKILLS, headers=ann)).json()
    assert body["skills"] == ["Rust", "Kotlin"]
    assert body["edited"] is False


async def test_deleting_the_cv_deletes_the_skills_too(db_client: AsyncClient) -> None:
    ann = await auth_headers(db_client, "ann@example.com")
    await upload(db_client, ann, synthetic_cv_text())

    await db_client.delete(CV, headers=ann)

    assert (await db_client.get(SKILLS, headers=ann)).status_code == 404


async def test_a_user_only_ever_reaches_their_own_skills(db_client: AsyncClient) -> None:
    ann = await auth_headers(db_client, "ann@example.com")
    bob = await auth_headers(db_client, "bob@example.com")
    await upload(db_client, ann, synthetic_cv_text())
    await upload(db_client, bob, "Elixir developer")

    await db_client.put(SKILLS, json={"skills": ["Ann only"]}, headers=ann)

    assert (await db_client.get(SKILLS, headers=bob)).json()["skills"] == ["Elixir"]
    assert (await db_client.get(SKILLS, headers=ann)).json()["skills"] == ["Ann only"]


# --- bad input -------------------------------------------------------------------------------


async def test_invalid_skill_bodies_are_422_and_change_nothing(db_client: AsyncClient) -> None:
    ann = await auth_headers(db_client, "ann@example.com")
    await upload(db_client, ann, synthetic_cv_text())
    before = (await db_client.get(SKILLS, headers=ann)).json()

    bad_bodies = [
        {"skills": ["x" * 51]},  # too long
        {"skills": [f"skill {n}" for n in range(101)]},  # too many
        {"skills": "Python"},  # not a list
        {"skills": [1, 2]},  # not strings
        {"skills": [], "extra": 1},  # unknown field
        {},  # missing
    ]
    for body in bad_bodies:
        response = await db_client.put(SKILLS, json=body, headers=ann)
        assert response.status_code == 422, body
        assert response.json()["code"] == "validation_error"

    assert (await db_client.get(SKILLS, headers=ann)).json() == before
