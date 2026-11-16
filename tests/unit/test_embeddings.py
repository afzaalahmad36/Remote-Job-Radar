"""The pure parts of the embedding code: what is embedded, its identity, and the fake embedder."""

import math

import pytest

from app.matching.embeddings import (
    CV_CHUNK_CHARS,
    CV_MAX_CHUNKS,
    EMBEDDING_DIM,
    EmbeddingError,
    check_vector,
    cv_chunks,
    job_embedding_text,
    normalised_mean,
    text_hash,
)
from app.matching.fake_embedder import FakeEmbedder
from tests.cv_helpers import synthetic_cv_text


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))  # both have length 1


# --- the text that stands for a job ------------------------------------------------------------


def test_a_job_is_embedded_as_its_structured_fields_then_the_start_of_the_post() -> None:
    text = job_embedding_text(
        title="Senior Python Developer",
        company="Acme",
        seniority="senior",
        remote_policy="remote",
        stack=["Python", "Django"],
        raw_text="We are   hiring.\n\nApply today.",
    )

    assert text.splitlines() == [
        "Title: Senior Python Developer",
        "Company: Acme",
        "Seniority: senior",
        "Remote: remote",
        "Stack: Python, Django",
        "We are hiring. Apply today.",  # whitespace collapsed
    ]


def test_missing_fields_are_left_blank_and_a_long_post_is_cut() -> None:
    text = job_embedding_text(
        title="Dev", company=None, seniority=None, remote_policy=None, stack=[], raw_text="x" * 5000
    )

    assert "Company: \n" in text and "Stack: \n" in text
    assert len(text.splitlines()[-1]) == 1_800


# --- identity ----------------------------------------------------------------------------------


def test_the_hash_depends_on_both_the_text_and_the_model() -> None:
    assert text_hash("m@1", "hello") == text_hash("m@1", "hello")
    assert text_hash("m@1", "hello") != text_hash("m@2", "hello")
    assert text_hash("m@1", "hello") != text_hash("m@1", "hello!")
    assert len(text_hash("m@1", "hello")) == 64


# --- cutting a CV ------------------------------------------------------------------------------


def test_a_short_cv_is_one_chunk() -> None:
    result = cv_chunks(synthetic_cv_text())

    assert len(result.chunks) == 1 and result.total == 1 and not result.truncated


def test_a_long_cv_is_cut_into_chunks_that_fit_and_break_at_lines() -> None:
    line = "Built and operated backend services in Python and PostgreSQL.\n"
    text = line * 200  # about 12,000 characters

    result = cv_chunks(text)

    assert 2 <= len(result.chunks) <= CV_MAX_CHUNKS
    assert all(len(chunk) <= CV_CHUNK_CHARS for chunk in result.chunks)
    assert all(chunk.endswith("PostgreSQL.") for chunk in result.chunks)  # cut at line ends
    assert not result.truncated


def test_a_cv_past_the_cap_keeps_the_first_chunks_and_says_it_was_cut() -> None:
    text = ("word " * 200 + "\n") * 60  # about 60,000 characters

    result = cv_chunks(text)

    assert len(result.chunks) == CV_MAX_CHUNKS
    assert result.total > CV_MAX_CHUNKS and result.truncated


def test_a_text_without_line_breaks_is_still_cut() -> None:
    result = cv_chunks("a" * 6000)

    assert [len(chunk) for chunk in result.chunks] == [2500, 2500, 1000]


def test_an_empty_cv_has_no_chunks() -> None:
    assert cv_chunks("  \n ").chunks == []


# --- vectors -----------------------------------------------------------------------------------


def test_check_vector_accepts_384_finite_numbers_and_returns_floats() -> None:
    assert check_vector([1] * EMBEDDING_DIM) == [1.0] * EMBEDDING_DIM


@pytest.mark.parametrize("size", [0, 3, 383, 385, 768])
def test_check_vector_refuses_a_wrong_length(size: int) -> None:
    with pytest.raises(EmbeddingError, match="dimensions"):
        check_vector([0.1] * size)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_check_vector_refuses_values_that_are_not_numbers(bad: float) -> None:
    with pytest.raises(EmbeddingError):
        check_vector([0.1] * (EMBEDDING_DIM - 1) + [bad])


def test_the_mean_of_vectors_has_length_one() -> None:
    a = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    b = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)

    mean = normalised_mean([a, b])

    assert math.isclose(sum(x * x for x in mean), 1.0)
    assert math.isclose(mean[0], mean[1]) and mean[0] > 0


def test_the_mean_of_opposite_vectors_has_no_direction_and_is_refused() -> None:
    a = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    b = [-1.0] + [0.0] * (EMBEDDING_DIM - 1)

    with pytest.raises(EmbeddingError):
        normalised_mean([a, b])
    with pytest.raises(EmbeddingError):
        normalised_mean([])


# --- the fake embedder -------------------------------------------------------------------------


def test_the_fake_embedder_is_deterministic_384_dimensional_and_normalised() -> None:
    fake = FakeEmbedder()

    first = fake.embed(["Senior Python developer"])[0]
    again = FakeEmbedder().embed(["Senior Python developer"])[0]

    assert first == again
    assert len(first) == EMBEDDING_DIM == fake.dimension
    assert math.isclose(sum(x * x for x in first), 1.0)
    assert fake.model_id == "fake-hash@v1"


def test_texts_that_share_words_are_closer_than_texts_that_do_not() -> None:
    fake = FakeEmbedder()
    job, similar, other = fake.embed(
        [
            "backend python developer postgresql docker",
            "python developer with docker and postgresql",
            "pastry chef croissant bakery sourdough",
        ]
    )

    assert cosine(job, similar) > 0.5 > cosine(job, other)


def test_the_fake_embedder_records_what_it_was_asked_and_copes_with_empty_text() -> None:
    fake = FakeEmbedder("fake@other")

    vectors = fake.embed(["", "!!!"])

    assert fake.calls == [["", "!!!"]] and fake.model_id == "fake@other"
    assert all(
        len(v) == EMBEDDING_DIM and math.isclose(sum(x * x for x in v), 1.0) for v in vectors
    )
