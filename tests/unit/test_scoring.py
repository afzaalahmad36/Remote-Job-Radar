"""The score: skill coverage, similarity mapping, weights, the cap, and the explanation's shape.

Pure functions with invented values: no database, no model, no network.
"""

import pytest

from app.matching.reasons import NOTE_MESSAGES, MatchReasons
from app.matching.scoring import (
    SCORER_VERSION,
    normalise_similarity,
    score_match,
    skill_key,
)
from app.models.job import RemotePolicy, Seniority
from app.models.job_match import MAX_MISSING_SKILLS, MAX_SKILL_NAME_CHARS
from tests.language_posts import GERMAN_PROSE
from tests.matching_helpers import config, cv_facts, job_facts, preferences

FLOOR, CEILING = 0.715, 0.86


def score(*, cv=None, prefs=None, job=None, cosine=0.8, cfg=None):
    return score_match(
        cv=cv or cv_facts(),
        preferences=prefs,
        job=job or job_facts(),
        cosine=cosine,
        config=cfg or config(),
    )


# --- similarity mapping ---
@pytest.mark.parametrize(
    ("cosine", "expected"),
    [
        (0.2, 0.0),  # far below the floor
        (FLOOR, 0.0),
        (0.7875, 0.5),  # halfway
        (CEILING, 1.0),
        (0.99, 1.0),  # above the ceiling
    ],
)
def test_similarity_is_clamped_between_floor_and_ceiling(cosine: float, expected: float) -> None:
    assert normalise_similarity(cosine, FLOOR, CEILING) == pytest.approx(expected)


# --- skills ---
def test_skills_match_through_aliases_and_ignore_case() -> None:
    cv = cv_facts(["postgres", "PYTHON", "  docker  "])
    job = job_facts(stack=["PostgreSQL", "Python", "Docker", "Terraform"])

    result = score(cv=cv, job=job, cosine=None)

    assert result is not None
    assert result.reasons.skills.matched == ["PostgreSQL", "Python", "Docker"]
    assert result.reasons.skills.missing == ["Terraform"]
    assert skill_key("postgres") == skill_key("PostgreSQL")


def test_an_unknown_technology_is_compared_as_folded_text() -> None:
    result = score(
        cv=cv_facts(["Obscure  Tool"]), job=job_facts(stack=["obscure tool", "Python"]), cosine=None
    )

    assert result is not None
    assert result.reasons.skills.matched == ["obscure tool"]
    assert result.reasons.skills.missing == ["Python"]


def test_a_repeated_stack_entry_counts_once() -> None:
    result = score(job=job_facts(stack=["Python", "python", "PYTHON", "Docker"]), cosine=None)

    assert result is not None
    assert result.reasons.components.skills is not None
    assert result.reasons.components.skills.total == 2


def test_coverage_is_matched_over_the_jobs_distinct_entries() -> None:
    result = score(
        cv=cv_facts(["Python"]), job=job_facts(stack=["Python", "Go", "Rust", "Zig"]), cosine=None
    )

    assert result is not None
    assert result.reasons.components.skills is not None
    assert result.reasons.components.skills.coverage == 0.25
    assert result.score == 25  # skills only, so 100 * 0.25


# --- weights ---
def test_both_components_use_the_configured_weights() -> None:
    # coverage 1.0, similarity 0.5 -> 0.6 * 1.0 + 0.4 * 0.5 = 0.8
    result = score(cosine=0.7875)

    assert result is not None
    assert result.score == 80
    assert result.reasons.weights.effective.skills == 0.6
    assert result.reasons.notes == []


def test_no_job_stack_moves_the_whole_weight_to_the_embedding() -> None:
    result = score(job=job_facts(stack=[]), cosine=0.7875)

    assert result is not None
    assert result.reasons.weights.effective.skills == 0.0
    assert result.reasons.weights.effective.embedding == 1.0
    assert result.score == 50
    assert "no_job_stack" in result.reasons.notes
    assert result.reasons.components.skills is None


def test_a_cv_without_skills_moves_the_whole_weight_to_the_embedding() -> None:
    result = score(cv=cv_facts([]), cosine=0.7875)

    assert result is not None
    assert result.score == 50
    assert "no_cv_skills" in result.reasons.notes


def test_no_vector_moves_the_whole_weight_to_the_skills() -> None:
    result = score(cosine=None)

    assert result is not None
    assert result.score == 100
    assert result.reasons.weights.effective.skills == 1.0
    assert result.reasons.notes == ["no_embedding"]
    assert result.reasons.components.embedding is None


def test_a_post_that_does_not_look_english_gets_no_embedding_weight_but_shows_the_cosine() -> None:
    result = score(job=job_facts(text_start=GERMAN_PROSE), cosine=0.86)

    assert result is not None
    assert result.reasons.weights.effective.embedding == 0.0
    assert result.reasons.notes == ["not_english"]
    assert result.reasons.components.embedding is not None
    assert result.score == 100  # coverage 1.0 on the skills alone


def test_nothing_to_compare_gives_no_score_at_all() -> None:
    assert score(job=job_facts(stack=[]), cosine=None) is None
    assert score(cv=cv_facts([]), cosine=None) is None
    assert score(job=job_facts(stack=[], text_start=GERMAN_PROSE), cosine=0.9) is None


def test_the_score_stays_within_0_and_100() -> None:
    low = score(cv=cv_facts(["Go"]), job=job_facts(stack=["Python", "Docker"]), cosine=0.1)
    high = score(cosine=0.99)

    assert low is not None and high is not None
    assert low.score == 0
    assert high.score == 100


def test_rounding_goes_up_at_a_half() -> None:
    # skills only, coverage 1/8 = 0.125 -> 12.5 -> 13
    stack = ["Python", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    result = score(cv=cv_facts(["Python"]), job=job_facts(stack=stack), cosine=None)

    assert result is not None
    assert result.score == 13


# --- the cap ---
def test_a_failed_filter_caps_the_score_but_keeps_the_job() -> None:
    prefs = preferences(remote_policies=[RemotePolicy.REMOTE])
    job = job_facts(remote_policy=RemotePolicy.ONSITE)

    result = score(prefs=prefs, job=job, cosine=0.86)

    assert result is not None
    assert result.score == 30
    assert result.filters_failed is True
    assert result.reasons.cap_applied is True
    assert "filters_failed" in result.reasons.notes


def test_a_failed_filter_does_not_raise_a_low_score() -> None:
    prefs = preferences(seniority_levels=[Seniority.SENIOR])
    job = job_facts(stack=["Go", "Rust"], seniority=Seniority.JUNIOR)

    result = score(prefs=prefs, job=job, cosine=0.5)

    assert result is not None
    assert result.score == 0
    assert result.filters_failed is True
    assert result.reasons.cap_applied is False  # the cap changed nothing


def test_unknown_never_fails_and_never_caps() -> None:
    prefs = preferences(
        remote_policies=[RemotePolicy.REMOTE], seniority_levels=[Seniority.SENIOR]
    )  # the job states neither

    result = score(prefs=prefs, cosine=0.86)

    assert result is not None
    assert result.score == 100
    assert result.filters_failed is False
    assert [f.outcome for f in result.reasons.filters] == ["unknown", "unknown", "unknown"]


def test_the_cap_is_a_setting() -> None:
    prefs = preferences(remote_policies=[RemotePolicy.REMOTE])
    job = job_facts(remote_policy=RemotePolicy.ONSITE)

    result = score(prefs=prefs, job=job, cosine=0.86, cfg=config(failed_filter_cap=10))

    assert result is not None
    assert result.score == 10


# --- the explanation ---
def test_the_explanation_has_exactly_the_documented_keys() -> None:
    result = score(prefs=preferences(remote_policies=[RemotePolicy.REMOTE]))

    assert result is not None
    document = result.reasons.to_json()
    assert set(document) == {
        "scorer_version",
        "score",
        "cap_applied",
        "weights",
        "components",
        "skills",
        "filters",
        "notes",
    }
    assert document["scorer_version"] == SCORER_VERSION
    assert set(document["components"]) == {"skills", "embedding"}
    assert set(document["skills"]) == {"matched", "missing"}
    assert {f["name"] for f in document["filters"]} == {"remote_policy", "seniority", "salary"}
    # The CV is described by a count, never by its skills.
    assert document["components"]["skills"]["cv_skills"] == 2


def test_the_explanation_round_trips_through_its_schema() -> None:
    result = score(prefs=preferences(remote_policies=[RemotePolicy.REMOTE]))

    assert result is not None
    document = result.reasons.to_json()
    assert MatchReasons.model_validate(document).to_json() == document


def test_the_schema_refuses_a_key_it_does_not_declare() -> None:
    result = score()
    assert result is not None
    document = result.reasons.to_json()

    with pytest.raises(ValueError):
        MatchReasons.model_validate({**document, "cv_text": "anything"})


def test_every_note_has_a_message() -> None:
    from typing import get_args

    from app.matching.reasons import Note

    assert set(get_args(Note)) == set(NOTE_MESSAGES)


# --- bounds on the stored skill lists ---
def test_a_skill_name_that_is_too_long_is_left_out_and_noted() -> None:
    too_long = "x" * (MAX_SKILL_NAME_CHARS + 1)

    result = score(job=job_facts(stack=["Python", too_long]), cosine=None)

    assert result is not None
    assert too_long not in result.missing_skills
    assert "skills_truncated" in result.reasons.notes


def test_the_missing_list_is_capped() -> None:
    stack = [f"Tool{i}" for i in range(MAX_MISSING_SKILLS + 10)]

    result = score(cv=cv_facts(["Python"]), job=job_facts(stack=stack), cosine=None)

    assert result is not None
    assert len(result.missing_skills) == MAX_MISSING_SKILLS
    assert len(result.reasons.skills.missing) == MAX_MISSING_SKILLS
    assert "skills_truncated" in result.reasons.notes
    # The coverage still counts every entry, only the lists are cut.
    assert result.reasons.components.skills is not None
    assert result.reasons.components.skills.total == MAX_MISSING_SKILLS + 10


def test_a_list_within_the_bounds_is_not_marked_truncated() -> None:
    result = score(job=job_facts(stack=["Python", "Docker", "Go"]), cosine=None)

    assert result is not None
    assert "skills_truncated" not in result.reasons.notes
