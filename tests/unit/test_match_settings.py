"""The scoring settings: defaults, validation, and the example env file."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.settings_helpers import settings_with as make_settings

ROOT = Path(__file__).resolve().parents[2]


def test_the_defaults_are_the_ones_fixed_before_any_evaluation() -> None:
    settings = make_settings()

    assert settings.match_weight_skills == 0.6
    assert settings.match_weight_embedding == 0.4
    assert settings.match_similarity_floor == 0.715  # P50 of random job-job cosines
    assert settings.match_similarity_ceiling == 0.86  # P99 of the same
    assert settings.match_failed_filter_cap == 30
    assert settings.match_candidates == 200


def test_the_weights_must_add_up_to_one() -> None:
    with pytest.raises(ValidationError, match="add up to 1"):
        make_settings(match_weight_skills=0.7, match_weight_embedding=0.4)

    assert make_settings(match_weight_skills=1.0, match_weight_embedding=0.0)
    assert make_settings(match_weight_skills=0.25, match_weight_embedding=0.75)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_a_weight_outside_zero_and_one_is_refused(value: float) -> None:
    with pytest.raises(ValidationError):
        make_settings(match_weight_skills=value, match_weight_embedding=1 - value)


def test_the_floor_must_be_below_the_ceiling() -> None:
    with pytest.raises(ValidationError, match="FLOOR must be smaller"):
        make_settings(match_similarity_floor=0.9, match_similarity_ceiling=0.8)
    with pytest.raises(ValidationError):
        make_settings(match_similarity_floor=0.8, match_similarity_ceiling=0.8)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("match_failed_filter_cap", 101),
        ("match_failed_filter_cap", -1),
        ("match_candidates", 0),
        ("match_candidates", 1_001),
    ],
)
def test_the_cap_and_the_candidate_count_have_bounds(name: str, value: int) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{name: value})


def test_the_example_env_file_documents_every_scoring_setting_with_the_same_defaults() -> None:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()

    for line in (
        "MATCH_WEIGHT_SKILLS=0.6",
        "MATCH_WEIGHT_EMBEDDING=0.4",
        "MATCH_SIMILARITY_FLOOR=0.715",
        "MATCH_SIMILARITY_CEILING=0.86",
        "MATCH_FAILED_FILTER_CAP=30",
        "MATCH_CANDIDATES=200",
    ):
        assert line in lines
