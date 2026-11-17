"""The pure parts of scripts/calibrate_similarity.py: percentiles, pair sampling, the CV report.

No database and no model: the vectors are small invented ones.
"""

import math

import pytest

from scripts.calibrate_similarity import (
    cosine,
    percentile,
    random_pair_cosines,
    report_cv,
    table,
)


def test_percentile_interpolates_between_values() -> None:
    values = [0.0, 1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0) == 0.0
    assert percentile(values, 50) == 2.0
    assert percentile(values, 100) == 4.0
    assert percentile(values, 25) == 1.0
    assert percentile(values, 12.5) == pytest.approx(0.5)


def test_percentile_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)


def test_cosine_of_equal_orthogonal_and_opposite_vectors() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def vectors(count: int) -> list[list[float]]:
    return [[math.cos(i), math.sin(i), 0.5] for i in range(count)]


def test_the_pair_sample_is_repeatable_for_a_seed_and_differs_for_another() -> None:
    first = random_pair_cosines(vectors(30), pairs=100, seed=1)

    assert first == random_pair_cosines(vectors(30), pairs=100, seed=1)
    assert first != random_pair_cosines(vectors(30), pairs=100, seed=2)
    assert first == sorted(first)
    assert len(first) == 100


def test_with_fewer_pairs_than_asked_every_pair_is_used_once() -> None:
    result = random_pair_cosines(vectors(5), pairs=5_000, seed=1)

    assert len(result) == 10  # 5 * 4 / 2


def test_the_table_lists_the_percentiles() -> None:
    text = table("Title", [0.1, 0.2, 0.3, 0.4])

    assert text.startswith("Title")
    for label in ("min", "P50", "P99", "max"):
        assert label in text


def test_the_cv_report_says_when_the_whole_distribution_is_below_the_floor() -> None:
    jobs = [[1.0, 0.0], [0.9, 0.1]]
    far_cv = [0.0, 1.0]

    text, all_below = report_cv(far_cv, jobs, floor=0.7, ceiling=0.9)

    assert all_below is True
    assert "0 of 2" in text


def test_the_cv_report_counts_jobs_above_the_floor_and_the_ceiling() -> None:
    jobs = [[1.0, 0.0], [0.0, 1.0]]
    cv = [1.0, 0.0]

    text, all_below = report_cv(cv, jobs, floor=0.5, ceiling=0.9)

    assert all_below is False
    assert "1 of 2" in text
