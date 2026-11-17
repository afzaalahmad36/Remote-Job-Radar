"""The worked example of docs/MATCHING.md, run through the real scorer.

The synthetic CV (`tests/fixtures/cv/alex_example.txt`) goes through the real skills extractor, and
two invented jobs are scored against Alex's preferences. The expected explanations are the JSON
files in `tests/fixtures/matching/`, worked out by hand from the measured floor (0.715) and ceiling
(0.86), not copied from the scorer's output:

    job A: coverage 5/6 = 0.833, similarity (0.78 - 0.715) / 0.145 = 0.448
           score = round(100 * (0.6 * 0.8333 + 0.4 * 0.4483)) = 68
    job B: coverage 0, similarity 0 (0.58 is under the floor) -> 0; remote and seniority fail

The cosines (0.78, 0.58) are invented inputs: the model is not involved.
"""

import json
from pathlib import Path

from app.cv.skills import find_skills
from app.cv.text import clean_cv_text, mask_cv_text
from app.matching.scoring import CvFacts, PreferenceFacts, score_match
from app.models.job import RemotePolicy, SalaryPeriod, Seniority
from tests.matching_helpers import CONFIG, ENGLISH_TEXT, job_facts

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"

ALEX_PREFERENCES = PreferenceFacts(
    seniority_levels=[Seniority.SENIOR, Seniority.LEAD],
    remote_policies=[RemotePolicy.REMOTE],
    min_salary=100_000,
    salary_currency="USD",
    salary_period=SalaryPeriod.YEAR,
)


def alex() -> CvFacts:
    text = (FIXTURES / "cv" / "alex_example.txt").read_text(encoding="utf-8")
    return CvFacts(skills=find_skills(mask_cv_text(clean_cv_text(text))), vector_id="m|alex")


def expected(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / "matching" / name).read_text(encoding="utf-8"))


def test_alex_has_the_fifteen_skills_the_fixture_counts() -> None:
    assert len(alex().skills) == 15


def test_the_hand_arithmetic_for_job_a_gives_68_without_the_scorer() -> None:
    """The expected 68 is plain arithmetic on the documented numbers, no scorer call."""
    coverage = 5 / 6  # 5 of the job's 6 stack entries are in Alex's skills
    similarity = (0.78 - 0.715) / (0.86 - 0.715)  # 0.4483
    weighted = 0.6 * coverage + 0.4 * similarity  # 0.5 + 0.1793 = 0.6793
    assert round(100 * weighted) == 68


def test_job_a_matches_the_worked_example() -> None:
    job = job_facts(
        stack=["Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes", "Terraform"],
        seniority=Seniority.SENIOR,
        remote_policy=RemotePolicy.REMOTE,
        salary_max=130_000,
        salary_currency="USD",
        salary_period=SalaryPeriod.YEAR,
        text_start=ENGLISH_TEXT,
    )

    result = score_match(
        cv=alex(), preferences=ALEX_PREFERENCES, job=job, cosine=0.78, config=CONFIG
    )

    assert result is not None
    assert result.reasons.to_json() == expected("alex_vs_job_a.json")
    assert result.score == 68
    assert result.missing_skills == ["Terraform"]
    assert result.filters_failed is False


def test_job_b_matches_the_worked_example() -> None:
    job = job_facts(
        stack=["Java", "Spring", "Angular", "MySQL"],
        seniority=Seniority.JUNIOR,
        remote_policy=RemotePolicy.ONSITE,
        salary_max=40_000,
        salary_currency="EUR",
        salary_period=SalaryPeriod.YEAR,
        text_start=ENGLISH_TEXT,
    )

    result = score_match(
        cv=alex(), preferences=ALEX_PREFERENCES, job=job, cosine=0.58, config=CONFIG
    )

    assert result is not None
    assert result.reasons.to_json() == expected("alex_vs_job_b.json")
    assert result.score == 0
    assert result.filters_failed is True
    assert result.missing_skills == ["Java", "Spring", "Angular", "MySQL"]


def test_the_explanations_hold_no_cv_text_only_job_skill_names_and_a_count() -> None:
    job = job_facts(stack=["Python", "FastAPI", "Terraform"], text_start=ENGLISH_TEXT)
    result = score_match(
        cv=alex(), preferences=ALEX_PREFERENCES, job=job, cosine=0.78, config=CONFIG
    )

    assert result is not None
    stored = json.dumps(result.reasons.to_json())
    for cv_only in ("Django", "Celery", "Redis", "React", "GitHub Actions", "Northwind", "Alex"):
        assert cv_only not in stored
