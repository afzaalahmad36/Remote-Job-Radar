"""Small builders for the scoring tests: facts with sensible defaults, so each test names only
what it is about. Everything here is invented."""

from dataclasses import replace
from typing import Any

from app.matching.scoring import CvFacts, JobFacts, PreferenceFacts, ScoringConfig

# The defaults of docs/MATCHING.md (floor and ceiling measured by scripts/calibrate_similarity.py).
CONFIG = ScoringConfig(
    weight_skills=0.6,
    weight_embedding=0.4,
    similarity_floor=0.715,
    similarity_ceiling=0.86,
    failed_filter_cap=30,
)

ENGLISH_TEXT = (
    "We are looking for an engineer to join our team. You will build services that are used by "
    "customers every day, and you will work with the people who design the product."
)


def cv_facts(skills: list[str] | None = None, vector_id: str | None = "m|cv") -> CvFacts:
    return CvFacts(skills=["Python", "Docker"] if skills is None else skills, vector_id=vector_id)


def job_facts(**overrides: Any) -> JobFacts:
    base = JobFacts(
        stack=["Python", "Docker"],
        seniority=None,
        remote_policy=None,
        salary_min=None,
        salary_max=None,
        salary_currency=None,
        salary_period=None,
        text_start=ENGLISH_TEXT,
        vector_id="m|job",
        raw_text_hash="h" * 64,
        rules_version="rules_v3",
    )
    return replace(base, **overrides)


def preferences(**overrides: Any) -> PreferenceFacts:
    return replace(PreferenceFacts(), **overrides)


def config(**overrides: Any) -> ScoringConfig:
    return replace(CONFIG, **overrides)
