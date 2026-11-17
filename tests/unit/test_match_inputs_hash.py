"""inputs_hash: it changes with exactly the inputs the score reads, and with nothing else."""

import pytest

from app.matching.inputs_hash import match_inputs_hash
from app.matching.scoring import PreferenceFacts
from app.models.job import RemotePolicy, SalaryPeriod, Seniority
from tests.matching_helpers import config, cv_facts, job_facts, preferences


def digest(cv=None, prefs=None, job=None, cfg=None) -> str:
    return match_inputs_hash(
        cv=cv or cv_facts(), preferences=prefs, job=job or job_facts(), config=cfg or config()
    )


BASE = digest()


def test_it_is_a_64_character_hex_digest_and_stable() -> None:
    assert len(BASE) == 64
    assert int(BASE, 16) >= 0
    assert digest() == BASE


@pytest.mark.parametrize(
    "changed",
    [
        digest(cv=cv_facts(["Python", "Docker", "Go"])),
        digest(cv=cv_facts(vector_id="m|another")),
        digest(cv=cv_facts(vector_id=None)),
        digest(prefs=preferences(remote_policies=[RemotePolicy.REMOTE])),
        digest(job=job_facts(stack=["Python", "Docker", "Go"])),
        digest(job=job_facts(seniority=Seniority.SENIOR)),
        digest(job=job_facts(remote_policy=RemotePolicy.REMOTE)),
        digest(
            job=job_facts(salary_max=100, salary_currency="USD", salary_period=SalaryPeriod.YEAR)
        ),
        digest(job=job_facts(vector_id="m|changed")),
        digest(job=job_facts(raw_text_hash="z" * 64)),
        digest(job=job_facts(rules_version="rules_v4")),
        digest(cfg=config(weight_skills=0.5, weight_embedding=0.5)),
        digest(cfg=config(similarity_floor=0.7)),
        digest(cfg=config(similarity_ceiling=0.9)),
        digest(cfg=config(failed_filter_cap=20)),
    ],
)
def test_each_scored_input_changes_the_hash(changed: str) -> None:
    assert changed != BASE


def test_preferences_each_field_changes_the_hash() -> None:
    saved = digest(prefs=preferences())
    variants = [
        preferences(seniority_levels=[Seniority.SENIOR]),
        preferences(min_salary=1, salary_currency="USD", salary_period=SalaryPeriod.YEAR),
        preferences(min_salary=1, salary_currency="EUR", salary_period=SalaryPeriod.YEAR),
        preferences(min_salary=1, salary_currency="USD", salary_period=SalaryPeriod.MONTH),
    ]

    assert saved != BASE  # a saved but empty row differs from no row at all
    assert len({digest(prefs=v) for v in variants} | {saved}) == len(variants) + 1


def test_order_and_case_of_skills_do_not_matter() -> None:
    assert digest(cv=cv_facts(["docker", "PYTHON"])) == digest(cv=cv_facts(["Python", "Docker"]))
    assert digest(job=job_facts(stack=["Docker", "Python"])) == BASE
    assert digest(prefs=preferences(seniority_levels=[Seniority.LEAD, Seniority.SENIOR])) == digest(
        prefs=preferences(seniority_levels=[Seniority.SENIOR, Seniority.LEAD])
    )


def test_a_stored_skill_alias_hashes_like_its_canonical_name() -> None:
    assert digest(cv=cv_facts(["postgres"])) == digest(cv=cv_facts(["PostgreSQL"]))


def test_what_the_scorer_does_not_read_does_not_change_the_hash() -> None:
    # The notes are not part of PreferenceFacts at all, and the post text only enters through its
    # hash and the language check's slice (which the text hash already covers).
    assert not hasattr(PreferenceFacts(), "notes")
    assert digest(job=job_facts(text_start="another start of the same post")) == BASE


def test_a_float_that_rounds_the_same_hashes_the_same() -> None:
    assert digest(cfg=config(similarity_floor=0.715 + 1e-9)) == BASE
