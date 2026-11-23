"""The invented personas of the matching evaluation: tests/fixtures/cv/personas.toml.

They must be invented (reserved addresses), differ from each other in skills and seniority, and
read the way the protocol in docs/MATCHING.md section 10 says.
"""

import re
from pathlib import Path

from app.cv.skills import find_skills
from app.cv.text import clean_cv_text, mask_cv_text
from app.models.job import RemotePolicy, SalaryPeriod, Seniority
from scripts import match_eval as ev

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "cv"
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def specs() -> dict[str, ev.PersonaSpec]:
    return {s.name: s for s in ev.load_personas(FIXTURES / "personas.toml")}


def skills_of(spec: ev.PersonaSpec) -> set[str]:
    return set(find_skills(mask_cv_text(clean_cv_text(spec.cv_text))))


def test_the_three_personas_of_the_protocol_are_there_and_synthetic() -> None:
    loaded = specs()

    assert set(loaded) == {"alex", "sam", "jordan"} == set(ev.SEEN_EXAMPLES)
    assert all(spec.kind == "synthetic" for spec in loaded.values())


def test_the_preferences_are_the_ones_the_protocol_states() -> None:
    loaded = specs()
    alex, sam, jordan = loaded["alex"], loaded["sam"], loaded["jordan"]

    assert alex.preferences.seniority_levels == (Seniority.SENIOR, Seniority.LEAD)
    assert alex.preferences.min_salary == 100_000
    assert alex.preferences.salary_currency == "USD"
    assert alex.preferences.salary_period == SalaryPeriod.YEAR
    assert sam.preferences.seniority_levels == (Seniority.JUNIOR, Seniority.MID)
    assert jordan.preferences.seniority_levels == (Seniority.SENIOR,)
    for spec in loaded.values():
        assert spec.preferences.remote_policies == (RemotePolicy.REMOTE,)
    assert sam.preferences.min_salary is None and jordan.preferences.min_salary is None


def test_every_address_is_an_example_address() -> None:
    for name in ("alex", "sam", "jordan"):
        text = (FIXTURES / f"{name}_example.txt").read_text(encoding="utf-8")
        addresses = EMAIL.findall(text)

        assert addresses and all(a.endswith("@example.com") for a in addresses)
        assert "+1 555 010" in text


def test_the_masked_text_keeps_no_address_or_phone() -> None:
    for spec in specs().values():
        masked = mask_cv_text(clean_cv_text(spec.cv_text))

        assert not EMAIL.search(masked)
        assert "555 010" not in masked


def test_the_personas_differ_in_skills() -> None:
    alex, sam, jordan = (skills_of(s) for s in (specs()[n] for n in ("alex", "sam", "jordan")))

    assert min(len(alex), len(sam), len(jordan)) >= 8
    assert {"Next.js", "Tailwind CSS"} & sam and not {"Next.js"} & alex
    assert {"Pandas", "Airflow", "Snowflake"} <= jordan
    assert not {"Pandas", "Airflow", "Snowflake"} & (alex | sam)
    assert not {"FastAPI", "Celery"} & (sam | jordan)
    assert len(alex & sam) < len(sam) / 2 and len(alex & jordan) < len(jordan) / 2


def test_the_personas_differ_in_seniority() -> None:
    loaded = specs()
    levels = {n: set(s.preferences.seniority_levels) for n, s in loaded.items()}

    assert levels["sam"].isdisjoint(levels["alex"]) and levels["sam"].isdisjoint(levels["jordan"])
    assert (
        "senior" in loaded["alex"].cv_text.lower() and "senior" in loaded["jordan"].cv_text.lower()
    )
    assert "senior" not in loaded["sam"].cv_text.lower()
