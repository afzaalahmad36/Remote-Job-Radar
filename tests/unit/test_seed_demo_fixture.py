"""The demo fixture and the seed script's guards: no database, no network."""

import json
import re
from pathlib import Path

import pytest

from app.core.config import Settings
from app.extraction.rules import extract_with_rules
from scripts import seed_demo
from scripts.db_guard import ScriptError

URL_HOST = re.compile(r"https?://([^/\s]+)")
EMAIL_DOMAIN = re.compile(r"[\w.+-]+@([\w.-]+)")


def test_the_fixture_has_twenty_distinct_invented_posts() -> None:
    jobs = seed_demo.load_fixture()

    assert len(jobs) == 20
    assert len({j.external_id for j in jobs}) == 20
    assert sum(len(j.text) < 300 for j in jobs) == 4  # the headline-only stubs


def test_every_address_and_link_in_the_fixture_is_on_example_com() -> None:
    text = "\n".join(j.text for j in seed_demo.load_fixture())

    assert set(URL_HOST.findall(text)) == {"example.com"}
    assert set(EMAIL_DOMAIN.findall(text)) == {"example.com"}


def test_the_rules_find_a_title_in_every_post_and_a_stack_in_most() -> None:
    results = [extract_with_rules(j.text) for j in seed_demo.load_fixture()]

    assert all(r.title is not None and r.title.value for r in results)
    assert sum(bool(r.stack) for r in results) >= 14


def test_a_fixture_with_an_unknown_field_or_a_repeated_id_is_refused(tmp_path: Path) -> None:
    job = {"external_id": "a", "days_before": 1, "text": "Backend Engineer at Example Co. " * 3}
    path = tmp_path / "f.json"

    path.write_text(json.dumps({"note": "n", "jobs": [{**job, "extra": 1}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="extra"):
        seed_demo.load_fixture(path)
    path.write_text(json.dumps({"note": "n", "jobs": [job, job]}), encoding="utf-8")
    with pytest.raises(ScriptError, match="repeated external_id"):
        seed_demo.load_fixture(path)


def test_production_is_refused(settings: Settings) -> None:
    production = settings.model_copy(update={"env": "production"})

    with pytest.raises(ScriptError, match="ENV=production"):
        seed_demo.check_target(production)


def test_a_database_not_named_jobradar_is_refused(settings: Settings) -> None:
    # the test database is named jobradar_test
    with pytest.raises(ScriptError, match="only reads a database named"):
        seed_demo.check_target(settings)


def test_a_remote_database_host_is_refused(settings: Settings) -> None:
    remote = settings.model_copy(
        update={"database_url": "postgresql+asyncpg://u:p@db.example.com:5432/jobradar"}
    )

    with pytest.raises(ScriptError, match="only writes to this machine"):
        seed_demo.check_target(remote)


def test_the_fixed_password_needs_an_explicit_flag() -> None:
    parser = seed_demo.build_parser()

    assert parser.parse_args([]).demo_password is False
    assert parser.parse_args(["--demo-password"]).demo_password is True
