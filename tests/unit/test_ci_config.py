"""Guards for the properties of .github/workflows/ci.yml that must not silently change."""

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def jobs() -> dict[str, dict[str, Any]]:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8"))
    result: dict[str, dict[str, Any]] = workflow["jobs"]
    return result


def test_every_ci_job_has_a_timeout_of_15_minutes_or_less(jobs: dict[str, Any]) -> None:
    # Without one a hung step runs for GitHub's default 6 hours (a Celery test once did).
    assert jobs  # there are jobs to check
    for name, job in jobs.items():
        assert 0 < job.get("timeout-minutes", 0) <= 15, f"job {name!r} needs timeout-minutes"


def test_ci_has_no_redis_service_because_tests_never_use_one(jobs: dict[str, Any]) -> None:
    services = jobs["quality"].get("services", {})

    assert set(services) == {"postgres"}
