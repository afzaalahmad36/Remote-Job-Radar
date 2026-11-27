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


# web-e2e builds an image, a stack and the web app, and may download a browser: allowed longer
# until its real duration on GitHub is known.
LONGER_JOBS = {"web-e2e": 25}


def test_every_ci_job_has_a_short_timeout(jobs: dict[str, Any]) -> None:
    # Without one a hung step runs for GitHub's default 6 hours (a Celery test once did).
    assert jobs  # there are jobs to check
    for name, job in jobs.items():
        limit = LONGER_JOBS.get(name, 15)
        assert 0 < job.get("timeout-minutes", 0) <= limit, f"job {name!r} needs timeout-minutes"


def test_ci_has_no_redis_service_because_tests_never_use_one(jobs: dict[str, Any]) -> None:
    services = jobs["quality"].get("services", {})

    assert set(services) == {"postgres"}
