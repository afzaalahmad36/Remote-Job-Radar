import pytest
from pydantic import ValidationError

from app.core.config import MAX_LLM_REQUEST_TIMEOUT_SECONDS
from app.extraction.limits import (
    EXTRACTION_HARD_TIME_LIMIT_SECONDS,
    EXTRACTION_SOFT_TIME_LIMIT_SECONDS,
    JOB_LOCK_TTL_SECONDS,
    MAX_REQUESTS_PER_JOB,
    run_deadline_seconds,
)
from tests.settings_helpers import settings_with


def test_each_limit_is_longer_than_the_one_inside_it() -> None:
    chain = [
        MAX_LLM_REQUEST_TIMEOUT_SECONDS,
        run_deadline_seconds(MAX_LLM_REQUEST_TIMEOUT_SECONDS),
        EXTRACTION_SOFT_TIME_LIMIT_SECONDS,
        EXTRACTION_HARD_TIME_LIMIT_SECONDS,
        JOB_LOCK_TTL_SECONDS,
    ]

    assert chain == sorted(set(chain))  # strictly increasing


def test_the_celery_limits_are_about_60_and_90_seconds() -> None:
    assert (EXTRACTION_SOFT_TIME_LIMIT_SECONDS, EXTRACTION_HARD_TIME_LIMIT_SECONDS) == (60, 90)


def test_the_job_lock_outlives_the_hard_time_limit() -> None:
    assert JOB_LOCK_TTL_SECONDS > EXTRACTION_HARD_TIME_LIMIT_SECONDS


def test_the_default_timeout_gives_a_deadline_inside_the_soft_limit() -> None:
    timeout = settings_with().llm_request_timeout_seconds

    assert timeout == 20.0
    assert MAX_REQUESTS_PER_JOB == 2
    assert run_deadline_seconds(timeout) == 45.0 < EXTRACTION_SOFT_TIME_LIMIT_SECONDS


def test_a_request_timeout_above_the_cap_is_rejected_so_the_chain_cannot_break() -> None:
    assert settings_with(llm_request_timeout_seconds=MAX_LLM_REQUEST_TIMEOUT_SECONDS)
    with pytest.raises(ValidationError):
        settings_with(llm_request_timeout_seconds=MAX_LLM_REQUEST_TIMEOUT_SECONDS + 1)
