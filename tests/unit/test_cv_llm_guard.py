"""The hosted-provider side of the CV privacy rules: a CV is personal data and must never reach a
hosted LLM. These two layers need no CV code at all:
  1. the allow-list of purposes a hosted provider may receive is pinned exactly;
  2. the guard refuses every other purpose, and any request marked as personal data.
The other two layers are in test_cv_privacy.py (the CV code cannot import an LLM client) and
tests/integration/test_cv_canary.py (a marker text must never reach an LLM call, a log or an error).
"""

import pytest

from app.core.config import Settings
from app.llm.base import HOSTED_ALLOWED_PURPOSES, LLMPrivacyError, LLMRequest, Purpose
from app.llm.client import GuardedClient
from app.llm.providers.fake import FakeLLMClient


def test_a_hosted_provider_may_receive_exactly_two_purposes() -> None:
    # If you are here because this failed: a hosted free-tier provider may read what it receives.
    # Do not add a purpose that touches a CV or any other personal data.
    assert HOSTED_ALLOWED_PURPOSES == {Purpose.JOB_POST_EXTRACTION, Purpose.CONNECTIVITY_CHECK}


def test_the_cv_purpose_is_not_on_the_hosted_allow_list() -> None:
    assert Purpose.CV_ANALYSIS in set(Purpose)
    assert Purpose.CV_ANALYSIS not in HOSTED_ALLOWED_PURPOSES


@pytest.mark.parametrize("purpose", sorted(set(Purpose) - HOSTED_ALLOWED_PURPOSES))
@pytest.mark.parametrize("marked_personal", [True, False])
async def test_a_hosted_provider_refuses_every_purpose_that_is_not_allowed(
    purpose: Purpose, marked_personal: bool
) -> None:
    inner = FakeLLMClient(local=False)
    request = LLMRequest(
        purpose=purpose,
        system="",
        user="Alex Example, Python developer, alex@example.com",
        json_schema=None,
        contains_personal_data=marked_personal,
    )

    with pytest.raises(LLMPrivacyError):
        await GuardedClient(inner).complete(request)

    assert inner.requests == []  # nothing reached the provider


async def test_cv_text_is_refused_by_a_hosted_provider_even_when_sent_as_a_job_post() -> None:
    inner = FakeLLMClient(local=False)
    request = LLMRequest(
        purpose=Purpose.JOB_POST_EXTRACTION,
        system="",
        user="my CV",
        json_schema=None,
        contains_personal_data=True,  # the one honest label a CV can carry
    )

    with pytest.raises(LLMPrivacyError):
        await GuardedClient(inner).complete(request)

    assert inner.requests == []


def test_no_setting_ties_the_cv_to_a_language_model() -> None:
    names = set(Settings.model_fields)

    assert not {name for name in names if "cv" in name and ("llm" in name or "provider" in name)}
