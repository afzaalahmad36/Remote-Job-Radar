"""The model path of extraction against a real Postgres. Only fake providers are used, and the
network guard stays on: nothing here can reach a real provider."""

import asyncio
import json
import logging
import uuid
from datetime import timedelta
from typing import Any

import pytest

from app.extraction.plan import ExtractionError
from app.llm import client as client_module
from app.llm.base import (
    LLMBadResponse,
    LLMConfigError,
    LLMError,
    LLMRateLimited,
    LLMRejected,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    Purpose,
)
from app.llm.client import GuardedClient, build_client
from app.llm.prompts import EXTRACT_JOB_V1
from app.llm.providers.fake import DEFAULT_RESPONSE, FakeLLMClient
from app.models import ExtractionMethod, ExtractionStatus, LlmEnrichment
from app.services import enrichment_service
from app.services.cooldown import UNAVAILABLE_RETRY_SECONDS
from app.services.extraction_service import extract_job
from app.services.extraction_types import (
    ExtractionOutcome,
    ExtractionResult,
    SessionFactory,
)
from app.services.llm_budget import DeferralReason
from tests.extraction_helpers import (
    NEEDS_STACK,
    NOTHING_LONG,
    NOW,
    RICH,
    add_job,
    add_usage,
    add_user,
    checked_out_connections,
    owner_of,
    row,
    usage_rows,
)
from tests.fakes import FakeCooldown
from tests.settings_helpers import settings_with

SETTINGS = settings_with(llm_min_post_chars=500)


def answer(**fields: Any) -> str:
    """A model reply: a valid, otherwise empty extraction with `fields` filled in."""
    return json.dumps({**json.loads(DEFAULT_RESPONSE), **fields})


async def run(
    factory: SessionFactory,
    job_id: uuid.UUID,
    llm: Any,
    cooldown: FakeCooldown | None = None,
    settings: Any = SETTINGS,
) -> ExtractionResult:
    result = await extract_job(
        factory,
        job_id,
        settings=settings,
        llm=llm,
        cooldown=cooldown if cooldown is not None else FakeCooldown(),
        now=NOW,
    )
    if result.outcome is ExtractionOutcome.DEFERRED:
        assert result.deferral is not None and (result.retry_after_seconds or 0) >= 1.0
    return result


async def waiting_job(factory: SessionFactory, text: str = NEEDS_STACK) -> uuid.UUID:
    """A job the rules have handled that now waits for the model (pending / pending)."""
    job_id = await add_job(factory, text)
    assert (await run(factory, job_id, None)).outcome in {
        ExtractionOutcome.FAILED,
        ExtractionOutcome.AWAITING_MODEL,
    }
    return job_id


# --- a valid answer ------------------------------------------------------------------------------


async def test_a_valid_answer_is_merged_stored_and_counted(factory: SessionFactory) -> None:
    llm = FakeLLMClient([answer(is_job_posting=True, stack=["Zigzag Engine", "Quux"])])
    job_id = await add_job(factory, NEEDS_STACK)

    result = await run(factory, job_id, llm)

    assert result == ExtractionResult(ExtractionOutcome.ENRICHED)
    job = await row(factory, job_id)
    assert (job.extraction_status, job.llm_enrichment) == (
        ExtractionStatus.EXTRACTED,
        LlmEnrichment.DONE,
    )
    assert job.extraction_method is ExtractionMethod.RULES_LLM
    assert (job.extraction_model, job.extraction_prompt_version) == (
        "fake-model",
        EXTRACT_JOB_V1.version,
    )
    assert job.title == "Marketing Manager" and job.stack == ["Zigzag Engine", "Quux"]
    assert (job.extraction_error, job.extraction_attempts, job.extracted_at) == (None, 1, NOW)
    [entry] = await usage_rows(factory)
    assert (entry.outcome, entry.provider, entry.model) == ("ok", "fake", "fake-model")
    assert (entry.input_tokens, entry.output_tokens) == (10, 5)
    assert entry.job_id == job_id and entry.user_id == await owner_of(factory, job_id)
    assert len(llm.requests) == 1


async def test_the_model_supplies_a_title_the_rules_could_not_find(factory: SessionFactory) -> None:
    llm = FakeLLMClient([answer(title="Backend Engineer", stack=[])])
    job_id = await add_job(factory, NOTHING_LONG)

    assert (await run(factory, job_id, llm)).outcome is ExtractionOutcome.ENRICHED

    job = await row(factory, job_id)
    assert (job.title, job.extraction_status) == ("Backend Engineer", ExtractionStatus.EXTRACTED)


async def test_a_model_that_finds_no_title_either_fails_the_job_with_a_reason(
    factory: SessionFactory,
) -> None:
    job_id = await add_job(factory, NOTHING_LONG)

    result = await run(factory, job_id, FakeLLMClient([answer()]))

    assert result.outcome is ExtractionOutcome.FAILED
    job = await row(factory, job_id)
    assert job.extraction_status is ExtractionStatus.EXTRACTION_FAILED
    assert job.llm_enrichment is LlmEnrichment.DONE
    assert job.extraction_error == ExtractionError.NO_TITLE.value


async def test_a_not_a_job_verdict_is_stored(factory: SessionFactory) -> None:
    job_id = await add_job(factory, NEEDS_STACK)

    result = await run(factory, job_id, FakeLLMClient([answer(is_job_posting=False)]))

    assert result.outcome is ExtractionOutcome.NOT_A_JOB
    job = await row(factory, job_id)
    assert (job.extraction_status, job.llm_enrichment) == (
        ExtractionStatus.NOT_A_JOB,
        LlmEnrichment.DONE,
    )
    assert job.title == "Marketing Manager"  # what the rules found is kept


async def test_a_stack_entry_the_post_does_not_contain_is_dropped_and_only_counted_in_the_log(
    factory: SessionFactory, caplog: pytest.LogCaptureFixture
) -> None:
    llm = FakeLLMClient([answer(stack=["Zigzag Engine", "Kubernetes", "Haskell"])])
    job_id = await add_job(factory, NEEDS_STACK)

    with caplog.at_level(logging.INFO, logger="app.services.enrichment_service"):
        await run(factory, job_id, llm)

    assert (await row(factory, job_id)).stack == ["Zigzag Engine"]
    assert f"Job {job_id}: grounding dropped field=stack count=2" in caplog.text
    assert "Kubernetes" not in caplog.text and "Haskell" not in caplog.text


async def test_a_job_that_waits_for_the_model_is_resumed_without_running_the_rules_again(
    factory: SessionFactory,
) -> None:
    job_id = await waiting_job(factory)
    before = await row(factory, job_id)
    assert (before.extraction_status, before.llm_enrichment) == (
        ExtractionStatus.EXTRACTED,
        LlmEnrichment.PENDING,
    )

    result = await run(factory, job_id, FakeLLMClient([answer(stack=["Quux"])]))

    assert result.outcome is ExtractionOutcome.ENRICHED
    assert (await row(factory, job_id)).extraction_attempts == before.extraction_attempts == 1


async def test_one_call_does_the_rules_and_then_the_model(factory: SessionFactory) -> None:
    llm = FakeLLMClient([answer(stack=["Quux"])])
    job_id = await add_job(factory, NEEDS_STACK)

    await run(factory, job_id, llm)

    job = await row(factory, job_id)
    assert job.extraction_method is ExtractionMethod.RULES_LLM and job.stack == ["Quux"]


async def test_enrich_job_ignores_a_job_that_is_missing_or_not_waiting(
    factory: SessionFactory,
) -> None:
    llm, cooldown = FakeLLMClient(), FakeCooldown()
    settings = SETTINGS
    done = await add_job(factory, RICH)
    await extract_job(factory, done, settings=settings, now=NOW)

    missing = await enrichment_service.enrich_job(
        factory, uuid.uuid4(), settings=settings, llm=llm, cooldown=cooldown
    )
    not_waiting = await enrichment_service.enrich_job(
        factory, done, settings=settings, llm=llm, cooldown=cooldown
    )

    assert missing.outcome is ExtractionOutcome.NOT_FOUND
    assert not_waiting.outcome is ExtractionOutcome.ALREADY_DONE
    assert llm.requests == []


async def test_identical_text_finished_meanwhile_is_reused_instead_of_asking(
    factory: SessionFactory,
) -> None:
    job_id = await waiting_job(factory)
    donor = await add_job(factory, NEEDS_STACK)
    await run(factory, donor, FakeLLMClient([answer(stack=["Quux"], title="Donor Title")]))
    second = FakeLLMClient()

    result = await run(factory, job_id, second)

    assert result.outcome is ExtractionOutcome.REUSED
    assert second.requests == []
    assert (await row(factory, job_id)).stack == ["Quux"]


# --- the validation retry ------------------------------------------------------------------------


async def test_an_invalid_answer_is_retried_once_with_field_names_only(
    factory: SessionFactory,
) -> None:
    secret_value = "SECRET-VALUE-FROM-THE-MODEL"
    bad = json.dumps({"is_job_posting": True, "stack": secret_value, "bogus": 1})
    llm = FakeLLMClient([bad, answer(stack=["Quux"])], strict=True)
    job_id = await add_job(factory, NEEDS_STACK)

    result = await run(factory, job_id, llm)

    assert result.outcome is ExtractionOutcome.ENRICHED
    assert len(llm.requests) == 2
    retry_text = llm.requests[1].user
    assert "rejected by validation" in retry_text and "stack" in retry_text
    assert secret_value not in retry_text  # field names and error kinds, never the received value
    outcomes = sorted(entry.outcome for entry in await usage_rows(factory))
    assert outcomes == ["invalid_response", "ok"]  # rows share one fixed timestamp: order is free


@pytest.mark.parametrize(
    ("text", "status"),
    [
        (NEEDS_STACK, ExtractionStatus.EXTRACTED),  # has a title: stays usable
        (NOTHING_LONG, ExtractionStatus.EXTRACTION_FAILED),  # nothing left that could add one
    ],
)
async def test_two_invalid_answers_in_a_row_give_up_after_exactly_two_requests(
    factory: SessionFactory, text: str, status: ExtractionStatus
) -> None:
    llm = FakeLLMClient(["not json at all"], strict=False)
    job_id = await add_job(factory, text)

    result = await run(factory, job_id, llm)

    assert result.outcome is ExtractionOutcome.SKIPPED
    assert len(llm.requests) == 2
    job = await row(factory, job_id)
    assert (job.extraction_status, job.llm_enrichment) == (status, LlmEnrichment.SKIPPED)
    assert job.extraction_error == ExtractionError.INVALID_REPLY.value
    assert [entry.outcome for entry in await usage_rows(factory)] == ["invalid_response"] * 2


async def test_an_empty_or_cut_off_reply_counts_as_invalid_too(factory: SessionFactory) -> None:
    llm = FakeLLMClient([LLMBadResponse("empty output"), answer(stack=["Quux"])], strict=True)
    job_id = await add_job(factory, NEEDS_STACK)

    result = await run(factory, job_id, llm)

    assert result.outcome is ExtractionOutcome.ENRICHED
    assert "empty output" in llm.requests[1].user


# --- failures that leave the job pending ---------------------------------------------------------


@pytest.mark.parametrize(
    ("retry_after", "pause"),
    [(None, 60.0), (5.0, 30.0), (120.0, 120.0), (5000.0, 900.0)],
)
async def test_a_429_leaves_the_job_pending_pauses_everyone_and_counts_the_request(
    factory: SessionFactory, retry_after: float | None, pause: float
) -> None:
    cooldown = FakeCooldown()
    job_id = await add_job(factory, NEEDS_STACK)

    result = await run(factory, job_id, FakeLLMClient([LLMRateLimited(retry_after)]), cooldown)

    assert result.outcome is ExtractionOutcome.DEFERRED
    assert (result.deferral, result.retry_after_seconds) == (DeferralReason.RATE_LIMITED, pause)
    assert cooldown.started == [pause]
    job = await row(factory, job_id)
    assert (job.extraction_status, job.llm_enrichment) == (
        ExtractionStatus.EXTRACTED,
        LlmEnrichment.PENDING,
    )
    assert [entry.outcome for entry in await usage_rows(factory)] == ["rate_limited"]


@pytest.mark.parametrize(
    ("reason", "outcome", "deferral"),
    [
        ("server error 503", "unavailable", DeferralReason.UNAVAILABLE),
        ("connection failed", "unavailable", DeferralReason.UNAVAILABLE),
        ("provider reported a failure", "unavailable", DeferralReason.UNAVAILABLE),
        ("no response in time", "timeout", DeferralReason.TIMEOUT),
        ("connect timeout", "timeout", DeferralReason.TIMEOUT),
    ],
)
async def test_a_transient_failure_leaves_the_job_pending_with_a_short_pause_and_a_counted_row(
    factory: SessionFactory, reason: str, outcome: str, deferral: DeferralReason
) -> None:
    cooldown = FakeCooldown()
    job_id = await add_job(factory, NEEDS_STACK)

    result = await run(factory, job_id, FakeLLMClient([LLMTransientError(reason)]), cooldown)

    assert (result.outcome, result.deferral) == (ExtractionOutcome.DEFERRED, deferral)
    assert cooldown.started == [30.0]
    assert (await row(factory, job_id)).llm_enrichment is LlmEnrichment.PENDING
    assert [entry.outcome for entry in await usage_rows(factory)] == [outcome]


async def test_a_provider_that_ignores_its_own_timeout_is_cut_off_by_the_run_deadline(
    factory: SessionFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Hanging(FakeLLMClient):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            await asyncio.sleep(30)
            return await super().complete(request)

    monkeypatch.setattr(enrichment_service, "run_deadline_seconds", lambda _: 0.05)
    cooldown = FakeCooldown()
    job_id = await add_job(factory, NEEDS_STACK)

    result = await run(factory, job_id, Hanging(), cooldown)

    assert (result.outcome, result.deferral) == (ExtractionOutcome.DEFERRED, DeferralReason.TIMEOUT)
    assert cooldown.started == [30.0]
    assert [entry.outcome for entry in await usage_rows(factory)] == ["timeout"]


async def test_a_failed_request_is_not_retried_inside_the_run(factory: SessionFactory) -> None:
    llm = FakeLLMClient([LLMTransientError("server error 503")], strict=True)
    job_id = await add_job(factory, NEEDS_STACK)

    await run(factory, job_id, llm)

    assert len(llm.requests) == 1


async def test_a_pause_that_cannot_be_started_does_not_lose_the_result(
    factory: SessionFactory, caplog: pytest.LogCaptureFixture
) -> None:
    job_id = await add_job(factory, NEEDS_STACK)

    with caplog.at_level(logging.WARNING, logger="app.services.enrichment_service"):
        result = await run(
            factory, job_id, FakeLLMClient([LLMRateLimited(None)]), FakeCooldown(fail_start=True)
        )

    assert result.outcome is ExtractionOutcome.DEFERRED
    assert "pause could not be started" in caplog.text


# --- the shared pause, and Redis being down ------------------------------------------------------


async def test_while_a_pause_is_in_effect_no_request_is_made_and_nothing_is_reserved(
    factory: SessionFactory,
) -> None:
    llm = FakeLLMClient()
    job_id = await add_job(factory, NEEDS_STACK)

    result = await run(factory, job_id, llm, FakeCooldown(seconds_left=42.0))

    assert (result.outcome, result.deferral, result.retry_after_seconds) == (
        ExtractionOutcome.DEFERRED,
        DeferralReason.COOLDOWN,
        42.0,
    )
    assert llm.requests == [] and await usage_rows(factory) == []


async def test_when_redis_is_down_the_model_is_not_asked_but_the_rules_still_work(
    factory: SessionFactory, caplog: pytest.LogCaptureFixture
) -> None:
    llm = FakeLLMClient()
    job_id = await add_job(factory, NEEDS_STACK)

    with caplog.at_level(logging.WARNING, logger="app.services.enrichment_service"):
        result = await run(factory, job_id, llm, FakeCooldown(unavailable=True))

    assert (result.outcome, result.deferral) == (
        ExtractionOutcome.DEFERRED,
        DeferralReason.COOLDOWN_UNAVAILABLE,
    )
    # A floor on the hint, so a caller that honours it cannot loop tightly while Redis is down.
    assert (result.retry_after_seconds or 0) >= UNAVAILABLE_RETRY_SECONDS
    assert llm.requests == [] and await usage_rows(factory) == []
    job = await row(factory, job_id)
    assert job.title == "Marketing Manager" and job.extraction_method is ExtractionMethod.RULES
    assert (job.extraction_status, job.llm_enrichment) == (
        ExtractionStatus.EXTRACTED,
        LlmEnrichment.PENDING,
    )
    assert "cooldown store is unavailable" in caplog.text


async def test_every_deferral_carries_a_hint_of_at_least_a_second(factory: SessionFactory) -> None:
    for cooldown in (FakeCooldown(seconds_left=0.01), FakeCooldown(unavailable=True)):
        job_id = await add_job(factory, NEEDS_STACK)

        result = await extract_job(
            factory, job_id, settings=SETTINGS, llm=FakeLLMClient(), cooldown=cooldown, now=NOW
        )

        assert result.outcome is ExtractionOutcome.DEFERRED
        assert (result.retry_after_seconds or 0) >= 1.0


# --- caps and the throttle -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("settings", "reason"),
    [
        (settings_with(llm_daily_request_cap_per_user=1), DeferralReason.USER_CAP),
        (settings_with(llm_daily_request_cap_global=1), DeferralReason.GLOBAL_CAP),
        (settings_with(llm_requests_per_minute=1), DeferralReason.THROTTLE),
    ],
)
async def test_a_full_budget_defers_the_job_without_a_request(
    factory: SessionFactory, settings: Any, reason: DeferralReason
) -> None:
    job_id = await add_job(factory, NEEDS_STACK)
    await add_usage(factory, await owner_of(factory, job_id), NOW - timedelta(seconds=10))
    llm = FakeLLMClient()

    result = await run(factory, job_id, llm, settings=settings)

    assert (result.outcome, result.deferral) == (ExtractionOutcome.DEFERRED, reason)
    assert llm.requests == []
    assert len(await usage_rows(factory)) == 1  # only the one we put there
    assert (await row(factory, job_id)).llm_enrichment is LlmEnrichment.PENDING


async def test_the_users_budget_is_not_used_up_by_someone_elses_requests(
    factory: SessionFactory,
) -> None:
    stranger = await add_user(factory)
    for _ in range(3):
        await add_usage(factory, stranger, NOW - timedelta(minutes=30))
    job_id = await add_job(factory, NEEDS_STACK)

    result = await run(
        factory,
        job_id,
        FakeLLMClient([answer(stack=["Quux"])]),
        settings=settings_with(llm_daily_request_cap_per_user=1),
    )

    assert result.outcome is ExtractionOutcome.ENRICHED


# --- requests that never leave, are rejected, or die ---------------------------------------------


async def test_a_request_refused_by_the_guard_leaves_no_row_and_nothing_is_sent(
    factory: SessionFactory, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(client_module, "HOSTED_ALLOWED_PURPOSES", frozenset())
    inner = FakeLLMClient(local=False)
    job_id = await add_job(factory, NEEDS_STACK)

    with caplog.at_level(logging.ERROR, logger="app.services.enrichment_service"):
        result = await run(factory, job_id, GuardedClient(inner))

    assert result.outcome is ExtractionOutcome.SKIPPED
    assert inner.requests == []
    assert await usage_rows(factory) == []  # it never left, so it was never counted
    job = await row(factory, job_id)
    assert (job.extraction_status, job.llm_enrichment) == (
        ExtractionStatus.EXTRACTED,
        LlmEnrichment.SKIPPED,
    )
    assert job.extraction_error == ExtractionError.REFUSED_LOCALLY.value
    assert "refused locally" in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        LLMRejected(401),  # bad key
        LLMRejected(403),  # no access
        LLMRejected(404),  # unknown model
        LLMRejected(402),  # any other 4xx is not about this job either
        LLMRejected(422),
        LLMConfigError("unexpected redirect"),  # no status at all
    ],
    ids=lambda error: str(error),
)
async def test_a_configuration_failure_leaves_the_job_pending_and_pauses_everyone_for_long(
    factory: SessionFactory, error: LLMConfigError, caplog: pytest.LogCaptureFixture
) -> None:
    cooldown = FakeCooldown()
    llm = FakeLLMClient([error], strict=True)
    job_id = await add_job(factory, NEEDS_STACK)

    with caplog.at_level(logging.ERROR, logger="app.services.enrichment_service"):
        result = await run(factory, job_id, llm, cooldown)

    assert result.outcome is ExtractionOutcome.DEFERRED
    assert (result.deferral, result.retry_after_seconds) == (DeferralReason.CONFIGURATION, 900.0)
    assert cooldown.started == [900.0]  # clamped to 15 minutes
    job = await row(factory, job_id)
    assert (job.extraction_status, job.llm_enrichment, job.extraction_error) == (
        ExtractionStatus.EXTRACTED,
        LlmEnrichment.PENDING,
        None,
    )  # nothing wrong with the job, so nothing is written against it
    assert [entry.outcome for entry in await usage_rows(factory)] == ["rejected"]  # still counted
    [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
    status = getattr(error, "status_code", None)
    assert str(job_id) in record.getMessage()
    assert f"status={status if status is not None else 'none'}" in record.getMessage()
    assert "request rejected" not in caplog.text and "redirect" not in caplog.text


async def test_after_a_configuration_failure_no_job_asks_again_until_the_pause_ends(
    factory: SessionFactory,
) -> None:
    cooldown = FakeCooldown()
    llm = FakeLLMClient([LLMRejected(401)], strict=True)
    first, second = await add_job(factory, NEEDS_STACK), await add_job(factory, NEEDS_STACK)

    await run(factory, first, llm, cooldown)
    result = await run(factory, second, llm, cooldown)

    assert (result.outcome, result.deferral) == (
        ExtractionOutcome.DEFERRED,
        DeferralReason.COOLDOWN,
    )
    assert len(llm.requests) == 1  # strict: a second request would have failed the test


async def test_only_a_400_is_about_the_jobs_content_and_ends_in_skipped(
    factory: SessionFactory, caplog: pytest.LogCaptureFixture
) -> None:
    cooldown = FakeCooldown()
    llm = FakeLLMClient([LLMRejected(400)], strict=True)
    job_id = await add_job(factory, NEEDS_STACK)

    with caplog.at_level(logging.WARNING, logger="app.services.enrichment_service"):
        result = await run(factory, job_id, llm, cooldown)

    assert result.outcome is ExtractionOutcome.SKIPPED
    assert cooldown.started == []  # one job's bad content is no reason to stop the others
    assert [entry.outcome for entry in await usage_rows(factory)] == ["rejected"]
    job = await row(factory, job_id)
    assert (job.llm_enrichment, job.extraction_error) == (
        LlmEnrichment.SKIPPED,
        ExtractionError.REJECTED.value,
    )
    assert [r.levelno for r in caplog.records] == [logging.WARNING]  # visible, but not an error


async def test_an_unknown_provider_error_is_treated_as_a_transient_failure(
    factory: SessionFactory,
) -> None:
    result = await run(
        factory, await add_job(factory, NEEDS_STACK), FakeLLMClient([LLMError("something odd")])
    )

    assert (result.outcome, result.deferral) == (
        ExtractionOutcome.DEFERRED,
        DeferralReason.UNAVAILABLE,
    )


async def test_a_worker_that_dies_after_reserving_leaves_a_row_that_keeps_counting(
    factory: SessionFactory,
) -> None:
    class Dies(FakeLLMClient):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            raise RuntimeError("the worker is killed here")

    job_id = await add_job(factory, NEEDS_STACK)

    with pytest.raises(RuntimeError, match="killed"):
        await run(factory, job_id, Dies())

    [entry] = await usage_rows(factory)
    assert entry.outcome == "reserved"
    job = await row(factory, job_id)
    assert job.llm_enrichment is LlmEnrichment.PENDING  # untouched: a later run tries again
    # ...and the dead request keeps counting against the cap.
    capped = settings_with(llm_daily_request_cap_per_user=1)
    result = await run(factory, job_id, FakeLLMClient(), settings=capped)
    assert result.deferral is DeferralReason.USER_CAP


# --- no transaction while the provider works -----------------------------------------------------


async def test_no_database_connection_is_in_use_while_the_provider_is_called(
    factory: SessionFactory,
) -> None:
    seen: list[int] = []

    class Probe(FakeLLMClient):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            seen.append(checked_out_connections(factory))
            return await super().complete(request)

    job_id = await add_job(factory, NEEDS_STACK)

    result = await run(factory, job_id, Probe([answer(stack=["Quux"])]))

    assert result.outcome is ExtractionOutcome.ENRICHED
    assert seen == [0]  # a session or transaction left open would show as a checked-out connection
    assert checked_out_connections(factory) == 0


# --- two workers ---------------------------------------------------------------------------------


async def test_two_workers_asking_at_once_store_one_answer_and_both_requests_are_counted(
    factory: SessionFactory,
) -> None:
    job_id = await waiting_job(factory)
    first = FakeLLMClient([answer(stack=["Quux"])], model="first")
    second = FakeLLMClient([answer(stack=["Zigzag Engine"])], model="second")
    cooldown = FakeCooldown()

    results = await asyncio.gather(
        run(factory, job_id, first, cooldown), run(factory, job_id, second, cooldown)
    )

    outcomes = sorted(result.outcome for result in results)
    assert outcomes == [ExtractionOutcome.ENRICHED, ExtractionOutcome.LOST_RACE]
    job = await row(factory, job_id)
    assert job.extraction_model in {"first", "second"}  # one consistent writer, not a mixture
    assert job.stack == (["Quux"] if job.extraction_model == "first" else ["Zigzag Engine"])
    assert [entry.outcome for entry in await usage_rows(factory)] == ["ok", "ok"]


# --- what may leave the machine ------------------------------------------------------------------


async def test_only_masked_job_post_text_is_ever_sent_to_a_hosted_provider(
    factory: SessionFactory,
) -> None:
    post = NEEDS_STACK + "\nContact hr@acme.com or call +1 415 555 0100 to apply."
    inner = FakeLLMClient([answer(stack=["Quux"])], local=False)
    job_id = await add_job(factory, post)

    result = await run(factory, job_id, GuardedClient(inner))

    assert result.outcome is ExtractionOutcome.ENRICHED
    [request] = inner.requests
    assert request.purpose is Purpose.JOB_POST_EXTRACTION
    assert request.store is False and request.contains_personal_data is False
    sent = request.system + request.user
    assert "hr@acme.com" not in sent and "415 555 0100" not in sent
    assert "Zigzag Engine" in request.user  # the public post text itself does go
    assert (await row(factory, job_id)).apply_email == "hr@acme.com"  # the rules own the contact


async def test_every_request_in_a_busy_run_uses_the_extraction_purpose_only(
    factory: SessionFactory,
) -> None:
    inner = FakeLLMClient(
        ["bad", answer(stack=["Quux"]), answer(is_job_posting=False)], local=False
    )
    guarded = GuardedClient(inner)
    for text in (NEEDS_STACK, NEEDS_STACK + " Extra words.", RICH, NOTHING_LONG):
        await run(factory, await add_job(factory, text), guarded)

    assert inner.requests  # something was asked
    assert {request.purpose for request in inner.requests} == {Purpose.JOB_POST_EXTRACTION}
    assert all(request.store is False for request in inner.requests)


async def test_the_default_setup_never_builds_a_client_so_nothing_can_be_sent() -> None:
    assert build_client(settings_with()) is None
