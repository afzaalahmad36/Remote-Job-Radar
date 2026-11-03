"""The model path of extraction: ask a provider for what the rules missed, and store the answer.

A run reads the job, then (for each request) checks the shared cooldown, reserves a place in the
budget, calls the provider with **no database transaction open**, and finally writes the result
with a compare-and-set. See `llm_budget` for the reservation and `cooldown` for the pause.

How each way a request can end is treated:

* a valid answer: merged with the rules' result and stored (`done`);
* an invalid answer: one corrective retry, then the model is given up on (`skipped`);
* 429, 5xx, a timeout, or no budget left: the job is left exactly as it is (`pending`) and a
  shared pause is started, so a later run tries again;
* a rejected request, or one our own guard refused: the model is given up on (`skipped`).

Every request that may have reached the provider keeps its `llm_usage` row; one that never left
this process has its row removed.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.config import Settings
from app.extraction.limits import MAX_REQUESTS_PER_JOB, run_deadline_seconds
from app.extraction.merge import merge
from app.extraction.plan import ExtractionError, model_result_state
from app.extraction.rules import extract_with_rules
from app.extraction.schema import ExtractionFormatError, JobExtraction, parse_extraction
from app.extraction.text_hash import raw_text_hash
from app.llm.base import (
    LLMBadResponse,
    LLMClient,
    LLMConfigError,
    LLMError,
    LLMRateLimited,
    LLMRejected,
    LLMResponse,
    LLMTransientError,
    LLMUsage,
)
from app.llm.prompts import EXTRACT_JOB_V1, build_extraction_request
from app.models import ExtractionStatus, LlmEnrichment
from app.repositories import extraction_repository as repository
from app.repositories.extraction_repository import JobSnapshot
from app.services import llm_budget
from app.services.cooldown import (
    CONFIGURATION_FAILURE_SECONDS,
    TRANSIENT_SECONDS,
    UNAVAILABLE_RETRY_SECONDS,
    Cooldown,
    CooldownUnavailable,
    rate_limit_pause,
)
from app.services.extraction_types import (
    ExtractionOutcome,
    ExtractionResult,
    SessionFactory,
    transaction,
)
from app.services.llm_budget import DeferralReason, Outcome, Refusal, Reservation

logger = logging.getLogger(__name__)

HTTP_BAD_REQUEST = 400


@dataclass(frozen=True)
class _Answer:
    """A request that came back valid, not yet stored (its usage row is finished with the job)."""

    extraction: JobExtraction
    reservation: Reservation
    usage: LLMUsage


@dataclass(frozen=True)
class _Invalid:
    """A request that came back unusable. `summary` is a fixed phrase or field names, never text."""

    summary: str


async def enrich_job(
    factory: SessionFactory,
    job_id: uuid.UUID,
    *,
    settings: Settings,
    llm: LLMClient,
    cooldown: Cooldown,
    now: datetime | None = None,
) -> ExtractionResult:
    """Ask the model about a job that waits for it (`llm_enrichment = pending`), once.

    Safe to run again and safe to run twice at once: every write is a compare-and-set, so a
    second worker that finishes later finds the job changed and its answer is dropped.
    """
    async with transaction(factory) as session:
        snapshot = await repository.load_snapshot(session, job_id)
    if snapshot is None:
        return ExtractionResult(ExtractionOutcome.NOT_FOUND)
    if snapshot.enrichment is not LlmEnrichment.PENDING:
        return ExtractionResult(ExtractionOutcome.ALREADY_DONE)
    reused = await _reuse_if_finished_elsewhere(factory, snapshot, now)
    if reused is not None:
        return reused
    asked = await _ask(factory, snapshot, settings, llm, cooldown, now)
    if isinstance(asked, ExtractionResult):
        return asked
    return await _store(factory, snapshot, asked, llm, now)


async def _reuse_if_finished_elsewhere(
    factory: SessionFactory, snapshot: JobSnapshot, now: datetime | None
) -> ExtractionResult | None:
    """A second look for identical text, right before spending a request: another job may have
    been finished since this one was planned."""
    text_hash = raw_text_hash(snapshot.raw_text)
    async with transaction(factory) as session:
        donor = await repository.find_reusable(
            session,
            text_hash=text_hash,
            exclude_id=snapshot.id,
            prompt_version=EXTRACT_JOB_V1.version,
        )
    if donor is None:
        return None
    async with transaction(factory) as session:
        won = await repository.write_reused_result(
            session, snapshot, donor, text_hash, now or datetime.now(UTC)
        )
    return ExtractionResult(ExtractionOutcome.REUSED if won else ExtractionOutcome.LOST_RACE)


async def _ask(
    factory: SessionFactory,
    snapshot: JobSnapshot,
    settings: Settings,
    llm: LLMClient,
    cooldown: Cooldown,
    now: datetime | None,
) -> _Answer | ExtractionResult:
    """Up to two requests: the first, and one corrective retry after an invalid answer."""
    deadline = asyncio.get_running_loop().time() + run_deadline_seconds(
        settings.llm_request_timeout_seconds
    )
    previous_error: str | None = None
    for _ in range(MAX_REQUESTS_PER_JOB):
        step = await _request_once(
            factory, snapshot, settings, llm, cooldown, now, deadline, previous_error
        )
        if not isinstance(step, _Invalid):
            return step
        previous_error = step.summary
    return await _skip(factory, snapshot, ExtractionError.INVALID_REPLY)


async def _request_once(
    factory: SessionFactory,
    snapshot: JobSnapshot,
    settings: Settings,
    llm: LLMClient,
    cooldown: Cooldown,
    now: datetime | None,
    deadline: float,
    previous_error: str | None,
) -> _Answer | ExtractionResult | _Invalid:
    if (paused := await _pause_in_effect(cooldown)) is not None:
        return paused
    request = build_extraction_request(
        snapshot.raw_text, max_chars=settings.llm_max_input_chars, previous_error=previous_error
    ).request
    async with transaction(factory) as session:  # short: it only counts and inserts a row
        reservation = await llm_budget.reserve(
            session,
            user_id=snapshot.user_id,
            job_id=snapshot.id,
            provider=llm.name,
            model=llm.model,
            prompt_version=EXTRACT_JOB_V1.version,
            settings=settings,
            now=now,
        )
    if isinstance(reservation, Refusal):
        return _deferred(reservation.reason, reservation.retry_after_seconds)
    try:
        # No session is open here. The deadline also covers a provider call that ignores its own.
        async with asyncio.timeout_at(deadline):
            response = await llm.complete(request)
    except LLMError as error:
        return await _after_error(factory, snapshot, cooldown, reservation, error)
    except TimeoutError:
        return await _fail_softly(
            factory,
            cooldown,
            reservation,
            (Outcome.TIMEOUT, DeferralReason.TIMEOUT),
            TRANSIENT_SECONDS,
        )
    return await _after_response(factory, reservation, response)


async def _after_response(
    factory: SessionFactory, reservation: Reservation, response: LLMResponse
) -> _Answer | _Invalid:
    try:
        extraction = parse_extraction(response.text)
    except ExtractionFormatError as error:
        await _finish_row(factory, reservation, Outcome.INVALID_RESPONSE, response.usage)
        return _Invalid(error.summary)
    return _Answer(extraction, reservation, response.usage)


async def _after_error(
    factory: SessionFactory,
    snapshot: JobSnapshot,
    cooldown: Cooldown,
    reservation: Reservation,
    error: LLMError,
) -> ExtractionResult | _Invalid:
    if error.never_sent:  # our own guard said no: the provider never saw it, so it never counted
        async with transaction(factory) as session:
            await llm_budget.discard(session, reservation)
        logger.error("A model request was refused locally for job %s", snapshot.id)
        return await _skip(factory, snapshot, ExtractionError.REFUSED_LOCALLY)
    if isinstance(error, LLMRateLimited):
        return await _fail_softly(
            factory,
            cooldown,
            reservation,
            (Outcome.RATE_LIMITED, DeferralReason.RATE_LIMITED),
            rate_limit_pause(error.retry_after_seconds),
        )
    if isinstance(error, LLMBadResponse):
        await _finish_row(factory, reservation, Outcome.INVALID_RESPONSE)
        return _Invalid(error.reason)  # a fixed phrase from the adapter, e.g. "empty output"
    if isinstance(error, LLMConfigError):  # the provider refused the request itself
        return await _after_rejection(factory, snapshot, cooldown, reservation, error)
    timed_out = isinstance(error, LLMTransientError) and (
        "timeout" in error.reason or "no response" in error.reason
    )
    kind = (
        (Outcome.TIMEOUT, DeferralReason.TIMEOUT)
        if timed_out
        else (Outcome.UNAVAILABLE, DeferralReason.UNAVAILABLE)
    )
    return await _fail_softly(factory, cooldown, reservation, kind, TRANSIENT_SECONDS)


async def _after_rejection(
    factory: SessionFactory,
    snapshot: JobSnapshot,
    cooldown: Cooldown,
    reservation: Reservation,
    error: LLMConfigError,
) -> ExtractionResult:
    """Only an HTTP 400 can be about this job's content, so only it ends the job (`skipped`).
    Anything else (401 bad key, 403 no access, 404 unknown model, any other 4xx, or an error
    with no status) is the configuration's fault: the job is left alone, everyone pauses for as
    long as we allow, and an error-level log makes it visible."""
    status = error.status_code if isinstance(error, LLMRejected) else None
    await _finish_row(factory, reservation, Outcome.REJECTED)
    if status == HTTP_BAD_REQUEST:
        logger.warning(
            "The provider rejected the request for job %s as invalid (status=400)", snapshot.id
        )
        return await _skip(factory, snapshot, ExtractionError.REJECTED)
    logger.error(
        "The provider refused the request for job %s (status=%s): check the key, access and model",
        snapshot.id,
        status if status is not None else "none",
    )
    await _start_pause(cooldown, CONFIGURATION_FAILURE_SECONDS)
    return _deferred(DeferralReason.CONFIGURATION, CONFIGURATION_FAILURE_SECONDS)


async def _fail_softly(
    factory: SessionFactory,
    cooldown: Cooldown,
    reservation: Reservation,
    kind: tuple[Outcome, DeferralReason],
    pause: float,
) -> ExtractionResult:
    """A failure that says nothing about the job: count the request, pause everyone, move on."""
    outcome, reason = kind
    await _finish_row(factory, reservation, outcome)
    await _start_pause(cooldown, pause)
    return _deferred(reason, pause)


async def _store(
    factory: SessionFactory,
    snapshot: JobSnapshot,
    answer: _Answer,
    llm: LLMClient,
    now: datetime | None,
) -> ExtractionResult:
    merged = merge(extract_with_rules(snapshot.raw_text), answer.extraction, snapshot.raw_text)
    status, error = model_result_state(merged)
    async with transaction(factory) as session:  # the usage row and the result, together
        await llm_budget.finalize(session, answer.reservation, Outcome.OK, answer.usage)
        won = await repository.write_model_result(
            session,
            snapshot,
            merged,
            status=status,
            error=error,
            model=llm.model,
            prompt_version=EXTRACT_JOB_V1.version,
            now=now or datetime.now(UTC),
        )
    if merged.dropped_stack:  # field name and count only, never the dropped text
        logger.info(
            "Job %s: grounding dropped field=stack count=%d", snapshot.id, merged.dropped_stack
        )
    if not won:
        logger.info("Job %s: another worker wrote first, leaving its result", snapshot.id)
        return ExtractionResult(ExtractionOutcome.LOST_RACE)
    return ExtractionResult(_outcome_for(status))


def _outcome_for(status: ExtractionStatus) -> ExtractionOutcome:
    if status is ExtractionStatus.NOT_A_JOB:
        return ExtractionOutcome.NOT_A_JOB
    if status is ExtractionStatus.EXTRACTION_FAILED:
        return ExtractionOutcome.FAILED
    return ExtractionOutcome.ENRICHED


async def _skip(
    factory: SessionFactory, snapshot: JobSnapshot, error: ExtractionError
) -> ExtractionResult:
    async with transaction(factory) as session:
        won = await repository.write_skipped(session, snapshot, error)
    return ExtractionResult(ExtractionOutcome.SKIPPED if won else ExtractionOutcome.LOST_RACE)


async def _finish_row(
    factory: SessionFactory,
    reservation: Reservation,
    outcome: Outcome,
    usage: LLMUsage | None = None,
) -> None:
    async with transaction(factory) as session:
        await llm_budget.finalize(session, reservation, outcome, usage)


async def _pause_in_effect(cooldown: Cooldown) -> ExtractionResult | None:
    """A DEFERRED result if no request may be made now, else None. Fails closed: if the pause
    cannot be read (Redis is down) nothing is sent."""
    try:
        remaining = await cooldown.remaining()
    except CooldownUnavailable:
        logger.warning("The cooldown store is unavailable; not asking the model now")
        return _deferred(DeferralReason.COOLDOWN_UNAVAILABLE, UNAVAILABLE_RETRY_SECONDS)
    if remaining is None:
        return None
    return _deferred(DeferralReason.COOLDOWN, remaining)


async def _start_pause(cooldown: Cooldown, seconds: float) -> None:
    try:
        await cooldown.start(seconds)
    except CooldownUnavailable:
        logger.warning("The cooldown store is unavailable; the pause could not be started")


def _deferred(reason: DeferralReason, retry_after_seconds: float) -> ExtractionResult:
    return ExtractionResult(ExtractionOutcome.DEFERRED, reason, max(retry_after_seconds, 1.0))
