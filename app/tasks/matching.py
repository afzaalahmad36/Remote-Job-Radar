"""Celery tasks that score a user's jobs. They run on the main worker (the `celery` queue).

Scoring needs no model: it reads vectors that the embed worker stored earlier, so this module and
everything it imports stay free of fastembed and onnxruntime (a test imports it in a clean
interpreter and checks). Tasks receive only a user id. The CV text, skills and preference notes
never travel through Celery or Redis: the task reads what it needs from the database, in
`services/match_service.py`, which selects skills and vectors only.

How the pieces fit (docs/MATCHING.md, "Keeping matches fresh"):

* A change that makes matches stale calls `match_status_service.request_scoring` in its own
  transaction. That row is the truth: it survives a lost message and a Redis outage.
* `enqueue_match_scoring` then publishes a task, after a short delay, and drops the publish when
  one was sent a few seconds ago: a burst of changes becomes one task. The delay also lets the
  change commit first. The task reads the latest data whenever it runs, so it covers them all.
* The run lock and the dedup key only save work. Correctness is the status row and the checks in
  the write step.
* `sweep_matches` queues users whose request has been waiting without a task (see
  `services/match_sweep.py`).
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from typing import Protocol

from redis import Redis

from app.core.config import Settings, get_settings
from app.services import match_runner, match_status_service, match_sweep
from app.tasks.celery_app import (
    HARD_TIME_LIMIT_SECONDS,
    REDIS_CONNECT_TIMEOUT_SECONDS,
    REDIS_SOCKET_TIMEOUT_SECONDS,
    SOFT_TIME_LIMIT_SECONDS,
    celery_app,
)
from app.tasks.common import ENQUEUE_ERRORS, RETRYABLE_ERRORS, session_factory
from app.tasks.locking import SourceLocks, redis_match_locks_for

logger = logging.getLogger(__name__)

SCORE_DELAY_SECONDS = 5.0  # lets the change commit, and gathers a burst into one run
FOLLOW_UP_DELAY_SECONDS = 5.0  # a run that ended with a request still open queues the next one
DEDUP_KEY_PREFIX = "match:queued:"
SWEEP_SOFT_TIME_LIMIT_SECONDS = 120
SWEEP_HARD_TIME_LIMIT_SECONDS = 150

# What leaves no usable job behind: nothing to score. The same outcomes as
# `tasks.jobs.NO_EMBEDDING_AFTER` (a test checks they stay equal); this module cannot import that
# one, because the extraction tasks reach the LLM client.
NO_SCORING_AFTER = frozenset({"not_found", "failed", "not_a_job", "lost_race", "skipped_locked"})
SCORE_AFTER_JOB_EMBEDDING = frozenset({"embedded", "copied", "unchanged"})
SCORE_AFTER_CV_EMBEDDING = frozenset({"embedded", "unchanged"})

Enqueue = Callable[[uuid.UUID, float], None]


# --- publishing -------------------------------------------------------------------------------


class PublishDedup(Protocol):
    """Remembers for a few seconds that a task was just published for a user."""

    def claim(self, user_id: uuid.UUID, ttl_seconds: int) -> bool:
        """True if this call is the first within the window (the caller should publish)."""
        ...


class RedisPublishDedup:
    """`SET key 1 NX EX ttl`: succeeds for the first caller only. Holds no data but a flag."""

    def __init__(self, client: Redis) -> None:
        self._client = client

    @classmethod
    def from_url(cls, url: str) -> "RedisPublishDedup":
        return cls(
            Redis.from_url(
                url,
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT_SECONDS,
                socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
            )
        )

    def claim(self, user_id: uuid.UUID, ttl_seconds: int) -> bool:
        return bool(self._client.set(f"{DEDUP_KEY_PREFIX}{user_id}", "1", nx=True, ex=ttl_seconds))


_dedup_by_url: dict[str, PublishDedup] = {}


def redis_dedup_for(redis_url: str) -> PublishDedup:
    """One dedup client per process (and so one connection pool)."""
    if redis_url not in _dedup_by_url:
        _dedup_by_url[redis_url] = RedisPublishDedup.from_url(redis_url)
    return _dedup_by_url[redis_url]


def enqueue_match_scoring(
    user_id: uuid.UUID,
    *,
    delay: float = SCORE_DELAY_SECONDS,
    force: bool = False,
    dedup: PublishDedup | None = None,
) -> bool:
    """Publish a scoring task for the user, unless one was published in the last few seconds.

    True if a task was published. `force` skips the dedup (the sweep and the follow-up run have
    their own caps). May raise ENQUEUE_ERRORS (the queue or Redis is unreachable).
    """
    if not force:
        dedup = dedup or redis_dedup_for(get_settings().redis_url)
        if not dedup.claim(user_id, int(delay) + 5):
            return False
    score_user_matches.apply_async(args=[str(user_id)], countdown=delay)
    return True


def _publish(user_id: uuid.UUID, enqueue: Callable[[uuid.UUID], bool] | None) -> bool:
    """Publish, and only log a failure: the request is stored, the sweep will queue it."""
    try:
        return (enqueue or enqueue_match_scoring)(user_id)
    except ENQUEUE_ERRORS:
        logger.warning("Scoring for user %s was not queued; the sweep will", user_id)
        return False


# --- requesting a re-score from a background task ------------------------------------------------


async def _request_for_job(settings: Settings, job_id: uuid.UUID) -> uuid.UUID | None:
    async with session_factory(settings) as factory, factory() as session:
        owner = await match_status_service.request_scoring_for_job(session, job_id)
        await session.commit()
        return owner


async def _request_for_user(settings: Settings, user_id: uuid.UUID) -> None:
    async with session_factory(settings) as factory, factory() as session:
        await match_status_service.request_scoring(session, user_id)
        await session.commit()


def request_scoring_for_job(
    job_id: uuid.UUID,
    *,
    settings: Settings | None = None,
    enqueue: Callable[[uuid.UUID], bool] | None = None,
) -> bool:
    """Record a re-score request for the job's owner, then publish. True if a task was published.

    A database error is not caught: the calling task is retried (all of them are idempotent). A
    publish error only logs.
    """
    owner = asyncio.run(_request_for_job(settings or get_settings(), job_id))
    return owner is not None and _publish(owner, enqueue)


def request_scoring_for_user(
    user_id: uuid.UUID,
    *,
    settings: Settings | None = None,
    enqueue: Callable[[uuid.UUID], bool] | None = None,
) -> bool:
    """The same for a user directly (their CV was embedded)."""
    asyncio.run(_request_for_user(settings or get_settings(), user_id))
    return _publish(user_id, enqueue)


def request_scoring_after_extraction(job_id: uuid.UUID, outcome: str) -> bool:
    """After an extraction run, with embeddings OFF: the job is usable now, so re-score.

    With embeddings on, the embedding task requests the re-score when it finishes (it changes what
    the score reads), so this does nothing then.
    """
    if outcome in NO_SCORING_AFTER or get_settings().embedding_provider != "none":
        return False
    return request_scoring_for_job(job_id)


def request_scoring_after_job_embedding(job_id: uuid.UUID, outcome: str) -> bool:
    """After an embed_job run that left a usable vector (or confirmed the stored one)."""
    return outcome in SCORE_AFTER_JOB_EMBEDDING and request_scoring_for_job(job_id)


def request_scoring_after_cv_embedding(user_id: uuid.UUID, outcome: str) -> bool:
    """After an embed_cv run that left a usable vector."""
    return outcome in SCORE_AFTER_CV_EMBEDDING and request_scoring_for_user(user_id)


# --- score one user ----------------------------------------------------------------------------


def run_score_user(
    user_id: uuid.UUID,
    *,
    settings: Settings | None = None,
    locks: SourceLocks | None = None,
    enqueue: Enqueue | None = None,
) -> str:
    """Score one user's jobs unless a run of theirs holds the lock. Returns the outcome as text.

    Only a database or Redis outage raises (Celery retries the task). A run that ends with a
    request still open (a change arrived meanwhile, or the inputs changed while it ran) queues
    one more run itself, so the user does not wait for the next sweep.
    """
    settings = settings or get_settings()
    locks = locks or redis_match_locks_for(settings.redis_url)
    if not locks.acquire(user_id):
        logger.info("Skipping scoring for user %s: another run holds the lock", user_id)
        return "skipped_locked"
    try:
        result = asyncio.run(_score(settings, user_id))
    finally:
        locks.release(user_id)
    if result.pending_after:
        _follow_up(user_id, enqueue)
    return result.outcome.value


async def _score(settings: Settings, user_id: uuid.UUID) -> match_runner.RunResult:
    async with session_factory(settings) as factory:
        return await match_runner.run_user_scoring(factory, user_id, settings)


def _publish_forced(user_id: uuid.UUID, delay: float) -> None:
    """Publish without the dedup: the callers (the follow-up and the sweep) have their own caps."""
    enqueue_match_scoring(user_id, delay=delay, force=True)


def _follow_up(user_id: uuid.UUID, enqueue: Enqueue | None) -> None:
    publish = enqueue or _publish_forced
    try:
        publish(user_id, FOLLOW_UP_DELAY_SECONDS)
    except ENQUEUE_ERRORS:
        logger.warning("The next scoring run of user %s was not queued; the sweep will", user_id)


@celery_app.task(
    name="score_user_matches",
    soft_time_limit=SOFT_TIME_LIMIT_SECONDS,
    time_limit=HARD_TIME_LIMIT_SECONDS,
    autoretry_for=RETRYABLE_ERRORS,
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=3,
)
def score_user_matches(user_id: str) -> str:
    """Celery task: score one user's jobs. The only argument is the user id."""
    return run_score_user(uuid.UUID(user_id))


# --- the sweep ---------------------------------------------------------------------------------


def run_sweep_matches(
    *, settings: Settings | None = None, enqueue: Enqueue | None = None
) -> dict[str, int]:
    """One pass of the safety net; see `services/match_sweep.py` for the steps.

    The database steps always run. Publishing stops at the first failure with one warning (the
    queue is down: the next pass, five minutes later, is the retry; there is no loop here).
    """
    settings = settings or get_settings()
    work = asyncio.run(_collect(settings))
    publish = enqueue or _publish_forced
    plan = [(user_id, 0.0) for user_id in [*work.waiting, *work.recovered]]
    plan += list(zip(work.refreshed, match_sweep.spread_delays(len(work.refreshed)), strict=True))
    queued = 0
    try:
        for user_id, delay in plan:
            publish(user_id, delay)
            queued += 1
    except ENQUEUE_ERRORS:
        logger.warning(
            "The queue is unavailable: %d of %d scoring tasks were not queued; "
            "the next sweep will pick them up",
            len(plan) - queued,
            len(plan),
        )
    return {
        "waiting": len(work.waiting),
        "refreshed": len(work.refreshed),
        "recovered": len(work.recovered),
        "queued": queued,
    }


async def _collect(settings: Settings) -> match_sweep.SweepWork:
    async with session_factory(settings) as factory:
        return await match_sweep.collect_work(factory, settings)


@celery_app.task(
    name="sweep_matches",
    soft_time_limit=SWEEP_SOFT_TIME_LIMIT_SECONDS,
    time_limit=SWEEP_HARD_TIME_LIMIT_SECONDS,
)
def sweep_matches() -> dict[str, int]:
    """Celery task, run by beat every five minutes: the safety net for scoring."""
    return run_sweep_matches()
