"""The safety net for extraction: what the periodic sweep looks at and changes in the database.

The sweep never talks to the queue itself. It decides which jobs need a task (this module, which
only touches the database and the shared cooldown) and the Celery task queues them, so a broker
outage can never leave this part half done.

Steps, in order:

1. relabel `reserved` usage rows nobody finished as `abandoned` (they keep counting);
2. give up on jobs that have waited for the model longer than LLM_ENRICHMENT_GIVE_UP_DAYS;
3. pick jobs whose rules phase has not run (new jobs whose task was lost, and retried jobs);
4. pick jobs waiting for the model, but only when asking could succeed: a client exists, no pause
   is in effect, and the global budget has room.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.core.config import Settings
from app.repositories import extraction_repository as repository
from app.services import llm_budget
from app.services.cooldown import Cooldown, CooldownUnavailable
from app.services.extraction_types import SessionFactory, transaction

logger = logging.getLogger(__name__)

SWEEP_BATCH = 100  # jobs queued per step per sweep


@dataclass(frozen=True)
class SweepWork:
    abandoned: int
    gave_up: int
    rules_job_ids: list[uuid.UUID]  # step 3
    enrichment_job_ids: list[uuid.UUID]  # step 4


async def collect_work(
    factory: SessionFactory,
    *,
    settings: Settings,
    cooldown: Cooldown | None,
    now: datetime | None = None,
) -> SweepWork:
    """Do the database steps of the sweep and say which jobs need a task.

    `cooldown` is None when no model client exists (provider `none` or the kill switch off): then
    no waiting job is picked, because a task could only defer it again.
    """
    now = now or datetime.now(UTC)
    async with transaction(factory) as session:
        abandoned = await llm_budget.abandon_stale_reservations(session, now)
    async with transaction(factory) as session:
        gave_up = await repository.give_up_waiting(
            session, older_than=now - timedelta(days=settings.llm_enrichment_give_up_days)
        )
    async with transaction(factory) as session:
        rules_ids = await repository.job_ids_without_rules(session, SWEEP_BATCH)
    enrichment_ids = await _waiting_that_could_be_asked(factory, settings, cooldown, now)
    logger.info(
        "Extraction sweep: abandoned=%d gave_up=%d rules_due=%d enrichment_due=%d",
        abandoned,
        gave_up,
        len(rules_ids),
        len(enrichment_ids),
    )
    return SweepWork(abandoned, gave_up, rules_ids, enrichment_ids)


async def _waiting_that_could_be_asked(
    factory: SessionFactory, settings: Settings, cooldown: Cooldown | None, now: datetime
) -> list[uuid.UUID]:
    if cooldown is None:
        return []
    try:
        remaining = await cooldown.remaining()
    except CooldownUnavailable:
        # Once, and no retry here: the next scheduled sweep looks again.
        logger.warning("The cooldown store is unavailable; not queueing model work this sweep")
        return []
    if remaining is not None:
        return []
    async with transaction(factory) as session:
        places = await llm_budget.places_left(session, settings, now)
        if places <= 0:
            return []
        return await repository.waiting_job_ids(
            session,
            limit=min(SWEEP_BATCH, places),
            user_cap=settings.llm_daily_request_cap_per_user,
            since=now - llm_budget.DAY,
        )
