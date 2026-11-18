"""The scoring tasks: publishing and coalescing, the run lock, the hooks and the sweep.

Synchronous on purpose, like the other task tests: a task runs its own event loop. No broker and
no Redis: publishing is replaced by a recorder, locks and the dedup key by in-memory fakes.
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models import ExtractionStatus, JobMatch, MatchStatus
from app.services import match_status_service as status
from app.services.match_runner import RunOutcome, RunResult
from app.tasks import embedding, jobs, matching
from app.tasks.celery_app import build_config
from app.tasks.matching import (
    FOLLOW_UP_DELAY_SECONDS,
    NO_SCORING_AFTER,
    enqueue_match_scoring,
    request_scoring_after_cv_embedding,
    request_scoring_after_extraction,
    request_scoring_after_job_embedding,
    request_scoring_for_job,
    request_scoring_for_user,
    run_score_user,
    run_sweep_matches,
)
from tests.extraction_helpers import RICH
from tests.fakes import FakeSourceLocks, ScoringRequests
from tests.integration.test_match_service import PYTHON_POST, add_cv, add_job, two_jobs
from tests.settings_helpers import settings_with
from tests.task_helpers import Seeder


@pytest.fixture
def fake_settings(settings: Settings) -> Settings:
    """Fake embeddings, and a floor and ceiling that map the fake cosines straight onto 0..1."""
    return settings.model_copy(
        update={
            "embedding_provider": "fake",
            "match_similarity_floor": 0.0,
            "match_similarity_ceiling": 1.0,
        }
    )


class Published:
    """Stands in for `enqueue_match_scoring`: records (user, delay) and can fail."""

    def __init__(self, fail_with: Exception | None = None) -> None:
        self.calls: list[tuple[uuid.UUID, float]] = []
        self.fail_with = fail_with

    def __call__(self, user_id: uuid.UUID, delay: float = 0.0) -> None:
        self.calls.append((user_id, delay))
        if self.fail_with:
            raise self.fail_with


class UserOnly:
    """The one-argument form `_publish` uses (the dedup lives inside the real function)."""

    def __init__(self, result: bool = True, fail_with: Exception | None = None) -> None:
        self.calls: list[uuid.UUID] = []
        self.result = result
        self.fail_with = fail_with

    def __call__(self, user_id: uuid.UUID) -> bool:
        self.calls.append(user_id)
        if self.fail_with:
            raise self.fail_with
        return self.result


class FakeDedup:
    def __init__(self) -> None:
        self.claims: list[tuple[uuid.UUID, int]] = []
        self._taken: set[uuid.UUID] = set()

    def claim(self, user_id: uuid.UUID, ttl_seconds: int) -> bool:
        self.claims.append((user_id, ttl_seconds))
        if user_id in self._taken:
            return False
        self._taken.add(user_id)
        return True


def scorable_user(seed: Seeder) -> tuple[uuid.UUID, list[uuid.UUID]]:
    """A user with a CV (three skills, a vector) and two extracted jobs; nothing requested yet."""
    user_id = seed.user()

    async def work(session: AsyncSession) -> list[uuid.UUID]:
        await add_cv(session, user_id, skills=["Python", "Django", "PostgreSQL"])
        jobs_ = await two_jobs(session, user_id)
        await session.commit()
        return [job.id for job in jobs_]

    return user_id, seed.run(work)


def status_row(seed: Seeder, user_id: uuid.UUID) -> MatchStatus | None:
    async def work(session: AsyncSession) -> MatchStatus | None:
        return await session.get(MatchStatus, user_id)

    row: MatchStatus | None = seed.run(work)
    return row


def request(seed: Seeder, user_id: uuid.UUID) -> None:
    async def work(session: AsyncSession) -> None:
        await status.request_scoring(session, user_id)
        await session.commit()

    seed.run(work)


def match_count(seed: Seeder, user_id: uuid.UUID) -> int:
    async def work(session: AsyncSession) -> int:
        rows = await session.scalars(select(JobMatch).where(JobMatch.user_id == user_id))
        return len(rows.all())

    count: int = seed.run(work)
    return count


# --- publishing and coalescing -----------------------------------------------------------------


def test_a_burst_of_triggers_publishes_one_task(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[list[str], float]] = []
    monkeypatch.setattr(
        matching.score_user_matches,
        "apply_async",
        lambda args, countdown: sent.append((args, countdown)),
    )
    dedup, user_id = FakeDedup(), uuid.uuid4()

    results = [enqueue_match_scoring(user_id, dedup=dedup) for _ in range(100)]

    assert results.count(True) == 1 and results.count(False) == 99
    assert sent == [([str(user_id)], matching.SCORE_DELAY_SECONDS)]  # only the user id travels
    assert {claim[1] for claim in dedup.claims} == {int(matching.SCORE_DELAY_SECONDS) + 5}


def test_different_users_are_coalesced_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[list[str]] = []
    monkeypatch.setattr(
        matching.score_user_matches, "apply_async", lambda args, countdown: sent.append(args)
    )
    dedup = FakeDedup()
    first, second = uuid.uuid4(), uuid.uuid4()

    assert enqueue_match_scoring(first, dedup=dedup) and enqueue_match_scoring(second, dedup=dedup)
    assert not enqueue_match_scoring(first, dedup=dedup)

    assert sent == [[str(first)], [str(second)]]


def test_force_skips_the_dedup_for_the_follow_up_and_the_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[float] = []
    monkeypatch.setattr(
        matching.score_user_matches, "apply_async", lambda args, countdown: sent.append(countdown)
    )
    dedup, user_id = FakeDedup(), uuid.uuid4()
    enqueue_match_scoring(user_id, dedup=dedup)

    assert enqueue_match_scoring(user_id, delay=12.0, force=True, dedup=dedup)

    assert sent == [matching.SCORE_DELAY_SECONDS, 12.0]
    assert len(dedup.claims) == 1  # the forced publish never asked


def test_the_scoring_task_is_on_the_main_queue_and_takes_only_a_user_id() -> None:
    task = matching.score_user_matches
    assert task.name == "score_user_matches"
    routes = build_config(settings_with())["task_routes"]  # only the model tasks have a route
    assert "score_user_matches" not in routes and "sweep_matches" not in routes
    assert task.__wrapped__.__annotations__ == {"user_id": str, "return": str}


# --- the task ----------------------------------------------------------------------------------


def test_a_run_scores_the_user_and_does_not_publish_again_when_nothing_is_open(
    seed: Seeder, fake_settings: Settings
) -> None:
    user_id, _ = scorable_user(seed)
    request(seed, user_id)
    locks, published = FakeSourceLocks(), Published()

    outcome = run_score_user(user_id, settings=fake_settings, locks=locks, enqueue=published)

    assert outcome == "completed"
    assert match_count(seed, user_id) == 2
    assert published.calls == []
    assert locks.events == [("acquire", user_id), ("release", user_id)]
    row = status_row(seed, user_id)
    assert row is not None and row.finished_seq == row.requested_seq


def test_a_run_that_finds_the_lock_held_does_nothing(seed: Seeder, fake_settings: Settings) -> None:
    user_id, _ = scorable_user(seed)
    request(seed, user_id)
    locks = FakeSourceLocks()
    locks.held.add(user_id)

    outcome = run_score_user(user_id, settings=fake_settings, locks=locks, enqueue=Published())

    assert outcome == "skipped_locked"
    assert match_count(seed, user_id) == 0
    assert locks.held == {user_id}  # the other run's lock is left alone


def test_the_lock_is_released_even_when_the_run_raises(
    seed: Seeder, fake_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = seed.user()
    locks = FakeSourceLocks()

    async def boom(settings: Settings, uid: uuid.UUID) -> RunResult:
        raise RuntimeError("database down")

    monkeypatch.setattr(matching, "_score", boom)

    with pytest.raises(RuntimeError):
        run_score_user(user_id, settings=fake_settings, locks=locks, enqueue=Published())

    assert locks.held == set()


def test_a_run_that_ends_with_a_request_still_open_queues_one_more(
    seed: Seeder, fake_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = seed.user()

    async def pending(settings: Settings, uid: uuid.UUID) -> RunResult:
        return RunResult(RunOutcome.COMPLETED, pending_after=True)

    monkeypatch.setattr(matching, "_score", pending)
    published = Published()

    run_score_user(user_id, settings=fake_settings, locks=FakeSourceLocks(), enqueue=published)

    assert published.calls == [(user_id, FOLLOW_UP_DELAY_SECONDS)]


def test_a_failed_follow_up_publish_is_only_logged(
    seed: Seeder,
    fake_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    user_id = seed.user()

    async def pending(settings: Settings, uid: uuid.UUID) -> RunResult:
        return RunResult(RunOutcome.ABORTED, pending_after=True)

    monkeypatch.setattr(matching, "_score", pending)
    caplog.set_level(logging.WARNING)

    outcome = run_score_user(
        user_id,
        settings=fake_settings,
        locks=FakeSourceLocks(),
        enqueue=Published(fail_with=OSError("no route")),
    )

    assert outcome == "aborted"
    assert any("the sweep will" in r.getMessage() for r in caplog.records)


# --- requests from background tasks ------------------------------------------------------------


def test_a_job_embedding_requests_a_rescore_for_the_jobs_owner_only(
    seed: Seeder, fake_settings: Settings
) -> None:
    alice, jobs_ = scorable_user(seed)
    bob = seed.user()
    published = UserOnly()

    assert request_scoring_for_job(jobs_[0], settings=fake_settings, enqueue=published)
    assert request_scoring_for_job(jobs_[1], settings=fake_settings, enqueue=published)

    assert published.calls == [alice, alice]
    row = status_row(seed, alice)
    assert row is not None and row.requested_seq == 2
    assert status_row(seed, bob) is None


def test_an_unknown_job_requests_nothing(seed: Seeder, fake_settings: Settings) -> None:
    published = UserOnly()

    assert not request_scoring_for_job(uuid.uuid4(), settings=fake_settings, enqueue=published)
    assert published.calls == []


def test_a_cv_embedding_requests_a_rescore_for_the_user(
    seed: Seeder, fake_settings: Settings
) -> None:
    user_id, _ = scorable_user(seed)
    published = UserOnly()

    assert request_scoring_for_user(user_id, settings=fake_settings, enqueue=published)

    row = status_row(seed, user_id)
    assert row is not None and row.requested_seq == 1 and published.calls == [user_id]


def test_a_failed_publish_after_a_request_is_only_logged_and_the_request_stays(
    seed: Seeder, fake_settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    user_id, jobs_ = scorable_user(seed)
    caplog.set_level(logging.WARNING)

    sent = request_scoring_for_job(
        jobs_[0], settings=fake_settings, enqueue=UserOnly(fail_with=OSError("down"))
    )

    assert sent is False
    assert any("the sweep will" in r.getMessage() for r in caplog.records)
    row = status_row(seed, user_id)
    assert row is not None and row.requested_seq > row.finished_seq  # stored: the sweep finds it


@pytest.mark.parametrize("outcome", ["extracted", "reused", "enriched", "already_done", "skipped"])
def test_with_embeddings_off_an_extraction_requests_a_rescore(
    outcome: str, monkeypatch: pytest.MonkeyPatch, celery_env: None
) -> None:
    requested: list[uuid.UUID] = []
    monkeypatch.setattr(
        matching, "request_scoring_for_job", lambda job_id: requested.append(job_id)
    )
    job_id = uuid.uuid4()

    request_scoring_after_extraction(job_id, outcome)

    assert requested == [job_id]


@pytest.mark.parametrize("outcome", sorted(NO_SCORING_AFTER))
def test_an_extraction_that_leaves_no_usable_job_requests_nothing(
    outcome: str, monkeypatch: pytest.MonkeyPatch, celery_env: None
) -> None:
    requested: list[uuid.UUID] = []
    monkeypatch.setattr(
        matching, "request_scoring_for_job", lambda job_id: requested.append(job_id)
    )

    assert request_scoring_after_extraction(uuid.uuid4(), outcome) is False
    assert requested == []


def test_with_embeddings_on_the_embedding_asks_not_the_extraction(
    monkeypatch: pytest.MonkeyPatch, celery_env: None
) -> None:
    requested: list[uuid.UUID] = []
    monkeypatch.setattr(
        matching, "request_scoring_for_job", lambda job_id: requested.append(job_id)
    )
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    from app.core.config import get_settings

    get_settings.cache_clear()

    assert request_scoring_after_extraction(uuid.uuid4(), "extracted") is False
    assert requested == []


def test_the_outcomes_that_skip_scoring_equal_the_ones_that_skip_embedding() -> None:
    assert NO_SCORING_AFTER == jobs.NO_EMBEDDING_AFTER  # one list, kept in two places on purpose


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("embedded", True),
        ("copied", True),
        ("unchanged", True),
        ("not_found", False),
        ("not_ready", False),
        ("lost_race", False),
        ("disabled", False),
    ],
)
def test_a_job_embedding_requests_a_rescore_only_when_it_left_a_usable_vector(
    outcome: str, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[uuid.UUID] = []
    monkeypatch.setattr(
        matching, "request_scoring_for_job", lambda job_id: requested.append(job_id) or True
    )

    request_scoring_after_job_embedding(uuid.uuid4(), outcome)

    assert bool(requested) is expected


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [("embedded", True), ("unchanged", True), ("lost_race", False), ("not_found", False)],
)
def test_a_cv_embedding_requests_a_rescore_only_when_it_left_a_usable_vector(
    outcome: str, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[uuid.UUID] = []
    monkeypatch.setattr(
        matching, "request_scoring_for_user", lambda uid: requested.append(uid) or True
    )

    request_scoring_after_cv_embedding(uuid.uuid4(), outcome)

    assert bool(requested) is expected


def test_the_embed_and_extract_tasks_pass_their_outcome_to_the_scoring_hook(
    monkeypatch: pytest.MonkeyPatch, scoring_requests: ScoringRequests
) -> None:
    job_id, user_id = uuid.uuid4(), uuid.uuid4()
    monkeypatch.setattr(embedding, "run_embed_job", lambda _id: "embedded")
    monkeypatch.setattr(embedding, "run_embed_cv", lambda _id: "unchanged")
    monkeypatch.setattr(jobs, "run_extract_job", lambda _id: "extracted")
    monkeypatch.setattr(jobs, "enqueue_embed_job", lambda _id: None)

    assert embedding.embed_job.run(str(job_id)) == "embedded"
    assert embedding.embed_cv.run(str(user_id)) == "unchanged"
    assert jobs.extract_job.run(str(job_id)) == "extracted"

    assert scoring_requests.after_job_embedding == [(job_id, "embedded")]
    assert scoring_requests.after_cv_embedding == [(user_id, "unchanged")]
    assert scoring_requests.after_extraction == [(job_id, "extracted")]


# --- the sweep ---------------------------------------------------------------------------------


def backdate(seed: Seeder, user_id: uuid.UUID, **ages: timedelta | None) -> None:
    """Move columns of the status row into the past (a value of None clears the column)."""
    now = datetime.now(UTC)
    values = {name: None if age is None else now - age for name, age in ages.items()}

    async def work(session: AsyncSession) -> None:
        await session.execute(
            update(MatchStatus).where(MatchStatus.user_id == user_id).values(**values)
        )
        await session.commit()

    seed.run(work)


def set_status(seed: Seeder, user_id: uuid.UUID, **values: object) -> None:
    async def work(session: AsyncSession) -> None:
        await session.execute(
            update(MatchStatus).where(MatchStatus.user_id == user_id).values(**values)
        )
        await session.commit()

    seed.run(work)


def waiting_user(seed: Seeder, age: timedelta = timedelta(minutes=3)) -> uuid.UUID:
    user_id, _ = scorable_user(seed)
    request(seed, user_id)
    backdate(seed, user_id, requested_at=age)
    return user_id


def test_the_sweep_queues_a_user_whose_request_has_waited_and_remembers_it(
    seed: Seeder, fake_settings: Settings
) -> None:
    user_id = waiting_user(seed)
    published = Published()

    first = run_sweep_matches(settings=fake_settings, enqueue=published)
    again = run_sweep_matches(settings=fake_settings, enqueue=published)

    assert user_id in [call[0] for call in published.calls]
    assert [call[0] for call in published.calls].count(user_id) == 1  # once per 5 minutes
    assert first["waiting"] >= 1 and again["waiting"] == 0
    row = status_row(seed, user_id)
    assert row is not None and row.queued_at is not None


def test_the_sweep_queues_the_same_user_again_after_five_minutes(
    seed: Seeder, fake_settings: Settings
) -> None:
    user_id = waiting_user(seed)
    published = Published()
    run_sweep_matches(settings=fake_settings, enqueue=published)

    backdate(seed, user_id, queued_at=timedelta(minutes=6))
    run_sweep_matches(settings=fake_settings, enqueue=published)

    assert [call[0] for call in published.calls].count(user_id) == 2


def test_the_sweep_leaves_a_fresh_request_alone_its_task_is_still_coming(
    seed: Seeder, fake_settings: Settings
) -> None:
    user_id, _ = scorable_user(seed)
    request(seed, user_id)  # just now
    published = Published()

    run_sweep_matches(settings=fake_settings, enqueue=published)

    assert user_id not in [call[0] for call in published.calls]


def test_the_sweep_stops_queueing_a_user_after_three_failures(
    seed: Seeder, fake_settings: Settings
) -> None:
    user_id = waiting_user(seed)
    set_status(seed, user_id, failure_count=status.MAX_FAILURES)
    published = Published()

    run_sweep_matches(settings=fake_settings, enqueue=published)

    assert user_id not in [call[0] for call in published.calls]


def test_the_sweep_leaves_a_user_alone_while_a_run_is_in_progress_but_not_after_it_died(
    seed: Seeder, fake_settings: Settings
) -> None:
    user_id = waiting_user(seed)
    backdate(seed, user_id, running_since=timedelta(seconds=30))
    published = Published()

    run_sweep_matches(settings=fake_settings, enqueue=published)
    assert user_id not in [call[0] for call in published.calls]

    backdate(seed, user_id, running_since=status.RUNNING_STALE + timedelta(minutes=1))
    run_sweep_matches(settings=fake_settings, enqueue=published)
    assert user_id in [call[0] for call in published.calls]


def test_a_changed_scorer_fingerprint_requests_a_rescore_for_up_to_date_users(
    seed: Seeder, fake_settings: Settings
) -> None:
    user_id, _ = scorable_user(seed)
    request(seed, user_id)
    run_score_user(user_id, settings=fake_settings, locks=FakeSourceLocks(), enqueue=Published())
    published = Published()
    quiet = run_sweep_matches(settings=fake_settings, enqueue=published)
    assert user_id not in [call[0] for call in published.calls]  # same fingerprint: untouched
    assert quiet["refreshed"] == 0 or user_id not in [c[0] for c in published.calls]

    changed = fake_settings.model_copy(update={"match_similarity_ceiling": 0.9})
    result = run_sweep_matches(settings=changed, enqueue=published)

    assert user_id in [call[0] for call in published.calls]
    assert result["refreshed"] >= 1
    row = status_row(seed, user_id)
    assert row is not None and row.requested_seq > row.finished_seq and row.queued_at is not None
    # The user is pending now, so the next pass does not request them again.
    again = Published()
    run_sweep_matches(settings=changed, enqueue=again)
    assert user_id not in [call[0] for call in again.calls]


def test_users_are_refreshed_in_small_batches_and_spread_over_a_window(
    seed: Seeder, fake_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import match_sweep

    monkeypatch.setattr(match_sweep, "REFRESH_BATCH", 2)
    users = []
    for _ in range(3):
        user_id, _ = scorable_user(seed)
        request(seed, user_id)
        run_score_user(
            user_id, settings=fake_settings, locks=FakeSourceLocks(), enqueue=Published()
        )
        users.append(user_id)
    changed = fake_settings.model_copy(update={"match_failed_filter_cap": 20})
    first = Published()

    run_sweep_matches(settings=changed, enqueue=first)

    refreshed = [call for call in first.calls if call[0] in users]
    assert len(refreshed) == 2  # only a batch, not everybody at once
    delays = [call[1] for call in first.calls]
    assert delays == sorted(delays) and delays[0] == 0.0
    assert max(delays) < match_sweep.SPREAD_WINDOW_SECONDS  # spread inside the window
    assert len(set(delays)) == len(delays)  # each user gets a different delay
    second = Published()
    run_sweep_matches(settings=changed, enqueue=second)
    assert len([call for call in second.calls if call[0] in users]) == 1  # the third, next pass


def test_a_user_with_a_cv_and_no_status_row_gets_one(seed: Seeder, fake_settings: Settings) -> None:
    user_id, _ = scorable_user(seed)
    assert status_row(seed, user_id) is None
    published = Published()

    run_sweep_matches(settings=fake_settings, enqueue=published)

    assert user_id in [call[0] for call in published.calls]
    row = status_row(seed, user_id)
    assert row is not None and row.requested_seq == 1 and row.queued_at is not None


def test_a_user_without_a_cv_is_not_refreshed(seed: Seeder, fake_settings: Settings) -> None:
    user_id = seed.user()
    set_row: list[int] = []

    async def work(session: AsyncSession) -> None:
        await status.request_scoring(session, user_id)
        await session.commit()
        set_row.append(1)

    seed.run(work)
    set_status(seed, user_id, requested_seq=1, finished_seq=1, fingerprint="old")
    published = Published()

    run_sweep_matches(settings=fake_settings, enqueue=published)

    assert user_id not in [call[0] for call in published.calls]


def test_a_queue_that_is_down_stops_the_sweep_with_one_warning_and_no_retry(
    seed: Seeder, fake_settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    waiting_user(seed)
    waiting_user(seed)
    published = Published(fail_with=OSError("no route"))
    caplog.set_level(logging.WARNING)

    result = run_sweep_matches(settings=fake_settings, enqueue=published)

    assert result["queued"] == 0
    assert len(published.calls) == 1  # it stopped at the first failure
    warnings = [r for r in caplog.records if "were not queued" in r.getMessage()]
    assert len(warnings) == 1


def test_the_spread_delays_fill_the_window_evenly() -> None:
    from app.services.match_sweep import spread_delays

    assert spread_delays(0) == []
    assert spread_delays(1) == [0.0]
    assert spread_delays(4, window=60.0) == [0.0, 15.0, 30.0, 45.0]


# --- the sweep recovers a request that was never recorded ---------------------------------------


def finished_user(seed: Seeder, settings: Settings) -> uuid.UUID:
    """A user whose matches were scored and are up to date."""
    user_id, _ = scorable_user(seed)
    request(seed, user_id)
    run_score_user(user_id, settings=settings, locks=FakeSourceLocks(), enqueue=Published())
    return user_id


def embed_job_later(seed: Seeder, user_id: uuid.UUID, *, minutes: int = 1) -> uuid.UUID:
    """A job of the user whose vector was stored after their last run finished (`minutes` ahead
    of now, so the test needs no sleeping), with no scoring request recorded for it."""

    async def work(session: AsyncSession) -> uuid.UUID:
        job = await add_job(session, user_id, body=PYTHON_POST + " later", stack=["Python"])
        job.embedded_at = datetime.now(UTC) + timedelta(minutes=minutes)
        await session.commit()
        return job.id

    job_id: uuid.UUID = seed.run(work)
    return job_id


def test_a_lost_embed_job_request_is_recovered_by_the_sweep(
    seed: Seeder, fake_settings: Settings
) -> None:
    user_id = finished_user(seed, fake_settings)
    embed_job_later(seed, user_id)  # the embed_job task died before it could record a request
    before = status_row(seed, user_id)
    assert before is not None and before.requested_seq == before.finished_seq
    published = Published()

    result = run_sweep_matches(settings=fake_settings, enqueue=published)

    assert user_id in [call[0] for call in published.calls]
    assert result["recovered"] >= 1
    after = status_row(seed, user_id)
    assert after is not None and after.requested_seq == before.requested_seq + 1
    assert after.queued_at is not None
    # The run it queues scores the new job, and then the sweep has nothing left to recover.
    run_score_user(user_id, settings=fake_settings, locks=FakeSourceLocks(), enqueue=Published())
    assert match_count(seed, user_id) == 3
    again = Published()
    run_sweep_matches(settings=fake_settings, enqueue=again)
    assert user_id not in [call[0] for call in again.calls]


def test_the_recovery_ignores_jobs_embedded_before_the_last_run(
    seed: Seeder, fake_settings: Settings
) -> None:
    user_id = finished_user(seed, fake_settings)  # its jobs were embedded long before the run
    published = Published()

    run_sweep_matches(settings=fake_settings, enqueue=published)

    assert user_id not in [call[0] for call in published.calls]


def test_the_recovery_queues_a_user_at_most_once_per_five_minutes(
    seed: Seeder, fake_settings: Settings
) -> None:
    user_id = finished_user(seed, fake_settings)
    embed_job_later(seed, user_id)
    backdate(seed, user_id, queued_at=timedelta(minutes=1))
    published = Published()

    run_sweep_matches(settings=fake_settings, enqueue=published)
    assert user_id not in [call[0] for call in published.calls]  # queued a minute ago

    backdate(seed, user_id, queued_at=timedelta(minutes=6))
    run_sweep_matches(settings=fake_settings, enqueue=published)
    assert user_id in [call[0] for call in published.calls]


def test_the_recovery_respects_failures_and_runs_in_progress(
    seed: Seeder, fake_settings: Settings
) -> None:
    failing, busy = finished_user(seed, fake_settings), finished_user(seed, fake_settings)
    for user_id in (failing, busy):
        embed_job_later(seed, user_id)
    set_status(seed, failing, failure_count=status.MAX_FAILURES)
    backdate(seed, busy, running_since=timedelta(seconds=30))
    published = Published()

    run_sweep_matches(settings=fake_settings, enqueue=published)

    assert not {failing, busy} & {call[0] for call in published.calls}


def test_the_recovery_is_capped_per_pass(
    seed: Seeder, fake_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import match_sweep

    monkeypatch.setattr(match_sweep, "RECOVER_BATCH", 1)
    users = [finished_user(seed, fake_settings) for _ in range(2)]
    for user_id in users:
        embed_job_later(seed, user_id)

    first = Published()
    run_sweep_matches(settings=fake_settings, enqueue=first)
    second = Published()
    run_sweep_matches(settings=fake_settings, enqueue=second)

    assert len([c for c in first.calls if c[0] in users]) == 1
    assert len([c for c in second.calls if c[0] in users]) == 1  # the other one, next pass
    assert {c[0] for c in first.calls + second.calls} >= set(users)


# --- the real triggers, against the test database, without the recording fixture ---------------


@pytest.fixture
def real_hooks(
    monkeypatch: pytest.MonkeyPatch, celery_env: None, test_database_url: str
) -> list[uuid.UUID]:
    """Put the real re-score hooks back (the autouse fixture replaces them), point the app at the
    test database and the fake embedder, and record only the publish (there is no broker)."""
    from app.core.config import get_settings

    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    get_settings.cache_clear()
    monkeypatch.setattr(
        jobs, "request_scoring_after_extraction", matching.request_scoring_after_extraction
    )
    monkeypatch.setattr(
        embedding,
        "request_scoring_after_job_embedding",
        matching.request_scoring_after_job_embedding,
    )
    monkeypatch.setattr(
        embedding,
        "request_scoring_after_cv_embedding",
        matching.request_scoring_after_cv_embedding,
    )
    published: list[uuid.UUID] = []
    monkeypatch.setattr(
        matching, "enqueue_match_scoring", lambda user_id: published.append(user_id) or True
    )
    monkeypatch.setattr(jobs, "enqueue_embed_job", lambda job_id: None)
    monkeypatch.setattr(jobs, "redis_job_locks_for", lambda url: FakeSourceLocks())
    return published


EXTRACTED_JOB = {
    "extraction_status": ExtractionStatus.EXTRACTED,
    "title": "Senior Python Developer",
    "stack": ["Python"],
    "extracted_at": datetime(2026, 11, 3, 12, 0, tzinfo=UTC),
    "extraction_attempts": 1,
}


def test_the_embed_job_task_records_a_request_and_publishes_for_the_owner(
    seed: Seeder, real_hooks: list[uuid.UUID]
) -> None:
    user_id = seed.user()
    job_id = seed.job(seed.source(user_id), RICH, **EXTRACTED_JOB)

    assert embedding.embed_job.run(str(job_id)) == "embedded"

    row = status_row(seed, user_id)
    assert row is not None and row.requested_seq == 1
    assert real_hooks == [user_id]


def test_the_embed_cv_task_records_a_request_and_publishes_for_the_user(
    seed: Seeder, real_hooks: list[uuid.UUID]
) -> None:
    user_id = seed.user()
    seed.cv(user_id, "Python developer with Django experience.")

    assert embedding.embed_cv.run(str(user_id)) == "embedded"

    row = status_row(seed, user_id)
    assert row is not None and row.requested_seq == 1
    assert real_hooks == [user_id]


def test_the_extract_task_with_embeddings_off_records_a_request_for_the_owner(
    seed: Seeder, real_hooks: list[uuid.UUID], monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.core.config import get_settings

    monkeypatch.setenv("EMBEDDING_PROVIDER", "none")
    get_settings.cache_clear()
    user_id = seed.user()
    job_id = seed.job(seed.source(user_id), RICH)

    assert jobs.extract_job.run(str(job_id)) == "extracted"  # the rules run for real

    row = status_row(seed, user_id)
    assert row is not None and row.requested_seq == 1
    assert real_hooks == [user_id]


def test_the_extract_task_with_embeddings_on_leaves_the_request_to_the_embedding(
    seed: Seeder, real_hooks: list[uuid.UUID]
) -> None:
    user_id = seed.user()
    job_id = seed.job(seed.source(user_id), RICH)

    assert jobs.extract_job.run(str(job_id)) == "extracted"

    assert status_row(seed, user_id) is None  # embed_job asks, when the vector is stored
    assert real_hooks == []
