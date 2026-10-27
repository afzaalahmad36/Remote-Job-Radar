import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from celery.beat import Scheduler
from celery.schedules import crontab, schedule
from pydantic import ValidationError

from app.core.config import Settings
from app.tasks.celery_app import (
    HARD_TIME_LIMIT_SECONDS,
    SOFT_TIME_LIMIT_SECONDS,
    build_config,
    celery_app,
)

ROOT = Path(__file__).resolve().parent.parent.parent
EXPECTED_TASKS = {"ingest_source", "ingest_all_active_sources", "purge_expired_refresh_tokens"}


def settings_with(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "postgresql+asyncpg://u:p@db:5432/x",
        "redis_url": "redis://redis:6379/0",
        "secret_key": "s3cret",
    }
    return Settings(_env_file=None, **(values | overrides))  # type: ignore[arg-type]


# --- settings ------------------------------------------------------------------------------------


def test_the_ingest_interval_defaults_to_30_minutes() -> None:
    assert settings_with().ingest_interval_minutes == 30


@pytest.mark.parametrize("minutes", [5, 6, 30, 1440])
def test_intervals_of_five_minutes_or_more_are_accepted(minutes: int) -> None:
    assert settings_with(ingest_interval_minutes=minutes).ingest_interval_minutes == minutes


@pytest.mark.parametrize("minutes", [4, 1, 0, -5])
def test_an_interval_below_the_five_minute_minimum_refuses_to_start(minutes: int) -> None:
    with pytest.raises(ValidationError, match="ingest_interval_minutes"):
        settings_with(ingest_interval_minutes=minutes)


def test_the_interval_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/x")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
    monkeypatch.setenv("SECRET_KEY", "s3cret")
    monkeypatch.setenv("INGEST_INTERVAL_MINUTES", "15")

    assert Settings(_env_file=None).ingest_interval_minutes == 15  # type: ignore[call-arg]


def test_the_worker_concurrency_defaults_to_two() -> None:
    assert settings_with().celery_worker_concurrency == 2


@pytest.mark.parametrize("value", [1, 2, 4, 16])
def test_sensible_worker_concurrency_values_are_accepted(value: int) -> None:
    assert settings_with(celery_worker_concurrency=value).celery_worker_concurrency == value


@pytest.mark.parametrize("value", [0, -1, 17, 500])
def test_worker_concurrency_outside_one_to_sixteen_refuses_to_start(value: int) -> None:
    with pytest.raises(ValidationError, match="celery_worker_concurrency"):
        settings_with(celery_worker_concurrency=value)


def test_worker_concurrency_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/x")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
    monkeypatch.setenv("SECRET_KEY", "s3cret")
    monkeypatch.setenv("CELERY_WORKER_CONCURRENCY", "4")

    assert Settings(_env_file=None).celery_worker_concurrency == 4  # type: ignore[call-arg]


def test_the_celery_config_carries_the_worker_concurrency() -> None:
    config = build_config(settings_with(celery_worker_concurrency=3))

    assert config["worker_concurrency"] == 3


# --- the Celery configuration --------------------------------------------------------------------


def test_the_broker_is_redis_from_the_settings() -> None:
    config = build_config(settings_with(redis_url="redis://cache:6379/4"))

    assert config["broker_url"] == "redis://cache:6379/4"


def test_there_is_no_result_backend_and_results_are_ignored() -> None:
    config = build_config(settings_with())

    assert "result_backend" not in config
    assert config["task_ignore_result"] is True


def test_only_json_is_accepted_or_produced() -> None:
    config = build_config(settings_with())

    assert config["accept_content"] == ["json"]  # a pickle message is refused, not loaded
    assert config["task_serializer"] == config["result_serializer"] == "json"
    assert config["event_serializer"] == "json"


def test_tasks_are_acknowledged_late_and_requeued_if_the_worker_dies() -> None:
    config = build_config(settings_with())

    assert config["task_acks_late"] is True
    assert config["task_reject_on_worker_lost"] is True


def test_workers_take_one_task_at_a_time() -> None:
    assert build_config(settings_with())["worker_prefetch_multiplier"] == 1


def test_every_task_has_a_soft_limit_below_its_hard_limit_by_default() -> None:
    config = build_config(settings_with())

    assert config["task_soft_time_limit"] == SOFT_TIME_LIMIT_SECONDS
    assert config["task_time_limit"] == HARD_TIME_LIMIT_SECONDS
    assert 0 < SOFT_TIME_LIMIT_SECONDS < HARD_TIME_LIMIT_SECONDS


def test_publishing_fails_fast_when_the_broker_is_down() -> None:
    policy = build_config(settings_with())["task_publish_retry_policy"]

    assert policy["max_retries"] <= 3 and policy["interval_max"] <= 1


# --- the beat schedule ---------------------------------------------------------------------------


def test_the_schedule_has_the_two_jobs_with_their_task_names() -> None:
    entries = build_config(settings_with())["beat_schedule"]

    assert set(entries) == {"ingest-all-active-sources", "purge-expired-refresh-tokens"}
    assert entries["ingest-all-active-sources"]["task"] == "ingest_all_active_sources"
    assert entries["purge-expired-refresh-tokens"]["task"] == "purge_expired_refresh_tokens"


@pytest.mark.parametrize(("minutes", "seconds"), [(30, 1800.0), (5, 300.0), (90, 5400.0)])
def test_the_sync_runs_every_ingest_interval(minutes: int, seconds: float) -> None:
    entry = build_config(settings_with(ingest_interval_minutes=minutes))["beat_schedule"][
        "ingest-all-active-sources"
    ]

    assert entry["schedule"] == seconds
    assert entry["options"]["expires"] == seconds  # a run older than one interval is stale


def test_the_default_sync_interval_is_30_minutes() -> None:
    entry = build_config(settings_with())["beat_schedule"]["ingest-all-active-sources"]

    assert entry["schedule"] == 30 * 60


def test_the_purge_runs_once_a_day() -> None:
    entry = build_config(settings_with())["beat_schedule"]["purge-expired-refresh-tokens"]
    when = entry["schedule"]

    assert isinstance(when, crontab)
    assert (when.hour, when.minute) == ({3}, {17})  # one fixed time of day, every day
    assert when.day_of_week == set(range(7)) and when.day_of_month == set(range(1, 32))


def test_the_sync_schedule_is_a_plain_interval() -> None:
    config = build_config(settings_with())

    assert not isinstance(
        config["beat_schedule"]["ingest-all-active-sources"]["schedule"], schedule
    )
    assert isinstance(config["beat_schedule"]["ingest-all-active-sources"]["schedule"], float)


# --- restarting beat: no burst of catch-up runs --------------------------------------------------


class RecordingScheduler(Scheduler):
    """Celery's real beat scheduler, except that "sending" a task just records its name."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.sent: list[str] = []
        super().__init__(*args, **kwargs)

    def apply_async(
        self, entry: Any, producer: Any = None, advance: bool = True, **kw: Any
    ) -> None:
        self.sent.append(entry.name)


OURS = ("ingest-all-active-sources", "purge-expired-refresh-tokens")


def ticks(scheduler: RecordingScheduler, count: int = 10) -> list[str]:
    for _ in range(count):  # beat's main loop: tick, sleep, tick, ...
        scheduler.tick()
    return [name for name in scheduler.sent if name in OURS]


def test_a_freshly_started_beat_waits_a_full_interval_instead_of_running_at_once(
    celery_env: None,
) -> None:
    scheduler = RecordingScheduler(app=celery_app)  # no saved schedule: every entry starts "now"

    assert ticks(scheduler) == []


def test_a_beat_that_was_down_for_days_runs_each_job_once_not_once_per_missed_interval(
    celery_env: None,
) -> None:
    scheduler = RecordingScheduler(app=celery_app)
    long_ago = celery_app.now() - timedelta(days=3)  # 144 missed 30-minute intervals
    for name in OURS:
        scheduler.schedule[name].last_run_at = long_ago
    scheduler._heap = None  # make beat re-read the schedule, as it does when one is loaded

    sent = ticks(scheduler, count=20)

    assert sorted(sent) == sorted(OURS)  # exactly one run of each, however long it was down


def test_after_the_catch_up_run_the_next_one_is_a_full_interval_away(celery_env: None) -> None:
    scheduler = RecordingScheduler(app=celery_app)
    long_ago = celery_app.now() - timedelta(days=3)
    scheduler.schedule["ingest-all-active-sources"].last_run_at = long_ago
    scheduler._heap = None
    assert ticks(scheduler).count("ingest-all-active-sources") == 1

    everything_sent = ticks(scheduler, count=50)  # many more ticks, with no time passing

    assert everything_sent.count("ingest-all-active-sources") == 1  # still just the catch-up run


# --- the registered tasks ------------------------------------------------------------------------


def test_exactly_the_expected_tasks_are_registered(celery_env: None) -> None:
    celery_app.loader.import_default_modules()  # what a worker does on start-up

    names = {name for name in celery_app.tasks if not name.startswith("celery.")}

    assert names == EXPECTED_TASKS


def test_every_task_has_explicit_time_limits(celery_env: None) -> None:
    celery_app.loader.import_default_modules()

    for name in EXPECTED_TASKS:
        task = celery_app.tasks[name]
        soft = task.soft_time_limit or SOFT_TIME_LIMIT_SECONDS
        hard = task.time_limit or HARD_TIME_LIMIT_SECONDS
        assert 0 < soft < hard, name


def test_every_scheduled_task_exists(celery_env: None) -> None:
    celery_app.loader.import_default_modules()

    for entry in build_config(settings_with())["beat_schedule"].values():
        assert entry["task"] in celery_app.tasks


# --- wiring, checked in a fresh process so nothing from this test run leaks in -------------------


def run_python(code: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_importing_the_celery_app_needs_no_environment_variables() -> None:
    clean = {
        k: v for k, v in os.environ.items() if k not in {"DATABASE_URL", "REDIS_URL", "SECRET_KEY"}
    }

    result = run_python("import app.tasks.celery_app, app.tasks.jobs, app.api_deps", clean)

    assert result.returncode == 0, result.stderr


def test_a_real_worker_process_reads_its_config_from_the_settings() -> None:
    env = os.environ | {
        "DATABASE_URL": "postgresql+asyncpg://u:p@db:5432/x",
        "REDIS_URL": "redis://redis-host:6379/3",
        "SECRET_KEY": "x" * 40,
        "INGEST_INTERVAL_MINUTES": "7",
    }
    code = (
        "import json; from app.tasks.celery_app import celery_app as a; c = a.conf; "
        "print(json.dumps({'broker': c.broker_url, 'serializer': c.task_serializer, "
        "'accept': list(c.accept_content), 'late': c.task_acks_late, "
        "'prefetch': c.worker_prefetch_multiplier, 'soft': c.task_soft_time_limit, "
        "'hard': c.task_time_limit, 'backend': a.backend.__class__.__name__, "
        "'interval': c.beat_schedule['ingest-all-active-sources']['schedule']}))"
    )

    result = run_python(code, env)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "broker": "redis://redis-host:6379/3",
        "serializer": "json",
        "accept": ["json"],
        "late": True,
        "prefetch": 1,
        "soft": SOFT_TIME_LIMIT_SECONDS,
        "hard": HARD_TIME_LIMIT_SECONDS,
        "backend": "DisabledBackend",
        "interval": 420.0,  # 7 minutes
    }


def test_the_app_refuses_to_start_with_too_short_an_interval() -> None:
    env = os.environ | {
        "DATABASE_URL": "postgresql+asyncpg://u:p@db:5432/x",
        "REDIS_URL": "redis://redis:6379/0",
        "SECRET_KEY": "x" * 40,
        "INGEST_INTERVAL_MINUTES": "2",
    }

    result = run_python("from app.core.config import get_settings; get_settings()", env)

    assert result.returncode != 0
    assert "ingest_interval_minutes" in result.stderr
