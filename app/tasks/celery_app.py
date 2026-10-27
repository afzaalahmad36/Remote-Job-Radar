from collections.abc import Iterator, Mapping
from functools import cached_property
from typing import Any

from celery import Celery
from celery.schedules import crontab

from app.core.config import Settings, get_settings

# Every task gets these limits unless it sets its own. The soft limit raises an exception inside
# the task (so it can clean up); the hard limit kills the worker process running it.
SOFT_TIME_LIMIT_SECONDS = 60
HARD_TIME_LIMIT_SECONDS = 90

PURGE_AT_UTC = {"hour": 3, "minute": 17}  # daily, at a quiet time


def build_config(settings: Settings) -> dict[str, Any]:
    """Celery's settings, built from the app settings (one place, easy to test)."""
    ingest_interval = settings.ingest_interval_minutes * 60
    return {
        "broker_url": settings.redis_url,
        # No result backend, on purpose. Nothing ever waits for a task's return value: the
        # outcome of a sync is stored where users can see it (jobs, sources.last_synced_at,
        # sources.last_error). A backend would only add Redis keys to store and expire.
        "task_ignore_result": True,
        # JSON only: a message can never carry a pickle, which would execute code on load.
        "task_serializer": "json",
        "result_serializer": "json",
        "event_serializer": "json",
        "accept_content": ["json"],
        # Acknowledge a message after the task finishes, not when it starts, and requeue it if
        # the worker process dies mid-task. Safe because every task is idempotent.
        "task_acks_late": True,
        "task_reject_on_worker_lost": True,
        "task_soft_time_limit": SOFT_TIME_LIMIT_SECONDS,
        "task_time_limit": HARD_TIME_LIMIT_SECONDS,
        # A worker process takes one task at a time instead of hoarding a batch, so one slow
        # feed cannot hold up tasks that another process could be running.
        "worker_prefetch_multiplier": 1,
        "worker_concurrency": settings.celery_worker_concurrency,
        "timezone": "UTC",
        "enable_utc": True,
        "broker_connection_retry_on_startup": True,
        # Publishing from the API must fail fast when Redis is down (the API then answers 503).
        "task_publish_retry_policy": {
            "max_retries": 2,
            "interval_start": 0,
            "interval_step": 0.2,
            "interval_max": 0.5,
        },
        "broker_transport_options": {"socket_connect_timeout": 5, "socket_timeout": 5},
        "beat_schedule": {
            "ingest-all-active-sources": {
                "task": "ingest_all_active_sources",
                "schedule": float(ingest_interval),
                # If the workers were down for a while, don't replay the missed runs when they
                # come back: a run that is older than one interval is stale.
                "options": {"expires": ingest_interval},
            },
            "purge-expired-refresh-tokens": {
                "task": "purge_expired_refresh_tokens",
                "schedule": crontab(**PURGE_AT_UTC),
                "options": {"expires": 12 * 3600},
            },
        },
    }


class _LazyConfig(Mapping[str, Any]):
    """The app's Celery config, built the first time Celery asks for it, not at import time.

    Importing this module (as tests and tools do) therefore needs no environment variables;
    a worker, beat or the API reads the real settings when it actually starts using Celery.
    """

    @cached_property
    def _values(self) -> dict[str, Any]:
        return build_config(get_settings())

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


# `include` makes workers import the task module on start-up, which registers the tasks.
celery_app = Celery("job_radar", include=["app.tasks.jobs"])
celery_app.config_from_object(_LazyConfig())
