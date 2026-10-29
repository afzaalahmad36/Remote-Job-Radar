"""Time limits of one extraction run. They are chosen together so they cannot contradict each other.

Each limit is longer than the one inside it, so the innermost one always fires first and the
others are only safety nets:

    one provider request timeout        (setting, at most MAX_LLM_REQUEST_TIMEOUT_SECONDS)
    < the run's overall deadline        (at most MAX_REQUESTS_PER_JOB requests + a margin)
    < Celery soft time limit            (the task gets a chance to clean up)
    < Celery hard time limit            (the worker process is killed)
    < the per-job lock's expiry         (the lock must outlive any run that is still allowed)
"""

from app.core.config import MAX_LLM_REQUEST_TIMEOUT_SECONDS

# One request, plus either a retry after a timeout/5xx or the one retry after invalid output.
MAX_REQUESTS_PER_JOB = 2
# Room for the pause before a retry, the rules and the database writes.
DEADLINE_MARGIN_SECONDS = 5.0

EXTRACTION_SOFT_TIME_LIMIT_SECONDS = 60
EXTRACTION_HARD_TIME_LIMIT_SECONDS = 90
JOB_LOCK_TTL_SECONDS = EXTRACTION_HARD_TIME_LIMIT_SECONDS + 30


def run_deadline_seconds(request_timeout: float = MAX_LLM_REQUEST_TIMEOUT_SECONDS) -> float:
    """The most one extraction run may take, given the timeout of a single request."""
    return MAX_REQUESTS_PER_JOB * request_timeout + DEADLINE_MARGIN_SECONDS
