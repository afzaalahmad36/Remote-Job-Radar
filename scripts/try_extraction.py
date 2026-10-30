"""Manual smoke test for job extraction. Run it by hand; app code never imports it and CI never
runs it (the tests only cover its argument parsing and its request-budget logic).

Default: a rules-only dry run. It reads jobs from the local database (read-only), runs the
deterministic rules and prints per-field coverage and how many posts WOULD need a provider
request. No network, nothing is written.

    uv run python -m scripts.try_extraction [--source-id UUID] [--limit 50]

Provider mode sends real requests to the configured provider, so it needs BOTH flags. The budget is
mandatory and capped at HARD_MAX_REQUESTS; the run stops at the budget or on the first rate-limit
or rejected request, and never retries:

    uv run python -m scripts.try_extraction --allow-provider --max-requests 3 [--yes]

It builds its own client from the settings (LLM_PROVIDER, LLM_MODEL_EXTRACTION, GEMINI_API_KEY)
and turns the kill switch on for this process only: .env is never modified. A run record
(no post text, no key) is written under .local/, which git ignores.

Evaluation (stub): `--eval labeled.json` only checks the file's format for now. Format:

    {"posts": [{"text": "full post text...",
                "expected": {"title": "Backend Engineer", "company": "Acme",
                             "stack": ["Python", "Django"], "seniority": "senior",
                             "remote_policy": "remote"}}]}

Every key inside "expected" is optional; a key left out means "not labeled" for that post.
"""

import argparse
import asyncio
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import Settings, get_settings
from app.extraction.merge import (
    MergedExtraction,
    is_too_short_to_enrich,
    merge,
    missing_key_fields,
)
from app.extraction.rules import RuleResult, extract_with_rules
from app.extraction.schema import ExtractionFormatError, parse_extraction
from app.llm.base import (
    LLMBadResponse,
    LLMClient,
    LLMConfigError,
    LLMError,
    LLMPrivacyError,
    LLMRateLimited,
    LLMResponse,
)
from app.llm.client import build_client
from app.llm.prompts import EXTRACT_JOB_V1, build_extraction_request
from app.models import Job

HARD_MAX_REQUESTS = 10  # not a setting and not a flag: raising it means editing this file
DEFAULT_LIMIT = 50
MAX_LIMIT = 500
RUN_RECORD_DIR = Path(__file__).resolve().parent.parent / ".local" / "extraction-runs"
MAX_RECORDED_MODEL_CHARS = 4_000
LOW_SUFFIX = " (low)"  # shown after a rules value that is only a guess
COVERAGE_FIELDS = (
    "title", "company", "remote_policy", "stack", "apply_email", "apply_url", "salary", "seniority"
)  # fmt: skip
COMPARED_FIELDS = (
    "title", "company", "stack", "seniority", "salary", "remote_policy", "apply_url", "apply_email"
)  # fmt: skip


class ScriptError(Exception):
    """A problem the user can fix; main() prints it without a traceback."""


# --- arguments ---------------------------------------------------------------------------------


def _request_budget(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number") from None
    if not 1 <= number <= HARD_MAX_REQUESTS:
        raise argparse.ArgumentTypeError(f"must be between 1 and {HARD_MAX_REQUESTS}")
    return number


def _limit(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number") from None
    if not 1 <= number <= MAX_LIMIT:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_LIMIT}")
    return number


def _uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a UUID") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.try_extraction",
        description="Try job extraction on stored jobs. Rules only unless --allow-provider.",
    )
    parser.add_argument("--source-id", type=_uuid, help="only jobs of this source")
    parser.add_argument("--limit", type=_limit, default=DEFAULT_LIMIT, help="newest N jobs")
    parser.add_argument(
        "--allow-provider",
        action="store_true",
        help="send real requests to the configured provider (needs --max-requests)",
    )
    parser.add_argument(
        "--max-requests",
        type=_request_budget,
        help=f"most provider requests this run may make (1-{HARD_MAX_REQUESTS})",
    )
    parser.add_argument("--yes", action="store_true", help="skip the confirmation question")
    parser.add_argument("--eval", type=Path, metavar="LABELED.json", help="check a labels file")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.allow_provider and args.max_requests is None:
        parser.error("--allow-provider needs --max-requests N")
    if args.max_requests is not None and not args.allow_provider:
        parser.error("--max-requests only makes sense with --allow-provider")
    if args.eval is not None and (args.allow_provider or args.source_id):
        parser.error("--eval cannot be combined with --allow-provider or --source-id")
    return args


# --- analysing posts (no network) ----------------------------------------------------------------


@dataclass(frozen=True)
class PostAnalysis:
    job_id: uuid.UUID
    text: str
    rules: RuleResult
    missing: tuple[str, ...]
    too_short: bool

    @property
    def needs_request(self) -> bool:
        """The same decision the extraction service will make."""
        return bool(self.missing) and not self.too_short


def analyze(job_id: uuid.UUID, text: str, min_chars: int) -> PostAnalysis:
    rules = extract_with_rules(text)
    return PostAnalysis(
        job_id=job_id,
        text=text,
        rules=rules,
        missing=missing_key_fields(rules),
        too_short=is_too_short_to_enrich(text, min_chars),
    )


def field_coverage(analyses: list[PostAnalysis]) -> dict[str, int]:
    """How many posts have each field (any confidence)."""

    def has(post: PostAnalysis, name: str) -> bool:
        value = getattr(post.rules, name)
        return bool(value) if name == "stack" else value is not None

    return {name: sum(1 for a in analyses if has(a, name)) for name in COVERAGE_FIELDS}


def select_candidates(analyses: list[PostAnalysis], max_requests: int) -> list[PostAnalysis]:
    """Posts to send: those that need a request and pass the length gate, in the given
    (newest-first) order, never more than the budget."""
    return [a for a in analyses if a.needs_request][:max_requests]


def drop_duplicate_texts(rows: list["PostRow"]) -> tuple[list["PostRow"], int]:
    """Keep the first (newest) row of each identical text: a repeat would waste a request."""
    seen: set[str] = set()
    unique: list[PostRow] = []
    for row in rows:
        if row.text not in seen:
            seen.add(row.text)
            unique.append(row)
    return unique, len(rows) - len(unique)


# --- the request budget ------------------------------------------------------------------------


@dataclass
class RequestBudget:
    """Counts provider requests. Attempted requests count, successful or not."""

    max_requests: int
    used: int = 0
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.max_requests <= HARD_MAX_REQUESTS:
            raise ValueError(f"max_requests must be between 1 and {HARD_MAX_REQUESTS}")

    @property
    def can_send(self) -> bool:
        return self.stop_reason is None and self.used < self.max_requests

    def stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason


@dataclass
class RequestRecord:
    job_id: str
    outcome: str  # ok | rate_limited | rejected | privacy | transient | bad_response
    reason: str | None
    latency_ms: int
    sent_chars: int
    truncated: bool
    tokens: dict[str, int | None] = field(default_factory=dict)
    validated: bool = False
    validation_error: str | None = None
    rules_values: dict[str, str] = field(default_factory=dict)
    merged_values: dict[str, str] = field(default_factory=dict)
    conflicts: tuple[str, ...] = ()
    model_text: str | None = None  # untrusted; capped


async def run_requests(
    client: LLMClient,
    candidates: list[PostAnalysis],
    budget: RequestBudget,
    *,
    max_chars: int,
) -> list[RequestRecord]:
    """One request per candidate, in order, until the budget is used or something stops the run.

    A 429 and a rejected request (quota, key, model) stop everything; nothing is retried.
    """
    records: list[RequestRecord] = []
    for post in candidates:
        if not budget.can_send:
            break
        prepared = build_extraction_request(post.text, max_chars=max_chars)
        budget.used += 1
        started = time.perf_counter()
        try:
            response = await client.complete(prepared.request)
        except LLMError as error:
            latency = round((time.perf_counter() - started) * 1000)
            records.append(_failed(post, prepared.request.user, prepared.truncated, latency, error))
            _stop_if_fatal(budget, error)
            continue
        latency = round((time.perf_counter() - started) * 1000)
        records.append(
            _answered(post, prepared.request.user, prepared.truncated, latency, response)
        )
    return records


def _stop_if_fatal(budget: RequestBudget, error: LLMError) -> None:
    if isinstance(error, LLMRateLimited):
        budget.stop("rate limited (HTTP 429): stopped, nothing retried")
    elif isinstance(error, LLMConfigError):
        budget.stop(f"provider rejected the request ({error.reason}): stopped, nothing retried")
    elif isinstance(error, LLMPrivacyError):
        budget.stop("the privacy guard refused a request")


def _outcome(error: LLMError) -> str:
    if isinstance(error, LLMRateLimited):
        return "rate_limited"
    if isinstance(error, LLMConfigError):
        return "rejected"
    if isinstance(error, LLMPrivacyError):
        return "privacy"
    if isinstance(error, LLMBadResponse):
        return "bad_response"
    return "transient"


def _failed(
    post: PostAnalysis, sent: str, truncated: bool, latency: int, error: LLMError
) -> RequestRecord:
    return RequestRecord(
        job_id=str(post.job_id),
        outcome=_outcome(error),
        reason=error.reason,
        latency_ms=latency,
        sent_chars=len(sent),
        truncated=truncated,
        rules_values=rules_values(post),
    )


def _answered(
    post: PostAnalysis, sent: str, truncated: bool, latency: int, response: LLMResponse
) -> RequestRecord:
    usage = response.usage
    record = RequestRecord(
        job_id=str(post.job_id),
        outcome="ok",
        reason=None,
        latency_ms=latency,
        sent_chars=len(sent),
        truncated=truncated,
        tokens={
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "thought": usage.thought_tokens,
        },
        rules_values=rules_values(post),
        model_text=response.text[:MAX_RECORDED_MODEL_CHARS],
    )
    try:
        extraction = parse_extraction(response.text)
    except ExtractionFormatError as error:
        record.validation_error = error.summary  # field names and error kinds only
        record.merged_values = merged_values(merge(post.rules, None, post.text))
        return record
    record.validated = True
    merged = merge(post.rules, extraction, post.text)
    record.merged_values = merged_values(merged)
    record.conflicts = merged.conflicts
    return record


# --- showing values ----------------------------------------------------------------------------


def _found(found: Any) -> str:
    if found is None:
        return "-"
    suffix = "" if found.is_high else LOW_SUFFIX
    return f"{found.value}{suffix}"


def _salary_text(low: object, high: object, currency: object, period: object) -> str:
    if low is None and high is None:
        return "-"
    return f"{low}-{high} {currency or '?'}/{period or '?'}"


def _rules_salary(rules: RuleResult) -> str:
    if rules.salary is None:
        return "-"
    amounts = rules.salary.value
    text = _salary_text(amounts.minimum, amounts.maximum, amounts.currency, amounts.period)
    return text if rules.salary.is_high else f"{text}{LOW_SUFFIX}"


def rules_values(post: PostAnalysis) -> dict[str, str]:
    rules = post.rules
    return {
        "title": _found(rules.title),
        "company": _found(rules.company),
        "stack": ", ".join(rules.stack) or "-",
        "seniority": _found(rules.seniority),
        "salary": _rules_salary(rules),
        "remote_policy": _found(rules.remote_policy),
        "apply_url": _found(rules.apply_url),
        "apply_email": _found(rules.apply_email),
    }


def merged_values(merged: MergedExtraction) -> dict[str, str]:
    return {
        "title": merged.title or "-",
        "company": merged.company or "-",
        "stack": ", ".join(merged.stack) or "-",
        "seniority": str(merged.seniority or "-"),
        "salary": _salary_text(
            merged.salary_min, merged.salary_max, merged.salary_currency, merged.salary_period
        ),
        "remote_policy": str(merged.remote_policy or "-"),
        "apply_url": merged.apply_url or "-",
        "apply_email": merged.apply_email or "-",
    }


def print_coverage(analyses: list[PostAnalysis], duplicates_skipped: int) -> None:
    total = len(analyses)
    print(f"\nRules-only coverage over {total} posts ({duplicates_skipped} repeated texts skipped)")
    for name, count in field_coverage(analyses).items():
        print(f"  {name:<14}{count:>4}  ({100 * count // total}%)")
    needing = sum(1 for a in analyses if a.needs_request)
    short = sum(1 for a in analyses if a.too_short)
    incomplete = sum(1 for a in analyses if a.missing)
    print(f"\n  key fields missing (title/stack): {incomplete}")
    print(f"  too short to send (length gate):  {short}")
    print(f"  would trigger a provider request: {needing}")


def print_result(index: int, record: RequestRecord) -> None:
    tokens = record.tokens
    token_text = "tokens n/a"
    if tokens:
        counts = "/".join(str(tokens.get(kind)) for kind in ("input", "output", "thought"))
        token_text = f"tokens in/out/thought {counts}"
    print(
        f"\n[{index}] job {record.job_id}: {record.outcome} ({record.reason or 'ok'}), "
        f"{record.latency_ms} ms, {token_text}"
    )
    if record.outcome != "ok":
        return
    verdict = "yes" if record.validated else f"NO ({record.validation_error})"
    print(f"    validated against JobExtraction: {verdict}")
    print(f"    {'field':<14}{'rules':<34}merged")
    for name in COMPARED_FIELDS:
        before, after = record.rules_values[name], record.merged_values[name]
        mark = "*" if before.removesuffix(LOW_SUFFIX) != after else " "
        print(f"  {mark} {name:<14}{before[:32]:<34}{after[:60]}")
    if record.conflicts:
        print(f"    model disagreed with a HIGH rule value on: {', '.join(record.conflicts)}")


# --- reading jobs (SELECT only) ----------------------------------------------------------------


@dataclass(frozen=True)
class PostRow:
    job_id: uuid.UUID
    text: str


async def load_posts(database_url: str, source_id: uuid.UUID | None, limit: int) -> list[PostRow]:
    """The newest jobs. The connection itself is read-only (Postgres refuses any write)."""
    engine = create_async_engine(
        database_url, connect_args={"server_settings": {"default_transaction_read_only": "on"}}
    )
    query = select(Job.id, Job.raw_text).order_by(
        Job.posted_at.desc().nulls_last(), Job.created_at.desc()
    )
    if source_id is not None:
        query = query.where(Job.source_id == source_id)
    try:
        async with engine.connect() as connection:
            result = await connection.execute(query.limit(limit))
            return [PostRow(row.id, row.raw_text) for row in result]
    except (OSError, SQLAlchemyError, asyncpg.PostgresError) as error:
        # Only the error's type: its message could contain the database URL and password.
        raise ScriptError(f"could not read the database ({type(error).__name__})") from None
    finally:
        await engine.dispose()


# --- provider setup ----------------------------------------------------------------------------


def provider_settings(settings: Settings) -> Settings:
    """A copy of the settings with the kill switch on, for this process only (.env is untouched).
    Refuses clearly when a provider cannot be used."""
    if settings.llm_provider == "none":
        raise ScriptError("LLM_PROVIDER is 'none': set it to 'gemini' (or 'fake') in your .env")
    if not settings.llm_model_extraction.strip():
        raise ScriptError("LLM_MODEL_EXTRACTION is empty: look up a current model in the docs")
    if settings.llm_provider == "gemini" and not settings.gemini_api_key.get_secret_value():
        raise ScriptError("GEMINI_API_KEY is empty: add your free key to .env")
    return settings.model_copy(update={"llm_extraction_enabled": True})


def make_client(settings: Settings) -> LLMClient:
    try:
        client = build_client(provider_settings(settings))
    except LLMConfigError as error:
        raise ScriptError(error.reason) from None
    if client is None:  # cannot happen after provider_settings; kept so mypy and readers agree
        raise ScriptError("no provider client could be built")
    return client


def confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


# --- labeled data (stub) -----------------------------------------------------------------------


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    company: str | None = None
    stack: list[str] | None = None
    seniority: str | None = None
    remote_policy: str | None = None


class LabeledPost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    expected: Expected


class LabeledFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    posts: list[LabeledPost] = Field(min_length=1, max_length=MAX_LIMIT)


def load_labels(path: Path) -> LabeledFile:
    try:
        return LabeledFile.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ScriptError(f"cannot read {path}: {type(error).__name__}") from None
    except ValidationError as error:
        problems = sorted({".".join(str(p) for p in e["loc"]) for e in error.errors()})
        raise ScriptError(f"{path} does not match the format: {', '.join(problems[:6])}") from None


# --- the run record ----------------------------------------------------------------------------


def save_record(
    directory: Path, *, client: LLMClient, budget: RequestBudget, records: list[RequestRecord]
) -> Path:
    """A local JSON file with the run: ids, lengths, outcomes, values. No post text, no key."""
    started = datetime.now(UTC)
    payload = {
        "finished_at": started.isoformat(),
        "provider": client.name,
        "model": client.model,
        "prompt_version": EXTRACT_JOB_V1.version,
        "budget": {"max": budget.max_requests, "used": budget.used, "stop": budget.stop_reason},
        "requests": [asdict(record) for record in records],
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{started:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --- the command -------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    if args.eval is not None:
        labels = load_labels(args.eval)
        print(f"{args.eval}: format OK, {len(labels.posts)} labeled posts.")
        print("--eval is a stub: accuracy scoring is not built yet.")
        return 0

    settings = get_settings()
    client = make_client(settings) if args.allow_provider else None  # fail before reading
    rows = asyncio.run(load_posts(settings.database_url, args.source_id, args.limit))
    if not rows:
        print("No jobs found. Sync a source first (the database is only read).")
        return 0
    rows, duplicates = drop_duplicate_texts(rows)
    analyses = [analyze(r.job_id, r.text, settings.llm_min_post_chars) for r in rows]
    print_coverage(analyses, duplicates)

    if client is None:
        print(
            "\nDry run: rules only, no network. To ask a model: --allow-provider --max-requests N"
        )
        return 0
    return run_provider(args, settings, client, analyses)


def run_provider(
    args: argparse.Namespace, settings: Settings, client: LLMClient, analyses: list[PostAnalysis]
) -> int:
    candidates = select_candidates(analyses, args.max_requests)
    if not candidates:
        print("\nNo post needs a provider request (or all are below the length gate).")
        return 0

    print(
        f"\nAbout to send {len(candidates)} request(s) (budget {args.max_requests}, hard cap "
        f"{HARD_MAX_REQUESTS}) to {client.name} / {client.model}:"
    )
    for post in candidates:
        prepared = build_extraction_request(post.text, max_chars=settings.llm_max_input_chars)
        note = ", middle omitted" if prepared.truncated else ""
        print(
            f"  job {post.job_id}  post {len(post.text)} chars -> sent "
            f"{len(prepared.request.user)} chars (emails/phones masked){note}"
        )
    if not args.yes and not confirm("Send these requests?"):
        print("Cancelled. Nothing was sent.")
        return 0

    budget = RequestBudget(args.max_requests)
    records = asyncio.run(
        run_requests(client, candidates, budget, max_chars=settings.llm_max_input_chars)
    )
    for index, record in enumerate(records, start=1):
        print_result(index, record)
    path = save_record(RUN_RECORD_DIR, client=client, budget=budget, records=records)
    print(f"\n{budget.used} request(s) made. Stopped early: {budget.stop_reason or 'no'}")
    print(f"Run record: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except ScriptError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
