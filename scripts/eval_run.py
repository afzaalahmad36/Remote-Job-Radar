"""`try_extraction.py --eval`: score the extraction against hand labels.

For every labelled post it takes two answers and scores both the same way (the pass rule is in
CLAUDE.md, section 6): the extraction stored in the database, and the rules recomputed offline from
the post text. It reads the database (read-only) and makes NO provider request: this module does
not import the LLM client at all, and a test checks that.
"""

import asyncio
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import asyncpg  # type: ignore[import-untyped]
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import Settings
from app.extraction.merge import merge
from app.extraction.rules import extract_with_rules
from app.extraction.text_hash import raw_text_hash
from app.models import Job
from scripts.db_guard import ScriptError, database_target, ensure_jobs_table, read_only_engine
from scripts.eval_labelfile import RUN_DIR, LabelFile, load_label_file
from scripts.eval_scoring import (
    FieldValues,
    PostScore,
    Summary,
    field_table,
    headline,
    normalise_title,
    score_post,
    summarise,
)


class Exclusion(StrEnum):
    """Why a selected post is not scored."""

    SKIPPED = "skipped"
    STALE_HASH = "stale_hash"  # the post text changed since it was labelled
    BLANK_TITLE = "blank_title"  # the label has no title, so there is nothing to compare
    JOB_NOT_FOUND = "job_not_found"  # the job was deleted
    NOT_LABELLED = "not_labelled"  # selected but never labelled or skipped


@dataclass(frozen=True)
class StoredJob:
    raw_text: str
    stored: FieldValues  # the extraction columns exactly as the database holds them


@dataclass(frozen=True)
class Evaluation:
    selected: int
    excluded: Mapping[Exclusion, int]
    stored: Summary
    rules: Summary
    stored_failures: Mapping[uuid.UUID, tuple[str, ...]]
    rules_failures: Mapping[uuid.UUID, tuple[str, ...]]


def rules_values(text: str) -> FieldValues:
    """What the rules alone give for a post, merged exactly as the service stores it."""
    merged = merge(extract_with_rules(text), None, text)
    return FieldValues(
        title=merged.title,
        company=merged.company,
        stack=merged.stack,
        remote_policy=merged.remote_policy.value if merged.remote_policy else None,
        seniority=merged.seniority.value if merged.seniority else None,
        salary_min=merged.salary_min,
        salary_max=merged.salary_max,
        salary_currency=merged.salary_currency,
        salary_period=merged.salary_period.value if merged.salary_period else None,
    )


def evaluate(label_file: LabelFile, jobs: Mapping[uuid.UUID, StoredJob]) -> Evaluation:
    """Pure: the label file and the jobs read from the database in, the scores out."""
    excluded = dict.fromkeys(Exclusion, 0)
    stored_scores: dict[uuid.UUID, PostScore] = {}
    rules_scores: dict[uuid.UUID, PostScore] = {}
    excluded[Exclusion.NOT_LABELLED] = len(label_file.pending())
    for entry in label_file.posts:
        job = jobs.get(entry.job_id)
        reason = None
        if entry.labels is None:
            reason = Exclusion.SKIPPED
        elif job is None:
            reason = Exclusion.JOB_NOT_FOUND
        elif raw_text_hash(job.raw_text) != entry.raw_text_hash:
            reason = Exclusion.STALE_HASH
        elif normalise_title(entry.labels.title) is None:
            reason = Exclusion.BLANK_TITLE
        if reason is not None:
            excluded[reason] += 1
            continue
        assert entry.labels is not None and job is not None  # narrowed by the checks above
        label = entry.labels.values()
        stored_scores[entry.job_id] = score_post(label, job.stored)
        rules_scores[entry.job_id] = score_post(label, rules_values(job.raw_text))
    return Evaluation(
        selected=len(label_file.selected),
        excluded=excluded,
        stored=summarise(stored_scores.values()),
        rules=summarise(rules_scores.values()),
        stored_failures={k: v.failures for k, v in stored_scores.items() if v.failures},
        rules_failures={k: v.failures for k, v in rules_scores.items() if v.failures},
    )


# --- reading the stored extraction -------------------------------------------------------------


async def read_stored_jobs(
    database_url: str, job_ids: list[uuid.UUID]
) -> dict[uuid.UUID, StoredJob]:
    query = select(
        Job.id,
        Job.raw_text,
        Job.title,
        Job.company,
        Job.stack,
        Job.remote_policy,
        Job.seniority,
        Job.salary_min,
        Job.salary_max,
        Job.salary_currency,
        Job.salary_period,
    ).where(Job.id.in_(job_ids))
    engine = read_only_engine(database_url)
    try:
        async with engine.connect() as connection:
            await ensure_jobs_table(connection)
            rows = (await connection.execute(query)).all()
    except (OSError, SQLAlchemyError, asyncpg.PostgresError) as error:
        raise ScriptError(f"could not read the database ({type(error).__name__})") from None
    finally:
        await engine.dispose()
    return {
        row.id: StoredJob(
            raw_text=row.raw_text,
            stored=FieldValues(
                title=row.title,
                company=row.company,
                stack=tuple(row.stack or ()),
                remote_policy=row.remote_policy.value if row.remote_policy else None,
                seniority=row.seniority.value if row.seniority else None,
                salary_min=row.salary_min,
                salary_max=row.salary_max,
                salary_currency=row.salary_currency,
                salary_period=row.salary_period.value if row.salary_period else None,
            ),
        )
        for row in rows
    }


# --- output ------------------------------------------------------------------------------------


def format_exclusions(evaluation: Evaluation) -> str:
    total = sum(evaluation.excluded.values())
    scored = evaluation.stored.scored
    reasons = ", ".join(f"{reason.value} {count}" for reason, count in evaluation.excluded.items())
    return f"Posts: {evaluation.selected} selected, {scored} scored, {total} excluded ({reasons})"


def summary_record(summary: Summary) -> dict[str, Any]:
    """A summary as plain data for the run record: numbers and codes only."""
    return {
        "verdict": summary.verdict,
        "scored": summary.scored,
        "parse_correct": summary.passed,
        "pass_share": summary.pass_share,
        "interval": summary.interval,
        "invented_posts": summary.invented_posts,
        "gate_failed": summary.gate_failed,
        "fail_reasons": list(summary.fail_reasons),
        "fields": {
            name: {outcome.value: count for outcome, count in field.counts.items()}
            for name, field in summary.fields.items()
        },
        "stack": {
            "posts_ok": summary.stack.posts_ok,
            "true_positives": summary.stack.true_positives,
            "false_positives": summary.stack.false_positives,
            "false_negatives": summary.stack.false_negatives,
            "dictionary_gaps": summary.stack.dictionary_gaps,
        },
        "failure_counts": dict(summary.failure_counts),
    }


def save_eval_record(
    directory: Path, *, labels_path: Path, target: str, evaluation: Evaluation
) -> Path:
    """A local JSON file with the numbers and the ids of failing posts. No post text, no labels."""
    finished = datetime.now(UTC)
    payload = {
        "finished_at": finished.isoformat(),
        "mode": "eval",
        "provider_requests": 0,
        "labels_file": labels_path.name,
        "database": target,
        "selected": evaluation.selected,
        "excluded": {reason.value: count for reason, count in evaluation.excluded.items()},
        "stored": summary_record(evaluation.stored),
        "rules": summary_record(evaluation.rules),
        "stored_failures": {str(k): list(v) for k, v in evaluation.stored_failures.items()},
        "rules_failures": {str(k): list(v) for k, v in evaluation.rules_failures.items()},
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{finished:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def print_evaluation(evaluation: Evaluation) -> None:
    print()
    print(format_exclusions(evaluation))
    print()
    print(headline("Stored extraction (what the product shows)", evaluation.stored))
    print(headline("Rules recomputed offline", evaluation.rules))
    for name, summary in (
        ("Stored extraction", evaluation.stored),
        ("Rules offline", evaluation.rules),
    ):
        if summary.scored:
            print(f"\n{name}: per field")
            print(field_table(summary))
            for code, count in sorted(summary.failure_counts.items()):
                print(f"  failed on {code}: {count} post(s)")
    if evaluation.stored.scored and evaluation.stored_failures != evaluation.rules_failures:
        print(
            "\nThe stored extraction and the offline rules differ on some posts (a rules version "
            "change, or a model result)."
        )


def run_eval(labels_path: Path, settings: Settings) -> int:
    """Score `labels_path`. Reads the database; never contacts a provider."""
    label_file = load_label_file(labels_path)
    target = database_target(settings.database_url)  # refuses the wrong database up front
    print(f"Reading (read-only) from database {target}")
    ids = [entry.job_id for entry in label_file.posts]
    jobs = asyncio.run(read_stored_jobs(settings.database_url, ids))
    evaluation = evaluate(label_file, jobs)
    print_evaluation(evaluation)
    path = save_eval_record(RUN_DIR, labels_path=labels_path, target=target, evaluation=evaluation)
    print(f"\nRun record: {path}")
    return 0
