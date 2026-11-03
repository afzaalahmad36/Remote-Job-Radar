"""The decision for the rules phase of one job. Pure: no database, no network, no clock.

Given a post's text, it says what to store (the rules' values), what the job's state becomes and
whether the model may be asked later. The service only applies the answer, so every case of the
state table can be tested here without a database.
"""

from dataclasses import dataclass
from enum import StrEnum

from app.extraction.merge import (
    MergedExtraction,
    is_too_short_to_enrich,
    merge,
    missing_key_fields,
)
from app.extraction.rules import RuleResult, extract_with_rules
from app.extraction.text_hash import raw_text_hash
from app.models import ExtractionStatus, LlmEnrichment


class ExtractionError(StrEnum):
    """The only texts that may be stored in `jobs.extraction_error`.

    A fixed set on purpose: nothing from the post, from a provider or from an exception can reach
    that column, because the column only accepts a member of this enum.
    """

    TOO_SHORT = "post too short for the model"
    MODEL_NOT_ENABLED = "no title found; the model is not enabled"
    UNEXPECTED = "unexpected error"


@dataclass(frozen=True)
class RulesPlan:
    text_hash: str
    values: MergedExtraction  # what the rules found; fields they did not find are None / empty
    status: ExtractionStatus
    enrichment: LlmEnrichment
    error: ExtractionError | None  # why the model will not be asked (or the job failed)


def plan_rules_phase(text: str, *, can_enrich: bool, min_chars: int) -> RulesPlan:
    """Run the rules on `text` and decide the job's state.

    `can_enrich` is whether a model client exists (provider configured and kill switch on);
    `min_chars` is the length gate (`LLM_MIN_POST_CHARS`).
    """
    rules = extract_with_rules(text)
    values = merge(rules, None, text)  # rules only: a model is never asked in this phase
    has_title = values.title is not None
    enrichment, error = _decide_enrichment(rules, text, has_title, can_enrich, min_chars)
    return RulesPlan(
        text_hash=raw_text_hash(text),
        values=values,
        status=_decide_status(has_title, enrichment),
        enrichment=enrichment,
        error=error,
    )


def failed_plan(text: str) -> RulesPlan:
    """What is stored when the rules phase itself broke. It can be revived by a retry."""
    nothing = MergedExtraction(
        is_job_posting=True,
        title=None,
        company=None,
        stack=(),
        seniority=None,
        salary_min=None,
        salary_max=None,
        salary_currency=None,
        salary_period=None,
        remote_policy=None,
        apply_email=None,
        apply_url=None,
        conflicts=(),
    )
    return RulesPlan(
        text_hash=raw_text_hash(text),
        values=nothing,
        status=ExtractionStatus.EXTRACTION_FAILED,
        enrichment=LlmEnrichment.SKIPPED,
        error=ExtractionError.UNEXPECTED,
    )


def _decide_enrichment(
    rules: RuleResult, text: str, has_title: bool, can_enrich: bool, min_chars: int
) -> tuple[LlmEnrichment, ExtractionError | None]:
    if not missing_key_fields(rules):
        return LlmEnrichment.NOT_NEEDED, None
    if is_too_short_to_enrich(text, min_chars):
        return LlmEnrichment.SKIPPED, ExtractionError.TOO_SHORT
    if can_enrich or has_title:
        # With a title the job is already usable, so it may wait for a model that is switched
        # on later (the sweep gives up after LLM_ENRICHMENT_GIVE_UP_DAYS).
        return LlmEnrichment.PENDING, None
    # No title and nothing can ever add one: fail at once instead of waiting for nothing.
    return LlmEnrichment.SKIPPED, ExtractionError.MODEL_NOT_ENABLED


def _decide_status(has_title: bool, enrichment: LlmEnrichment) -> ExtractionStatus:
    if has_title:
        return ExtractionStatus.EXTRACTED  # the database requires a title for this status
    if enrichment is LlmEnrichment.PENDING:
        return ExtractionStatus.PENDING  # the model may still find one
    return ExtractionStatus.EXTRACTION_FAILED
