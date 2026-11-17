"""The job score: a 0-100 number and the reasons for it. Pure functions, no I/O, no language model.

Read docs/MATCHING.md for the design. In short:

    score = round(100 * (w_skills * skill_coverage + w_embedding * similarity))

capped at `failed_filter_cap` when a hard filter failed. A component that cannot be computed gives
its weight to the other one, and the explanation says so; with neither there is no score at all.

Privacy: this module sees the CV only as a list of skill names and an already computed cosine
similarity, never the CV text (a test checks the service that feeds it). It imports no language
model client and no HTTP library (a test checks it).

Python idiom: `@dataclass(frozen=True)` makes a small immutable record (like a TypeScript
`readonly` interface that also builds its own constructor).
"""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.extraction.stack import canonical_name
from app.matching.reasons import (
    Components,
    EmbeddingComponent,
    FilterOutcome,
    FilterReason,
    FilterResult,
    MatchReasons,
    Note,
    SalaryOfJob,
    SalaryWanted,
    SkillLists,
    SkillsComponent,
    WeightPair,
    Weights,
)
from app.models.job import RemotePolicy, SalaryPeriod, Seniority
from app.models.job_match import MAX_MISSING_SKILLS, MAX_SKILL_NAME_CHARS

# Bump when the formula, a constant below or the filters change. It is part of every stored match's
# inputs_hash, so a new version re-scores everything.
SCORER_VERSION = "score_v1"

# Amounts are compared per year. Fixed constants: 52 weeks, 5 working days, 8 working hours.
PERIODS_PER_YEAR: dict[SalaryPeriod, int] = {
    SalaryPeriod.YEAR: 1,
    SalaryPeriod.MONTH: 12,
    SalaryPeriod.WEEK: 52,
    SalaryPeriod.DAY: 260,
    SalaryPeriod.HOUR: 2080,
}

# --- the language check ------------------------------------------------------------------------
# The embedding model reads English. A post is "English" when enough of its words are English
# function words. The threshold was chosen from invented posts in five languages only, never from
# stored jobs (tests/language_posts.py, docs/MATCHING.md): every English post scored at least 0.267
# and every other-language post at most 0.013; 0.10 sits in the gap, nearer the non-English side so
# an English post that is mostly a technology list keeps its embedding.
ENGLISH_MIN_RATIO = 0.10
ENGLISH_MIN_WORDS = 15  # a shorter text gives no verdict and counts as English
ENGLISH_SAMPLE_CHARS = 1_800  # the part of a post that is embedded
# Function words that are not also common words of German, French, Spanish or Portuguese
# ("a", "in", "an", "on", "as" are left out for that reason).
ENGLISH_WORDS = frozenset(
    "the and of to for with you your we our will are be is at by or from that this have has it "
    "their they who which all can not more us what about if but been would should into also "
    "other".split()
)
_WORD = re.compile(r"[^\W\d_]+")  # runs of letters, accents included


def looks_english(text: str) -> bool:
    """Whether the start of a post reads as English. Short texts give no verdict (True)."""
    words = _WORD.findall(text[:ENGLISH_SAMPLE_CHARS].lower())
    if len(words) < ENGLISH_MIN_WORDS:
        return True
    return sum(word in ENGLISH_WORDS for word in words) / len(words) >= ENGLISH_MIN_RATIO


# --- inputs ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoringConfig:
    """The numbers that shape a score (from settings); all of them are part of the inputs_hash."""

    weight_skills: float
    weight_embedding: float
    similarity_floor: float
    similarity_ceiling: float
    failed_filter_cap: int


@dataclass(frozen=True)
class CvFacts:
    """What the scorer may know about the CV: its skills and which vector it has. Never its text."""

    skills: Sequence[str]
    vector_id: str | None  # "model|text hash" of a vector made by the current model, else None


@dataclass(frozen=True)
class PreferenceFacts:
    """The user's scored preferences. The free-text notes are deliberately not here."""

    seniority_levels: Sequence[Seniority] = ()
    remote_policies: Sequence[RemotePolicy] = ()
    min_salary: int | None = None
    salary_currency: str | None = None
    salary_period: SalaryPeriod | None = None


@dataclass(frozen=True)
class JobFacts:
    """The extracted fields of a job that the score reads, plus what identifies its inputs."""

    stack: Sequence[str]
    seniority: Seniority | None
    remote_policy: RemotePolicy | None
    salary_min: int | None
    salary_max: int | None
    salary_currency: str | None
    salary_period: SalaryPeriod | None
    text_start: str  # the first characters of the post, for the language check
    vector_id: str | None  # "model|text hash" of the job's vector when it is usable, else None
    raw_text_hash: str | None
    rules_version: str | None


@dataclass(frozen=True)
class MatchResult:
    score: int
    filters_failed: bool
    missing_skills: list[str]
    reasons: MatchReasons


# --- skills ------------------------------------------------------------------------------------
def skill_key(name: str) -> str:
    """The comparison key of a skill: the dictionary's canonical name, else the folded text."""
    canonical = canonical_name(name)
    return canonical.casefold() if canonical else " ".join(name.casefold().split())


def _display_name(name: str) -> str:
    return canonical_name(name) or " ".join(name.split())


def _unique_skills(names: Sequence[str]) -> list[tuple[str, str]]:
    """(key, display name) per distinct skill, in the order given."""
    seen: dict[str, str] = {}
    for name in names:
        if name.strip():
            seen.setdefault(skill_key(name), _display_name(name))
    return list(seen.items())


def skill_keys(skills: Sequence[str]) -> frozenset[str]:
    return frozenset(key for key, _ in _unique_skills(skills))


def _bounded(names: list[str]) -> tuple[list[str], bool]:
    """Drop names that are empty or too long, keep at most `MAX_MISSING_SKILLS`."""
    usable = [n for n in names if 0 < len(n) <= MAX_SKILL_NAME_CHARS]
    kept = usable[:MAX_MISSING_SKILLS]
    return kept, len(kept) < len(names)


# --- hard filters ------------------------------------------------------------------------------
def _choice_filter(
    name: Literal["remote_policy", "seniority"], job_value: str | None, wanted: Sequence[str]
) -> FilterResult:
    """Pass when the job's value is one the user listed, fail when it is not. An empty list is
    "no preference", and a post that does not say is unknown: neither can fail."""
    chosen = list(wanted)
    if not chosen:
        return FilterResult(
            name=name, outcome="unknown", job=job_value, preference=chosen, reason="no_preference"
        )
    if job_value is None:
        return FilterResult(
            name=name, outcome="unknown", preference=chosen, reason="job_value_missing"
        )
    outcome: FilterOutcome = "pass" if job_value in chosen else "fail"
    return FilterResult(name=name, outcome=outcome, job=job_value, preference=chosen)


def _offered_salary(job: JobFacts) -> SalaryOfJob | None:
    if job.salary_min is None and job.salary_max is None:
        return None
    return SalaryOfJob(
        min=job.salary_min,
        max=job.salary_max,
        currency=job.salary_currency,
        period=job.salary_period.value if job.salary_period else None,
    )


def salary_filter(job: JobFacts, preferences: PreferenceFacts | None) -> FilterResult:
    """Compare the top of the job's salary range with the user's minimum, both per year.

    Currencies are never converted: a different currency is `unknown`. A range counts by its top
    (`salary_max`, else `salary_min`): it passes when the top reaches the minimum; it fails only
    when the post gave a maximum and even that is too low; "from X" under the minimum stays
    unknown, because the post says the pay may be higher.
    """
    if (
        preferences is None
        or preferences.min_salary is None
        or preferences.salary_currency is None
        or preferences.salary_period is None
    ):
        return FilterResult(name="salary", outcome="unknown", reason="no_preference")
    wanted = SalaryWanted(
        min=preferences.min_salary,
        currency=preferences.salary_currency,
        period=preferences.salary_period.value,
    )
    offered = _offered_salary(job)

    def undecided(reason: FilterReason) -> FilterResult:
        return FilterResult(
            name="salary", outcome="unknown", job=offered, preference=wanted, reason=reason
        )

    top = job.salary_max if job.salary_max is not None else job.salary_min
    if top is None or job.salary_currency is None:
        return undecided("job_value_missing")
    if job.salary_currency != wanted.currency:
        return undecided("currency_differs")
    if job.salary_period is None:
        return undecided("salary_period_missing")
    job_per_year = top * PERIODS_PER_YEAR[job.salary_period]
    wanted_per_year = wanted.min * PERIODS_PER_YEAR[preferences.salary_period]
    outcome: FilterOutcome = "pass"
    reason: FilterReason | None = None
    if job_per_year < wanted_per_year:
        # Only a stated maximum that is too low fails; "from X" may still pay enough.
        outcome = "fail" if job.salary_max is not None else "unknown"
        reason = None if job.salary_max is not None else "open_ended_below_minimum"
    return FilterResult(
        name="salary",
        outcome=outcome,
        job=offered,
        preference=wanted,
        job_per_year=job_per_year,
        preference_per_year=wanted_per_year,
        reason=reason,
    )


def run_filters(job: JobFacts, preferences: PreferenceFacts | None) -> list[FilterResult]:
    prefs = preferences or PreferenceFacts()
    return [
        _choice_filter(
            "remote_policy",
            job.remote_policy.value if job.remote_policy else None,
            [p.value for p in prefs.remote_policies],
        ),
        _choice_filter(
            "seniority",
            job.seniority.value if job.seniority else None,
            [s.value for s in prefs.seniority_levels],
        ),
        salary_filter(job, preferences),
    ]


# --- the score ---------------------------------------------------------------------------------
def normalise_similarity(cosine: float, floor: float, ceiling: float) -> float:
    """Map a cosine similarity to 0..1: the floor and below is 0, the ceiling and above is 1."""
    return min(1.0, max(0.0, (cosine - floor) / (ceiling - floor)))


def _effective_weights(
    config: ScoringConfig, skills_ok: bool, embedding_ok: bool
) -> WeightPair | None:
    if skills_ok and embedding_ok:
        return WeightPair(skills=config.weight_skills, embedding=config.weight_embedding)
    if skills_ok:
        return WeightPair(skills=1.0, embedding=0.0)
    if embedding_ok:
        return WeightPair(skills=0.0, embedding=1.0)
    return None


def score_match(
    *,
    cv: CvFacts,
    preferences: PreferenceFacts | None,
    job: JobFacts,
    cosine: float | None,
    config: ScoringConfig,
) -> MatchResult | None:
    """Score one job for one user, or None when there is nothing to compare at all.

    `cosine` is the cosine similarity of the CV vector and the job vector, or None when either
    vector is missing or made by another model.
    """
    job_skills = _unique_skills(job.stack)
    cv_keys = skill_keys(cv.skills)
    matched = [name for key, name in job_skills if key in cv_keys]
    missing = [name for key, name in job_skills if key not in cv_keys]
    skills_ok = bool(job_skills) and bool(cv_keys)
    english = looks_english(job.text_start)
    embedding_ok = cosine is not None and english

    notes: list[Note] = []
    if not job_skills:
        notes.append("no_job_stack")
    elif not cv_keys:
        notes.append("no_cv_skills")
    if cosine is None:
        notes.append("no_embedding")
    elif not english:
        notes.append("not_english")

    effective = _effective_weights(config, skills_ok, embedding_ok)
    if effective is None:
        return None

    coverage = len(matched) / len(job_skills) if skills_ok else 0.0
    normalised = (
        normalise_similarity(cosine, config.similarity_floor, config.similarity_ceiling)
        if cosine is not None
        else 0.0
    )
    raw = effective.skills * coverage + effective.embedding * (normalised if embedding_ok else 0.0)
    uncapped = min(100, max(0, math.floor(raw * 100 + 0.5)))

    filters = run_filters(job, preferences)
    filters_failed = any(f.outcome == "fail" for f in filters)
    score = min(uncapped, config.failed_filter_cap) if filters_failed else uncapped
    if filters_failed:
        notes.append("filters_failed")

    matched_out, matched_cut = _bounded(matched)
    missing_out, missing_cut = _bounded(missing)
    if matched_cut or missing_cut:
        notes.append("skills_truncated")

    reasons = MatchReasons(
        scorer_version=SCORER_VERSION,
        score=score,
        cap_applied=score < uncapped,
        weights=Weights(
            configured=WeightPair(skills=config.weight_skills, embedding=config.weight_embedding),
            effective=effective,
        ),
        components=Components(
            skills=SkillsComponent(
                coverage=round(coverage, 3),
                matched=len(matched),
                total=len(job_skills),
                cv_skills=len(cv_keys),
            )
            if skills_ok
            else None,
            embedding=EmbeddingComponent(
                cosine=round(cosine, 4),
                floor=config.similarity_floor,
                ceiling=config.similarity_ceiling,
                normalised=round(normalised, 3),
            )
            if cosine is not None
            else None,
        ),
        skills=SkillLists(matched=matched_out, missing=missing_out),
        filters=filters,
        notes=notes,
    )
    return MatchResult(
        score=score, filters_failed=filters_failed, missing_skills=missing_out, reasons=reasons
    )
