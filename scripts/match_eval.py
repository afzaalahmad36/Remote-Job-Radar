"""Shared pieces of the matching evaluation (protocol: docs/MATCHING.md, section 10).

Pure functions and plain records, no database and no model: the three eval_match_* scripts do the
input and output. Nothing here makes a provider request or imports a language-model client.

What is computed, per persona (an invented CV plus fixed preferences, or the owner's own CV):

* five rankings of the frozen pool: the shipped hybrid score (A), the hybrid with a floor and a
  ceiling taken from OTHER personas' CV-job cosines (B), skills only, embedding only, newest first;
* a blind sheet: the top 5 of each ranking, the documented worked examples, and 10 random jobs,
  shuffled with a fixed seed and numbered;
* an answer key, kept apart from the sheets, that says where every job stood in every ranking.

"Skills only" uses the shipped scorer with no cosine, so the hard filters (and their cap) still
apply to it; "embedding only" is the raw cosine and applies no filter. Ties are broken by the newer
job and then by job id, so a ranking is reproducible.
"""

import hashlib
import random
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from app.matching.scoring import (
    ENGLISH_SAMPLE_CHARS,
    SCORER_VERSION,
    CvFacts,
    JobFacts,
    PreferenceFacts,
    ScoringConfig,
    score_match,
)
from app.models.job import RemotePolicy, SalaryPeriod, Seniority
from scripts.calibrate_similarity import cosine, percentile
from scripts.db_guard import ScriptError

# The frozen pool of docs/MATCHING.md ("The frozen pool snapshot"): recorded before any sheet.
PROTOCOL_POOL_HASH = "7cd0b36da7ede96de6a8e61396ef0ca969958237bb3c67ba25f42cce0183470c"
PROTOCOL_POOL_SIZE = 108

TOP_N = 5  # the top of each method that goes on a sheet
RANDOM_N = 10  # random jobs per sheet
DEFAULT_SEED = 1
SHEET_TEXT_CHARS = 1_500  # the start of the post shown to the labeller
B_FLOOR_PERCENTILE = 50.0  # the rule of section 2, applied to CV-job cosines
B_CEILING_PERCENTILE = 99.0

# The methods whose top jobs go on a sheet (random is drawn separately).
METHODS = ("hybrid_a", "hybrid_b", "skills_only", "embedding_only", "newest_first")

LABELS = ("relevant", "maybe", "no")

# The nine worked examples of the protocol (first 8 characters of the job id -> documented label).
# They stay on the sheets and the key marks them `seen_example`.
SEEN_EXAMPLES: Mapping[str, Mapping[str, str]] = {
    "alex": {"4cf4674d": "relevant", "14a2b4d7": "maybe", "ce090770": "no"},
    "sam": {"ac5106d6": "relevant", "3c85f760": "maybe", "0c02a211": "no"},
    "jordan": {"14a2b4d7": "relevant", "7b9c2242": "maybe", "0922063b": "no"},
}

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")  # becomes a file name: nothing else is allowed

PersonaKind = Literal["synthetic", "real"]


@dataclass(frozen=True)
class PoolJob:
    id: str
    title: str
    company: str | None
    seniority: Seniority | None
    remote_policy: RemotePolicy | None
    stack: tuple[str, ...]
    salary_min: int | None
    salary_max: int | None
    salary_currency: str | None
    salary_period: SalaryPeriod | None
    posted_at: datetime | None
    created_at: datetime
    text: str  # the start of the (public) post
    raw_text_hash: str | None
    rules_version: str | None
    vector: Sequence[float]

    @property
    def recency(self) -> float:
        return (self.posted_at or self.created_at).timestamp()


@dataclass(frozen=True)
class PersonaSpec:
    """What a persona file says. `cv_text` is the CV (invented or the owner's): never printed."""

    name: str
    kind: PersonaKind
    cv_text: str
    preferences: PreferenceFacts


@dataclass(frozen=True)
class Persona:
    """A persona ready to score: skills read from the masked CV text, and the CV vector."""

    name: str
    kind: PersonaKind
    skills: tuple[str, ...]
    vector: Sequence[float]
    preferences: PreferenceFacts


@dataclass(frozen=True)
class PersonaPlan:
    """Everything decided for one persona. Nothing here holds CV text or skills."""

    name: str
    kind: PersonaKind
    b_floor: float
    b_ceiling: float
    rankings: Mapping[str, list[str]]  # method -> every pool job id, best first
    union: list[str]
    random_ids: list[str]
    examples: Mapping[str, str]  # job id -> documented label
    items: list[str]  # job ids in sheet order: item number = position + 1


# --- the pool ----------------------------------------------------------------------------------


def pool_hash(ids: Sequence[str]) -> str:
    """SHA-256 of the sorted ids joined with one newline (the identity in docs/MATCHING.md)."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def check_pool(pool: Sequence[PoolJob], expected_hash: str = PROTOCOL_POOL_HASH) -> None:
    """Refuse a pool that is not the frozen one."""
    actual = pool_hash([job.id for job in pool])
    if actual != expected_hash:
        raise ScriptError(
            f"the pool is not the frozen one ({len(pool)} jobs, hash {actual[:12]}...; "
            f"the protocol recorded {expected_hash[:12]}...): nothing was built"
        )


# --- personas ----------------------------------------------------------------------------------


def _enum_list[E](make: Callable[[str], E], values: object, field: str) -> tuple[E, ...]:
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ScriptError(f"{field} must be a list of text values")
    try:
        return tuple(make(v) for v in values)
    except ValueError:
        raise ScriptError(f"{field} has a value that is not allowed") from None


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def load_personas(path: Path, *, local_root: Path | None = None) -> list[PersonaSpec]:
    """Read a persona file (TOML). A CV path is relative to the file. Messages never quote a CV.

    Without `local_root` every persona is synthetic (a "real" kind is refused: an owner's CV must
    not be named in a file that could be committed). With `local_root` (the git-ignored `.local/`),
    both the file and every CV it names must lie inside it, and every persona is a real one.
    """
    if local_root is not None and not _inside(path, local_root):
        raise ScriptError("a real-CV persona file must be inside .local/")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        raise ScriptError(f"cannot read the persona file {path.name}") from None
    entries = data.get("persona")
    if not isinstance(entries, list) or not entries:
        raise ScriptError(f"{path.name} has no [[persona]] entry")
    specs = [_load_one(entry, path.parent, local_root) for entry in entries]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ScriptError("persona names must be unique")
    return specs


def _load_one(entry: object, base: Path, local_root: Path | None) -> PersonaSpec:
    if not isinstance(entry, dict):
        raise ScriptError("a persona entry must be a table")
    name = entry.get("name")
    if not isinstance(name, str) or not _NAME.match(name):
        raise ScriptError("a persona name must be lower case letters, digits, - or _")
    kind: PersonaKind = "synthetic" if local_root is None else "real"
    if entry.get("kind", kind) != kind:
        raise ScriptError(f"persona {name}: kind must be {kind} in this file")
    cv_file = entry.get("cv")
    if not isinstance(cv_file, str):
        raise ScriptError(f"persona {name}: cv must be a file name")
    cv_path = base / cv_file
    if local_root is not None and not _inside(cv_path, local_root):
        raise ScriptError(f"persona {name}: the CV file must be inside .local/")
    try:
        cv_text = cv_path.read_text(encoding="utf-8")
    except OSError:
        raise ScriptError(f"persona {name}: cannot read the CV file") from None
    return PersonaSpec(name, kind, cv_text, _preferences(name, entry))


def _preferences(name: str, entry: Mapping[str, object]) -> PreferenceFacts:
    minimum = entry.get("min_salary")
    currency = entry.get("salary_currency")
    period = entry.get("salary_period")
    given = [v is not None for v in (minimum, currency, period)]
    if any(given) and not all(given):
        raise ScriptError(f"persona {name}: a salary needs min_salary, currency and period")
    if minimum is not None and (not isinstance(minimum, int) or isinstance(minimum, bool)):
        raise ScriptError(f"persona {name}: min_salary must be a whole number")
    if currency is not None and not isinstance(currency, str):
        raise ScriptError(f"persona {name}: salary_currency must be text")
    return PreferenceFacts(
        seniority_levels=_enum_list(
            Seniority, entry.get("seniority_levels", []), "seniority_levels"
        ),
        remote_policies=_enum_list(
            RemotePolicy, entry.get("remote_policies", []), "remote_policies"
        ),
        min_salary=minimum,
        salary_currency=currency.upper() if isinstance(currency, str) else None,
        salary_period=SalaryPeriod(period) if isinstance(period, str) else None,
    )


# --- scoring one persona against the pool -------------------------------------------------------


def job_facts(job: PoolJob) -> JobFacts:
    return JobFacts(
        stack=job.stack,
        seniority=job.seniority,
        remote_policy=job.remote_policy,
        salary_min=job.salary_min,
        salary_max=job.salary_max,
        salary_currency=job.salary_currency,
        salary_period=job.salary_period,
        text_start=job.text[:ENGLISH_SAMPLE_CHARS],
        vector_id="pool",
        raw_text_hash=job.raw_text_hash,
        rules_version=job.rules_version,
    )


def cv_job_cosines(persona: Persona, pool: Sequence[PoolJob]) -> list[float]:
    return [cosine(persona.vector, job.vector) for job in pool]


def b_constants(
    held_out: Persona, personas: Sequence[Persona], pool: Sequence[PoolJob]
) -> tuple[float, float]:
    """Floor and ceiling for method B: P50 and P99 of the CV-job cosines of the OTHER synthetic
    personas. The held-out persona's own cosines never take part (and a real CV never does)."""
    values = sorted(
        c
        for other in personas
        if other.kind == "synthetic" and other.name != held_out.name
        for c in cv_job_cosines(other, pool)
    )
    if not values:
        raise ScriptError(f"persona {held_out.name}: no other synthetic persona to calibrate from")
    floor = percentile(values, B_FLOOR_PERCENTILE)
    ceiling = percentile(values, B_CEILING_PERCENTILE)
    if floor >= ceiling:
        raise ScriptError(f"persona {held_out.name}: the calibration gave floor >= ceiling")
    return floor, ceiling


def _order(values: Mapping[str, float], pool: Sequence[PoolJob]) -> list[str]:
    """Job ids, best value first; ties go to the newer job, then to the smaller id."""
    by_id = {job.id: job for job in pool}
    return sorted(values, key=lambda i: (-values[i], -by_id[i].recency, i))


def rank_methods(
    persona: Persona,
    pool: Sequence[PoolJob],
    config: ScoringConfig,
    b_floor: float,
    b_ceiling: float,
) -> dict[str, list[str]]:
    """The five full rankings of the pool for one persona."""
    cv = CvFacts(skills=persona.skills, vector_id="persona")
    config_b = ScoringConfig(
        weight_skills=config.weight_skills,
        weight_embedding=config.weight_embedding,
        similarity_floor=b_floor,
        similarity_ceiling=b_ceiling,
        failed_filter_cap=config.failed_filter_cap,
    )
    hybrid_a: dict[str, float] = {}
    hybrid_b: dict[str, float] = {}
    skills: dict[str, float] = {}
    embedding: dict[str, float] = {}
    newest: dict[str, float] = {}
    for job in pool:
        facts = job_facts(job)
        cos = cosine(persona.vector, job.vector)
        for target, cfg, value in (
            (hybrid_a, config, cos),
            (hybrid_b, config_b, cos),
            (skills, config, None),
        ):
            result = score_match(
                cv=cv, preferences=persona.preferences, job=facts, cosine=value, config=cfg
            )
            target[job.id] = float(result.score) if result is not None else -1.0
        embedding[job.id] = cos
        newest[job.id] = job.recency
    return {
        "hybrid_a": _order(hybrid_a, pool),
        "hybrid_b": _order(hybrid_b, pool),
        "skills_only": _order(skills, pool),
        "embedding_only": _order(embedding, pool),
        "newest_first": _order(newest, pool),
    }


# --- the sheet ---------------------------------------------------------------------------------


def resolve_examples(name: str, pool_ids: Sequence[str]) -> dict[str, str]:
    """The documented worked examples of a persona as full job ids (each is one pool job)."""
    found: dict[str, str] = {}
    for prefix, label in SEEN_EXAMPLES.get(name, {}).items():
        matches = [i for i in pool_ids if i.startswith(prefix)]
        if len(matches) != 1:
            raise ScriptError(f"worked example {prefix} of {name} is not exactly one pool job")
        found[matches[0]] = label
    return found


def plan_persona(
    persona: Persona,
    rankings: Mapping[str, list[str]],
    pool_ids: Sequence[str],
    b: tuple[float, float],
    seed: int,
) -> PersonaPlan:
    """The sheet content of one persona: the tops, the examples, a random draw, shuffled."""
    union = sorted({job_id for method in METHODS for job_id in rankings[method][:TOP_N]})
    examples = resolve_examples(persona.name, pool_ids)
    taken = set(union) | set(examples)
    rest = sorted(i for i in pool_ids if i not in taken)
    if len(rest) < RANDOM_N:
        raise ScriptError(f"persona {persona.name}: not enough jobs left for the random draw")
    random_ids = sorted(random.Random(f"{seed}:{persona.name}:random").sample(rest, RANDOM_N))
    items = sorted(taken | set(random_ids))
    random.Random(f"{seed}:{persona.name}:shuffle").shuffle(items)
    return PersonaPlan(
        name=persona.name,
        kind=persona.kind,
        b_floor=b[0],
        b_ceiling=b[1],
        rankings=dict(rankings),
        union=union,
        random_ids=random_ids,
        examples=examples,
        items=items,
    )


def plan_all(
    personas: Sequence[Persona], pool: Sequence[PoolJob], config: ScoringConfig, seed: int
) -> list[PersonaPlan]:
    pool_ids = sorted(job.id for job in pool)
    plans = []
    for persona in personas:
        b = b_constants(persona, personas, pool)
        rankings = rank_methods(persona, pool, config, b[0], b[1])
        plans.append(plan_persona(persona, rankings, pool_ids, b, seed))
    return plans


# The only fields of a job a sheet may carry: no rank, no score, no method, no group.
SHEET_ITEM_FIELDS = (
    "item",
    "job_id",
    "title",
    "company",
    "seniority",
    "remote_policy",
    "stack",
    "salary_min",
    "salary_max",
    "salary_currency",
    "salary_period",
    "text",
)


def sheet_document(plan: PersonaPlan, pool: Sequence[PoolJob]) -> dict[str, object]:
    by_id = {job.id: job for job in pool}
    items = []
    for number, job_id in enumerate(plan.items, start=1):
        job = by_id[job_id]
        items.append(
            {
                "item": number,
                "job_id": job.id,
                "title": job.title,
                "company": job.company,
                "seniority": job.seniority.value if job.seniority else None,
                "remote_policy": job.remote_policy.value if job.remote_policy else None,
                "stack": list(job.stack),
                "salary_min": job.salary_min,
                "salary_max": job.salary_max,
                "salary_currency": job.salary_currency,
                "salary_period": job.salary_period.value if job.salary_period else None,
                "text": job.text[:SHEET_TEXT_CHARS],
            }
        )
    return {"persona": plan.name, "pool_hash": pool_hash([j.id for j in pool]), "items": items}


def key_document(
    plans: Sequence[PersonaPlan], pool: Sequence[PoolJob], config: ScoringConfig, seed: int
) -> dict[str, object]:
    """The answer key: where every job stood in every ranking, and what each item is."""
    personas: dict[str, object] = {}
    for plan in plans:
        personas[plan.name] = {
            "kind": plan.kind,
            "b_floor": plan.b_floor,
            "b_ceiling": plan.b_ceiling,
            "items": {str(n): job_id for n, job_id in enumerate(plan.items, start=1)},
            "union": plan.union,
            "random": plan.random_ids,
            "seen_examples": dict(plan.examples),
            "rankings": {method: plan.rankings[method] for method in METHODS},
        }
    return {
        "protocol": "docs/MATCHING.md section 10",
        "scorer_version": SCORER_VERSION,
        "seed": seed,
        "top_n": TOP_N,
        "random_n": RANDOM_N,
        "config": {
            "weight_skills": config.weight_skills,
            "weight_embedding": config.weight_embedding,
            "similarity_floor": config.similarity_floor,
            "similarity_ceiling": config.similarity_ceiling,
            "failed_filter_cap": config.failed_filter_cap,
        },
        "pool": {
            "count": len(pool),
            "hash": pool_hash([j.id for j in pool]),
            "ids": sorted(j.id for j in pool),
        },
        "personas": personas,
    }


def describe(plans: Sequence[PersonaPlan], pool: Sequence[PoolJob]) -> list[str]:
    """Counts only: no job id, title, skill or CV text."""
    lines = [f"pool: {len(pool)} jobs, hash {pool_hash([j.id for j in pool])[:12]}... (frozen)"]
    for plan in plans:
        tops = len(plan.union)
        extra = len(set(plan.examples) - set(plan.union))
        lines.append(
            f"{plan.name} ({plan.kind}): {len(plan.items)} items = {tops} distinct jobs in the "
            f"tops (top {TOP_N} of {len(METHODS)} methods, {len(METHODS) * TOP_N - tops} repeats) "
            f"+ {extra} worked examples not in the tops + {len(plan.random_ids)} random; "
            f"B floor {plan.b_floor:.4f}, ceiling {plan.b_ceiling:.4f}"
        )
    return lines
