"""The identity of everything one score read, so an unchanged match is never rewritten.

`inputs_hash` is the SHA-256 of a canonical JSON document holding exactly the inputs of
`score_match` and nothing else:

- the CV's skills (as comparison keys, order ignored) and the identity of its vector;
- the user's scored preferences (not the free-text notes: the scorer never reads them);
- the job's scored fields, the identity of its vector, its text hash and the rules version;
- the scorer version and every number in `ScoringConfig` (weights, floor, ceiling, cap).

Change any of them and the hash changes; change anything else (a preference note, a column the
scorer does not read) and it does not. The hash is a digest: it cannot be turned back into a skill.
"""

import hashlib
import json

from app.matching.scoring import (
    SCORER_VERSION,
    CvFacts,
    JobFacts,
    PreferenceFacts,
    ScoringConfig,
    skill_keys,
)

_PRECISION = 6  # decimals kept of the weights, floor and ceiling (floats must hash the same)


def _config_part(config: ScoringConfig) -> list[float | int]:
    return [
        round(config.weight_skills, _PRECISION),
        round(config.weight_embedding, _PRECISION),
        round(config.similarity_floor, _PRECISION),
        round(config.similarity_ceiling, _PRECISION),
        config.failed_filter_cap,
    ]


def scoring_fingerprint(config: ScoringConfig) -> str:
    """A digest of the scorer version and every setting a score depends on.

    Stored in `match_status` after a run. When it differs from the current one (a setting or the
    scorer version changed), the sweep asks for every user to be scored again.
    """
    canonical = json.dumps(
        {"scorer": SCORER_VERSION, "config": _config_part(config)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def match_inputs_hash(
    *, cv: CvFacts, preferences: PreferenceFacts | None, job: JobFacts, config: ScoringConfig
) -> str:
    """A 64-character hex digest of the inputs of one (CV, preferences, job) score."""
    document = {
        "scorer": SCORER_VERSION,
        "config": _config_part(config),
        "cv": {"skills": sorted(skill_keys(cv.skills)), "vector": cv.vector_id},
        "preferences": _preference_part(preferences),
        "job": {
            "stack": sorted(skill_keys(job.stack)),
            "seniority": job.seniority.value if job.seniority else None,
            "remote": job.remote_policy.value if job.remote_policy else None,
            "salary": [
                job.salary_min,
                job.salary_max,
                job.salary_currency,
                job.salary_period.value if job.salary_period else None,
            ],
            "vector": job.vector_id,
            "text": job.raw_text_hash,
            "rules": job.rules_version,
        },
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _preference_part(preferences: PreferenceFacts | None) -> dict[str, object] | None:
    if preferences is None:
        return None
    return {
        "seniority": sorted(level.value for level in preferences.seniority_levels),
        "remote": sorted(policy.value for policy in preferences.remote_policies),
        "salary": [
            preferences.min_salary,
            preferences.salary_currency,
            preferences.salary_period.value if preferences.salary_period else None,
        ],
    }
