# Job matching: how a job gets its 0-100 score

The calibration section (section 2) was written **before** any scoring code or test existed, so the
floor and the ceiling cannot have been tuned to a result. This file describes the scorer built in
Phase 4, PR 4.

No LLM is used anywhere in scoring: the CV must never reach a hosted provider, and nothing here
needs a model that runs at request time (the vectors are stored by the embedding worker, PR 3).

## 1. The score

```
score = round_half_up( 100 * ( w_skills * skill_coverage + w_embedding * similarity ) )
if any hard filter failed: score = min(score, MATCH_FAILED_FILTER_CAP)
```

- `skill_coverage` = job stack entries the CV has / distinct job stack entries. Both sides go
  through `canonical_name()` (the stack dictionary, aliases, case-insensitive); an entry the
  dictionary does not know is compared as its casefolded text.
- `similarity` = `clamp((cosine - floor) / (ceiling - floor), 0, 1)`, where `cosine` is
  `1 - (job.embedding <=> cv.embedding)`.
- Hard filters (remote policy, seniority, salary) each return `pass`, `fail` or `unknown`. `unknown`
  never fails and never changes the score. A job with a failed filter is **capped, not deleted**; the
  list hides it by default and shows it with `include_filtered` (PR 5).

**Required and optional skills weigh the same.** A job's stack is a list of technologies the post
names. Posts do not reliably say which are required and which are "nice to have", the extraction
rules do not separate them, and guessing would invent precision. So every entry counts once. A post
that lists ten technologies where two are essential gives a lower coverage than it deserves; the
explanation always shows which skills matched and which are missing, so the reader can judge.

Configured weights (`MATCH_WEIGHT_SKILLS` 0.6, `MATCH_WEIGHT_EMBEDDING` 0.4) are replaced by
effective weights when a component cannot be computed: no stack on the job or no skills on the CV
moves the whole weight to the embedding; no comparable vector, or a post that does not look
English, moves it to the skills; with neither there is no score and no row is written.

## 2. Calibration of the floor and the ceiling

`floor` and `ceiling` come from the stored jobs only, with no relevance judgement and no CV:

- `floor` = P50 of the cosine similarity of random pairs of jobs ("no more similar than two jobs
  chosen at random: no signal"),
- `ceiling` = P99 of the same pairs ("closer than 99 % of random job pairs").

Script: `scripts/calibrate_similarity.py` (read-only database connection, no provider request, the
invented CV only). Sample: one job per distinct post text (the same post stored by two sources is
not compared with itself), extracted, with a vector of the current model, 5,000 random pairs, seed 1.
Run on 2026-11-17 against the development database, in the Docker image (the real model, offline):

```text
model BAAI/bge-small-en-v1.5@aa8f8b060edb00e03bfdd08813a2949946c8ba55
102 jobs with distinct text, 5000 random pairs, seed 1

Random job-job pairs (this is what floor and ceiling come from)
  n = 5000   mean = 0.7188
    min  0.5791
     P1  0.6225
     P5  0.6444
    P10  0.6578
    P25  0.6846
    P50  0.7147
    P75  0.7462
    P90  0.7825
    P95  0.8090
    P99  0.8602
  P99.9  0.9685
    max  0.9982

Chosen: floor = P50 = 0.715, ceiling = P99 = 0.86
```

**Chosen defaults (fixed before any scoring test):** `MATCH_SIMILARITY_FLOOR=0.715`,
`MATCH_SIMILARITY_CEILING=0.86`.

### What the synthetic CV looks like against the same jobs (statement only, not used to tune)

The invented CV `tests/fixtures/cv/alex_example.txt`, embedded exactly like an upload (clean, mask,
chunk, embed, normalised mean), against all 102 distinct-text stored jobs:

```text
Synthetic CV (alex_example) against every distinct stored job
  n = 102   mean = 0.6975
    min  0.5633
     P1  0.5728
     P5  0.6205
    P10  0.6325
    P25  0.6634
    P50  0.6975
    P75  0.7279
    P90  0.7621
    P95  0.7777
    P99  0.8174
  P99.9  0.8238
    max  0.8245
  jobs above the floor (0.7150):   39 of 102
  jobs at/above the ceiling (0.8600): 0 of 102
```

Reading it plainly, without changing anything:

- The whole distribution does **not** sit below the floor (39 of 102 jobs are above it), so the
  stop rule did not trigger.
- The CV's median (0.6975) is slightly below the floor (0.715): a CV and a job are different kinds
  of text, so CV-job cosines run a little lower than job-job ones, as expected.
- No job reaches the ceiling: the best similarity a job can get for this CV is
  (0.8245 - 0.715) / (0.86 - 0.715) = 0.755, so the embedding part alone cannot reach 100 for this
  CV. That is a property of the calibration chosen in advance, and it is **not** adjusted.
- The upper tail of the job-job pairs (P99.9 0.9685, max 0.9982) comes from near-duplicate posts
  (the same role posted twice with small differences); P99 is the ceiling, so those pairs sit above it.

Any later change to the floor, the ceiling or the weights is a new `SCORER_VERSION` and a recorded
decision, never a quiet edit (the version is part of every stored `inputs_hash`).

### How to read a score (in plain words)

- **A score ranks jobs for one CV. It is not a probability.** 68 does not mean "68 % chance of a
  fit", and a 68 for one user's CV says nothing about a 68 for another user's. Only the order of the
  jobs for one CV, and the explanation next to each score, carry meaning.
- **The embedding half runs low for CV-job pairs.** The floor and the ceiling were calibrated on
  job-job pairs, because that is the only thing that can be measured without judging relevance. A CV
  and a job post are different kinds of text, so their cosines sit lower: for the invented Alex CV the
  median is 0.698 against a floor of 0.715, and the maximum is 0.825 against a ceiling of 0.86. Most
  CV-job pairs therefore get a similarity part near 0, and none gets 1.
- **This was found at calibration time and was not tuned.** The numbers above were read after the
  floor and the ceiling were fixed, and nothing was changed in response (no re-tuning, no new
  `SCORER_VERSION`).
- **PR 6 will measure whether a CV-job recalibration helps**, on held-out labels (jobs a person
  judged for a CV, kept apart from anything used to choose a constant). Until that measurement
  exists, the calibration stays as it is.

## 3. Missing inputs

| Situation | Effective weights | The explanation says |
|---|---|---|
| The job lists no stack | skills 0, embedding 1 | note `no_job_stack` |
| The CV has no skills (for example one stored before PR 2) | skills 0, embedding 1 | note `no_cv_skills` |
| The CV or the job has no vector of the current model | skills 1, embedding 0 | note `no_embedding` |
| The post does not look English | skills 1, embedding 0 | note `not_english` (the cosine is still shown) |
| None of the above leaves a component | no score | no row is written (a row that existed is removed) |
| The user has no preferences row | the score is unchanged | every filter is `unknown`, reason `no_preference` |

A failed hard filter adds the note `filters_failed`. `NOTE_MESSAGES` (in `app/matching/reasons.py`)
has a sentence for each note; the endpoints of PR 5 will show them.

**Language.** The embedding model reads English. `looks_english` counts how many of the words in the
first 1,800 characters are English function words (the, and, of, to, for, with, ...). It is a pure
function: no dictionary download, no new dependency. The threshold is **0.10**, and a text of fewer
than 15 words gives no verdict (treated as English). It was fixed from **invented posts only**
(`tests/language_posts.py`: English, German, French, Spanish and Portuguese, prose and bullet-style
lists full of English technology names), never from a stored job:

| Group | Ratio of English function words |
|---|---|
| English posts (prose 0.466, bullets 0.271, short stub 0.267) | at least 0.267 |
| Other-language posts (7 posts: German, French, Spanish, Portuguese) | at most 0.013 |
| Threshold | 0.10: in the gap and nearer the non-English side, so an English post that is mostly a technology list keeps its embedding |

The skills part does not depend on language: the technology dictionary is language-neutral, so a
German post that names Python and Docker still gets its coverage. **Known gap:** a non-English CV
cannot be detected, because the scorer never reads CV text. The skills still work; the embedding
similarity of such a CV is simply less reliable. An upload-time language flag was considered and left
out.

## 4. The stored explanation

`job_matches.reasons` is validated by a Pydantic model (`MatchReasons`, `extra="forbid"`). It holds
only: matched and missing skills (the **job's own stack entries**, never the CV's other skills), a
count of the CV's skills, every filter with the values it compared, the component values, the
configured and effective weights, notes (fixed codes) and the scorer version. It never holds CV text
or preference notes (a test plants a canary in both and in a CV skill and checks it never appears).
Parts that do not apply are left out of the JSON instead of being `null`.

The worked example, with the **measured** floor and ceiling. The CV is the invented Alex CV through
the real skills extractor (15 skills); the preferences are senior or lead, remote, at least 100,000
USD per year; the two cosines (0.78 and 0.58) are invented inputs. These are the files compared in
`tests/unit/test_alex_vs_two_jobs.py`, worked out by hand before the scorer produced them (job A:
coverage 5/6, similarity (0.78 - 0.715) / 0.145 = 0.448, 0.6 * 0.8333 + 0.4 * 0.4483 = 0.679, score
68; job B: coverage 0, cosine under the floor, score 0).

Job A, Python / FastAPI / PostgreSQL / Docker / Kubernetes / Terraform, remote, senior, up to 130,000
USD a year:

```json
{
  "scorer_version": "score_v1", "score": 68, "cap_applied": false,
  "weights": {"configured": {"skills": 0.6, "embedding": 0.4}, "effective": {"skills": 0.6, "embedding": 0.4}},
  "components": {
    "skills": {"coverage": 0.833, "matched": 5, "total": 6, "cv_skills": 15},
    "embedding": {"cosine": 0.78, "floor": 0.715, "ceiling": 0.86, "normalised": 0.448}
  },
  "skills": {"matched": ["Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes"], "missing": ["Terraform"]},
  "filters": [
    {"name": "remote_policy", "outcome": "pass", "job": "remote", "preference": ["remote"]},
    {"name": "seniority", "outcome": "pass", "job": "senior", "preference": ["senior", "lead"]},
    {"name": "salary", "outcome": "pass", "job": {"max": 130000, "currency": "USD", "period": "year"},
     "preference": {"min": 100000, "currency": "USD", "period": "year"},
     "job_per_year": 130000, "preference_per_year": 100000}
  ],
  "notes": []
}
```

Job B, Java / Spring / Angular / MySQL, on-site, junior, up to 40,000 EUR a year:

```json
{
  "scorer_version": "score_v1", "score": 0, "cap_applied": false,
  "weights": {"configured": {"skills": 0.6, "embedding": 0.4}, "effective": {"skills": 0.6, "embedding": 0.4}},
  "components": {
    "skills": {"coverage": 0.0, "matched": 0, "total": 4, "cv_skills": 15},
    "embedding": {"cosine": 0.58, "floor": 0.715, "ceiling": 0.86, "normalised": 0.0}
  },
  "skills": {"matched": [], "missing": ["Java", "Spring", "Angular", "MySQL"]},
  "filters": [
    {"name": "remote_policy", "outcome": "fail", "job": "onsite", "preference": ["remote"]},
    {"name": "seniority", "outcome": "fail", "job": "junior", "preference": ["senior", "lead"]},
    {"name": "salary", "outcome": "unknown", "job": {"max": 40000, "currency": "EUR", "period": "year"},
     "preference": {"min": 100000, "currency": "USD", "period": "year"}, "reason": "currency_differs"}
  ],
  "notes": ["filters_failed"]
}
```

Job B already scores 0, so the cap of 30 changes nothing there (`cap_applied` is false); a job that
would have scored 80 with the same failed filters would be stored as 30 with `cap_applied` true.

**Filters, exactly.** Each returns `pass`, `fail` or `unknown`; `unknown` never fails and never
changes the score. Remote policy and seniority: no preference or a post that does not say is
`unknown`; otherwise `pass` when the job's value is in the user's list, else `fail`. Salary: amounts
are compared per year (month 12, week 52, day 260, hour 2,080); currencies are never converted
(`currency_differs` is `unknown`); a range is judged by its top (`salary_max`, else `salary_min`); it
fails only when the post states a maximum that is below the minimum, and "from X" under the minimum
stays `unknown` (`open_ended_below_minimum`).

**Bounds.** A skill name over 100 characters is left out of the stored lists, and each list keeps at
most 30 names (a note `skills_truncated` says so); a check constraint also caps `missing_skills` at
30. The coverage still counts every entry.

## 5. `job_matches` and `inputs_hash`

One row per (user, job): `score` (0 to 100, `SMALLINT`, checked), `filters_failed`, `missing_skills`,
`reasons`, `scorer_version`, `inputs_hash`, timestamps. Unique on (`user_id`, `job_id`); indexed on
(`user_id`, `score`) for the ranking and on `job_id`; both foreign keys cascade on delete.

`inputs_hash` is the SHA-256 of canonical JSON of exactly what the score read: the CV's skills (as
comparison keys, order and case ignored) and the identity of its vector (`model|text hash`); the
user's five scored preference fields (**not** the notes); the job's stack, seniority, remote policy,
salary fields, vector identity, `raw_text_hash` and rules version; the scorer version; and the
weights, floor, ceiling and cap. A run computes the hash first and skips the job when the stored one
is equal, so nothing is written (and nothing is scored) twice. The upsert is
`INSERT ... ON CONFLICT (user_id, job_id) DO UPDATE ... WHERE job_matches.inputs_hash <> EXCLUDED.inputs_hash`.
Change a setting, the scorer version, a CV skill, a preference value, a job field or a vector, and the
affected matches are rewritten on the next run; change a preference note and nothing happens.

**Whose matches.** Every repository function takes the `user_id` and filters on it; jobs are reached
only through `sources.user_id`; the list joins jobs and sources again, so even a stray row pointing
at another user's job is never returned. A single-job request for another user's job is the same
"not found" as a missing one (PR 5's endpoints will answer 404, never 403).

## 6. Retrieval and why there is no vector index yet

Two measured findings, not assumptions, are the reason for no vector index for now:

1. **The user filter does not shrink the scan.** The planner reads the whole `jobs` table and joins
   `sources`, so the cost follows all users' jobs, not the one user's (table below).
2. **An approximate (HNSW) index returned 0 rows for a user with 50 jobs** among 20,050, because the
   index picks its nearest neighbours across every user first and the `WHERE` removes them afterwards.

```sql
SELECT ... FROM jobs j JOIN sources s ON s.id = j.source_id
WHERE s.user_id = :user AND j.extraction_status = 'extracted'
  AND j.embedding IS NOT NULL AND j.embedding_model = :current_model
ORDER BY j.embedding <=> :cv_vector, j.id LIMIT :MATCH_CANDIDATES;
```

An exact scan (`<=>` is cosine distance). Measured with `EXPLAIN (ANALYZE)` on 2026-11-17:

| Data | Rows the scan sees | Execution time (warm) |
|---|---|---|
| My stored jobs, the user with the most embedded jobs | 95 candidates (290 jobs in the table) | about 2.5 ms |
| Throwaway database, one user with 20,000 jobs, LIMIT 200 | 20,000 | about 135 ms |
| Same database, a user with 50 jobs among 20,050 | 50 (the planner scans the table) | about 9 ms |

An approximate index would be wrong here, not just unnecessary. On the same throwaway data, after
`CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`, the same query for the user with 50
jobs returned **0 rows**: the index finds its nearest neighbours across everybody's jobs first and
the `WHERE` clause removes them afterwards, so a user whose jobs are not near the top gets nothing.
(pgvector's iterative scan could mitigate it; it is not needed at this size.) Revisit when one
scoring query passes about 100 ms for a typical user, or the `jobs` table passes about 100,000 rows:
the planner scans the whole table when it prefers to, so the cost grows with all users' jobs.

Jobs without a vector of the current model are not candidates until the embedding sweep embeds them.
A CV without a usable vector scores the newest `MATCH_CANDIDATES` jobs on skills alone.

## 7. Privacy

- No language model, no network: `app/matching/` and the match service import no `app.llm` and no
  HTTP library, directly or indirectly (`tests/unit/test_matching_isolation.py`).
- The CV text is never read by scoring. The service selects the CV's `skills`, vector and vector
  identity only, and the preferences without `notes`; a test records the SQL and checks it, and the
  input records (`CvFacts`, `PreferenceFacts`) have no field that could hold either.
- The display-only suggestions (years of experience, seniority guess) are not read: the existing
  scan of "every scoring module" now also covers the match service and repository.
- Logs carry the user id, counts and timing only. The explanation holds job skill names, enum
  values and fixed codes.

## 8. Limits

- Required and optional skills weigh the same (section 1).
- A CV vector is the mean of at most 6 chunks (PR 3), so it blurs on long CVs; the skills part exists
  partly for that reason.
- With the calibration chosen in advance, the embedding part of the score cannot reach 100 for the
  synthetic CV (section 2). Any change would be a new `SCORER_VERSION`.
- Only the nearest `MATCH_CANDIDATES` jobs are scored per run; a job outside them has no match until
  the CV or the jobs change enough to bring it in.
- A stale match row stays until its job is re-scored, becomes unscorable (the row is removed) or the
  job is deleted; the list only returns matches of jobs that are still extracted.
- Nothing runs the scorer by itself yet: the Celery task, the triggers and the endpoints are PR 5.
