# Job matching: how a job gets its 0-100 score

The calibration section (section 2) was written **before** any scoring code or test existed, so the
floor and the ceiling cannot have been tuned to a result. This file describes the scorer built in
Phase 4, PR 4, and how its results are kept fresh and served (section 9, PR 5).

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
- A request is stored in the database before anything is published, so a lost message is recovered by
  the sweep. A request that was never recorded (the database was down for every retry of an
  `embed_job` task after its vector was committed) is found by the sweep's fourth step: an up-to-date
  user with a job embedded after their last run finished. That step looks at each up-to-date user's
  jobs once per pass without an index on `jobs.embedded_at`; add one if the table grows large. An
  embedding that was merely re-checked also refreshes `embedded_at`, which can cause one cheap extra
  run (unchanged scores are not rewritten).
- Scores are a few seconds to a few minutes behind a change (a 5-second publish delay, then the
  queue); `scoring` in the API says so. A deployment with several API processes still has the
  per-process rate limiter (see CLAUDE.md), including the one on `POST /matches/rescore`.

## 9. Keeping matches fresh (Phase 4, PR 5)

Scores are computed in the background, stored, and only read by the API. Reading never recomputes.

### The request is stored first

`match_status` holds one row per user: `requested_seq`, `finished_seq`, timestamps, a failure count and
the fingerprint (scorer version plus settings) of the last finished run. It holds counters, timestamps
and fixed codes only: no CV text, no skills, no notes.

Anything that makes a user's matches stale raises `requested_seq` **in the same transaction as the
change itself**, so the change and the request land together, and a lost Celery message or a Redis
outage cannot lose the request. A run reads `requested_seq` when it starts and, when it finishes,
sets `finished_seq` to that number. A change that arrives during the run raised `requested_seq` past
it, so the user is still *pending* after the run ends (counters, not clocks, decide this: two
statements of one transaction can carry the same clock reading). Redis only delays and coalesces
messages; this row is the truth.

| What changed | Request raised by | Task published by |
|---|---|---|
| `PUT /cv` (new CV) | the upload's transaction (clears failures) | `embed_cv` when it stores the vector; the endpoint itself when `EMBEDDING_PROVIDER=none` |
| `PUT /cv/skills`, `PUT /preferences` | the same transaction (clears failures) | the endpoint, after the commit |
| `DELETE /cv` | the same transaction; the user's matches are deleted in it too | the endpoint (the run finds no CV and settles) |
| a job was embedded (`embed_job`) | the task, for the job's owner | the task |
| a job was extracted, with embeddings off | the `extract_job` task, for the owner | the task |
| `POST /matches/rescore` | the endpoint (clears failures) | the endpoint, before the commit |
| settings or `SCORER_VERSION` changed | the sweep (see below) | the sweep |

The scoring task is `score_user_matches(user_id)` on the main `celery` queue: it reads stored vectors
and never loads the embedding model (a test imports the scoring modules in a clean interpreter and
checks that `fastembed`, `onnxruntime` and `tokenizers` stay unloaded). **Its only argument is the user
id.** CV text, skills and preference notes are never a task argument, a Redis key or a Redis value:
the dedup key is `match:queued:<user id>` holding `1`, the run lock is `lock:match:<user id>`.

### Coalescing

A publish waits 5 seconds (so the change commits first) and is dropped when one was published for the
same user in the last few seconds (`SET key 1 NX EX`). A burst of 100 embedded jobs or five quick CV
edits is one task, and that task reads the latest data whenever it runs. A run that ends with a request
still open (a change arrived meanwhile, or its inputs changed while it ran) publishes one follow-up
itself, so nothing waits for the sweep.

### The run, and what makes it safe

1. `begin_run` is one UPDATE that refuses while another run is in progress (started less than 5
   minutes ago), so two runs of one user cannot overlap even if the Redis lock was lost. It returns
   the request number the run will satisfy.
2. Read the CV skills, vector, preferences and the nearest candidates, and **end that transaction**.
3. Compute the scores: pure CPU, nothing locked, no transaction open.
4. One short write transaction: lock the CV and preferences rows (`FOR SHARE`), read their
   timestamps again, and compare them with what step 2 read. If either changed, **write nothing**,
   clear the run and stay pending (the change already requested another run). Otherwise upsert the
   changed matches, delete matches that can no longer be scored, record the finish, and commit. The
   locked part is a handful of statements (a test records them and checks that every candidate read
   and every score came before the first lock, and that none came after).
5. After a **complete** run (all of the user's candidates, inputs unchanged, a ticket from the status
   table, at least one candidate) the user's matches for jobs that are no longer candidates are deleted:
   they were scored for an earlier CV or earlier jobs. A skipped or aborted run deletes nothing, and
   neither does a run with no candidates (that usually means the jobs have no vector of the current
   model yet, not that the old matches are wrong).

Job rows are not locked: a job that changes mid-run raised its own request, so the next run fixes it
(eventual consistency, one extra run at most).

### Failures

A database or Redis outage is raised to Celery, which retries (the run is cleared first so the retry
is not refused as "already running"). Any other exception is a bug: it is counted (`failure_count`,
code `unexpected_error`; the log has the exception *type* only, because the message and the traceback
can quote a skill or an explanation). After 3 failures in a row the sweep leaves the user alone and
the state is `failed`; saving a CV, skills or preferences, or `POST /matches/rescore`, clears the
count. A job being embedded does not.

### The sweep (`sweep_matches`, beat, every 5 minutes, one pass, no retry loop)

1. A user with a CV and no status row gets a request.
2. A user who is up to date under a different fingerprint (a setting or the scorer version changed)
   gets a request: at most 50 per pass, and the publishes are spread over a 60-second window, so a
   settings change does not queue everybody at once. A user already pending is skipped, so the same
   users are not picked again next pass.
3. A user whose request has waited more than 2 minutes with no task is queued, at most **once per 5
   minutes per user**, never while a run is in progress, never after 3 failures.
4. An up-to-date user who owns a job whose `embedded_at` is newer than their `finished_at` gets a
   request and is queued: this recovers a request that was never recorded. Same caps (once per 5
   minutes per user, never during a run, never after 3 failures), at most 100 users per pass.

If the queue is down the sweep logs one warning and stops; the next pass is the retry.

### What the API returns

`GET /matches` and `GET /matches/{job_id}` return the stored rows and a `scoring` object:
`up_to_date`, `pending` (a request is open), `running`, `no_cv` or `failed`, with `requested_at`,
`finished_at`, `failure_count`, `max_attempts` and a fixed `error` code. While a re-score is pending the
previous scores are returned unchanged, so a client can show them with an "updating" hint. After a new
CV the old scores stay visible, flagged pending, until the run replaces or removes them. After a
deleted CV the matches are gone at once and the state is `no_cv`.

`GET /matches` (ranked by score, then the newest job; `min_score`, `include_filtered`, `source_id`,
`limit` up to 100, `offset`), `GET /matches/status` (cheap, for polling), `GET /matches/{job_id}` (the
stored explanation exactly as stored, plus one fixed sentence per note; 404 for another user's job, a
missing job and a job with no match yet, all the same body) and `POST /matches/rescore` (202;
`queued`, `already_pending` or `already_running`; its own rate-limit bucket, the same numbers as a
sync; 404 without a CV; 503 and nothing changed if the queue is down). Every response model lists its
fields explicitly and a test pins the exact key set, so no field that could carry CV text can appear.

### With `EMBEDDING_PROVIDER=none`, scores are skills-only

When embeddings are off nothing is embedded, so no CV or job has a vector and the embedding part has
nothing to compare. Scoring still works: the skills part gets all the weight (effective weights
skills 1, embedding 0), a job without a stack is unscorable, and **every explanation says so** with
the note `no_embedding` ("No comparable embedding: the score is skills only."). Turning embeddings on
later changes the vector identity inside every `inputs_hash`, so the next run rewrites the matches; no
manual step is needed.

## 10. Evaluation protocol (Phase 4, PR 6)

This section is written and committed **before any sheet is built and before any label exists**.
Nothing below may change after the first label is written; if something must change, the evaluation
is restarted and the reason is recorded here. The evaluation measures one thing: for a given CV and
its preferences, does the scorer put jobs the person would apply to near the top of their list.

### Personas

| Persona | Profile (the CV is written from this) | Preferences |
|---|---|---|
| Alex | Backend: Python, FastAPI, PostgreSQL, Docker (exists: `tests/fixtures/cv/alex_example.txt`) | senior or lead, remote, at least 100,000 USD per year |
| Sam | Frontend: React, TypeScript, Next.js, CSS | junior or mid, remote, no salary |
| Jordan | Data and ML: Python, SQL, pandas, Airflow, AWS | senior, remote, no salary |

Sam's and Jordan's CVs are invented, committed under `tests/fixtures/cv/`, and written from the
profiles in this table only. Disclosure: the profiles were fixed in the plan before this section, but
while choosing the worked examples below the author read the titles, seniority, remote policy and
stack of the stored jobs (no scores, no post bodies), so the personas were not chosen blind to the
general content of the pool.

**The real CV is optional.** It is kept out of the pass rule, never committed, never uploaded to
anything but the local stack, and held to a final local check after everything else is fixed. It is
labelled and reported (counts only) only if its owner chooses to run it, with the privacy checklist
of section 11. Its preferences are written by its owner in `.local/` before labelling.

### The pool

Every stored job that is `extracted`, has a vector of the current embedding model, and has a distinct
`raw_text_hash` (the same post stored once per source counts once; the lowest id is kept). At the time
of writing this is 108 distinct texts out of 268 extracted rows (the snapshot below). The pool is **frozen** when the
sheets are built: the job ids, their count and a hash of the sorted id list are recorded in the answer
key, and a later run refuses to score if the pool differs. The pool contains jobs outside software
(sales, operations, design); they are kept, because a real feed contains them, and results are also
reported for the technical posts alone (jobs with a non-empty stack).

Personas are scored in memory with the pure scorer (`app/matching/scoring.py`). Nothing is uploaded,
no `job_matches` row is written, no provider is asked, and the database is only read.

### The frozen pool snapshot (recorded before any sheet or label exists)

Recorded on 2026-11-19, read-only, from the development database. The pool is the rows with
`extraction_status = 'extracted'` and a vector whose `embedding_model` is
`BAAI/bge-small-en-v1.5@aa8f8b060edb00e03bfdd08813a2949946c8ba55`, one row per `raw_text_hash` (the
lowest id, compared as text). It has **108** jobs (268 extracted rows in all). Its identity is the
SHA-256 of the sorted ids joined with a single newline (UTF-8, lower-case hyphenated UUIDs, no
trailing newline):

`7cd0b36da7ede96de6a8e61396ef0ca969958237bb3c67ba25f42cce0183470c`

The sheet script recomputes this hash from the database and **refuses to build when it differs**, and
the scoring script refuses a key whose pool hash differs. The ids:

```text
03005e8e-39a9-45c3-8d95-f59082ccc42d
06a4a005-6da7-4aa9-a212-f450a1df19e9
0759c33d-9866-4131-b98a-b7bdc8d0dcf1
0922063b-859b-41cd-804e-ecb7134224de
09aa1503-0e75-40bc-8795-cd016ab05498
09bf389f-d2bd-48ee-938a-241e2f8e1b59
09d8d8b3-ccb3-47a6-99be-75522f5e5d22
0c02a211-29b0-48c3-b8cc-e3106ec5dabd
0d80fc59-100e-48f9-9228-97f9a999054e
0e083a05-f135-40b8-82b2-7ffcc09b59e0
0e70dbbc-ca26-4a03-bb12-db14c250fc84
0f9120f4-2858-4d66-a87b-2fb78c9b8cd5
12b3bd55-ffa7-4035-af43-3fa2ad171f16
12f159c0-e2a8-4d01-9b26-61bef4310049
13585f81-5b37-4feb-ad85-a99c0b6a2034
144b0542-a984-4b61-bd85-7e158af7a743
14a2b4d7-c789-4584-a57b-15794c583c7f
1580df5f-53c7-4f52-9c9e-d43e90d840eb
16b06b6a-54c2-48ad-80fa-b943f558e47e
175adf5e-7c4a-4612-9b72-9232da743934
17f9f5fd-ecbe-445f-ab8e-019b727348c4
188d6927-04e6-4b18-bd69-b45459b55574
19ccdaa7-ec7d-46c9-8fb8-80a8b31139b6
1acb773d-913a-4228-b3fe-b8e6bb5e358c
1c0af5f0-2450-40a3-979a-73fd354f8633
1cab422d-bdbb-4aa2-9fad-1d1f5b5828df
1e698faf-1f64-4d14-a83d-6a65e69d03ce
1f12c377-7dd4-4398-8302-af1cb671c223
20b09792-3e98-499c-9540-e336442b2578
23b14c9a-cdad-4ddf-89d0-e361509c1c1e
244a9cf1-04a4-4ae2-ae0f-b70d023a92a3
26b28c74-cc82-40e9-b330-57d7d0a27d17
29a77ec8-130a-4770-9d6f-4c34ad8ae08c
2a542d49-d654-4a0d-a74b-f9d4e0af05f2
2a71a524-3724-49d4-bd6a-2448509cc073
2af9c0a6-ce2d-4084-a433-708b301d7381
2ba5b22a-ba2f-46f0-ae73-6fab4a41b433
2c8982cf-4ab0-44d9-b767-ead75ab4b6e3
2cf1755b-c277-4bc8-a40b-627ba800b2e3
2d3d6e91-292a-45c4-a2e7-c988fe823ad5
2d9a37e7-6817-44f3-b6b8-5532e1e644ac
2e49a9de-bd7b-4fef-b4fd-b48e33cf2700
2f286ec3-7545-44c8-81f6-7faeddbf4e1f
3125650d-2f3a-4ed1-8bda-0b34d6fb74a2
31b03179-a8a2-47cf-8ac8-925286b2ec30
32dbbd3b-6015-4c28-ab59-0d9f81c0bbd3
33e3aa3b-1dd4-4645-b605-5df5d4f52056
3548cd6f-be6a-4c87-96c1-1ed2fdd93501
3653ead3-9939-4d6f-bb1d-09dbc129f0c7
3869cddf-6059-4a4d-8aa9-a6636c3cb38a
3b282359-ad78-4346-b15e-7e3d67cf0946
3c5a15b1-7747-498d-96d4-ef2f2b7d40e4
3c85f760-6357-4a16-a63f-4ad3807500f5
401ad3e5-e8fa-4a12-9639-5f1e848604fa
4a2454b1-9740-4f7c-be7f-ed14eed18622
4b06f7de-a869-4625-a46e-d123a330fd4b
4c1976d9-a00c-49a9-8e20-eb5eadf8cc32
4cf4674d-0f56-43f1-949d-448d455633e4
4da5c847-baab-4278-85c2-dc36da82cbd8
4ec680c2-e589-4448-af9e-f7cc48e3b45d
4fb6f237-82a5-4475-a350-95ebf21288d2
50df5538-d970-4464-bfbc-86cefb95463c
515b3f61-919e-461d-a042-cbebad57c83f
53eeabc5-8c18-4112-8745-48a05c8f44a9
552a4dff-ac2b-4fbc-856c-978603594cb2
55d6cb75-c447-4d57-8eb4-c340623e0ea1
58575cda-b8b1-49e0-beb6-6cfb49846e70
58aa2c65-72cf-4f0c-aace-9b9263b603ad
5ba5ceb9-f1bc-4d3b-9a75-32e36bb89e16
5c9658f4-85d3-4d81-b648-78798d186ebf
66575841-12a3-4eb0-a1cb-b70649504057
6af6172f-d061-4c47-9567-8a160b389676
6d819a7f-2dbc-4b5d-9962-536d3cb65e51
6de4a48b-ae0e-4643-a9bc-5a2b17a86f3d
6ecaa3e3-9b37-4185-977c-75a51ca5d33d
6fa59e75-8fc4-4b17-97fb-01f52bc5dc19
72bb0870-945c-4390-a82a-fce5f8650652
73e6da4d-5c36-457f-b43c-c55e8413becb
7403a9ac-8703-47f9-993a-67b975f64f4b
7639bf68-5c43-4afb-b7ba-eb1858e72aa4
76ebef92-2562-40aa-95ad-88e3315d1cbc
79009475-3df3-4cd4-b44b-05b9a5eb1fda
7aa9e422-805b-4b3e-bdac-85ca2e29a986
7b9c2242-defa-48c9-b379-5cbd0e5bb637
7ece0a73-4dd5-4248-972e-6b722eb4268c
809fb8be-1303-4e47-a562-877f9fa66f27
827252bf-eac1-43fe-9376-6dfe776cc8f2
8542c463-550c-4bbf-96a4-080f59584926
85924571-875c-493f-b241-56adfc3e9373
87265f8e-03a7-4236-8043-e842723f6243
872a8fac-d89d-446b-b78e-a66954571628
942448ee-1e8b-4a8d-be52-bf5cc3742a9a
97955769-6947-4622-abf5-855696d699bc
98267790-ef44-47ba-8bad-3baf5c990a27
9839709d-47d6-4d4c-ae7a-d338dd192b6f
9a49aa37-dfef-440b-a4cf-d4f1f35b34ca
9e068281-fd95-4e26-89c8-4a244d206b3b
a4696cbe-0088-4fe9-a801-0d184ef76154
a78b11a1-045a-40f6-9c33-49695141fe4e
a8d7588f-161d-4a82-b53a-05b5e2d9a028
ab41d244-4eb7-4767-aace-7eda6365bcb6
ac5106d6-34d5-4067-b20b-d6f25f3a7669
b3a2831c-7275-4fd5-8436-13354b738572
b5e76a4b-bc02-43ec-8441-f33ddb4ed55c
ce090770-3dc5-4a0f-9c46-1ae8a4d3f661
d2706da8-39a8-4353-bfd4-3b967c4c0491
d2cccc32-0af5-4240-add2-d5bf653bee65
e92eab89-30c2-48b4-89e8-f1a3572c91f1
```

### Methods compared

| Method | What orders the jobs |
|---|---|
| A: hybrid | the shipped score, constants as in section 2 (floor 0.715, ceiling 0.86) |
| B: hybrid, CV-job calibration | the same formula with a floor and a ceiling taken from CV-job cosines (below) |
| skills only | skill coverage alone (the embedding weight set to 0) |
| embedding only | the raw cosine of CV and job vectors |
| newest first | `posted_at` descending (`created_at` when missing) |
| random | no order; reported as the relevant share of the labelled random jobs |

Ties are broken by the newer job, then by job id, in every method, so a ranking is reproducible.

### The sheets

Per persona, the sheet holds the union of **the top 5 of each of A, B, skills only, embedding only and
newest first**, plus **10 random jobs** drawn with a fixed seed from the pool jobs not already in that
union. That is about 20 to 30 jobs per persona. The union is used, and not the system's top jobs
alone, so that no baseline is judged on jobs nobody labelled. The sheet is shuffled with a fixed seed
and each item has an opaque number. The labelling tool shows a job's public fields (title, company,
seniority, remote policy, stack, salary) and the start of its post, and **never** a rank, a score, a
method or a number that could reveal either. The answer key (which jobs each method put where) is
written to a different directory and is not opened before labelling ends.

**P@10 is reported only for a persona and a method whose top 10 are all labelled.** The top 5 are
always labelled by construction; ranks 6 to 10 are labelled only where they fall in the union or the
random draw. Where a rank among the top 10 has no label, P@10 is reported as "not measured", never as
a count over fewer jobs.

### The relevance scale

Labels are made from the persona's point of view, using the persona's CV profile and preferences. One
label per job.

- **relevant**: the persona would apply. The main work of the job matches the persona's skills and
  seniority, and no stated preference is violated.
- **maybe**: adjacent. The domain or part of the stack matches but the main work differs, or the
  seniority is one level off, or one key requirement is missing.
- **no**: a different field or role, or a stated preference is violated (seniority more than one level
  off, on-site or hybrid when the persona wants remote, a stated maximum salary below the persona's
  minimum).

A job that does not say something (no seniority, no remote policy) does not violate it. Unknown never
makes a job `no` by itself.

Worked examples, chosen from the pool by title, seniority, remote policy and stack, before any sheet
existed and without looking at any score. The id is the first 8 characters of the job id.

| Persona | Label | Job | Why |
|---|---|---|---|
| Alex | relevant | `4cf4674d` Senior Backend Developer (Python); senior, remote; Python, Django, SQL, PostgreSQL, Docker, Kubernetes, AWS, GCP | backend Python work at Alex's level and preference |
| Alex | maybe | `14a2b4d7` Senior Data Engineer; senior, remote; SQL, Python, dbt, BigQuery, PostgreSQL | senior, remote, shares Python, SQL and PostgreSQL, but the main work is data pipelines, not backend services |
| Alex | no | `ce090770` Category Manager - Apparel; mid, remote | a different field |
| Sam | relevant | `ac5106d6` Software engineer; remote; TypeScript, React, GraphQL, PostgreSQL, GCP | TypeScript and React work, no seniority stated |
| Sam | maybe | `3c85f760` Senior React Full-stack Developer; senior, remote | the right stack, one level above Sam's junior or mid |
| Sam | no | `0c02a211` Senior Director, Field Enablement; senior; Salesforce | a different field and far above Sam's level |
| Jordan | relevant | `14a2b4d7` Senior Data Engineer; senior, remote; SQL, Python, dbt, BigQuery, PostgreSQL | data work with Python and SQL at Jordan's level and preference |
| Jordan | maybe | `7b9c2242` Machine Learning Engineer; remote; AWS, PyTorch, TensorFlow, Airflow, Python | shares Python, AWS and Airflow, but model building rather than data engineering; seniority not stated |
| Jordan | no | `0922063b` Framer Website Designer; onsite; HTML, CSS | a different field, and on-site |

These nine rows illustrate the scale; they are not exhaustive, and the labeller decides each job on
its post. The same job may carry different labels for different personas.

**Seen examples.** The nine jobs above stay in the sheets with the labels documented here, and the
sheet key marks them `seen_example`, so the scoring can report results with and without them. If the
labeller changes their mind about one after reading its post, the change is recorded below as a dated
note and the table above is never edited; the scoring then reports both the documented and the
changed label.

Dated notes on the worked examples: none yet.

### Who labels, and how

One labeller: the project's owner, who also built the system. This is a bias and is stated in the
limits. To keep it small: labels are made on the blind sheet only, the owner does not open
`GET /matches` or any scoring output for these personas before the last persona is labelled, and each
persona is labelled in one sitting where possible. The labelling tool saves after every label and
writes **one file per persona** (`.local/match_eval/labels/<persona>.json`: item number, job id and
label, nothing else). The answer key lives in a different directory (`.local/match_eval/key/`), and
the scoring script is the only reader of both.

### Metrics

- **P@5 and P@10** as raw counts (for example 3/5), per persona and method, never only as a ratio.
  Pooled P@5 is the sum over the three synthetic personas, out of 15.
- **Primary count: `relevant` only.** Secondary count: `relevant` plus `maybe`.
- The same counts for the technical-posts stratum (jobs with a non-empty stack): the top 5 of each
  method restricted to those jobs.
- **Random** is the number of `relevant` among the labelled random jobs of that persona, out of 10;
  the expected value for five random jobs is half of it.
- No confidence interval, test or p-value is reported: with 15 slots there is no meaningful one.

### The pass rule

Fixed now, not after seeing a result. The evaluation passes when **all** of these hold:

1. pooled P@5 (primary count, method A) over Alex, Sam and Jordan is **at least 9/15**;
2. it is **strictly more** than the pooled P@5 of newest first;
3. every persona is reported, including any persona where A does worse than a baseline.

The real CV, if used, is reported apart as counts only and is not part of the rule. A miss is reported
as FAIL. It is not repaired by changing the weights, the floor or the ceiling afterwards: any such
change is a new `SCORER_VERSION` in a separate PR, evaluated again from a fresh sheet.

### The CV-job calibration question

The floor and ceiling of section 2 come from job-job pairs, and CV-job cosines sit lower (section 2,
"How to read a score"). Whether recalibrating helps is tested **without using any label**.

- **A** is the shipped constants (0.715 and 0.86).
- **B** uses the same rule as section 2, **P50 and P99**, applied to CV-job cosines: the cosines of a
  persona's CV against every job of the pool.
- **Held-out rule.** For each synthetic persona, B's floor and ceiling come **only from the other two
  personas' CV-job cosines**, never from the held-out persona's own CV. The real CV, if used, is held
  out the same way: its B constants come from the three synthetic personas only.
- B is reported next to A, side by side, for every persona. **B is recommended only if it adds at
  least 2 relevant jobs to the pooled P@5 and loses relevant jobs in no persona.** Otherwise the
  result is "no evidence that it helps" and A stays.
- Limit: B's constants are computed on the same pool the sheets are drawn from (never on labels, never
  on the held-out persona), so this is not an out-of-sample test.

A change to the shipped constants is out of scope for this PR whatever B shows.

### Note, 2026-11-19: the baselines are not on equal footing

This note was added before any sheet was built and before any label exists; it clarifies the table of
methods and changes no rule.

- **Skills only** is the shipped scorer with the embedding weight at zero (no cosine is given to
  `score_match`). The hard filters and the cap of 30 therefore **still apply** to it.
- **Embedding only** is the raw cosine of the CV and job vectors and applies **no filter**.
- **Newest first** and **random** use neither skills nor filters.
- The five methods are therefore **not on equal footing**: A, B and skills only know the persona's
  seniority, remote policy and salary preferences; embedding only, newest first and random do not.
  Every result table and every statement of a result must say so next to the numbers, and a method
  that "loses" to the hybrid is not evidence about its signal alone.

### The scripts and their guards

`scripts/eval_match_sheet.py`, `eval_match_label.py` and `eval_match_score.py` are run by hand, never
by CI or app code. The sheet script reads the database read-only, refuses a database not named
`jobradar`, builds no LLM client and refuses `--allow-provider`. The scoring script needs no database,
model or provider: it reads the pool snapshot, the key and the label files. It prints counts and
opaque numbers only, never CV text, skills or notes, and writes a run record (constants, versions,
file hashes, counts) under `.local/match_eval/runs/`.

### Limits of this evaluation

- Tiny sample: three invented personas and at most one real CV, 5 slots per P@5. No claim about
  accuracy in general, and no significance.
- One labeller, who also built the system.
- One source mix: two public feeds, one with a board-specific rule, stored at one date.
- Non-software jobs are in the pool (hence the technical stratum).
- Skill weights are equal (section 8); duplicate texts are collapsed to one job.
- B is calibrated on the pool it is evaluated on.
- It scores the scorer's code on a frozen pool, not the Celery pipeline, the sweep or the API.
- The embedding model is an English-only int8 model; a non-English CV is outside the evaluation.

## 11. Privacy checklist for the optional real CV

Applies only if the owner chooses to include their own CV. Nothing in section 10 needs it.

- The file lives only under `.local/cv/` (git-ignored; check with `git check-ignore`). The pre-commit
  hook already rejects a CV-like file outside `tests/fixtures/cv`.
- It is uploaded only to the local stack, with `curl.exe --data-binary @.local\cv\<file>` to the local
  API, and only after the owner has checked that their `.env` names no hosted provider (this
  repository's tooling never reads `.env`). `worker-embed` has no route to the internet.
- Emails, phone numbers and links are masked; names are not.
- A `pg_dump` of the development database contains the `cvs` table: back up outside the repository and
  use `--exclude-table-data=cvs` while the real CV is stored.
- Afterwards: `DELETE /api/v1/cv`, check with a count query that no `cvs` row and no `job_matches` row
  remains for that account, delete the file, and clear the shell history of the upload line.
- Labels, answer key and run records hold item numbers, job ids and counts only.
- One canary run, once: upload a **copy** with a unique marker line to a throwaway local account, run
  the flow and every match endpoint, and count (print only counts) the marker in the API and worker
  logs, Redis keys and values, `match_status` and `job_matches` rows and the responses. After the
  `DELETE`, count it in a full `pg_dump`: it must be 0.
