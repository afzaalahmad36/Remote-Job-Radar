# Profile API examples: CV, skills and preferences

All paths are under `/api/v1` and need a login (`401` without one). The examples use `curl`,
assume the API runs on `http://localhost:8000` and use `jq` only to pick the token out of the
reply. Use an **invented** CV (see `tests/fixtures/cv/`), never a real one, in screenshots and docs.

```bash
BASE=http://localhost:8000/api/v1

# Register once, then log in. The access token goes in the Authorization header.
curl -s -X POST "$BASE/auth/register" -H 'Content-Type: application/json' \
  -d '{"email": "alex@example.com", "password": "correct-horse-battery"}'
TOKEN=$(curl -s -X POST "$BASE/auth/login" -H 'Content-Type: application/json' \
  -d '{"email": "alex@example.com", "password": "correct-horse-battery"}' | jq -r .access_token)
```

## Upload a CV: `PUT /cv`

**The request body is the file itself, not a multipart form.** There is no `-F` and no field
name: send the bytes with `--data-binary` and say what they are in `Content-Type`
(`application/pdf`, `text/plain` or `application/octet-stream`). The content decides the format
(`%PDF-` at the start is a PDF, anything else must be UTF-8 text), so a wrong header or file name
changes nothing.

```bash
# A PDF
curl -s -X PUT "$BASE/cv" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/pdf' \
  --data-binary @alex_example.pdf

# Plain text
curl -s -X PUT "$BASE/cv" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: text/plain' \
  --data-binary @tests/fixtures/cv/alex_example.txt
```

Use `--data-binary`, not `-d`: `-d` strips line breaks, which would flatten the CV. Uploading
again replaces the CV (one per user). The reply holds the stored text, with emails, phone numbers
and links masked, so you can check exactly what the system keeps. The file itself is never stored.

| Status | Code | Meaning |
|---|---|---|
| 200 | | stored (the reply is the `CvRead` document) |
| 413 | `cv_too_large` | more than 2 MB |
| 415 | `cv_unsupported_type` | not a PDF or UTF-8 text |
| 422 | `cv_unreadable` | damaged, password protected, no text (a scanned PDF), too many pages (10), too long (30,000 characters) |
| 429 | `rate_limited` | more than 5 uploads per minute |
| 503 | `cv_busy` | all PDF readers are in use; try again shortly |

```bash
curl -s "$BASE/cv" -H "Authorization: Bearer $TOKEN"                   # read it back
curl -s -X DELETE "$BASE/cv" -H "Authorization: Bearer $TOKEN" -i      # 204: delete it for good
```

## Skills: `GET` / `PUT /cv/skills`

When a CV is uploaded its skills are read from the text with the same technology dictionary the
job rules use (so a CV skill and a job's stack entry are spelled the same). Reading returns the
list, whether you edited it, and two **suggestions**, each with the kind of match that produced
it. Suggestions are only shown to you: nothing scores or filters with them.

```bash
curl -s "$BASE/cv/skills" -H "Authorization: Bearer $TOKEN"
# {"skills": ["Python", "FastAPI", ...], "edited": false,
#  "suggestions": {"years_of_experience": {"value": 6, "match": "years phrase"},
#                  "seniority": {"value": "senior", "match": "title keyword in the current role"}},
#  "updated_at": "..."}

# Replace the whole list. Names the dictionary knows are spelled its way ("postgres" is stored
# as "PostgreSQL"); any other name is kept as you typed it. Duplicates are dropped.
curl -s -X PUT "$BASE/cv/skills" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"skills": ["Python", "postgres", "Team leadership"]}'
```

> **A new upload discards your edited skills.** Uploading a CV again (`PUT /cv`) reads the skills
> from the new text and sets `edited` back to `false`: the edit was for the old document.
>
> **A CV stored before this feature has an empty skills list** (`"skills": []`) until you save a
> list with `PUT /cv/skills` or upload the CV again.

At most 100 skills of 50 characters each.

How skills are found: the same technology dictionary the job rules use. Words that are too
ambiguous to count in a sentence ("Go", "R", "C", "Swift", "Rust", "Ruby", ...) count only inside a
skills-style list: a line under a heading such as `Skills`, `Technologies`, `Tech stack` or
`Languages`, or a comma/slash separated line that names at least two other known technologies,
and only as a whole list item. `Skills: Python, Go, SQL` gives all three; "Go-To-Market" and "go to
production" give nothing. Anything the finder misses, add with `PUT /cv/skills`.

## Preferences: `GET` / `PUT /preferences`

Nothing is saved until you save it. Reading before the first save returns empty preferences
(`updated_at` is `null`) and creates no row. `PUT` **replaces** everything: a field you leave out
is cleared. An empty list means "no preference".

```bash
curl -s -X PUT "$BASE/preferences" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{
        "seniority_levels": ["senior", "lead"],
        "remote_policies": ["remote"],
        "min_salary": 90000,
        "salary_currency": "USD",
        "salary_period": "year",
        "notes": "No fintech, please."
      }'
curl -s "$BASE/preferences" -H "Authorization: Bearer $TOKEN"
```

Levels: `intern junior mid senior lead principal`. Policies: `remote hybrid onsite`. Periods:
`year month week day hour`. A minimum salary needs an amount, a 3-letter currency and a period
together (`422` otherwise). The notes are personal data: emails, phone numbers and links in them
are masked like in a CV, and they are never logged or sent to a model.
