# Demo screenshots and transcripts

**Status:** `01-swagger-overview.png` (a headless browser capture of the page only, so no address bar,
tabs or profile) and the three transcripts `03-matches.txt`, `04-explanation.txt` and `05-seed.txt`
(a real run on 2026-11-24, password redacted) are committed. The terminal PNGs `03` to `05` and
the Swagger response shot `02` below were **not** taken: the transcripts replace the terminal ones,
and the table remains the recipe if you want to add them.

Everything here comes from the **invented** demo data only: the demo account, the 20 invented posts
of `tests/fixtures/demo/demo_jobs.json` and the invented persona Alex
(`tests/fixtures/cv/alex_example.txt`). No real CV, no real feed and no real company appears.

Files are added only after the review checklist below has been done on each one.

## Shot list

Take these with the stack running (`docker compose up`), the demo seeded
(`uv run python -m scripts.seed_demo`) and Alex's CV uploaded as in the walkthrough in
[../API_EXAMPLES.md](../API_EXAMPLES.md). Save PNGs in this folder with the names below.

| File | Purpose | What to show, and the exact command or page |
|---|---|---|
| `01-swagger-overview.png` | Shows the API surface a reader can try | Browser at `http://localhost:8000/docs`, scrolled to the top, all route groups collapsed. Do **not** click Authorize. |
| `02-swagger-matches.png` | Shows a real ranked response | In Swagger, log in with the demo account through `POST /auth/login`, use the padlock with the access token, run `GET /api/v1/matches?limit=5`. Crop to the response body; **close the Authorize dialog and scroll so no token is visible**. |
| `03-terminal-matches.png` | Shows the PowerShell flow end to end | Terminal started from a neutral folder (see the checklist), running steps 4 and 5 of the walkthrough: the `scoring: up_to_date` loop and the `$page.items \| ForEach-Object ...` list. |
| `04-terminal-explanation.png` | Shows that every score is explained | Step 6 of the walkthrough: `curl.exe -s "$base/matches/$id" -H $auth \| ConvertFrom-Json \| ConvertTo-Json -Depth 8`, showing the matched and missing skills and the filters. Do not echo `$auth` or `$token`. |
| `05-terminal-seed.png` | Shows the demo seed and its guards | `uv run python -m scripts.seed_demo --reset` (the printed password must be hidden or the user reset again afterwards). Shows "Posts with a title after the rules phase: 20 of 20". |

## Transcripts (text, from the same runs)

Next to each terminal shot, save the text of the same run as `.txt` so it can be read, searched and
diffed. Capture from a neutral folder, with the token and password variables never printed:

```powershell
cd $env:TEMP                       # or any folder whose path holds no real name
Start-Transcript -Path .\walkthrough.txt
# ... run the steps of the walkthrough ...
Stop-Transcript
```

`Start-Transcript` writes the user name and the machine name in its header and the host in its
footer: **delete those lines** before copying the file here. Save the cleaned files as
`03-matches.txt`, `04-explanation.txt` and `05-seed.txt`.

## Review checklist (do it for every file, before it is committed)

For each PNG, open it at full size and tick every line. For each `.txt`, search it.

- [ ] **No real CV**: only Alex's invented text; no real name, employer, school or photo.
- [ ] **No real person's name or email**: only `Alex Example`, `demo@example.com` and other
      `@example.com` addresses; no phone number.
- [ ] **No token**: no `Authorization` header value, no `access_token`, `refresh_token` or `$token`
      value; Swagger's Authorize dialog is closed. A token starts with `eyJ`: search for it.
- [ ] **No password**: the demo password printed by the seed script is not visible, or has been
      replaced afterwards with `--reset`.
- [ ] **No `.env`**: no secret key, no API key (`AIza...`), no connection string with a password.
- [ ] **No Windows user name or path**: the terminal prompt and window title do not show
      `C:\Users\<name>`, the file explorer, `$env:USERPROFILE` or the machine name; run from a neutral
      folder.
- [ ] **Browser chrome is clean**: the address bar shows only `localhost:8000` (no other site,
      query with a token or account page), the tab titles are only the Swagger tab, and there are no
      bookmarks, extension icons, profile picture or other open tabs in the shot.
- [ ] **No real feed post**: job titles and companies are the fictional ones of the fixture; no
      real company, no real link (every link is `example.com`).
- [ ] **No other windows**: no notifications, chat, mail, editor tabs with real file names or
      desktop wallpaper in the shot.
- [ ] **Text transcripts**: search for `\Users\`, `eyJ`, `AIza`, `@gmail`, `password` and the machine
      name; each must be absent or be the word in a command that shows no value.
- [ ] **Size and format**: PNG, under about 500 KB each; no EXIF-bearing photo.

Mechanical help: a unit test (`tests/unit/test_demo_artifacts.py`) fails if a `.txt` transcript in
this folder holds a token, a key, a Windows user path or a Gmail address. It only reads text; the
PNGs still need the human check above.
