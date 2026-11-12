# Extraction accuracy

How well the extractor reads job posts, measured by hand on a small labelled sample. The pass rule
is fixed in `CLAUDE.md` (section 6, "Extraction accuracy evaluation") and was written before any post
was labelled. CI never labels or scores real posts, and the scoring makes no provider request.

**The 90% target was not met.** The headline measurement is the first-pass labels with the current
rules (`rules_v2`, recomputed offline): **13 of 30 posts parsed correctly (43.3%)**, and the
invented-value gate (at most 10% of posts with an invented value) also failed at 4 of 30
(`.local/eval/runs/20261106T081400Z.json`). On the completed labels the best result is 14 of 30
(46.7%, invented 1 of 30, `.local/eval/runs/20261111T071345Z.json`), and the same 14 of 30 after the
`rules_v3` remote-policy rule (`.local/eval/runs/20261112T121649Z.json`). The completed-label score is
less independent of the extractor (see "The second pass"), so it is reported next to the first-pass
number and does not replace it.

Why the target was not met:

- **The rule needs every field right.** A post passes only if the title, company, stack, remote
  policy, salary and seniority all pass, so field errors add up. On the completed labels 16 of 30
  posts fail on at least one field, and 12 of those 16 fail on the stack (record
  `20261111T071345Z.json`, `rules_failures`).
- **The stack.** 62 label entries are not in the technology dictionary: they cost recall without
  being a rule miss. Stack recall is 57.6% against the 70% the rule asks for, while precision is
  91.9%: the extractor mostly finds correct technologies but not all that the post names.
- **Remote policy.** The weakest field under `rules_v2`: 12 of the 28 non-blank labels were not found
  (recall 57.1%, counted from the record). A missing value does not fail a post under the rule (only
  a wrong or invented one does), so this was a real gap that the 90% figure does not show. The
  `rules_v3` rule closed part of it (recall 82.1%, accuracy 80.0%, record `20261112T121649Z.json`)
  without moving the parse-correct share; 5 labelled remote posts are still missed (see "The
  remote-policy rule of `rules_v3`").
- **Smaller errors**: 2 wrong titles, 2 wrong companies and 1 invented remote policy.

What was changed in response to the first run:

- **`rules_v2`** (from the diagnosis of the first run): benefit money is no longer read as a salary,
  a price or a perk is cut from the title, "WFH" as an occasional option is not remote, a customer
  list or an "X alternative" phrase is not a stack, a post naming more than 12 technologies reports
  no stack, ambiguous English words (swift, rust, ruby, ...) need a capital letter, and `$` with
  `CAN`/`CAD` is CAD.
- **`rules_v3`** (from a diagnosis of the stored posts): a `Headquarters:` header that names remote,
  anywhere or worldwide settles a post whose text gives mixed remote signals, unless the body states
  a concrete obligation. It is specific to one board's header format.
- **The evaluation tooling**: blind labelling (`eval_label`), scoring of the stored extraction and
  of the rules recomputed offline (`try_extraction --eval`), label checks, and a recompute script
  that resets old rules-only rows so the sweep redoes them.
- **Label cleaning**: the hand labels were corrected from the post text alone (headline split,
  salary amounts and periods, stack entries not in the text), and later completed in a second pass
  under fixed rules.

## Results

Each number names the run record it comes from. Run records are written by
`scripts/try_extraction.py --eval` to `.local/eval/runs/` (git-ignored; they hold numbers and reason
codes only, no post text or labels). A cell with no record is "to fill": it has not been measured.

| Labels | Output scored | Parse correctly (target 90%) | Invented-value gate (limit 10%) | Stack F1 | Run record |
|---|---|---|---|---|---|
| First pass | stored extraction (`rules_v1`; 4 of the 30 posts also enriched by the model) | 5/30 = 16.7% (95% interval 7.3% to 33.6%) | 9/30 = 30.0% | 58.7% | `.local/eval/runs/20261106T061103Z.json` |
| First pass | rules recomputed offline, `rules_v1` | 8/30 = 26.7% (14.2% to 44.4%) | 7/30 = 23.3% | 59.6% | `.local/eval/runs/20261106T061103Z.json` |
| First pass | rules recomputed offline, `rules_v2` | 13/30 = 43.3% (27.4% to 60.8%) | 4/30 = 13.3% | 67.2% | `.local/eval/runs/20261106T081400Z.json` |
| First pass | stored extraction after the `rules_v2` recompute | to fill | to fill | to fill | to fill |
| Completed (second pass) | rules recomputed offline, `rules_v2` | 14/30 = 46.7% (30.2% to 63.9%) | 1/30 = 3.3% | 70.8% | `.local/eval/runs/20261111T071345Z.json` |
| Completed (second pass) | stored extraction after the `rules_v2` recompute | 11/30 = 36.7% (21.9% to 54.5%) | 3/30 = 10.0% | 69.7% | `.local/eval/runs/20261111T071345Z.json` |
| Completed (second pass), after the remote-policy rule (`rules_v3`) | rules recomputed offline | 14/30 = 46.7% (30.2% to 63.9%) | 1/30 = 3.3% | 70.8% | `.local/eval/runs/20261112T121649Z.json` |
| Completed (second pass), after the remote-policy rule (`rules_v3`) | stored extraction | 11/30 = 36.7% (21.9% to 54.5%) | 3/30 = 10.0% | 69.7% | `.local/eval/runs/20261112T121649Z.json` |

Notes on the filled cells:

- The first-pass row at 13/30 is the headline. The completed-label rows are **less independent**:
  the completion rules were written after the first run was diagnosed, part of them mirrors rules
  the extractor now has, and the stack rules were revised twice after dry-runs showed false adds.
- **The invented-value gate for the stored extraction is at the limit, not above it**: 3 of 30 is
  exactly 10.0%, and the gate fails only above 10%. The record says `gate_failed: false`. Both
  completed-label rows still fail the 90% target.
- The stored extraction in `20261106T081400Z.json` is the same as in `20261106T061103Z.json`
  (5/30): nothing had been recomputed yet. `rules_v2` was merged after the first run, so the
  stored column was still `rules_v1` plus the model rows.
- In `20261111T071345Z.json` the stored extraction (11/30) and the rules offline (14/30) differ on
  three posts: two with an invented seniority and one with an extra stack entry in the stored
  extraction only. The record does not say why; the expected cause is the rows the model enriched
  earlier, which the recompute deliberately does not reset.
- The run `20261111T071345Z.json` made no provider request (`provider_requests: 0`).
- **The `rules_v3` rows did not move the parse-correct share**, as predicted: a missing remote policy
  never fails a post, and the remote-policy field was already at 0 wrong values. Only the field's
  accuracy changed (see the table below). The invented-value gate is also unchanged: 1 of 30 offline,
  and for the stored extraction 3 of 30 = 10.0%, at the limit and not above it
  (`gate_failed: false`).
- **The stored extraction trails the rules** in `20261112T121649Z.json` (11/30 against 14/30, the
  same three posts as before): two have an invented seniority and one more has an extra stack entry
  in the stored extraction only (stack false positives 12 against 8). The record does not hold rules
  versions, so the cause rests on the owner's check of the database: 16 model-enriched rows kept
  their `rules_v1` results, because the recompute deliberately does not reset rows the model
  touched. The two invented seniority values came from the model path on labelled posts. The
  remote-policy field is identical in both outputs (23 correct).
- The run `20261112T121649Z.json` also made no provider request (`provider_requests: 0`).
- Stack F1 is computed from the true/false positive and false negative counts in the record
  (micro-averaged over the 30 posts).
- "First pass" means the labels as corrected from the post text alone by `scripts/eval_fix_labels.py`
  (headline split, seniority word in the title, salary amounts and periods found in the text, stack
  entries not in the text removed). "Completed" means those labels after the second pass below; the
  first-pass file is kept as `.local/eval/labels.pass1.json`.

### Per field, rules offline, completed labels

From `.local/eval/runs/20261111T071345Z.json` (the `rules` block), 30 posts. "Correct" counts a
correct value and a correctly absent one.

| Field | Correct | Wrong | Missing | Invented | Accuracy |
|---|---|---|---|---|---|
| Title | 28 | 2 | 0 | 0 | 93.3% |
| Company | 28 | 2 | 0 | 0 | 93.3% |
| Salary | 28 | 0 | 2 | 0 | 93.3% |
| Seniority | 29 | 0 | 1 | 0 | 96.7% |
| Remote policy | 17 | 0 | 12 | 1 | 56.7% |
| Stack (posts passing the stack rule) | 18 | | | | 60.0% |

Stack entries over all 30 posts: 91 correct, 8 extra and 67 missed, so precision 91.9%, recall 57.6%
and F1 70.8%.

### Per field, rules offline, before and after `rules_v3`

Before: `20261111T071345Z.json`; after: `20261112T121649Z.json` (the `rules` block of each), same labels.

| Field | Before (`rules_v2`) | After (`rules_v3`) |
|---|---|---|
| Title | 93.3% | 93.3% |
| Company | 93.3% | 93.3% |
| Salary | 93.3% | 93.3% |
| Seniority | 96.7% | 96.7% |
| Remote policy, accuracy | 56.7% (17/30) | 80.0% (24/30) |
| Remote policy, recall on non-blank labels | 57.1% (16/28) | 82.1% (23/28) |
| Stack (posts passing the stack rule) | 60.0% | 60.0% |
| Posts parsing correctly | 14/30 = 46.7% | 14/30 = 46.7% |
| Invented-value posts | 1/30 | 1/30 |

## How to fill the remaining row

Only "First pass, stored extraction after the `rules_v2` recompute" has no record. Score the
first-pass labels, which are kept in `.local/eval/labels.pass1.json`:

```bash
uv run python -m scripts.try_extraction --eval .local/eval/labels.pass1.json
```

Then write the new record's file name next to its numbers.

## The remote-policy rule of `rules_v3`

`rules_v3` adds one rule for the remote policy. Read these limits before the numbers:

- **It is specific to this board's `Headquarters:` header.** The rule reads that header line only,
  and only when the rest of the text gives mixed signals (a remote word next to a hybrid or on-site
  word, which `rules_v2` leaves undecided). A feed without the header is unaffected.
- **It was written from a diagnosis of the stored posts, all 200, not only the 30 labelled ones**, so
  it is not independent evidence either, like the `rules_v2` changes. It changes 20 of the 200 stored
  posts from no policy to remote, and no other result.
- **It does not change the parse-correct share.** A missing remote policy never fails a post, so
  the 30-post share stays at 14/30 offline (the completed labels); only the field's accuracy moves.
- **Safeguard.** A body statement that names a concrete obligation ("three days a week", "days per
  week", "in the office N days", "must live within", "within N miles", "relocate") keeps the result at
  none. Three labelled posts are held back by it (a New York office with core days three days a week;
  in-person collaboration 2-3 days per week); their labels say remote.
- **Posts 8 and 27 may have unsupported remote labels.** The post text never says remote, anywhere
  or worldwide, so no text rule can find it; they are left as labelled.

Remote-policy field, rules offline, completed labels. Before the rule was merged these numbers were
computed in memory with the rule switched off and on; the run `20261112T121649Z.json` then measured
the same result (23 correct, 5 missing, 1 invented), so the in-memory prediction matched the
measurement:

| | correct | missing | invented | accuracy | recall on non-blank labels | invented-value posts |
|---|---|---|---|---|---|---|
| `rules_v2` behaviour | 16 | 12 | 1 | 56.7% | 57.1% (16/28) | 1/30 |
| `rules_v3` | 23 | 5 | 1 | 80.0% | 82.1% (23/28) | 1/30 |

Five labelled remote posts are still missed: three are held back by the obligation safeguard (posts
11, 26 and 29) and two name no remote word in the post at all (posts 8 and 27), where the label may
be unsupported. The sample has no hybrid label, so how the rule behaves on a genuinely hybrid post whose header says
remote is untested on real data; the unit tests cover it with invented text.

## The second pass

`scripts/eval_complete_labels.py` completes the labels under fixed rules (they are listed in the
script's docstring and in `CLAUDE.md`): a missing salary amount or currency is filled only when the
post states it, and a technology is added only when it is a known product named in the post as a
whole word, outside a customer list or an "X alternative" phrase, in a post that names 12 or fewer
technologies. The stack rules were revised twice, after dry-runs showed false adds: the employer's
name (the company before the colon in the headline, or its first word, ignored everywhere in the
post), a name such as "Spark Capital" or "Maven Clinic", "Go-To-Market" read as Go, and a technology
the post says the role is not about. A post that would gain more than 8 entries goes to a human
instead. The first-pass labels are kept, and both scores are reported, because the second-pass rules
were written after the first run had been diagnosed and part of them mirrors rules the extractor
now has. A score on the completed labels is therefore not independent evidence.

## Limits

- **One source and 30 posts**, picked with a fixed seed from one feed. A score here is a measurement
  on this sample, not a claim about accuracy in general; 13 of 30 has a 95% interval of 27% to 61%.
- **The label rules decide the score.** The pass rule is strict about the stack (no entry outside the
  label, at least 70% of the label found), so a label that is shorter or longer than what the post
  names moves the result a lot. Hand labels also contain mistakes; the first pass fixed some from the
  post text, and the second pass completes the rest under fixed rules that were themselves revised.
- **Dictionary gaps.** 62 label entries are unknown to the technology dictionary. Adding them would
  raise the score on this sample and say nothing about new posts, so the dictionary has not been
  fitted to the labels.
- **Stored and offline results differ** on posts the model enriched; the offline rules column is the
  one that measures the rules.
- The extractor and the lists were not changed to improve a score after a score was seen, except by
  the documented `rules_v2` change set, which came from the diagnosis of the first run, and the
  `rules_v3` remote-policy rule, which came from a diagnosis of the stored posts.
