# Extraction accuracy

How well the extractor reads job posts, measured by hand on a small labelled sample. The pass rule
is fixed in `CLAUDE.md` (section 6, "Extraction accuracy evaluation") and was written before any post
was labelled. CI never labels or scores real posts, and the scoring makes no provider request.

**The 90% target was not met.** On the first-pass labels the best result so far is 13 of 30 posts
parsed correctly (`rules_v2`, offline), and the invented-value gate (at most 10% of posts with an
invented value) also fails at 4 of 30. The main reasons, from the diagnosis of the first run
(`.local/eval/diagnosis.txt`, local only):

- **Stack.** Most failures. 62 label entries are not in the technology dictionary (they cost recall
  without being a rule miss), and most "extra" entries the extractor found were technologies the post
  really names but the hand label left out, although the label convention says to list every
  technology the post names.
- **Labels that were incomplete.** Three salaries (a range missing from one label, one end missing
  from two others) and one remote policy were blank or partial although the post states them.
- **A few real rule errors**, which `rules_v2` fixed: benefit money read as a salary, a price or a
  perk in the title, "WFH" as an occasional option read as remote, a customer list read as a stack.

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
| Completed (second pass) | rules recomputed offline, `rules_v2` | to fill | to fill | to fill | to fill |
| Completed (second pass) | stored extraction after the `rules_v2` recompute | to fill | to fill | to fill | to fill |

Notes on the filled cells:

- The stored extraction in `20261106T081400Z.json` is the same as in `20261106T061103Z.json`
  (5/30): nothing had been recomputed yet. `rules_v2` was merged after the first run, so the
  stored column is still `rules_v1` plus the model rows.
- Stack F1 is computed from the true/false positive and false negative counts in the record
  (micro-averaged over the 30 posts).
- "First pass" means the labels as corrected from the post text alone by `scripts/eval_fix_labels.py`
  (headline split, seniority word in the title, salary amounts and periods found in the text, stack
  entries not in the text removed). "Completed" means those labels after the second pass below; the
  first-pass file is kept as `.local/eval/labels.pass1.json` once the second pass has run.

## How to fill the remaining rows

1. Back up the database (the command is printed by `scripts/recompute_extractions.py --help`).
2. `uv run python -m scripts.recompute_extractions --dry-run`, then with `--confirm-backup-taken`;
   let the extraction sweep (beat and a worker) recompute the reset rows.
3. `uv run python -m scripts.try_extraction --eval .local/eval/labels.json` on the first-pass labels,
   and write the record's file name next to the numbers.
4. `uv run python -m scripts.eval_complete_labels --dry-run`, read the change table, then run it for
   real; score again with `--eval` and add the record.

## The second pass

`scripts/eval_complete_labels.py` completes the labels under rules fixed before it runs (they are
listed in the script's docstring and in `CLAUDE.md`): a missing salary amount or currency is filled
only when the post states it, and a technology is added only when it is a known product named in
the post as a whole word, outside a customer list or an "X alternative" phrase, in a post that names
12 or fewer technologies. The first-pass labels are kept, and both scores are reported, because the
second-pass rules were written after the first run had been diagnosed and part of them mirrors rules
the extractor now has. A score on the completed labels is therefore not independent evidence.

## Limits

- **One source and 30 posts**, picked with a fixed seed from one feed. A score here is a measurement
  on this sample, not a claim about accuracy in general; 13 of 30 has a 95% interval of 27% to 61%.
- **The label rules decide the score.** The pass rule is strict about the stack (no entry outside the
  label, at least 70% of the label found), so a label that is shorter or longer than what the post
  names moves the result a lot. Hand labels also contain mistakes; the first pass fixed some from the
  post text, and the second pass fixes the rest under fixed rules.
- **Dictionary gaps.** 62 label entries are unknown to the technology dictionary. Adding them would
  raise the score on this sample and say nothing about new posts, so the dictionary has not been
  fitted to the labels.
- **Stored and offline results differ** on posts the model enriched; the offline rules column is the
  one that measures the rules.
- The extractor and the lists were not changed to improve a score after a score was seen, except by
  the documented `rules_v2` change set, which came from the diagnosis of the first run.
