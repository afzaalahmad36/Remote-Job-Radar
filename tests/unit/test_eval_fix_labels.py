"""`scripts/eval_fix_labels.py`: the deterministic label fixer.

Every example is invented. The database read is replaced, so no database, network or provider is
used, and the fixer never touches a stored or recomputed extraction.
"""

import ast
import uuid
from collections.abc import Mapping
from pathlib import Path

import pytest

from app.extraction.stack import spellings
from app.extraction.text_hash import raw_text_hash
from app.models import RemotePolicy, SalaryPeriod, Seniority
from scripts import eval_fix_labels as fix
from scripts.db_guard import ScriptError
from scripts.eval_fix_labels import (
    Change,
    PostFix,
    Session,
    entry_names,
    fix_post,
    format_changes,
    format_counts,
    format_lists,
    periods_near,
    plan_fixes,
    run_fix,
    seniority_from_title,
    split_headline,
    verify_only_allowed_changes,
)
from scripts.eval_labelfile import LabelEntry, LabelFile, Labels, load_label_file, save_label_file
from tests.settings_helpers import settings_with

ROOT = Path(__file__).resolve().parent.parent.parent
TEXT = (
    "Acme Corp: Senior Backend Engineer\n\n"
    "We use Python and k8s. Pay: $90,000 - $120,000 per year. Fully remote."
)
SETTINGS = settings_with(database_url="postgresql+asyncpg://u:secret@db:5432/jobradar")
NO_ALIASES: Mapping[str, tuple[str, ...]] = {}


def messy(**changes: object) -> Labels:
    """Labels with the mistakes the fixer corrects, for TEXT."""
    values: dict[str, object] = {
        "title": "Acme Corp: Senior Backend Engineer",
        "company": "Acme",
        "stack": ["Python", "Kubernetes", "Rust"],
        "remote_policy": RemotePolicy.REMOTE,
        "seniority": Seniority.LEAD,
        "salary_min": 90000,
        "salary_max": 130000,
        "salary_currency": "USD",
        "salary_period": SalaryPeriod.MONTH,
    }
    return Labels(**{**values, **changes})  # type: ignore[arg-type]


def fixed(
    labels: Labels, text: str = TEXT, aliases: Mapping[str, tuple[str, ...]] | None = None
) -> PostFix:
    return fix_post(1, labels, text, NO_ALIASES if aliases is None else aliases)


# --- rule 1: title and company -------------------------------------------------------------------


def test_the_headline_is_the_first_non_empty_line() -> None:
    assert split_headline("\n  \n  Acme: Backend Engineer  \nbody: more") == (
        "Acme",
        "Backend Engineer",
    )


def test_the_split_is_at_the_first_colon_and_space_and_keeps_the_spelling() -> None:
    assert split_headline("A·Team: Staff Engineer: Platform") == (
        "A·Team",
        "Staff Engineer: Platform",
    )


@pytest.mark.parametrize(
    "text", ["", "   \n ", "Backend Engineer at Acme", "Acme:Engineer", ": Engineer", "Acme: "]
)
def test_a_headline_without_company_colon_role_cannot_be_split(text: str) -> None:
    assert split_headline(text) is None


def test_title_and_company_come_from_the_headline() -> None:
    result = fixed(messy())

    assert result.labels.title == "Senior Backend Engineer"
    assert result.labels.company == "Acme Corp"
    assert [(c.field, c.before, c.after) for c in result.changes[:2]] == [
        ("title", "Acme Corp: Senior Backend Engineer", "Senior Backend Engineer"),
        ("company", "Acme", "Acme Corp"),
    ]
    assert {c.rule for c in result.changes[:2]} == {fix.RULE_HEADLINE}


def test_a_blank_company_label_is_filled_from_the_headline() -> None:
    assert fixed(messy(company=None)).labels.company == "Acme Corp"


def test_an_unsplittable_headline_changes_neither_field_and_needs_a_human() -> None:
    result = fixed(messy(title="Backend Dev", company="Acme"), "Backend Engineer at Acme\n\nbody")

    assert (result.labels.title, result.labels.company) == ("Backend Dev", "Acme")
    assert (fix.NEEDS_HUMAN, "headline is not 'Company: Role'; title and company unchanged") in (
        result.flags
    )
    assert not [c for c in result.changes if c.rule == fix.RULE_HEADLINE]


def test_rules_after_the_headline_use_the_current_title_when_it_was_not_split() -> None:
    result = fixed(messy(title="Junior Dev", seniority=None), "Backend Engineer at Acme\n\nbody")

    assert result.labels.seniority == Seniority.JUNIOR


# --- rule 2: seniority ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Senior Backend Engineer", "senior"),
        ("SENIOR engineer", "senior"),
        ("Intern, Data", "intern"),
        ("Mid-level Developer", "mid"),
        ("Principal Engineer", "principal"),
        ("Lead Engineer", "lead"),
        ("Senior Senior Engineer", "senior"),
        ("Junior Developer", "junior"),
        ("Staff Engineer", None),
        ("Engineering Manager", None),
        ("Director of Engineering", None),
        ("Head of Platform", None),
        ("Senior Lead Engineer", None),
        ("Internal Tools Engineer", None),
        ("Backend Engineer", None),
        (None, None),
    ],
)
def test_seniority_is_the_one_enum_word_in_the_title(
    title: str | None, expected: str | None
) -> None:
    assert seniority_from_title(title) == expected


def test_a_wrong_seniority_is_replaced_and_a_missing_word_clears_it() -> None:
    assert fixed(messy()).labels.seniority == Seniority.SENIOR
    cleared = fixed(messy(), "Acme Corp: Staff Engineer\n\nbody")
    assert cleared.labels.seniority is None
    assert ("seniority", "lead", "(none)", fix.RULE_SENIORITY) in [
        (c.field, c.before, c.after, c.rule) for c in cleared.changes
    ]


# --- rule 3: salary ------------------------------------------------------------------------------


def test_an_amount_not_in_the_text_is_removed_and_listed_and_the_other_is_kept() -> None:
    result = fixed(messy())

    assert (result.labels.salary_min, result.labels.salary_max) == (90000, None)
    assert (fix.SALARY_REMOVED, "salary_max is not in the text") in result.flags
    assert ("salary_max", "130000", "(none)", fix.RULE_AMOUNT) in [
        (c.field, c.before, c.after, c.rule) for c in result.changes
    ]


@pytest.mark.parametrize("amount", ["$90,000", "90k", "90 000", "90.000"])
def test_amounts_are_found_with_separators_and_a_k_suffix(amount: str) -> None:
    text = f"Acme: Engineer\n\nPay {amount} per year."

    result = fixed(messy(salary_min=90000, salary_max=None), text)

    assert result.labels.salary_min == 90000
    assert not result.flags


@pytest.mark.parametrize(
    ("phrase", "period"),
    [
        ("per hour", "hour"),
        ("hourly", "hour"),
        ("$90/hr", "hour"),
        ("per year", "year"),
        ("annual", "year"),
        ("annually", "year"),
        ("yearly", "year"),
        ("a year", "year"),
        ("$90/yr", "year"),
        ("per month", "month"),
        ("monthly", "month"),
        ("per week", "week"),
        ("weekly", "week"),
        ("per day", "day"),
        ("daily", "day"),
    ],
)
def test_the_period_comes_from_the_word_next_to_the_amount(phrase: str, period: str) -> None:
    text = f"Acme: Engineer\n\nPay 90000 {phrase}."
    if phrase.startswith("$"):
        text = f"Acme: Engineer\n\nPay 90000{phrase[3:]}."

    result = fixed(messy(salary_min=90000, salary_max=None, salary_period=None), text)

    assert result.labels.salary_period is not None
    assert result.labels.salary_period.value == period


def test_a_period_word_must_be_within_sixty_characters_of_an_amount() -> None:
    near = "Acme: Engineer\n\n90000 " + "x" * 10 + " per year"
    far = "Acme: Engineer\n\n90000 " + "x" * 80 + " per year"

    assert fixed(messy(salary_min=90000, salary_max=None), near).labels.salary_period is not None
    assert fixed(messy(salary_min=90000, salary_max=None), far).labels.salary_period is None


@pytest.mark.parametrize("text", ["HR Manager 90000", "90000 for 5 years", "90000 in 4 weeks"])
def test_hr_the_department_and_plural_durations_are_not_periods(text: str) -> None:
    assert periods_near({90000}, text.casefold()) == set()


def test_a_period_label_with_no_period_word_near_the_amount_is_cleared() -> None:
    result = fixed(messy(salary_min=90000, salary_max=None), "Acme: Engineer\n\nPay 90000 flat.")

    assert result.labels.salary_period is None
    assert ("salary_period", "month", "(none)", fix.RULE_PERIOD) in [
        (c.field, c.before, c.after, c.rule) for c in result.changes
    ]


def test_two_different_periods_near_the_amounts_leave_the_period_and_need_a_human() -> None:
    text = "Acme: Engineer\n\n90000 per year, or 50 per hour. Overtime 90000 daily"
    result = fixed(messy(salary_min=90000, salary_max=None, salary_period=SalaryPeriod.YEAR), text)

    assert result.labels.salary_period == SalaryPeriod.YEAR
    assert (fix.NEEDS_HUMAN, "more than one salary period near the amounts; period unchanged") in (
        result.flags
    )


def test_with_no_amount_left_currency_and_period_go_too() -> None:
    result = fixed(messy(salary_min=70000, salary_max=80000), "Acme: Engineer\n\nCompetitive pay.")

    labels = result.labels
    assert (labels.salary_min, labels.salary_max) == (None, None)
    assert (labels.salary_currency, labels.salary_period) == (None, None)
    assert {c.field: c.rule for c in result.changes if c.field.startswith("salary")} == {
        "salary_min": fix.RULE_AMOUNT,
        "salary_max": fix.RULE_AMOUNT,
        "salary_currency": fix.RULE_NO_AMOUNTS,
        "salary_period": fix.RULE_NO_AMOUNTS,
    }


def test_a_stray_currency_without_amounts_is_cleared_without_a_listing() -> None:
    result = fixed(
        messy(salary_min=None, salary_max=None, salary_period=None), "Acme: Engineer\n\nbody"
    )

    assert result.labels.salary_currency is None
    assert not [flag for flag in result.flags if flag[0] == fix.SALARY_REMOVED]


# --- rule 4: stack -------------------------------------------------------------------------------


def test_an_entry_missing_from_the_text_is_removed_and_the_rest_keep_their_order() -> None:
    result = fixed(messy(stack=["Rust", "Python", "Kubernetes", "Go"]))

    assert result.labels.stack == ["Python", "Kubernetes"]
    assert [(c.before, c.after, c.rule) for c in result.changes if c.field == "stack"] == [
        ("Rust", "(removed)", fix.RULE_STACK),
        ("Go", "(removed)", fix.RULE_STACK),
    ]


def test_a_dictionary_alias_keeps_the_entry() -> None:
    # TEXT says "k8s", not "Kubernetes"; "Postgres" is a dictionary spelling of PostgreSQL
    text = TEXT + " Also postgres."

    assert fixed(messy(stack=["Kubernetes", "PostgreSQL"]), text).labels.stack == [
        "Kubernetes",
        "PostgreSQL",
    ]


def test_a_policy_alias_keeps_the_entry() -> None:
    text = "Acme: Engineer\n\nWe use Acme DB daily."  # the post never writes "AcmeDatabase"
    aliases = {"acme db": ("AcmeDatabase",)}

    assert fixed(messy(stack=["AcmeDatabase"]), text, aliases).labels.stack == ["AcmeDatabase"]
    assert fixed(messy(stack=["AcmeDatabase"]), text).labels.stack == []


def test_matching_is_by_whole_word_and_ignores_case() -> None:
    text = "Acme: Engineer\n\nWe use JavaScript and PYTHON."

    assert fixed(messy(stack=["Java", "python", "Script"]), text).labels.stack == ["python"]


def test_the_fixer_never_adds_a_stack_entry() -> None:
    assert fixed(messy(stack=[])).labels.stack == []


def test_a_name_for_several_products_is_kept_when_one_of_them_is_in_the_text() -> None:
    aliases = {"react vite": ("React", "Vite")}
    text = "Acme: Engineer\n\nWe use vite."

    assert fixed(messy(stack=["React Vite"]), text, aliases).labels.stack == ["React Vite"]


def test_an_alias_of_one_product_does_not_make_its_sibling_count() -> None:
    aliases = {"react vite": ("React", "Vite")}

    assert "vite" not in entry_names("React", aliases)
    assert "vite" in entry_names("React Vite", aliases)


def test_entry_names_include_dictionary_spellings_and_policy_aliases() -> None:
    aliases = {"google cloud platform": ("GCP",)}

    names = entry_names("Google Cloud Platform", aliases)

    assert {"google cloud platform", "gcp", "google cloud"} <= names
    assert entry_names("Mystery Tool", aliases) == {"mystery tool"}


def test_the_stack_dictionary_lists_the_spellings_of_a_known_technology() -> None:
    assert {"Kubernetes", "kubernetes", "k8s"} == set(spellings("k8s"))
    assert spellings("Mystery Tool") == ()


# --- rule 5: untouched, and the whole file -------------------------------------------------------


def test_remote_policy_is_never_touched() -> None:
    assert (
        fixed(messy(remote_policy=RemotePolicy.ONSITE)).labels.remote_policy == RemotePolicy.ONSITE
    )
    assert fixed(messy(remote_policy=None)).labels.remote_policy is None


def test_labels_that_already_follow_the_rules_produce_no_changes() -> None:
    clean = messy(
        title="Senior Backend Engineer",
        company="Acme Corp",
        stack=["Python", "Kubernetes"],
        seniority=Seniority.SENIOR,
        salary_max=120000,
        salary_period=SalaryPeriod.YEAR,
    )

    result = fixed(clean)

    assert result.changes == [] and result.flags == []
    assert result.labels == clean


def entry(labels: Labels, text: str = TEXT, job: int = 1) -> LabelEntry:
    return LabelEntry(
        job_id=uuid.UUID(int=job),
        raw_text_hash=raw_text_hash(text),
        status="labelled",
        labels=labels,
    )


def skipped(job: int) -> LabelEntry:
    return LabelEntry(job_id=uuid.UUID(int=job), raw_text_hash=raw_text_hash("s"), status="skipped")


def label_file(*entries: LabelEntry) -> LabelFile:
    return LabelFile(
        source_id=uuid.UUID(int=99),
        seed=1,
        selected=[e.job_id for e in entries],
        posts=list(entries),
    )


def texts(**by_job: str) -> dict[uuid.UUID, str]:
    return {uuid.UUID(int=int(job.removeprefix("j"))): text for job, text in by_job.items()}


def test_the_plan_numbers_labelled_posts_and_keeps_skipped_ones_as_they_are() -> None:
    skip = skipped(5)
    original = label_file(skip, entry(messy(), job=1), entry(messy(), job=2))

    plan = plan_fixes(original, texts(j1=TEXT, j2=TEXT), NO_ALIASES)

    assert plan.fixed.posts[0] == skip
    assert {c.post for c in plan.changes} == {1, 2}
    assert plan.fixed.posts[1].job_id == uuid.UUID(int=1)
    verify_only_allowed_changes(original, plan.fixed)


def test_a_missing_or_stale_post_is_listed_and_not_touched() -> None:
    changed = TEXT + " (edited)"
    original = label_file(entry(messy(), job=1), entry(messy(), job=2))

    plan = plan_fixes(original, {uuid.UUID(int=2): changed}, NO_ALIASES)

    assert plan.changes == []
    assert plan.fixed.posts == original.posts
    assert sorted(plan.lists[fix.NEEDS_HUMAN]) == [
        (1, "post not found; not touched"),
        (2, "post text changed since it was labelled; not touched"),
    ]


def test_posts_with_an_empty_stack_are_listed_as_info() -> None:
    original = label_file(entry(messy(stack=["Rust"]), job=1), entry(messy(), job=2))

    plan = plan_fixes(original, texts(j1=TEXT, j2=TEXT), NO_ALIASES)

    assert plan.lists[fix.INFO] == [(1, "stack is empty")]


def test_posts_that_end_up_with_identical_labels_are_listed_but_left_alone() -> None:
    original = label_file(
        entry(messy(), job=1), entry(messy(title="Other"), job=2), entry(messy(), job=3)
    )

    plan = plan_fixes(original, texts(j1=TEXT, j2=TEXT, j3=TEXT), NO_ALIASES)

    assert plan.lists[fix.IDENTICAL] == [(1, "same labels as posts 2, 3")]
    assert [e.labels for e in plan.fixed.posts][0] == [e.labels for e in plan.fixed.posts][1]


# --- the safety check ----------------------------------------------------------------------------


def with_labels(original: LabelFile, labels: Labels) -> LabelFile:
    post = original.posts[0].model_copy(update={"labels": labels})
    return original.model_copy(update={"posts": [post, *original.posts[1:]]})


def test_the_verification_accepts_changes_to_the_fixable_fields_only() -> None:
    original = label_file(entry(messy()))
    changed = messy(title="T", company="C", seniority=None, stack=["Python"], salary_min=None)

    verify_only_allowed_changes(original, with_labels(original, changed))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"remote_policy": RemotePolicy.HYBRID}, "outside the fixable ones"),
        ({"stack": ["Python", "Brand New"]}, "added or reordered"),
        ({"stack": ["Kubernetes", "Python"]}, "added or reordered"),
    ],
)
def test_the_verification_refuses_other_changes(changes: dict[str, object], message: str) -> None:
    original = label_file(entry(messy()))

    with pytest.raises(ScriptError, match=message):
        verify_only_allowed_changes(original, with_labels(original, messy(**changes)))


def test_the_verification_refuses_header_id_hash_status_and_count_changes() -> None:
    original = label_file(entry(messy(), job=1), skipped(2))
    first = original.posts[0]

    cases = [
        (original.model_copy(update={"seed": 2}), "header"),
        (original.model_copy(update={"posts": [first]}), "number of posts"),
        (
            original.model_copy(
                update={
                    "posts": [first.model_copy(update={"job_id": uuid.UUID(int=7)}), skipped(2)]
                }
            ),
            "id, hash or status",
        ),
        (
            original.model_copy(
                update={"posts": [first.model_copy(update={"raw_text_hash": "0" * 64}), skipped(2)]}
            ),
            "id, hash or status",
        ),
        (
            original.model_copy(
                update={"posts": [first, skipped(2).model_copy(update={"labels": messy()})]}
            ),
            "skipped post gained labels",
        ),
    ]
    for changed, message in cases:
        with pytest.raises(ScriptError, match=message):
            verify_only_allowed_changes(original, changed)


# --- the table and the lists ---------------------------------------------------------------------


def test_the_table_has_a_header_and_one_row_per_change() -> None:
    lines = format_changes([Change(3, "title", "Acme: Dev", "Dev", fix.RULE_HEADLINE)])

    assert lines[0].split(" | ")[0].strip() == "post"
    assert [cell.strip() for cell in lines[2].split(" | ")] == [
        "3",
        "title",
        "Acme: Dev",
        "Dev",
        fix.RULE_HEADLINE,
    ]


def test_counts_are_given_for_every_rule_even_when_zero() -> None:
    lines = format_counts([Change(1, "stack", "Rust", "(removed)", fix.RULE_STACK)])

    assert len(lines) == len(fix.RULES)
    assert "  rule 4 not in text: 1 change(s) in 1 post(s)" in lines
    assert "  rule 1 headline: 0 change(s) in 0 post(s)" in lines


def test_the_lists_are_printed_by_name_and_post() -> None:
    lines = format_lists({fix.INFO: [(4, "stack is empty")], fix.NEEDS_HUMAN: [(2, "why")]})

    assert lines == ["  NEEDS_HUMAN: post 2: why", "  INFO: post 4: stack is empty"]


# --- the command ---------------------------------------------------------------------------------


def fake_reader(monkeypatch: pytest.MonkeyPatch, by_job: dict[int, str]) -> list[list[uuid.UUID]]:
    asked: list[list[uuid.UUID]] = []

    async def fake(database_url: str, ids: list[uuid.UUID]) -> list[tuple[uuid.UUID, str]]:
        asked.append(ids)
        return [(uuid.UUID(int=n), text) for n, text in by_job.items()]

    monkeypatch.setattr(fix, "read_posts_by_id", fake)
    return asked


def session(answer: str | None, lines: list[str]) -> Session:
    def ask(prompt: str) -> str:
        lines.append(f"ASKED {prompt.strip()}")
        if answer is None:
            raise EOFError
        return answer

    return Session(ask=ask, out=lines.append)


def test_the_command_backs_up_shows_asks_writes_and_rechecks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    original = label_file(entry(messy()))
    save_label_file(path, original)
    before = path.read_bytes()
    asked = fake_reader(monkeypatch, {1: TEXT})
    lines: list[str] = []

    code = run_fix(path, SETTINGS, session("y", lines))

    output = "\n".join(lines)
    assert asked == [[uuid.UUID(int=1)]]
    assert (tmp_path / "labels.clean-v1.json").read_bytes() == before
    assert "Acme Corp: Senior Backend Engineer" in output  # the table may show text
    assert "ASKED Write these changes? [y/N]" in lines
    assert "secret" not in output
    written = load_label_file(path).posts[0].labels
    assert written is not None
    assert (written.title, written.company) == ("Senior Backend Engineer", "Acme Corp")
    assert written.remote_policy == RemotePolicy.REMOTE
    assert written.stack == ["Python", "Kubernetes"]
    assert code == 0  # the re-check finds nothing wrong with the corrected labels
    assert "Checked 1 labelled posts; 0 have problems." in lines


def test_the_recheck_after_writing_prints_codes_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(messy()), entry(messy(), job=2)))
    fake_reader(monkeypatch, {1: TEXT, 2: TEXT})
    lines: list[str] = []

    assert run_fix(path, SETTINGS, session("y", lines)) == 1

    start = lines.index("Written. Checking the labels against the post text again (codes only):")
    report = "\n".join(lines[start + 1 :])
    assert "identical_labels (with post 2)" in report
    assert "Backend" not in report and "Acme" not in report and "Python" not in report


def test_declining_changes_nothing_but_the_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(messy())))
    before = path.read_bytes()
    fake_reader(monkeypatch, {1: TEXT})
    lines: list[str] = []

    for answer in ("", "n", "no", "anything"):
        assert run_fix(path, SETTINGS, session(answer, lines)) == 1
        assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["labels.clean-v1.json", "labels.json"]


def test_end_of_input_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(messy())))
    before = path.read_bytes()
    fake_reader(monkeypatch, {1: TEXT})
    lines: list[str] = []

    assert run_fix(path, SETTINGS, session(None, lines)) == 1
    assert path.read_bytes() == before
    assert "Input ended. Nothing was changed." in "\n".join(lines)


def test_an_existing_backup_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(messy())))
    backup = tmp_path / "labels.clean-v1.json"
    backup.write_text("older copy", encoding="utf-8")
    fake_reader(monkeypatch, {1: TEXT})
    lines: list[str] = []

    run_fix(path, SETTINGS, session("n", lines))

    assert backup.read_text(encoding="utf-8") == "older copy"
    assert any("already there, kept as it is" in line for line in lines)


def test_nothing_to_change_asks_nothing_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    clean = messy(
        title="Senior Backend Engineer",
        company="Acme Corp",
        stack=["Python"],
        seniority=Seniority.SENIOR,
        salary_max=120000,
        salary_period=SalaryPeriod.YEAR,
    )
    save_label_file(path, label_file(entry(clean)))
    before = path.read_bytes()
    fake_reader(monkeypatch, {1: TEXT})
    lines: list[str] = []

    assert run_fix(path, SETTINGS, session("y", lines)) == 0
    assert path.read_bytes() == before
    assert "Nothing to change." in lines and not any(line.startswith("ASKED") for line in lines)


def test_the_wrong_database_is_refused_before_anything_is_read_or_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(messy())))
    asked = fake_reader(monkeypatch, {1: TEXT})
    wrong = settings_with(database_url="postgresql+asyncpg://u:secret@db:5432/jobradar_test")

    with pytest.raises(ScriptError, match="refusing to read database"):
        run_fix(path, wrong, session("y", []))

    assert asked == []
    assert [p.name for p in tmp_path.iterdir()] == ["labels.json"]


def test_a_missing_labels_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ScriptError, match="does not exist"):
        run_fix(tmp_path / "nope.json", SETTINGS, session("y", []))


def test_main_runs_the_fixer_and_reports_errors_with_exit_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(fix, "get_settings", lambda: SETTINGS)

    assert fix.main(["--labels", str(tmp_path / "nope.json")]) == 2
    assert "error:" in capsys.readouterr().err

    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(messy())))
    fake_reader(monkeypatch, {1: TEXT})
    assert fix.main(["--labels", str(path)], session("n", [])) == 1


# --- no provider, no extraction ------------------------------------------------------------------


def test_the_fixer_cannot_reach_a_provider_or_an_extraction() -> None:
    source = (ROOT / "scripts" / "eval_fix_labels.py").read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)

    forbidden = ("app.llm", "httpx", "requests", "aiohttp", "app.models", "app.extraction.rules")
    assert not {i for i in imported if i.startswith(forbidden)}
    assert not {i for i in imported if i.startswith(("scripts.eval_run", "scripts.try_extraction"))}
    assert not {i for i in imported if "build_client" in i or "merge" in i}
    for word in ("read_stored_jobs", "rules_values", "extraction_method", "llm_enrichment"):
        assert word not in source
