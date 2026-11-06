"""`eval_clean_labels.py --check-labels`: label problems against the post text.

Every example is invented. The database read is replaced, so no database is used; nothing here
can reach a provider, and the check never touches a stored or recomputed extraction.
"""

import ast
import uuid
from pathlib import Path

import pytest

from app.extraction.text_hash import raw_text_hash
from app.models import RemotePolicy, SalaryPeriod, Seniority
from scripts import eval_check_labels as check
from scripts import eval_clean_labels as clean
from scripts.db_guard import ScriptError
from scripts.eval_check_labels import (
    Problem,
    amount_spans,
    check_labels,
    check_post,
    fold,
    format_report,
    run_check,
)
from scripts.eval_labelfile import LabelEntry, LabelFile, Labels, save_label_file
from tests.settings_helpers import settings_with

ROOT = Path(__file__).resolve().parent.parent.parent
TEXT = (
    "Senior Backend Engineer at Acme Corp. We use Python, k8s and PostgreSQL. "
    "Salary: $90,000 - $120,000 per year. Fully remote."
)


def good(**changes: object) -> Labels:
    """Labels that agree with TEXT; each test breaks one thing."""
    values: dict[str, object] = {
        "title": "Senior Backend Engineer",
        "company": "Acme Corp",
        "stack": ["Python", "Kubernetes", "Postgres"],
        "remote_policy": RemotePolicy.REMOTE,
        "seniority": Seniority.SENIOR,
        "salary_min": 90000,
        "salary_max": 120000,
        "salary_currency": "USD",
        "salary_period": SalaryPeriod.YEAR,
    }
    return Labels(**{**values, **changes})  # type: ignore[arg-type]


def codes(labels: Labels, text: str = TEXT) -> list[str]:
    return [problem.code for problem in check_post(labels, text)]


def test_labels_that_agree_with_the_text_have_no_problems() -> None:
    assert check_post(good(), TEXT) == []


# --- numbers -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("$90,000 a year", 90000),
        ("90k", 90000),
        ("90K", 90000),
        ("1.5k", 1500),
        ("90 000 EUR", 90000),
        ("90.000 EUR", 90000),
        ("EUR 1,200,000", 1200000),
        ("75", 75),
        ("2024", 2024),
    ],
)
def test_amounts_are_read_with_separators_and_k(text: str, value: int) -> None:
    assert value in {v for v, _, _ in amount_spans(fold(text))}


def test_a_number_inside_a_word_is_not_an_amount() -> None:
    assert amount_spans(fold("sha256 and v2")) == []


# --- (1) title -----------------------------------------------------------------------------------


def test_a_title_not_in_the_text_is_reported() -> None:
    assert codes(good(title="Staff Platform Engineer")) == ["title_not_in_text"]


def test_a_title_matches_ignoring_case_and_spacing() -> None:
    assert codes(good(title="senior   BACKEND engineer")) == []


# --- (2) company ---------------------------------------------------------------------------------


def test_a_company_over_40_characters_is_reported() -> None:
    name = "Acme Corporation International Holdings Group Limited"
    text = f"Senior Backend Engineer at {name}. Python k8s Postgres $90,000 $120,000 per year"

    assert codes(good(company=name), text) == ["company_too_long"]


def test_a_company_with_a_comma_or_missing_from_the_text_is_reported() -> None:
    assert codes(good(company="Acme, Corp")) == ["company_comma", "company_not_in_text"]
    assert codes(good(company="Zeta Labs")) == ["company_not_in_text"]


def test_a_blank_company_is_not_checked() -> None:
    assert codes(good(company=None)) == []


# --- (3) stack -----------------------------------------------------------------------------------


def test_stack_entries_missing_from_the_text_are_counted_not_named() -> None:
    problems = check_post(good(stack=["Python", "Rust", "Docker", "Erlang"]), TEXT)

    assert problems == [Problem("stack_not_in_text", 3)]
    assert problems[0].text() == "stack_not_in_text x3"


def test_a_stack_entry_must_be_a_whole_word() -> None:
    assert codes(good(stack=["Java"]), TEXT + " We also use JavaScript.") == ["stack_not_in_text"]
    assert codes(good(stack=["Go"]), TEXT + " Good people.") == ["stack_not_in_text"]
    assert codes(good(stack=["Go"]), TEXT + " We write Go services.") == []


def test_aliases_work_both_ways_and_through_the_dictionary_spelling() -> None:
    assert codes(good(stack=["Kubernetes"])) == []  # the text says k8s
    assert codes(good(stack=["k8s"]), TEXT.replace("k8s", "Kubernetes")) == []
    assert codes(good(stack=["Golang"]), TEXT + " We write Go services.") == []
    assert codes(good(stack=["GCP"]), TEXT + " Google Cloud Platform.") == []
    assert codes(good(stack=["Kafka"]), TEXT + " Apache Kafka.") == []


def test_special_characters_in_a_stack_entry_are_matched_literally() -> None:
    assert codes(good(stack=["C++"]), TEXT + " We write C++ daily.") == []
    assert codes(good(stack=["Node.js"]), TEXT + " Node.js backend.") == []
    assert codes(good(stack=["C++"]), TEXT + " We write C daily.") == ["stack_not_in_text"]


# --- (4) salary amounts --------------------------------------------------------------------------


def test_a_salary_amount_not_in_the_text_is_reported() -> None:
    assert codes(good(salary_min=95000)) == ["salary_min_not_in_text"]
    assert codes(good(salary_max=130000)) == ["salary_max_not_in_text"]
    # With no labelled amount in the text, the period has no amount to sit next to either.
    assert codes(good(salary_min=95000, salary_max=130000)) == [
        "salary_min_not_in_text",
        "salary_max_not_in_text",
        "period_no_word_near_amount",
    ]


def test_amounts_may_be_written_with_k() -> None:
    text = "Senior Backend Engineer at Acme Corp. Python k8s Postgres. 90k-120k a year."

    assert codes(good(), text) == []


# --- (5) salary period ---------------------------------------------------------------------------


def test_a_period_needs_a_period_word_within_60_characters_of_an_amount() -> None:
    far = (
        "Senior Backend Engineer at Acme Corp. Python k8s Postgres. Pay is 90000 and 120000."
        + (" Great team and culture." * 6)
        + " Paid per year."
    )
    near = "Senior Backend Engineer at Acme Corp. Python k8s Postgres. 90000 and 120000 /hr"

    assert codes(good(), far) == ["period_no_word_near_amount"]
    assert codes(good(salary_period=SalaryPeriod.HOUR), near) == []


@pytest.mark.parametrize("word", ["year", "annual", "month", "hour", "per", "/hr", "/yr"])
def test_every_listed_period_word_counts(word: str) -> None:
    text = f"Senior Backend Engineer at Acme Corp. Python k8s Postgres. 90000 and 120000 {word}"

    assert codes(good(), text) == []


def test_a_word_that_only_contains_a_period_word_does_not_count() -> None:
    text = "Senior Backend Engineer at Acme Corp. Python k8s Postgres. 90000 and 120000 today"

    assert codes(good(), text) == ["period_no_word_near_amount"]


def test_a_period_with_no_amount_in_the_labels_or_text_is_reported() -> None:
    labels = good(salary_min=None, salary_max=None)

    assert codes(labels) == ["period_no_word_near_amount"]


def test_no_period_label_means_no_period_check() -> None:
    text = "Senior Backend Engineer at Acme Corp. Python k8s Postgres. 90000 and 120000"

    assert codes(good(salary_period=None), text) == []


# --- (6) seniority -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title", ["Junior Dev", "Mid-level Dev", "Lead Dev", "Staff Dev", "Principal Dev", "Intern"]
)
def test_a_level_word_in_the_title_satisfies_the_seniority_check(title: str) -> None:
    text = f"{title} at Acme Corp. Python k8s Postgres $90,000 $120,000 per year"

    assert codes(good(title=title), text) == []


def test_a_seniority_without_a_level_word_in_the_title_is_reported() -> None:
    text = "Backend Engineer at Acme Corp. Python k8s Postgres $90,000 $120,000 per year"

    assert codes(good(title="Backend Engineer"), text) == ["seniority_no_level_word_in_title"]
    assert codes(
        good(title="Internal Tools Engineer"), text.replace("Backend", "Internal Tools")
    ) == ["seniority_no_level_word_in_title"]


def test_no_seniority_label_means_no_seniority_check() -> None:
    text = "Backend Engineer at Acme Corp. Python k8s Postgres $90,000 $120,000 per year"

    assert codes(good(title="Backend Engineer", seniority=None), text) == []


# --- (7) identical labels and the whole file -----------------------------------------------------


def entry(labels: Labels, text: str = TEXT, job: int = 1) -> LabelEntry:
    return LabelEntry(
        job_id=uuid.UUID(int=job),
        raw_text_hash=raw_text_hash(text),
        status="labelled",
        labels=labels,
    )


def label_file(*entries: LabelEntry, skipped: int = 0) -> LabelFile:
    posts = list(entries)
    for n in range(skipped):
        posts.append(
            LabelEntry(
                job_id=uuid.UUID(int=900 + n), raw_text_hash=raw_text_hash(str(n)), status="skipped"
            )
        )
    return LabelFile(
        source_id=uuid.UUID(int=99), seed=1, selected=[e.job_id for e in posts], posts=posts
    )


def test_two_posts_with_the_same_labels_are_reported_with_their_partner() -> None:
    first = good(stack=["Python", "Kubernetes", "Postgres"])
    second = good(stack=["postgres", "PYTHON", "kubernetes"])  # same entries, other order and case
    other = good(title="Senior Backend Engineer II")
    texts = {uuid.UUID(int=n): TEXT for n in (1, 2, 3)}

    report = check_labels(
        label_file(entry(first, job=1), entry(second, job=2), entry(other, job=3)), texts
    )

    assert [p.text() for p in report.problems[1]] == ["identical_labels (with post 2)"]
    assert [p.text() for p in report.problems[2]] == ["identical_labels (with post 1)"]
    assert 3 in report.problems  # its own title is not in the text
    assert "identical_labels" not in [p.code for p in report.problems[3]]


def test_a_third_identical_post_names_both_partners() -> None:
    texts = {uuid.UUID(int=n): TEXT for n in (1, 2, 3)}

    report = check_labels(label_file(*(entry(good(), job=n) for n in (1, 2, 3))), texts)

    assert report.problems[1][0].detail == "with post 2, 3"


def test_a_missing_or_changed_text_is_reported_and_not_checked_further() -> None:
    entries = (entry(good(title="Nothing Like It"), job=1), entry(good(company="Zeta"), job=2))
    texts = {uuid.UUID(int=2): TEXT + " It changed."}

    report = check_labels(label_file(*entries), texts)

    assert [p.code for p in report.problems[1]] == ["job_not_found"]
    assert [p.code for p in report.problems[2]] == ["stale_text"]


def test_skipped_posts_are_not_checked_and_do_not_shift_the_numbers() -> None:
    entries = (entry(good(), job=1), entry(good(title="Staff Platform Engineer"), job=2))
    texts = {uuid.UUID(int=1): TEXT, uuid.UUID(int=2): TEXT}

    report = check_labels(label_file(*entries, skipped=2), texts)

    assert report.checked == 2 and list(report.problems) == [2]


# --- the report ----------------------------------------------------------------------------------


def test_the_report_has_post_numbers_and_codes_only_never_text() -> None:
    secret_title, secret_company = "Zebra Wrangler Supreme", "Quixotic Holdings"
    labels = good(title=secret_title, company=secret_company, stack=["Frobnicate"])
    report = check_labels(label_file(entry(labels)), {uuid.UUID(int=1): TEXT})

    lines = format_report(report)

    joined = "\n".join(lines)
    for secret in (secret_title, secret_company, "Frobnicate", "Acme", "Python"):
        assert secret not in joined
    assert lines[0] == (
        "post 1: title_not_in_text, company_not_in_text, stack_not_in_text, "
        "seniority_no_level_word_in_title"
    )
    assert "Checked 1 labelled posts; 1 have problems." in lines
    assert "  stack_not_in_text: 1 post(s)" in lines


def test_a_clean_file_reports_no_problems() -> None:
    report = check_labels(label_file(entry(good())), {uuid.UUID(int=1): TEXT})

    assert format_report(report) == ["Checked 1 labelled posts; 0 have problems."]


# --- the command ---------------------------------------------------------------------------------


def fake_reader(monkeypatch: pytest.MonkeyPatch, texts: dict[int, str]) -> list[list[uuid.UUID]]:
    asked: list[list[uuid.UUID]] = []

    async def fake(database_url: str, ids: list[uuid.UUID]) -> list[tuple[uuid.UUID, str]]:
        asked.append(ids)
        return [(uuid.UUID(int=n), text) for n, text in texts.items()]

    monkeypatch.setattr(check, "read_posts_by_id", fake)
    return asked


SETTINGS = settings_with(database_url="postgresql+asyncpg://u:secret@db:5432/jobradar")


def test_the_command_reads_text_prints_codes_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(good()), entry(good(title="Platform Dev"), job=2)))
    before = path.read_bytes()
    asked = fake_reader(monkeypatch, {1: TEXT, 2: TEXT})
    output: list[str] = []

    code = run_check(path, SETTINGS, output.append)

    assert code == 1 and asked == [[uuid.UUID(int=1), uuid.UUID(int=2)]]
    assert output[0] == "Reading post text (read-only) from database db:5432/jobradar"
    assert "post 2: title_not_in_text, seniority_no_level_word_in_title" in output
    assert "secret" not in "\n".join(output)
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["labels.json"]  # no backup, no report file


def test_the_command_exits_zero_when_nothing_is_wrong(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(good())))
    fake_reader(monkeypatch, {1: TEXT})

    assert run_check(path, SETTINGS, lambda line: None) == 0


def test_the_command_refuses_the_wrong_database_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(good())))
    asked = fake_reader(monkeypatch, {1: TEXT})
    wrong = settings_with(database_url="postgresql+asyncpg://u:secret@db:5432/jobradar_test")

    with pytest.raises(ScriptError, match="refusing to read database"):
        run_check(path, wrong, lambda line: None)
    assert asked == []


def test_check_labels_is_wired_into_the_cleaner_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def fake_run_check(path: Path, settings: object) -> int:
        calls.append(path)
        return 0

    monkeypatch.setattr(clean, "run_check", fake_run_check)
    monkeypatch.setattr(clean, "get_settings", lambda: SETTINGS)
    labels = tmp_path / "labels.json"

    assert clean.main(["--check-labels", "--labels", str(labels)]) == 0

    assert calls == [labels]
    assert not (tmp_path / "labels.raw.json").exists()  # checking never makes a backup


@pytest.mark.parametrize("extra", [["--policy"], ["--policy-file", "x.toml"]])
def test_check_labels_cannot_be_combined_with_policy_options(
    extra: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        clean.main(["--check-labels", *extra])

    assert "--check-labels cannot be combined" in capsys.readouterr().err


# --- no provider, no extraction ------------------------------------------------------------------


def test_the_check_module_cannot_reach_a_provider_or_an_extraction() -> None:
    source = (ROOT / "scripts" / "eval_check_labels.py").read_text(encoding="utf-8")
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
