"""`scripts/eval_complete_labels.py`: the second pass over the labels.

Every post and label is invented. The database read is replaced, so no database, network or
provider is used, and the script never touches a stored or recomputed extraction.
"""

import ast
import uuid
from pathlib import Path

import pytest

from app.extraction.text_hash import raw_text_hash
from scripts import eval_complete_labels as complete
from scripts.db_guard import ScriptError
from scripts.eval_clean_policy import Policy
from scripts.eval_complete_labels import (
    PostResult,
    complete_salary,
    complete_stack,
    find_candidates,
    format_changes,
    known_technologies,
    plan_completion,
    run_complete,
    verify_only_allowed_changes,
)
from scripts.eval_fix_labels import Session
from scripts.eval_labelfile import LabelEntry, LabelFile, Labels, load_label_file, save_label_file
from tests.settings_helpers import settings_with

ROOT = Path(__file__).resolve().parent.parent.parent
SETTINGS = settings_with(database_url="postgresql+asyncpg://u:secret@db:5432/jobradar")
POLICY = Policy(
    allow=frozenset({"decagon", "databricks", "maven"}),
    deny=frozenset({"llm"}),
    fragments=frozenset(),
)
ALLOW = [
    "Decagon",
    "Databricks",
    "Maven",
]  # the allow-list as written in the policy file (the casing is kept)
TECHNOLOGIES = known_technologies(POLICY, ALLOW)


def salary_of(labels: Labels, text: str) -> tuple[Labels, PostResult]:
    result = PostResult(labels)
    return complete_salary(1, labels, text, result), result


def stack_of(labels: Labels, text: str, max_adds: int = 8) -> tuple[Labels, PostResult]:
    result = PostResult(labels)
    return complete_stack(1, labels, text, TECHNOLOGIES, result, max_adds), result


# --- salary --------------------------------------------------------------------------------------


def test_a_blank_salary_is_filled_from_the_one_range_the_post_states() -> None:
    text = "Acme: Engineer\n\nSalary: $90,000 - $120,000 per year."

    labels, result = salary_of(Labels(), text)

    assert (labels.salary_min, labels.salary_max, labels.salary_currency) == (
        90_000,
        120_000,
        "USD",
    )
    assert labels.salary_period is None  # the period is not touched
    assert {c.field for c in result.changes} == {"salary_min", "salary_max", "salary_currency"}


@pytest.mark.parametrize(
    ("sentence", "currency"),
    [
        ("Competitive salary ($140K-$180K CAN) and stock options", "CAD"),
        ("Competitive salary ($140K-$180K CAD) and stock options", "CAD"),
        ("Competitive salary ($140K-$180K USD) and stock options", "USD"),
        ("Competitive salary $140K-$180K, and you can apply", "USD"),
    ],
)
def test_the_currency_comes_from_the_sign_or_code_in_the_sentence(
    sentence: str, currency: str | None
) -> None:
    labels, _ = salary_of(Labels(), sentence)

    assert labels.salary_currency == currency


def test_dots_and_commas_in_groups_of_three_are_thousands_and_other_dots_are_decimals() -> None:
    euro, _ = salary_of(Labels(), "Salary €50.000 - €65.000 a year")
    assert (euro.salary_min, euro.salary_max, euro.salary_currency) == (50_000, 65_000, "EUR")

    hourly, _ = salary_of(Labels(), "Pay is $12.5 per hour")
    assert hourly.salary_min == 12  # a decimal is cut, never read as 12,500


def test_80_to_120k_means_thousands_on_both_ends() -> None:
    labels, _ = salary_of(Labels(), "Base salary $80-120k a year.")

    assert (labels.salary_min, labels.salary_max) == (80_000, 120_000)


def test_a_min_only_label_gets_its_max_when_the_range_repeats_its_min() -> None:
    text = "Compensation\n$233,100—$366,300 USD\nPlus benefits."
    labels, result = salary_of(Labels(salary_min=233_100, salary_currency="USD"), text)

    assert (labels.salary_min, labels.salary_max) == (233_100, 366_300)
    assert [c.field for c in result.changes] == ["salary_max"]


def test_a_max_only_label_gets_its_min_when_the_range_repeats_its_max() -> None:
    text = "The base pay range for this role is $196,000-$294,000 plus equity."
    labels, _ = salary_of(Labels(salary_max=294_000, salary_currency="USD"), text)

    assert (labels.salary_min, labels.salary_max) == (196_000, 294_000)


def test_a_filled_salary_part_is_never_changed() -> None:
    text = "Salary: $90,000 - $120,000 per year."
    before = Labels(salary_min=1, salary_max=2, salary_currency="EUR")

    after, result = salary_of(before, text)

    assert after == before and result.changes == []


def test_a_label_that_matches_no_stated_range_is_left_alone() -> None:
    text = "Salary: $90,000 - $120,000 per year."
    before = Labels(salary_min=50_000, salary_currency="USD")

    after, result = salary_of(before, text)

    assert after == before and result.changes == []


def test_a_blank_currency_is_filled_when_the_amounts_match_the_post() -> None:
    text = "Salary: $90,000 - $120,000 per year."

    labels, _ = salary_of(Labels(salary_min=90_000, salary_max=120_000), text)

    assert labels.salary_currency == "USD"


def test_a_single_amount_is_min_equal_max_and_up_to_and_from_are_one_sided() -> None:
    assert salary_of(Labels(), "Salary $90,000 a year")[0].salary_min == 90_000
    assert salary_of(Labels(), "Salary $90,000 a year")[0].salary_max == 90_000
    up_to, _ = salary_of(Labels(), "Salary up to $90,000 a year")
    assert (up_to.salary_min, up_to.salary_max) == (None, 90_000)
    from_, _ = salary_of(Labels(), "Salary from $90,000 a year")
    assert (from_.salary_min, from_.salary_max) == (90_000, None)
    plus, _ = salary_of(Labels(), "Salary $90,000+ a year")
    assert (plus.salary_min, plus.salary_max) == (90_000, None)


def test_two_different_salaries_go_to_the_human_table_and_fill_nothing() -> None:
    text = "Salary $90,000 - $100,000 in Berlin.\nSalary $120,000 - $130,000 in London."

    labels, result = salary_of(Labels(), text)

    assert labels == Labels()
    assert [r.field for r in result.reviews] == ["salary"]
    assert "2 candidates" in result.reviews[0].shows


@pytest.mark.parametrize(
    "text",
    [
        "A 401(k) with 100% employer match (up to $6,000/year) in the U.S.",
        "$500 home office stipend",
        "Working from Home Allowance ($40 / Monthly)",
        "Signing bonus of $10,000",
        "We raised $20 million to grow.",  # no pay word
    ],
)
def test_benefit_money_and_money_without_a_pay_word_is_not_a_salary(text: str) -> None:
    assert find_candidates(text, Labels()) == []


def test_a_benefit_sentence_with_a_pay_word_is_still_a_candidate() -> None:
    assert len(find_candidates("Base salary $120,000 per year plus a bonus.", Labels())) == 1


# --- stack ---------------------------------------------------------------------------------------


def test_technologies_named_in_the_post_are_appended_in_order_after_the_existing_entries() -> None:
    text = "We use Docker, then Python and Django."

    labels, result = stack_of(Labels(stack=["Zebra Tool", "Python"]), text)

    assert labels.stack == ["Zebra Tool", "Python", "Docker", "Django"]
    assert [c.after for c in result.changes] == ["+ Docker", "+ Django"]
    assert all("Docker" in c.note or "Django" in c.note for c in result.changes)


def test_an_allow_listed_technology_is_added_and_a_deny_listed_one_is_not() -> None:
    labels, _ = stack_of(Labels(), "We integrate Decagon and use LLM tooling.")

    assert labels.stack == ["Decagon"]


def test_a_technology_only_named_in_a_customer_list_is_not_added() -> None:
    text = "Trusted by OpenAI, PayPal and Stripe. We build with Python."

    labels, _ = stack_of(Labels(), text)

    assert labels.stack == ["Python"]


def test_a_technology_in_a_customer_list_and_elsewhere_is_added() -> None:
    labels, _ = stack_of(Labels(), "Trusted by OpenAI. You will use OpenAI APIs daily.")

    assert labels.stack == ["OpenAI"]


def test_an_alternative_phrase_is_not_a_mention() -> None:
    labels, _ = stack_of(
        Labels(), "A Shopify alternative for small shops. We chose Postgres instead of MongoDB."
    )

    assert labels.stack == ["PostgreSQL"]


def test_twelve_technologies_are_added_and_thirteen_are_not() -> None:
    twelve = "Python, Java, Rust, Ruby, PHP, Dart, Scala, Kotlin, Swift, Elixir, Erlang, Haskell"
    labels, result = stack_of(Labels(), twelve, max_adds=20)
    assert len(labels.stack) == 12 and not result.reviews

    labels, result = stack_of(Labels(), twelve + ", Clojure", max_adds=20)
    assert labels.stack == []
    assert "13 technologies" in result.reviews[0].shows


def test_ambiguous_words_need_a_capital_letter_and_are_flagged() -> None:
    lower, lower_result = stack_of(Labels(), "We value swift execution and a rust belt town.")
    assert lower.stack == []
    assert any("'Swift'" in info for info in lower_result.infos)

    upper, upper_result = stack_of(Labels(), "Experience with Swift and Rust is required.")
    assert upper.stack == ["Swift", "Rust"]
    assert all(c.note.startswith("FLAG") for c in upper_result.changes)

    start, _ = stack_of(Labels(), "Swift fingers wanted.")  # a sentence start counts as capital
    assert start.stack == ["Swift"]


def test_a_dotted_name_is_not_read_as_its_last_part() -> None:
    labels, _ = stack_of(Labels(), "We use Node.js and Next.js, never plain js files.")

    assert labels.stack == ["Node.js", "Next.js", "JavaScript"]  # only the bare "js" counts


def test_javascript_is_not_java_and_golang_is_go() -> None:
    labels, _ = stack_of(Labels(), "JavaScript and golang services.")

    assert labels.stack == ["JavaScript", "Go"]


def test_a_technology_the_label_already_holds_under_another_spelling_is_not_added() -> None:
    text = "We run Airflow, Postgres and Docker."

    labels, _ = stack_of(Labels(stack=["Apache Airflow", "postgres"]), text)

    assert labels.stack == ["Apache Airflow", "postgres", "Docker"]


# --- stack: words that look like a technology but are not one here --------------------------------


def test_the_employer_named_in_the_headline_is_not_a_technology() -> None:
    text = "Databricks: Data Platform Engineer" + chr(10) + "You will build pipelines in Python."

    labels, result = stack_of(Labels(), text)

    assert labels.stack == ["Python"]
    assert "post 1: 'Databricks' not added: the employer's name" in result.infos


def test_the_employer_is_ignored_in_the_body_too() -> None:
    text = "Databricks: Data Engineer" + chr(10) + "You will use Databricks and Python daily."

    labels, result = stack_of(Labels(), text)

    assert labels.stack == ["Python"]
    assert result.infos.count("post 1: 'Databricks' not added: the employer's name") == 1


@pytest.mark.parametrize(
    "headline", ["Maven Clinic: Backend Engineer", "MAVEN CLINIC, Inc.: Backend Engineer"]
)
def test_a_technology_equal_to_the_first_word_of_the_employer_is_ignored(headline: str) -> None:
    text = headline + chr(10) + "Our builds use Maven and we write Python."

    labels, result = stack_of(Labels(), text)

    assert labels.stack == ["Python"]
    assert "post 1: 'Maven' not added: the employer's name" in result.infos


def test_a_technology_inside_the_employer_but_not_its_first_word_is_still_added() -> None:
    text = "Acme Maven: Engineer" + chr(10) + "We build with Maven."

    labels, _ = stack_of(Labels(), text)

    assert labels.stack == ["Maven"]


def test_a_headline_without_a_colon_names_no_employer() -> None:
    text = "Databricks" + chr(10) + "You will use Databricks and Python."

    labels, _ = stack_of(Labels(), text)

    assert labels.stack == ["Databricks", "Python"]


def test_the_employer_rule_only_looks_at_the_first_line() -> None:
    text = "Backend Engineer" + chr(10) + "Databricks: we use Databricks and Python."

    labels, _ = stack_of(Labels(), text)

    assert labels.stack == ["Databricks", "Python"]


@pytest.mark.parametrize(
    "word", ["Capital", "Clinic", "Catalog", "Catalogue", "Ventures", "Partners"]
)
def test_a_technology_followed_by_a_name_word_is_a_name_not_a_technology(word: str) -> None:
    labels, result = stack_of(Labels(), f"We were funded by Spark {word} last year.")

    assert labels.stack == []
    assert any("part of a proper name" in info for info in result.infos)


def test_unity_catalog_and_maven_clinic_are_not_unity_and_maven() -> None:
    labels, _ = stack_of(Labels(), "Maven Clinic runs on Unity Catalog and Spark Capital money.")

    assert labels.stack == []


@pytest.mark.parametrize(
    "text", ["A Go-To-Market plan.", "Our go-to-market team.", "A Python-based tool."]
)
def test_a_word_touching_a_hyphen_is_part_of_a_phrase(text: str) -> None:
    labels, result = stack_of(Labels(), text)

    assert labels.stack == []
    assert any("hyphenated" in info for info in result.infos) or "Go-To" in text


def test_a_capitalised_word_in_a_run_of_capitalised_words_is_a_name() -> None:
    labels, result = stack_of(
        Labels(), "Maven Health Group raised money. Senior Python Developer wanted."
    )

    assert labels.stack == []
    assert sum("multi-word capitalised name" in info for info in result.infos) == 2


def test_a_capitalised_word_next_to_lower_case_words_or_other_technologies_still_counts() -> None:
    labels, _ = stack_of(Labels(), "Experience with Maven builds.")
    assert labels.stack == ["Maven"]

    labels, _ = stack_of(Labels(), "Stack: Python Django PostgreSQL")  # a list, not a name
    assert labels.stack == ["Python", "Django", "PostgreSQL"]


@pytest.mark.parametrize(
    "text",
    [
        "A go-to-market lead and Go to market plans.",
        "Go beyond the usual.",
        "We use Python and Go.",  # one other technology and no comma: not a list
    ],
)
def test_go_without_a_list_go_paren_or_golang_is_not_the_language(text: str) -> None:
    labels, _ = stack_of(Labels(), text)

    assert "Go" not in labels.stack


@pytest.mark.parametrize(
    "text",
    [
        "Backend in Python, Go and Docker.",
        "Stack: Python/Go/Docker",
        "Experience with Go (the language) is a plus.",
        "We write Go; the golang toolchain is great.",
    ],
)
def test_go_is_the_language_in_a_list_as_go_paren_or_next_to_golang(text: str) -> None:
    labels, _ = stack_of(Labels(), text)

    assert "Go" in labels.stack


@pytest.mark.parametrize(
    "text",
    [
        "This role is NOT about WordPress sites.",
        "Everything except WordPress is in scope.",
        "We do it all, excluding WordPress.",
        "We maintain a simple WordPress blog.",
        "The role isn't about WordPress.",
    ],
)
def test_a_sentence_saying_the_role_is_not_about_it_adds_nothing(text: str) -> None:
    labels, result = stack_of(Labels(), text)

    assert labels.stack == []
    assert any("not, except, excluding or simple" in info for info in result.infos)


def test_a_negation_only_affects_its_own_sentence() -> None:
    labels, _ = stack_of(Labels(), "This is not a sales role. We use WordPress.")

    assert labels.stack == ["WordPress"]


def test_the_cases_from_the_first_dry_run_together_add_only_the_real_technologies() -> None:
    text = (
        "Databricks: Data Platform Engineer\n"
        "We are backed by Spark Capital. Maven Clinic partners with us. Our Go-To-Market team "
        "uses Unity Catalog; this role is NOT about WordPress sites.\n"
        "You will write Python and Docker code."
    )

    labels, _ = stack_of(Labels(), text)

    assert labels.stack == ["Python", "Docker"]


# --- stack: the match type and the safety limit ---------------------------------------------------


def test_every_added_entry_shows_its_sentence_and_how_it_matched() -> None:
    text = "Stack: Python, Django, Docker.\nWe deploy with Kubernetes.\nWe use Redis and Kafka."

    _, result = stack_of(Labels(), text)

    kinds = {c.after: c.match for c in result.changes}
    assert kinds["+ Python"] == "list"
    assert kinds["+ Kubernetes"] == "standalone"
    assert kinds["+ Redis"] == "other"
    table = format_changes(result.changes)
    assert "match" in table[0] and "sentence" in table[0]
    assert any("list" in row and "Stack: Python, Django, Docker" in row for row in table)


def test_a_salary_row_shows_a_dash_in_the_match_column() -> None:
    _, result = salary_of(Labels(), "Salary: $90,000 - $120,000 a year")

    assert format_changes(result.changes)[2].split(" | ")[5].strip() == "-"


NINE = "Python, Java, Docker, Django, Kubernetes, Terraform, PostgreSQL, Redis, Kafka"


def test_a_post_that_would_gain_more_entries_than_the_limit_goes_to_the_human_table() -> None:
    labels, result = stack_of(Labels(), NINE, max_adds=8)

    assert labels.stack == [] and not result.changes
    assert "9 entries would be added (over 8): none added" in result.reviews[0].shows

    labels, result = stack_of(Labels(), NINE, max_adds=9)
    assert len(labels.stack) == 9 and not result.reviews


def test_entries_the_label_already_holds_do_not_count_toward_the_limit() -> None:
    labels, result = stack_of(Labels(stack=["Python"]), NINE, max_adds=8)

    assert len(labels.stack) == 9 and not result.reviews


def test_the_limit_is_an_option_with_a_default_of_eight_and_zero_is_refused(tmp_path: Path) -> None:
    assert complete.build_parser().parse_args([]).max_adds_per_post == 8
    assert complete.build_parser().parse_args(["--max-adds-per-post", "3"]).max_adds_per_post == 3
    with pytest.raises(ScriptError, match="at least 1"):
        complete.run_complete(tmp_path / "x.json", SETTINGS, Session(), max_adds=0)


# --- the whole pass and its write guard ----------------------------------------------------------


def entry(labels: Labels, text: str, job: int = 1) -> LabelEntry:
    return LabelEntry(
        job_id=uuid.UUID(int=job),
        raw_text_hash=raw_text_hash(text),
        status="labelled",
        labels=labels,
    )


def label_file(*entries: LabelEntry) -> LabelFile:
    return LabelFile(
        source_id=uuid.UUID(int=99),
        seed=1,
        selected=[e.job_id for e in entries],
        posts=list(entries),
    )


TEXT = "Acme: Backend Engineer\n\nSalary: $90,000 - $120,000 per year. We use Python and Docker."


def test_the_plan_numbers_labelled_posts_lists_stale_ones_and_keeps_skipped_ones() -> None:
    skip = LabelEntry(job_id=uuid.UUID(int=5), raw_text_hash=raw_text_hash("s"), status="skipped")
    original = label_file(skip, entry(Labels(), TEXT, 1), entry(Labels(), TEXT, 2))

    plan = plan_completion(
        original, {uuid.UUID(int=1): TEXT, uuid.UUID(int=2): TEXT + " edited"}, POLICY, ALLOW
    )

    assert plan.fixed.posts[0] == skip
    assert {c.post for c in plan.changes} == {1}
    assert [(r.post, r.field) for r in plan.reviews] == [(2, "(whole post)")]
    verify_only_allowed_changes(original, plan.fixed)


def test_the_write_guard_rejects_everything_but_blank_salary_parts_and_appended_stack() -> None:
    base = Labels(title="T", stack=["Python", "Go"], salary_min=1)
    original = label_file(entry(base, TEXT))

    def after(**changes: object) -> LabelFile:
        return label_file(entry(base.model_copy(update=changes), TEXT))

    verify_only_allowed_changes(original, after(salary_max=2, stack=["Python", "Go", "Docker"]))
    for bad in (
        {"title": "Other"},
        {"salary_min": 5},
        {"stack": ["Go", "Python"]},
        {"stack": ["Python"]},
    ):
        with pytest.raises(ScriptError, match="refusing to write"):
            verify_only_allowed_changes(original, after(**bad))


def fake_reader(monkeypatch: pytest.MonkeyPatch, by_job: dict[int, str]) -> None:
    async def fake(database_url: str, ids: list[uuid.UUID]) -> list[tuple[uuid.UUID, str]]:
        return [(uuid.UUID(int=n), text) for n, text in by_job.items()]

    monkeypatch.setattr(complete, "read_posts_by_id", fake)
    monkeypatch.setattr(complete, "load_policy", lambda path: POLICY)
    monkeypatch.setattr(complete, "load_allow_names", lambda path: ALLOW)


def session(answer: str | None, lines: list[str]) -> Session:
    def ask(prompt: str) -> str:
        lines.append(f"ASKED {prompt.strip()}")
        if answer is None:
            raise EOFError
        return answer

    return Session(ask=ask, out=lines.append)


def saved(tmp_path: Path) -> tuple[Path, LabelFile]:
    path = tmp_path / "labels.json"
    original = label_file(entry(Labels(title="Backend Engineer", stack=["Python"]), TEXT))
    save_label_file(path, original)
    return path, original


def test_yes_backs_up_writes_and_rechecks_with_codes_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, original = saved(tmp_path)
    fake_reader(monkeypatch, {1: TEXT})
    lines: list[str] = []

    code = run_complete(path, SETTINGS, session("y", lines))

    written = load_label_file(path).posts[0].labels
    assert written is not None and written.stack == ["Python", "Docker"]
    assert (written.salary_min, written.salary_max) == (90_000, 120_000)
    assert load_label_file(tmp_path / "labels.pass1.json") == original
    assert any(line.startswith("ASKED Write these changes?") for line in lines)
    after_write = "\n".join(
        lines[
            lines.index("Written. Checking the labels against the post text again (codes only):") :
        ]
    )
    assert "Salary:" not in after_write and "Docker" not in after_write
    assert code in (0, 1)


def test_no_and_a_dry_run_write_nothing_and_make_no_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, original = saved(tmp_path)
    fake_reader(monkeypatch, {1: TEXT})
    for answer, dry in (("n", False), (None, False), ("y", True)):
        run_complete(path, SETTINGS, session(answer, []), dry_run=dry)

        assert load_label_file(path) == original
        assert not (tmp_path / "labels.pass1.json").exists()


def test_an_existing_backup_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = saved(tmp_path)
    backup = tmp_path / "labels.pass1.json"
    backup.write_text("the first backup", encoding="utf-8")
    fake_reader(monkeypatch, {1: TEXT})

    run_complete(path, SETTINGS, session("y", []))

    assert backup.read_text(encoding="utf-8") == "the first backup"


def test_a_database_not_named_jobradar_is_refused_before_anything_is_read(tmp_path: Path) -> None:
    path, _ = saved(tmp_path)
    wrong = settings_with(database_url="postgresql+asyncpg://u:p@db:5432/jobradar_test")

    with pytest.raises(ScriptError, match="jobradar"):
        run_complete(path, wrong, session("y", []))


# --- no extractor, no provider -------------------------------------------------------------------


def test_the_script_cannot_reach_the_extractor_a_provider_or_a_stored_extraction() -> None:
    source = (ROOT / "scripts" / "eval_complete_labels.py").read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)

    forbidden = ("app.llm", "httpx", "requests", "aiohttp", "app.models", "app.services")
    assert not {i for i in imported if i.startswith(forbidden)}
    modules = {i for i in imported if i.startswith("app.extraction")}
    # the stack dictionary and the text hash are the only extraction code it may use
    assert {m for m in modules if m.count(".") <= 2} <= {
        "app.extraction",
        "app.extraction.stack",
        "app.extraction.text_hash",
    }
    assert not {i for i in imported if i.startswith(("scripts.eval_run", "scripts.try_extraction"))}
    for word in (
        "read_stored_jobs",
        "rules_values",
        "extract_with_rules",
        "find_stack",
        "find_salary",
    ):
        assert word not in source
