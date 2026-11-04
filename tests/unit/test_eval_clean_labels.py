"""The one-off stack-label cleaner. Every example here is invented; no real labels are used.

Nothing here reads a database or a provider: the script imports neither (checked below).
"""

import ast
import hashlib
import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from scripts import eval_clean_labels as clean
from scripts.db_guard import ScriptError
from scripts.eval_clean_labels import (
    Session,
    apply_proposals,
    build_proposals,
    flag_reason,
    format_table,
    lint_entry,
    lint_file,
    propose_for_post,
    rejoin_split_brackets,
    split_slash_products,
    strip_brackets,
)
from scripts.eval_labelfile import LabelEntry, LabelFile, Labels, load_label_file, save_label_file

ROOT = Path(__file__).resolve().parent.parent.parent


def make_file(*stacks: list[str], skipped: int = 0) -> LabelFile:
    entries = []
    for number, stack in enumerate(stacks, start=1):
        labels = Labels(title=f"Title {number}", company=f"Company {number}", stack=stack)
        digest = hashlib.sha256(str(number).encode()).hexdigest()
        entries.append(
            LabelEntry(
                job_id=uuid.UUID(int=number), raw_text_hash=digest, status="labelled", labels=labels
            )
        )
    for extra in range(skipped):
        digest = hashlib.sha256(f"s{extra}".encode()).hexdigest()
        entries.append(
            LabelEntry(job_id=uuid.UUID(int=100 + extra), raw_text_hash=digest, status="skipped")
        )
    return LabelFile(
        source_id=uuid.UUID(int=999),
        seed=1,
        selected=[entry.job_id for entry in entries],
        posts=entries,
    )


def scripted(answers: list[str]) -> tuple[Session, list[str], list[str]]:
    """A session whose answers are given up front; returns it with the questions and output."""
    questions: list[str] = []
    output: list[str] = []
    queue: Iterator[str] = iter(answers)

    def ask(question: str) -> str:
        questions.append(question)
        try:
            return next(queue)
        except StopIteration:
            raise EOFError from None

    return Session(ask=ask, out=output.append), questions, output


def names(post: int, stack: list[str]) -> list[str]:
    return [i.name for p in propose_for_post(post, stack) for i in p.items]


# --- the rules ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("Java (Primary)", "Java"),
        ("Python (3.x) ", "Python"),
        ("Kafka [streaming]", "Kafka"),
        ("Go (Golang) (preferred)", "Go"),
        ("AWS (SageMaker", "AWS"),  # a bracket that never closes is cut
        ("EKS)", "EKS"),
        ("Plain", "Plain"),
    ],
)
def test_trailing_annotations_in_brackets_are_stripped(before: str, after: str) -> None:
    assert strip_brackets(before) == after


def test_a_bracket_split_by_a_comma_is_rejoined_then_stripped() -> None:
    stack = ["Java", "AWS (SageMaker", "EKS)", "Redis"]

    assert rejoin_split_brackets(stack) == [["Java"], ["AWS (SageMaker", "EKS)"], ["Redis"]]
    assert names(1, stack) == ["Java", "AWS", "Redis"]
    rules = [p.rules for p in propose_for_post(1, stack)]
    assert rules[1] == ["rejoin-brackets", "strip-brackets"]


def test_a_bracket_that_never_closes_does_not_swallow_the_rest_silently() -> None:
    proposals = propose_for_post(1, ["Spark (batch", "Flink"])

    assert [i.name for p in proposals for i in p.items] == ["Spark"]
    assert proposals[0].before == "Spark (batch, Flink"  # shown in the table, so nothing is hidden


def test_slash_joined_products_are_split_only_when_spaced() -> None:
    assert split_slash_products("Terraform / Pulumi") == ["Terraform", "Pulumi"]
    assert split_slash_products("CI/CD") == ["CI/CD"]
    assert names(1, ["Terraform / Pulumi"]) == ["Terraform", "Pulumi"]
    assert propose_for_post(1, ["Terraform / Pulumi"])[0].rules == ["split-slash"]


def test_a_bare_product_name_is_left_alone() -> None:
    proposals = propose_for_post(1, ["Python", "PostgreSQL"])

    assert [(p.changed, p.flagged, p.rules) for p in proposals] == [(False, False, [])] * 2


def test_duplicates_after_cleaning_are_dropped_once() -> None:
    proposals = propose_for_post(1, ["Java (Primary)", "java", "Java / Kotlin"])

    assert [i.name for p in proposals for i in p.items] == ["Java", "Kotlin"]
    assert "duplicate" in proposals[1].rules and "duplicate" in proposals[2].rules
    assert proposals[1].changed


def test_an_entry_with_nothing_left_is_dropped_by_rule() -> None:
    proposals = propose_for_post(1, ["(Primary)", "Go"])

    assert proposals[0].items == [] and "empty" in proposals[0].rules and proposals[0].changed


# --- flagging: asked about, never guessed -------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    [
        "Strong problem solving skills",  # description
        "Cloud architecture",  # concept
        "microservices",  # concept
        "experience with large systems and teams",  # sentence-like
        "A very long invented platform name here",  # too long
    ],
)
def test_concepts_and_descriptions_are_flagged(entry: str) -> None:
    assert flag_reason(entry) is not None


@pytest.mark.parametrize("entry", ["Python", "PostgreSQL", "SageMaker", "Amazon Web Services"])
def test_named_products_are_not_flagged(entry: str) -> None:
    assert flag_reason(entry) is None


def test_a_flagged_entry_is_not_changed_by_the_rules_only_marked() -> None:
    proposal = propose_for_post(1, ["Cloud architecture"])[0]

    assert [i.name for i in proposal.items] == ["Cloud architecture"]
    assert proposal.flagged and not proposal.changed and proposal.items[0].keep


# --- lint ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "problems"),
    [
        ("Java (Primary)", ["brackets"]),
        ("AWS (SageMaker", ["brackets"]),
        ("x" * 31, ["too_long"]),
        ("Use it a lot every single day", ["sentence"]),
        ("Knows Python well. Loves it", ["sentence"]),
        ("Node.js", []),
        ("CI/CD", []),
        ("", ["empty"]),
    ],
)
def test_lint_catches_brackets_sentences_and_long_entries(entry: str, problems: list[str]) -> None:
    assert lint_entry(entry) == problems


def test_lint_reports_the_post_number_among_labelled_posts() -> None:
    file = make_file(["Java"], ["Go (x)"], skipped=1)

    issues = lint_file(file)

    assert [(i.post, i.entry, i.problems) for i in issues] == [(2, "Go (x)", ("brackets",))]


# --- applying -----------------------------------------------------------------------------------


def test_only_the_stack_changes_and_dropped_entries_are_removed() -> None:
    original = make_file(["Java (Primary)", "Cloud architecture"], ["Go"], skipped=1)
    proposals = build_proposals(original)
    proposals[0].items[0].keep = True
    proposals[1].items[0].keep = False  # the user dropped the concept

    cleaned = apply_proposals(original, proposals)

    assert [e.labels.stack for e in cleaned.posts if e.labels] == [["Java"], ["Go"]]
    before, after = original.model_dump(mode="json"), cleaned.model_dump(mode="json")
    for a, b in zip(before["posts"], after["posts"], strict=True):
        if a["labels"]:
            a["labels"].pop("stack")
            b["labels"].pop("stack")
    assert before == after  # ids, hashes, titles, companies, skipped posts: untouched


def test_the_table_shows_before_after_and_rule_for_changed_rows_only() -> None:
    proposals = build_proposals(make_file(["Java (Primary)", "Python", "Cloud architecture"]))

    table = format_table(proposals)

    lines = table.splitlines()
    assert lines[0].split() == ["post", "before", "after", "rule"]
    assert "Java (Primary)" in lines[2] and "strip-brackets" in lines[2]
    assert "ASK: looks like a concept" in table
    assert "Python" not in table  # already clean: counted, not listed


# --- the command --------------------------------------------------------------------------------


@pytest.fixture
def labels_path(tmp_path: Path) -> Path:
    path = tmp_path / "labels.json"
    save_label_file(
        path,
        make_file(
            ["Java (Primary)", "AWS (SageMaker", "EKS)", "Terraform / Pulumi"],
            ["Go", "Cloud architecture"],
        ),
    )
    return path


def stacks(path: Path) -> list[list[str]]:
    return [e.labels.stack for e in load_label_file(path).posts if e.labels]


def test_nothing_is_written_until_the_user_confirms(labels_path: Path) -> None:
    original = labels_path.read_bytes()
    session, questions, _ = scripted(["k", "n"])

    assert clean.run(labels_path, session) == 1

    assert labels_path.read_bytes() == original
    assert questions[-1].startswith("Write these changes")


def test_quitting_at_a_question_changes_nothing(labels_path: Path) -> None:
    original = labels_path.read_bytes()
    session, _, output = scripted(["q"])

    assert clean.run(labels_path, session) == 1

    assert labels_path.read_bytes() == original and "Nothing was changed" in output[-1]


def test_end_of_input_changes_nothing(labels_path: Path) -> None:
    original = labels_path.read_bytes()
    session, _, _ = scripted([])

    assert clean.run(labels_path, session) == 1
    assert labels_path.read_bytes() == original


def test_confirmed_run_backs_up_first_writes_the_clean_labels_and_lints_clean(
    labels_path: Path,
) -> None:
    original = labels_path.read_bytes()
    session, questions, output = scripted(["d", "y"])  # drop "Cloud architecture"

    assert clean.run(labels_path, session) == 0

    assert (labels_path.parent / "labels.raw.json").read_bytes() == original
    assert stacks(labels_path) == [["Java", "AWS", "Terraform", "Pulumi"], ["Go"]]
    assert "Lint problems after: 0" in output[-1]
    assert "Cloud architecture" in questions[0]
    assert not list(labels_path.parent.glob("*.tmp"))


def test_keeping_a_flagged_entry_keeps_it(labels_path: Path) -> None:
    session, _, _ = scripted(["k", "y"])

    assert clean.run(labels_path, session) == 0

    assert stacks(labels_path)[1] == ["Go", "Cloud architecture"]


def test_an_unclear_answer_is_asked_again(labels_path: Path) -> None:
    session, questions, _ = scripted(["maybe", "", "d", "y"])

    assert clean.run(labels_path, session) == 0

    assert len([q for q in questions if "Keep or drop" in q]) == 3


def test_a_kept_entry_that_still_breaks_the_lint_is_reported_and_exits_nonzero(
    tmp_path: Path,
) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, make_file(["Strong ability to design large distributed systems"]))
    session, _, output = scripted(["k", "y"])

    assert clean.run(path, session) == 1

    assert any("still unclean: post 1" in line for line in output)


def test_an_existing_backup_is_never_overwritten(labels_path: Path) -> None:
    raw = labels_path.parent / "labels.raw.json"
    raw.write_text("the first backup", encoding="utf-8")
    session, _, output = scripted(["d", "y"])

    assert clean.run(labels_path, session) == 0

    assert raw.read_text(encoding="utf-8") == "the first backup"
    assert "already there" in output[0]


def test_a_second_run_finds_nothing_to_clean(labels_path: Path) -> None:
    clean.run(labels_path, scripted(["d", "y"])[0])
    session, _, output = scripted([])

    assert clean.run(labels_path, session) == 0

    assert "Nothing to clean" in output[-1]


def test_a_missing_or_invalid_file_is_an_error_and_writes_no_backup(tmp_path: Path) -> None:
    with pytest.raises(ScriptError, match="does not exist"):
        clean.run(tmp_path / "labels.json", scripted([])[0])
    (tmp_path / "labels.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ScriptError, match="does not match the format"):
        clean.run(tmp_path / "labels.json", scripted([])[0])


def test_main_reports_script_errors_with_exit_code_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert clean.main(["--labels", str(tmp_path / "nope.json")]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_main_uses_the_default_labels_path_in_the_ignored_folder() -> None:
    args = clean.build_parser().parse_args([])

    assert args.labels.relative_to(ROOT).parts[:2] == (".local", "eval")
    assert ".local/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_the_json_keeps_its_shape_after_cleaning(labels_path: Path) -> None:
    clean.run(labels_path, scripted(["d", "y"])[0])

    data = json.loads(labels_path.read_text(encoding="utf-8"))

    assert set(data) == {"version", "source_id", "seed", "selected", "posts"}


# --- no database, no provider -------------------------------------------------------------------


def test_the_cleaner_imports_no_database_provider_or_extraction_reader() -> None:
    tree = ast.parse((ROOT / "scripts" / "eval_clean_labels.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = ("app.llm", "httpx", "requests", "aiohttp", "sqlalchemy", "app.db", "app.models")
    assert not {i for i in imported if i.startswith(forbidden)}
    assert "scripts.eval_run" not in imported and "scripts.try_extraction" not in imported
