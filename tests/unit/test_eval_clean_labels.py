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


# --- the rules ---------------------------------------------------------------------------------


def kept(post: int, stack: list[str]) -> list[str]:
    """The entries that stay (flagged ones count as kept until the user answers)."""
    return [i.name for p in propose_for_post(post, stack) for i in p.items if i.keep]


def dropped(post: int, stack: list[str]) -> dict[str, str]:
    return {
        i.name: i.drop_reason
        for p in propose_for_post(post, stack)
        for i in p.items
        if i.drop_reason
    }


@pytest.mark.parametrize(
    ("entry", "stays", "note"),
    [
        ("Java (Primary)", ["Java"], "annotation"),
        ("Python (3.x) ", ["Python"], "version"),
        ("Go (Bonus)", ["Go"], "annotation"),
        ("Rust (preferred)", ["Rust"], "annotation"),
        ("Docker (nice to have)", ["Docker"], "annotation"),
        ("Kotlin (transitioning from Java to Kotlin)", ["Kotlin"], "annotation"),
        ("Go (used for the backend services)", ["Go"], "clause"),
    ],
)
def test_annotations_in_brackets_are_dropped_not_kept(
    entry: str, stays: list[str], note: str
) -> None:
    assert kept(1, [entry]) == stays
    assert note in dropped(1, [entry]).values()


def test_products_inside_brackets_become_separate_entries() -> None:
    assert kept(1, ["AWS (SageMaker, EC2, EKS)"]) == ["AWS", "SageMaker", "EC2", "EKS"]


def test_a_category_name_in_front_of_brackets_is_a_proposed_drop() -> None:
    assert kept(1, ["Vector Databases (Milvus, Faiss)"]) == ["Milvus", "Faiss"]
    assert dropped(1, ["Vector Databases (Milvus, Faiss)"]) == {"Vector Databases": "category name"}
    for category in ("Cloud Platforms", "Cloud platforms", "Databases", "Frameworks"):
        assert dropped(1, [f"{category} (Foo, Bar)"]) == {category: "category name"}


def test_spaced_and_unspaced_slashes_inside_brackets_split() -> None:
    assert kept(1, ["Cloud Platforms (AWS / GCP / Azure)"]) == ["AWS", "GCP", "Azure"]
    assert kept(1, ["Node.js (NestJS/Express)"]) == ["Node.js", "NestJS", "Express"]


def test_a_short_slash_name_inside_brackets_stays_whole() -> None:
    assert kept(1, ["Tooling (CI/CD, TCP/IP)"]) == ["Tooling", "CI/CD", "TCP/IP"]


def test_annotations_are_dropped_while_products_in_the_same_bracket_stay() -> None:
    stack = ["Frameworks (Django, FastAPI, preferred)"]

    assert kept(1, stack) == ["Django", "FastAPI"]
    assert dropped(1, stack) == {"Frameworks": "category name", "preferred": "annotation"}


def test_nested_brackets_are_walked_too() -> None:
    assert kept(1, ["Cloud Platforms (AWS (EC2, S3), GCP)"]) == ["AWS", "GCP", "EC2", "S3"]


def test_a_second_name_for_the_same_technology_is_a_duplicate() -> None:
    assert kept(1, ["Go (Golang) (preferred)"]) == ["Go"]
    assert dropped(1, ["Go (Golang) (preferred)"]) == {
        "Golang": "duplicate",
        "preferred": "annotation",
    }


def test_a_bracket_split_by_a_comma_is_rejoined_then_expanded() -> None:
    stack = ["Java", "AWS (SageMaker", "EKS)", "Redis"]

    assert rejoin_split_brackets(stack) == [["Java"], ["AWS (SageMaker", "EKS)"], ["Redis"]]
    assert kept(1, stack) == ["Java", "AWS", "SageMaker", "EKS", "Redis"]
    joined = propose_for_post(1, stack)[1]
    assert joined.before == "AWS (SageMaker, EKS)"
    assert all("rejoin-brackets" in i.rules for i in joined.items)


def test_a_bracket_that_never_closes_keeps_its_content_as_candidates() -> None:
    proposals = propose_for_post(1, ["Spark (batch", "Flink"])

    assert [i.name for p in proposals for i in p.items] == ["Spark", "batch", "Flink"]
    assert proposals[0].before == "Spark (batch, Flink"  # shown in the table, so nothing is hidden
    assert [i.flag for i in proposals[0].items] == [None, "unknown lowercase word", None]


def test_a_stray_closing_bracket_is_ignored() -> None:
    assert kept(1, ["EKS)"]) == ["EKS"]


def test_slash_joined_products_are_split_only_when_spaced() -> None:
    assert split_slash_products("Terraform / Pulumi") == ["Terraform", "Pulumi"]
    assert split_slash_products("CI/CD") == ["CI/CD"]
    assert kept(1, ["Terraform / Pulumi"]) == ["Terraform", "Pulumi"]
    assert propose_for_post(1, ["Terraform / Pulumi"])[0].items[0].rules == ["split-slash"]


def test_a_bare_product_name_is_left_alone() -> None:
    proposals = propose_for_post(1, ["Python", "PostgreSQL"])

    assert [(p.changed, p.flagged, p.items[0].rules) for p in proposals] == [(False, False, [])] * 2


def test_duplicates_after_cleaning_are_dropped_once() -> None:
    proposals = propose_for_post(1, ["Java (Primary)", "java", "Java / Kotlin"])

    assert kept(1, ["Java (Primary)", "java", "Java / Kotlin"]) == ["Java", "Kotlin"]
    assert proposals[1].items[0].drop_reason == "duplicate" and proposals[1].changed
    assert proposals[2].items[0].drop_reason == "duplicate"


def test_an_entry_that_is_only_a_note_leaves_nothing_behind() -> None:
    assert kept(1, ["(Primary)", "Go"]) == ["Go"]
    assert dropped(1, ["(Primary)", "Go"]) == {"Primary": "annotation"}
    assert propose_for_post(1, ["(Primary)"])[0].changed


def test_a_bare_category_name_is_a_proposed_drop_too() -> None:
    assert kept(1, ["Databases", "PostgreSQL"]) == ["PostgreSQL"]


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


def table_rows(stack: list[str]) -> list[list[str]]:
    """The table as cells: post, before, after, rule (columns are separated by two spaces)."""
    table = format_table(build_proposals(make_file(stack)))
    return [[c.strip() for c in line.split("  ") if c.strip()] for line in table.splitlines()[2:]]


def test_the_table_has_a_row_for_every_candidate_and_says_what_happened() -> None:
    rows = table_rows(["Vector Databases (Milvus, Faiss, preferred)"])

    assert [row[-2] for row in rows] == ["Vector Databases", "Milvus", "Faiss", "preferred"]
    assert rows[0][:2] == ["1", "Vector Databases (Milvus, Faiss, preferred)"]
    assert "DROP (category name)" in rows[0][-1] and "brackets-removed" in rows[0][-1]
    assert rows[1][-1] == "bracket-content" and rows[2][-1] == "bracket-content"
    assert rows[3][-1] == "bracket-content, DROP (annotation)"
    assert all(len(row) == 2 for row in rows[1:])  # before and post are shown once per group


def test_the_table_header_and_flags() -> None:
    table = format_table(
        build_proposals(make_file(["Java (Primary)", "Python", "Cloud architecture"]))
    )

    assert table.splitlines()[0].split() == ["post", "before", "after", "rule"]
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
    first_post = ["Java", "AWS", "SageMaker", "EKS", "Terraform", "Pulumi"]
    assert stacks(labels_path) == [first_post, ["Go"]]
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
