"""`eval_clean_labels.py --policy`: the keep-or-drop rule, its lists and the command.

Every example is invented. Nothing here reads a database or a provider.
"""

import ast
from pathlib import Path

import pytest

from scripts import eval_clean_labels as clean
from scripts.db_guard import ScriptError
from scripts.eval_clean_labels import apply_policy, build_proposals, format_table
from scripts.eval_clean_policy import (
    DEFAULT_POLICY_PATH,
    Decision,
    Policy,
    decide,
    load_policy,
    normal,
)
from scripts.eval_labelfile import load_label_file, save_label_file
from tests.unit.test_eval_clean_labels import make_file, scripted, stacks

ROOT = Path(__file__).resolve().parent.parent.parent
POLICY = Policy(
    allow=frozenset({"zorbo", "acme engine"}),
    deny=frozenset({"rag", "ci/cd"}),
    fragments=frozenset({"connect", "batch models"}),
)


# --- the decision, signal by signal -------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "flag", "keep", "reason"),
    [
        ("and Kubernetes", None, False, "fragment"),
        ("Or Terraform", None, False, "fragment"),
        ("Connect", None, False, "fragment"),
        ("Batch  Models", None, False, "fragment"),  # case and spacing do not matter
        ("RAG", None, False, "deny-list"),
        ("ci/cd", None, False, "deny-list"),
        ("Zorbo", None, True, "allow-list"),
        ("ACME Engine", "looks like a concept", True, "allow-list"),  # the list beats the flag
        ("SOC 2", None, False, "concept rule: standard"),
        ("ISO 27001", None, False, "concept rule: standard"),
        ("PCI DSS", None, False, "concept rule: standard"),
        ("Python", "unknown lowercase word", True, "dictionary"),
        ("Cloud architecture", "looks like a concept", False, "concept rule: looks like a concept"),
        ("batch", "unknown lowercase word", False, "concept rule: unknown lowercase word"),
    ],
)
def test_each_signal_decides_with_its_own_reason(
    name: str, flag: str | None, keep: bool, reason: str
) -> None:
    assert decide(name, flag, POLICY) == Decision(keep, reason)


def test_an_entry_no_signal_covers_and_nothing_flagged_is_left_alone() -> None:
    assert decide("Python", None, POLICY) is None
    assert decide("Quuxlang", None, POLICY) is None


def test_the_lists_cannot_overlap(tmp_path: Path) -> None:
    path = tmp_path / "policy.toml"
    path.write_text('allow = ["Foo"]\ndeny = ["foo"]\n', encoding="utf-8")

    with pytest.raises(ScriptError, match="on both the allow-list and a drop list: foo"):
        load_policy(path)


def test_names_are_compared_case_folded_with_single_spaces() -> None:
    assert normal("  Deep   Learning ") == "deep learning"


# --- the lists file -----------------------------------------------------------------------------


def test_the_shipped_lists_load_and_hold_the_agreed_names() -> None:
    policy = load_policy(DEFAULT_POLICY_PATH)

    for name in ("Decagon", "Kore.ai", "CoreML", "Confluent Platform", "Amazon Lex", "EKS"):
        assert normal(name) in policy.allow
    for name in ("LLMs", "RAG", "MLOps", "Deep Learning", "OSINT", "IaC", "CI/CD"):
        assert normal(name) in policy.deny
    for name in ("Connect", "Retrieval", "AWS-first", "3DS", "TIPs", "real-time", "abuse"):
        assert normal(name) in policy.fragments
    assert "batch models" in policy.fragments


def test_process_frameworks_and_sales_methods_are_denied_not_allowed() -> None:
    policy = load_policy(DEFAULT_POLICY_PATH)

    for name in ("Agile", "Scrum", "Kanban", "Challenger", "MEDDPICC", "Command of the Message"):
        assert normal(name) in policy.deny and normal(name) not in policy.allow
        assert decide(name, None, policy) == Decision(False, "deny-list")
        assert decide(name.upper(), "looks like a concept", policy) == Decision(False, "deny-list")


def test_the_shipped_lists_decide_the_agreed_examples() -> None:
    policy = load_policy(DEFAULT_POLICY_PATH)
    expected = {
        "Karpenter": True,
        "Pulumi": True,
        "SageMaker": True,
        "LLMs": False,
        "MLOps": False,
        "Retrieval": False,
        "3DS": False,
        "and Retrieval": False,
    }

    for name, keep in expected.items():
        decision = decide(name, None, policy)
        assert decision is not None and decision.keep is keep, name


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("allow = [", "is not valid TOML"),
        ('allowed = ["x"]', "unknown key"),
        ('allow = "x"', "must be a list of strings"),
        ("deny = [1]", "must be a list of strings"),
    ],
)
def test_a_bad_lists_file_is_reported(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "policy.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ScriptError, match=message):
        load_policy(path)


def test_a_missing_lists_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ScriptError, match="cannot read"):
        load_policy(tmp_path / "nope.toml")


def test_empty_lists_are_allowed(tmp_path: Path) -> None:
    path = tmp_path / "policy.toml"
    path.write_text("", encoding="utf-8")

    assert load_policy(path) == Policy(frozenset(), frozenset(), frozenset())


# --- applying it to proposals and the table -----------------------------------------------------

POST = [
    "Docker",
    "Cloud architecture",
    "and Kubernetes",
    "RAG",
    "Zorbo",
    "SOC 2",
    "Connect",
    "Strong ability to design large distributed systems",
    "Vector Databases (Quuxdb, Faiss)",
]


def proposals_after_policy() -> list[clean.Proposal]:
    proposals = build_proposals(make_file(POST))
    apply_policy(proposals, POLICY)
    return proposals


def verdicts(proposals: list[clean.Proposal]) -> dict[str, bool]:
    return {i.name: i.keep for p in proposals for i in p.items}


def test_the_policy_decides_every_entry_that_would_have_been_asked() -> None:
    proposals = proposals_after_policy()

    assert not [i for p in proposals for i in p.items if i.flag]  # nothing left to ask
    assert verdicts(proposals) == {
        "Docker": True,
        "Cloud architecture": False,
        "and Kubernetes": False,
        "RAG": False,
        "Zorbo": True,
        "SOC 2": False,
        "Connect": False,
        "Strong ability to design large distributed systems": False,
        "Vector Databases": False,  # a category name, dropped by the earlier rule
        "Quuxdb": True,
        "Faiss": True,
    }


def test_the_counts_are_what_the_lists_and_the_rule_decided() -> None:
    proposals = build_proposals(make_file(POST))

    # Zorbo is kept by the allow-list; six entries are dropped by the policy.
    assert apply_policy(proposals, POLICY) == (1, 6)


def test_every_decision_shows_in_the_table_with_its_reason() -> None:
    table = format_table(proposals_after_policy())

    for expected in (
        "DROP (concept rule: looks like a concept)",
        "DROP (fragment)",
        "DROP (deny-list)",
        "KEEP (allow-list)",
        "DROP (concept rule: standard)",
        "DROP (category name)",
    ):
        assert expected in table
    assert "ASK" not in table
    assert "Docker" not in table  # clean and undecided: not listed


def test_an_entry_already_dropped_by_a_rule_is_not_decided_again() -> None:
    proposals = build_proposals(make_file(["Java (Primary)"]))
    apply_policy(proposals, POLICY)

    primary = proposals[0].items[1]
    assert primary.drop_reason == "annotation" and primary.note is None


# --- the command --------------------------------------------------------------------------------


def test_policy_mode_asks_only_the_final_question_and_writes_on_yes(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, make_file(POST))
    session, questions, output = scripted(["y"])

    assert clean.run(path, session, POLICY) == 0

    assert len(questions) == 1 and questions[0].startswith("Write these changes")
    assert stacks(path) == [["Docker", "Zorbo", "Quuxdb", "Faiss"]]
    assert any(line.startswith("Policy: 1 entries kept and 6 dropped") for line in output)


def test_policy_mode_changes_nothing_without_a_final_yes(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, make_file(POST))
    original = path.read_bytes()

    for answers in (["n"], [""], []):
        assert clean.run(path, scripted(answers)[0], POLICY) == 1
        assert path.read_bytes() == original


def test_policy_mode_leaves_every_other_field_alone(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, make_file(POST, ["Go"], skipped=1))
    before = load_label_file(path)

    clean.run(path, scripted(["y"])[0], POLICY)

    after = load_label_file(path)
    assert [e.job_id for e in after.posts] == [e.job_id for e in before.posts]
    assert [e.labels.title for e in after.posts if e.labels] == ["Title 1", "Title 2"]
    assert after.posts[-1].status == "skipped"


def test_without_policy_the_questions_are_still_asked(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, make_file(["Cloud architecture"]))
    session, questions, _ = scripted(["d", "y"])

    clean.run(path, session)

    assert len(questions) == 2


def test_main_loads_the_lists_before_touching_any_file(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, make_file(POST))
    bad = tmp_path / "bad.toml"
    bad.write_text("allow = [", encoding="utf-8")

    assert clean.main(["--labels", str(path), "--policy", "--policy-file", str(bad)]) == 2

    assert not (tmp_path / "labels.raw.json").exists()  # no backup was made either


def test_main_runs_policy_mode_with_a_lists_file(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, make_file(["Docker", "Zorbo", "Cloud architecture"]))
    lists = tmp_path / "lists.toml"
    lists.write_text('allow = ["Zorbo"]\n', encoding="utf-8")

    code = clean.main(
        ["--labels", str(path), "--policy", "--policy-file", str(lists)], scripted(["y"])[0]
    )

    assert code == 0 and stacks(path) == [["Docker", "Zorbo"]]


def test_policy_file_without_policy_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        clean.main(["--policy-file", "x.toml"])

    assert "--policy-file only makes sense together with --policy" in capsys.readouterr().err


def test_the_policy_modules_import_no_database_or_provider() -> None:
    forbidden = ("app.llm", "httpx", "requests", "aiohttp", "sqlalchemy", "app.db", "app.models")
    for name in ("eval_clean_policy", "eval_clean_labels"):
        tree = ast.parse((ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not {i for i in imported if i.startswith(forbidden)}, name
