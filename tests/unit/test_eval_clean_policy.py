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
    strip_leading,
)
from scripts.eval_labelfile import load_label_file, save_label_file
from tests.unit.test_eval_clean_labels import make_file, scripted, stacks

ROOT = Path(__file__).resolve().parent.parent.parent
POLICY = Policy(
    allow=frozenset({"zorbo", "acme engine", "faiss"}),
    deny=frozenset({"rag", "ci/cd", "deep learning"}),
    fragments=frozenset({"connect", "batch models"}),
    aliases={"google cloud platform": "GCP"},
)
AND = "stripped 'and'"


# --- stripping a leading connector or adjective ---------------------------------------------------


@pytest.mark.parametrize(
    ("name", "rest", "rules"),
    [
        ("and AWS", "AWS", ("stripped 'and'",)),
        ("Or Go", "Go", ("stripped 'Or'",)),
        ("strong SQL", "SQL", ("stripped 'strong'",)),
        ("and strong SQL", "SQL", ("stripped 'and'", "stripped 'strong'")),
        ("Hands-On Docker", "Docker", ("stripped 'Hands-On'",)),
        ("solid Django", "Django", ("stripped 'solid'",)),
        ("Expert Python", "Python", ("stripped 'Expert'",)),
        ("proficient Go", "Go", ("stripped 'proficient'",)),
        ("deep Kubernetes", "Kubernetes", ("stripped 'deep'",)),
        ("Python", "Python", ()),
        ("android", "android", ()),  # starts with "and" but is not the word
        ("strong", "", ("stripped 'strong'",)),
        ("and", "", ("stripped 'and'",)),
    ],
)
def test_leading_connectors_and_skill_adjectives_are_stripped(
    name: str, rest: str, rules: tuple[str, ...]
) -> None:
    assert strip_leading(name) == (rest, rules)


# --- the decision, signal by signal ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "keep", "reason", "final", "rules"),
    [
        ("Connect", False, "fragment", "Connect", ()),
        ("Batch  Models", False, "fragment", "Batch  Models", ()),  # case and spacing ignored
        ("and", False, "fragment", "and", ("stripped 'and'",)),
        ("strong", False, "fragment", "strong", ("stripped 'strong'",)),
        ("RAG", False, "deny-list", "RAG", ()),
        ("ci/cd", False, "deny-list", "ci/cd", ()),
        ("and RAG", False, "deny-list", "RAG", (AND,)),
        ("Deep Learning", False, "deny-list", "Deep Learning", ()),  # not "Learning"
        ("Zorbo", True, "allow-list", "Zorbo", ()),
        ("and Zorbo", True, "allow-list", "Zorbo", (AND,)),
        ("Google Cloud Platform", True, "alias of GCP", "GCP", ()),
        ("or Google Cloud Platform", True, "alias of GCP", "GCP", ("stripped 'or'",)),
        ("SOC 2", False, "concept rule: standard", "SOC 2", ()),
        ("ISO 27001", False, "concept rule: standard", "ISO 27001", ()),
        ("PCI DSS", False, "concept rule: standard", "PCI DSS", ()),
        ("Python", True, "dictionary", "Python", ()),
        ("python", True, "dictionary", "python", ()),
        ("and AWS", True, "dictionary", "AWS", (AND,)),
        ("or Go", True, "dictionary", "Go", ("stripped 'or'",)),
        ("strong SQL", True, "dictionary", "SQL", ("stripped 'strong'",)),
        ("and strong SQL", True, "dictionary", "SQL", (AND, "stripped 'strong'")),
        ("hands-on Docker", True, "dictionary", "Docker", ("stripped 'hands-on'",)),
    ],
)
def test_each_signal_decides_with_its_own_reason(
    name: str, keep: bool, reason: str, final: str, rules: tuple[str, ...]
) -> None:
    assert decide(name, POLICY) == Decision(keep, reason, final, rules)


@pytest.mark.parametrize(
    "name",
    [
        "Quuxdb",  # a capitalised name nothing knows used to pass; now it is dropped
        "Liquidity Planning",
        "Pricing Strategy",
        "Data Storytelling",
        "Cross-Team Alignment",
        "Retail Analytics",
        "Cloud-native practices",
        "Classical Statistics",
        "modern data stack",
        "AI-assisted workflows",
        "Platform-Forward",
        "Agentic Workflows",
        "strong Quuxdb",
        "solid team player",
    ],
)
def test_an_entry_nothing_vouches_for_is_dropped_as_not_a_known_product(name: str) -> None:
    decision = decide(name, POLICY)

    assert (decision.keep, decision.reason) == (False, "not a known product")


def test_a_stripped_rest_that_is_a_list_hit_uses_the_list() -> None:
    assert decide("strong Zorbo", POLICY).reason == "allow-list"
    assert decide("expert RAG", POLICY).reason == "deny-list"
    assert decide("or Connect", POLICY).reason == "fragment"


# --- the lists file -------------------------------------------------------------------------------


def test_the_lists_cannot_overlap(tmp_path: Path) -> None:
    path = tmp_path / "policy.toml"
    path.write_text('allow = ["Foo"]\ndeny = ["foo"]\n', encoding="utf-8")

    with pytest.raises(ScriptError, match="on both a keep list and a drop list: foo"):
        load_policy(path)


def test_an_alias_cannot_also_be_on_a_drop_list(tmp_path: Path) -> None:
    path = tmp_path / "policy.toml"
    path.write_text('fragments = ["Bar"]\n[aliases]\nbar = "Baz"\n', encoding="utf-8")

    with pytest.raises(ScriptError, match="on both a keep list and a drop list: bar"):
        load_policy(path)


def test_names_are_compared_case_folded_with_single_spaces() -> None:
    assert normal("  Deep   Learning ") == "deep learning"


def test_the_shipped_lists_load_and_hold_the_agreed_names() -> None:
    policy = load_policy(DEFAULT_POLICY_PATH)

    allowed = (
        "Decagon", "Kore.ai", "CoreML", "Confluent Platform", "Amazon Lex", "EKS",
        "Cluster Autoscaler", "Karpenter", "Git", "GitLab", "Node.js", "NestJS", "Express",
    )  # fmt: skip
    for name in allowed:
        assert normal(name) in policy.allow
    denied = (
        "LLMs", "RAG", "MLOps", "Deep Learning", "OSINT", "IaC", "CI/CD", "Infrastructure as Code",
        "Retrieval-Augmented Generation", "Wrike", "Product", "Marketing", "CRM",
        "Agile", "Scrum", "Kanban", "Challenger", "MEDDPICC", "Command of the Message",
    )  # fmt: skip
    for name in denied:
        assert normal(name) in policy.deny
    for name in ("Connect", "Retrieval", "AWS-first", "3DS", "TIPs", "real-time", "abuse"):
        assert normal(name) in policy.fragments
    assert "batch models" in policy.fragments
    assert "meddpicc" not in policy.allow
    assert policy.aliases == {"google cloud platform": "GCP"}


def test_the_shipped_lists_decide_the_agreed_examples() -> None:
    policy = load_policy(DEFAULT_POLICY_PATH)
    expected = {
        "Karpenter": True,
        "Cluster Autoscaler": True,
        "Pulumi": True,
        "SageMaker": True,
        "GitLab": True,
        "Git": True,
        "Google Cloud Platform": True,
        "strong SQL": True,
        "and AWS": True,
        "or Go": True,
        "LLMs": False,
        "MLOps": False,
        "Infrastructure as Code": False,
        "Retrieval-Augmented Generation": False,
        "Wrike": False,
        "Product": False,
        "Marketing": False,
        "CRM": False,
        "MEDDPICC": False,
        "Retrieval": False,
        "3DS": False,
        "and Retrieval": False,
        "Liquidity Planning": False,
    }

    for name, keep in expected.items():
        assert decide(name, policy).keep is keep, name
    assert decide("Google Cloud Platform", policy).name == "GCP"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("allow = [", "is not valid TOML"),
        ('allowed = ["x"]', "unknown key"),
        ('allow = "x"', "must be a list of strings"),
        ("deny = [1]", "must be a list of strings"),
        ('aliases = "x"', "must map names to names"),
        ("[aliases]\nfoo = 1\n", "must map names to names"),
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

    assert load_policy(path) == Policy(frozenset(), frozenset(), frozenset(), {})


# --- applying it to proposals and the table -------------------------------------------------------

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


def proposals_after_policy(stack: list[str] = POST) -> list[clean.Proposal]:
    proposals = build_proposals(make_file(stack))
    apply_policy(proposals, POLICY)
    return proposals


def kept(stack: list[str]) -> list[str]:
    return [i.name for p in proposals_after_policy(stack) for i in p.items if i.keep]


def test_the_policy_decides_every_entry_and_asks_about_none() -> None:
    proposals = proposals_after_policy()

    assert not [i for p in proposals for i in p.items if i.flag]
    verdict = {i.name: i.keep for p in proposals for i in p.items}
    assert verdict == {
        "Docker": True,
        "Cloud architecture": False,  # nothing vouches for it
        "Kubernetes": True,  # "and Kubernetes" with the "and" stripped
        "RAG": False,
        "Zorbo": True,
        "SOC 2": False,
        "Connect": False,
        "Strong ability to design large distributed systems": False,  # dropped text is not edited
        "Vector Databases": False,  # a category name, dropped by the earlier rule
        "Quuxdb": False,  # an unknown capitalised name is no longer let through
        "Faiss": True,
    }


def test_the_counts_are_what_the_lists_and_the_rule_decided() -> None:
    proposals = build_proposals(make_file(POST))

    # Docker, Kubernetes, Zorbo and Faiss are kept; six entries are dropped by the policy.
    assert apply_policy(proposals, POLICY) == (4, 6)


def test_a_leading_and_or_or_keeps_the_technology_and_dedupes_in_the_same_post() -> None:
    assert kept(["and AWS", "Docker"]) == ["AWS", "Docker"]
    assert kept(["and AWS", "AWS"]) == ["AWS"]  # the plain one comes later: a duplicate
    assert kept(["AWS", "or AWS"]) == ["AWS"]
    assert kept(["Go", "or Go", "and Go"]) == ["Go"]
    proposals = proposals_after_policy(["and AWS", "AWS"])
    assert proposals[1].items[0].drop_reason == "duplicate"


def test_duplicates_are_only_removed_within_one_post() -> None:
    proposals = build_proposals(make_file(["and AWS"], ["AWS"]))
    apply_policy(proposals, POLICY)

    assert [[i.name for i in p.items if i.keep] for p in proposals] == [["AWS"], ["AWS"]]


def test_an_alias_is_kept_under_its_canonical_name_and_dedupes_against_it() -> None:
    assert kept(["Google Cloud Platform"]) == ["GCP"]
    assert kept(["Google Cloud Platform", "GCP"]) == ["GCP"]
    assert kept(["GCP", "Google Cloud Platform"]) == ["GCP"]


def test_a_skill_adjective_is_stripped_before_matching() -> None:
    assert kept(["strong SQL", "hands-on Docker", "solid Quuxdb"]) == ["SQL", "Docker"]


def test_deep_learning_is_denied_whole_not_stripped_to_learning() -> None:
    proposals = proposals_after_policy(["Deep Learning", "deep Kubernetes"])

    items = [i for p in proposals for i in p.items]
    assert [(i.name, i.keep) for i in items] == [("Deep Learning", False), ("Kubernetes", True)]
    assert items[0].drop_reason == "deny-list"


def test_every_decision_shows_in_the_table_with_its_reason() -> None:
    table = format_table(proposals_after_policy())

    for expected in (
        "DROP (not a known product)",
        "DROP (fragment)",
        "DROP (deny-list)",
        "KEEP (allow-list)",
        "KEEP (dictionary)",
        "DROP (concept rule: standard)",
        "DROP (category name)",
        "stripped 'and'",
    ):
        assert expected in table
    assert "ASK" not in table
    assert "Docker" in table  # kept entries are listed too, with their reason


def test_the_table_shows_the_alias_and_the_stripped_words() -> None:
    rows = format_table(proposals_after_policy(["Google Cloud Platform", "strong SQL"]))

    assert "KEEP (alias of GCP)" in rows
    assert "stripped 'strong', KEEP (dictionary)" in rows


def test_an_entry_already_dropped_by_a_rule_is_not_decided_again() -> None:
    proposals = build_proposals(make_file(["Java (Primary)"]))
    apply_policy(proposals, POLICY)

    primary = proposals[0].items[1]
    assert primary.drop_reason == "annotation" and primary.note is None


# --- the command ----------------------------------------------------------------------------------


def test_policy_mode_asks_only_the_final_question_and_writes_on_yes(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, make_file(POST))
    session, questions, output = scripted(["y"])

    assert clean.run(path, session, POLICY) == 0

    assert len(questions) == 1 and questions[0].startswith("Write these changes")
    assert stacks(path) == [["Docker", "Kubernetes", "Zorbo", "Faiss"]]
    assert any(line.startswith("Policy: 4 entries kept and 6 dropped") for line in output)


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


def test_without_policy_an_unknown_capitalised_name_is_still_left_alone(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, make_file(["Quuxdb", "Cloud architecture"]))

    clean.run(path, scripted(["d", "y"])[0])

    assert stacks(path) == [["Quuxdb"]]


def test_main_loads_the_lists_before_touching_any_file(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, make_file(POST))
    bad = tmp_path / "bad.toml"
    bad.write_text("allow = [", encoding="utf-8")

    assert clean.main(["--labels", str(path), "--policy", "--policy-file", str(bad)]) == 2

    assert not (tmp_path / "labels.raw.json").exists()  # no backup was made either


def test_main_runs_policy_mode_with_a_lists_file(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, make_file(["Docker", "and Zorbo", "Cloud architecture", "Quuxdb"]))
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
