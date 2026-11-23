"""The blind labelling tool (scripts/eval_match_label.py): invented sheets, scripted answers."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts import eval_match_label as label
from scripts import match_eval as ev
from scripts.db_guard import ScriptError
from tests.unit.test_embedding_isolation import forbidden_imports
from tests.unit.test_eval_match_sheet import CONFIG, make_pool, three

ROOT = Path(__file__).resolve().parent.parent.parent
PERSONAS = ROOT / "tests" / "fixtures" / "cv" / "personas.toml"


@pytest.fixture
def out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An output folder inside a `.local/` under the working directory, with alex's sheet."""
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / ".local" / "match_eval"
    (folder / "sheets").mkdir(parents=True)
    pool = make_pool()
    plan = ev.plan_all(three(), pool, CONFIG, seed=1)[0]
    document = ev.sheet_document(plan, pool)
    document["pool_hash"] = ev.PROTOCOL_POOL_HASH  # the real pool is not available in a test
    (folder / "sheets" / "alex.json").write_text(json.dumps(document), encoding="utf-8")
    return folder


def scripted(answers: list[str]) -> Callable[[str], str]:
    queue = iter(answers)

    def ask(_prompt: str) -> str:
        return next(queue)

    return ask


def read_labels(out: Path) -> dict[str, list[dict[str, object]]]:
    document: dict[str, list[dict[str, object]]] = json.loads(
        (out / "labels" / "alex.json").read_text(encoding="utf-8")
    )
    return document


def run(out: Path, answers: list[str], shown: list[str] | None = None) -> int:
    lines: list[str] = shown if shown is not None else []
    return label.run(label.parse_args(["alex", "--out", str(out)]), scripted(answers), lines.append)


def sheet_size(out: Path) -> int:
    return len(label.load_sheet(out / "sheets" / "alex.json", "alex"))


def test_every_item_is_labelled_and_saved_in_a_file_for_that_persona(out: Path) -> None:
    total = sheet_size(out)

    assert run(out, ["r", "m", "n"] * total) == 0

    saved = read_labels(out)
    entries = saved["labels"]
    assert len(entries) == total
    assert set(saved) == {"persona", "pool_hash", "labels"}
    assert all(set(e) == {"item", "job_id", "label"} for e in entries)
    assert {e["label"] for e in entries} <= set(ev.LABELS)
    assert [p.name for p in (out / "labels").iterdir()] == ["alex.json"]


def test_a_label_is_saved_before_the_next_question_so_a_crash_loses_nothing(out: Path) -> None:
    with pytest.raises(StopIteration):
        run(out, ["r", "n", "m"])  # the scripted input ends: like a closed terminal

    assert [e["label"] for e in read_labels(out)["labels"]] == ["relevant", "no", "maybe"]
    assert not list((out / "labels").glob("*.tmp"))


def test_running_again_resumes_at_the_first_unlabelled_item(out: Path) -> None:
    run(out, ["r", "m", "q"])
    shown: list[str] = []

    run(out, ["q"], shown)

    assert any("item 3 of" in line for line in shown)
    assert not any("item 1 of" in line for line in shown)
    assert len(read_labels(out)["labels"]) == 2


def test_back_relabels_the_previous_item(out: Path) -> None:
    run(out, ["r", "b", "n", "q"])

    assert [(e["item"], e["label"]) for e in read_labels(out)["labels"]] == [(1, "no")]


def test_an_unknown_answer_asks_again_and_q_stops_without_losing_labels(out: Path) -> None:
    shown: list[str] = []

    run(out, ["x", "", "r", "q"], shown)

    assert sum("not understood" in line for line in shown) == 2
    assert len(read_labels(out)["labels"]) == 1
    assert "stopped; run again to resume" in shown


def test_what_the_labeller_sees_has_no_rank_score_method_or_group(out: Path) -> None:
    shown: list[str] = []

    run(out, ["q"], shown)

    text = "\n".join(shown).lower()
    for word in ("rank", "score", "method", "hybrid", "newest", "random", "example", "union"):
        assert word not in text
    assert "title:" in text and "company:" in text and "stack:" in text


def test_the_tool_never_names_the_answer_key() -> None:
    source = (ROOT / "scripts" / "eval_match_label.py").read_text(encoding="utf-8")

    for name in ("key.json", "key_document", "rankings", "seen_examples", "union"):
        assert name not in source


def test_the_tool_works_when_no_answer_key_exists_at_all(out: Path) -> None:
    assert not (out / "key").exists()

    assert run(out, ["q"]) == 0


def test_a_sheet_of_another_persona_or_pool_is_refused(out: Path) -> None:
    path = out / "sheets" / "alex.json"
    document = json.loads(path.read_text(encoding="utf-8"))

    with pytest.raises(ScriptError, match="not this persona"):
        label.load_sheet(path, "sam")
    document["pool_hash"] = "0" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ScriptError, match="not this persona"):
        label.load_sheet(path, "alex")


def test_a_sheet_that_carries_an_extra_field_is_refused(out: Path) -> None:
    path = out / "sheets" / "alex.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["items"][0]["rank"] = 1
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ScriptError, match="unexpected shape"):
        label.load_sheet(path, "alex")


def test_a_labels_file_that_does_not_match_the_sheet_is_refused(out: Path) -> None:
    run(out, ["r", "q"])
    path = out / "labels" / "alex.json"
    first_job = str(label.load_sheet(out / "sheets" / "alex.json", "alex")[0]["job_id"])
    document = json.loads(path.read_text(encoding="utf-8"))

    document["labels"][0]["job_id"] = "someone-else"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ScriptError, match="does not match the sheet"):
        run(out, ["q"])
    document["labels"][0]["job_id"] = first_job
    document["labels"][0]["label"] = "great"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ScriptError, match="does not match the sheet"):
        run(out, ["q"])


def test_an_output_folder_outside_local_and_a_path_like_persona_are_refused(
    out: Path, tmp_path: Path
) -> None:
    elsewhere = label.parse_args(["alex", "--out", str(tmp_path / "elsewhere")])
    with pytest.raises(ScriptError, match="inside .local"):
        label.run(elsewhere, input, print)
    with pytest.raises(ScriptError, match="lower case"):
        label.run(label.parse_args(["../alex", "--out", str(out)]), input, print)


def test_the_preferences_are_shown_from_the_persona_file_without_cv_text() -> None:
    line = label.preference_line("alex", PERSONAS)

    assert (
        line == "seniority senior, lead; remote policy remote; minimum salary 100000 USD per year"
    )
    assert label.preference_line("nobody", PERSONAS) is None


def test_a_closed_terminal_ends_cleanly_with_progress_saved(
    out: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = iter(["r"])

    def fake_input(_prompt: str = "") -> str:
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)

    assert label.main(["alex", "--out", str(out)]) == 1

    assert "progress is saved" in capsys.readouterr().err
    assert len(read_labels(out)["labels"]) == 1


def test_the_labelling_tool_reaches_no_llm_and_no_http_client() -> None:
    assert forbidden_imports(ROOT / "app", ROOT / "scripts" / "eval_match_label.py") == []
