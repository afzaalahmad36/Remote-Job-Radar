"""`--eval`: which posts are scored or excluded, and what is printed and recorded.

The scoring is pure, so no database is needed: the jobs the command would read are built here, and
the one database function is replaced. Nothing here can reach a provider: the module under test
does not import the LLM client (checked below), and the network guard blocks any connection.
"""

import ast
import json
import uuid
from pathlib import Path

import pytest

from app.extraction.text_hash import raw_text_hash
from app.models import RemotePolicy, Seniority
from scripts import eval_run, try_extraction
from scripts.db_guard import ScriptError
from scripts.eval_labelfile import LabelEntry, LabelFile, Labels, save_label_file
from scripts.eval_run import (
    Evaluation,
    Exclusion,
    StoredJob,
    evaluate,
    format_exclusions,
    print_evaluation,
    rules_values,
    run_eval,
    save_eval_record,
)
from scripts.eval_scoring import FieldValues
from tests.settings_helpers import settings_with

ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE = uuid.UUID("22222222-2222-2222-2222-222222222222")
POST = (
    "Senior Python Developer at Acme Corp\n\nStack: Python, Django, PostgreSQL. "
    "Salary: $90,000 - $120,000 per year. Fully remote. " + "We build things. " * 10
)
SECRET = "Secret Title Marker"


def ids(count: int) -> list[uuid.UUID]:
    return [uuid.UUID(int=i + 1) for i in range(count)]


def label_matching_the_rules(text: str = POST, **changes: object) -> Labels:
    """A label equal to what the rules read from `text`: a post the system gets entirely right."""
    found = rules_values(text)
    values: dict[str, object] = {
        "title": found.title,
        "company": found.company,
        "stack": list(found.stack),
        "remote_policy": RemotePolicy(found.remote_policy) if found.remote_policy else None,
        "seniority": Seniority(found.seniority) if found.seniority else None,
        "salary_min": found.salary_min,
        "salary_max": found.salary_max,
        "salary_currency": found.salary_currency,
        "salary_period": found.salary_period,
    }
    return Labels(**{**values, **changes})  # type: ignore[arg-type]


def entry(
    job_id: uuid.UUID, text: str = POST, labels: Labels | None = None, skipped: bool = False
) -> LabelEntry:
    return LabelEntry(
        job_id=job_id,
        raw_text_hash=raw_text_hash(text),
        status="skipped" if skipped else "labelled",
        labels=None if skipped else (labels or label_matching_the_rules(text)),
    )


def job(text: str = POST, stored: FieldValues | None = None) -> StoredJob:
    return StoredJob(raw_text=text, stored=stored or rules_values(text))


def label_file(entries: list[LabelEntry], selected: int | None = None) -> LabelFile:
    chosen = ids(selected or len(entries))
    return LabelFile(source_id=SOURCE, seed=1, selected=chosen, posts=entries)


# --- the rules recomputed offline --------------------------------------------------------------


def test_the_offline_rules_read_the_fixture_post_as_expected() -> None:
    found = rules_values(POST)

    assert found.title == "Senior Python Developer"
    assert found.company == "Acme Corp"
    assert found.stack == ("Python", "Django", "PostgreSQL")
    assert (found.salary_min, found.salary_max, found.salary_currency) == (90000, 120000, "USD")
    assert found.salary_period == "year" and found.remote_policy == "remote"
    assert found.seniority == "senior"


def test_the_offline_rules_never_raise_on_odd_text() -> None:
    for text in ("", "   ", "\x00\x01", "word " * 5000):
        assert isinstance(rules_values(text), FieldValues)


# --- which posts are scored --------------------------------------------------------------------


def test_a_post_the_system_gets_right_passes_for_both_outputs() -> None:
    a, b = ids(2)
    jobs = {a: job(), b: job()}

    result = evaluate(label_file([entry(a), entry(b)]), jobs)

    assert result.selected == 2 and sum(result.excluded.values()) == 0
    assert result.stored.passed == result.rules.passed == 2
    assert result.stored_failures == {} == result.rules_failures


def test_the_stored_extraction_and_the_offline_rules_are_scored_separately() -> None:
    a = ids(1)[0]
    wrong_in_the_database = FieldValues(title="Junior Cook")

    result = evaluate(label_file([entry(a)]), {a: job(stored=wrong_in_the_database)})

    assert result.rules.passed == 1
    assert result.stored.passed == 0
    assert "title_wrong" in result.stored_failures[a]
    assert result.rules_failures == {}


def test_a_stored_job_with_nothing_extracted_fails_on_the_title() -> None:
    a = ids(1)[0]

    result = evaluate(label_file([entry(a)]), {a: job(stored=FieldValues())})

    assert result.stored.passed == 0
    assert "title_missing" in result.stored_failures[a]


def test_every_exclusion_reason_is_counted_and_nothing_is_lost() -> None:
    skipped, stale, blank, gone, fine, unlabelled = ids(6)
    entries = [
        entry(skipped, skipped=True),
        entry(stale, text="the text when it was labelled"),
        entry(blank, labels=Labels(title="  ")),
        entry(gone),
        entry(fine),
    ]
    jobs = {skipped: job(), stale: job(POST), blank: job(), fine: job()}

    result = evaluate(label_file(entries, selected=6), jobs)

    assert result.excluded == {
        Exclusion.SKIPPED: 1,
        Exclusion.STALE_HASH: 1,
        Exclusion.BLANK_TITLE: 1,
        Exclusion.JOB_NOT_FOUND: 1,
        Exclusion.NOT_LABELLED: 1,
    }
    assert result.stored.scored == 1
    assert result.selected == 6 == result.stored.scored + sum(result.excluded.values())
    assert unlabelled not in result.stored_failures


def test_a_stale_post_is_not_scored_against_a_label_made_for_other_text() -> None:
    a = ids(1)[0]

    result = evaluate(label_file([entry(a, text="old text")]), {a: job(POST)})

    assert result.excluded[Exclusion.STALE_HASH] == 1 and result.stored.scored == 0
    assert result.stored.verdict == "NO DATA"


def test_the_exclusion_line_gives_the_count_and_every_reason() -> None:
    a, b = ids(2)
    result = evaluate(label_file([entry(a, skipped=True), entry(b)]), {b: job()})

    line = format_exclusions(result)

    assert line.startswith("Posts: 2 selected, 1 scored, 1 excluded (")
    for reason in (
        "skipped 1",
        "stale_hash 0",
        "blank_title 0",
        "job_not_found 0",
        "not_labelled 0",
    ):
        assert reason in line


# --- what is printed and recorded --------------------------------------------------------------


def failing_run() -> Evaluation:
    a, b, c = ids(3)
    entries = [entry(a), entry(b, labels=label_matching_the_rules(title=SECRET)), entry(c)]
    return evaluate(label_file(entries), {a: job(), b: job(), c: job()})


def test_the_report_leads_with_the_share_the_gate_the_gaps_and_the_exclusions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_evaluation(failing_run())

    out = capsys.readouterr().out
    assert "Posts: 3 selected, 3 scored, 0 excluded" in out
    assert "Stored extraction (what the product shows): FAIL (on this sample)" in out
    assert "Rules recomputed offline: FAIL (on this sample)" in out
    assert "2/3 = 66.7%" in out and "invented gate" in out and "stack dictionary" in out
    assert "failed on title_wrong: 1 post(s)" in out
    assert "recall = how often a non-blank label value was found correctly" in out


def test_the_record_holds_numbers_and_ids_but_no_text_and_no_labels(tmp_path: Path) -> None:
    path = save_eval_record(
        tmp_path / "runs",
        labels_path=Path("somewhere/labels.json"),
        target="db:5432/jobradar",
        evaluation=failing_run(),
    )

    raw = path.read_text(encoding="utf-8")
    record = json.loads(raw)
    assert SECRET not in raw and "We build things" not in raw
    assert record["provider_requests"] == 0 and record["mode"] == "eval"
    assert record["labels_file"] == "labels.json" and record["database"] == "db:5432/jobradar"
    assert record["selected"] == 3 and record["stored"]["parse_correct"] == 2
    assert record["stored"]["verdict"] == "FAIL"
    assert list(record["stored_failures"].values()) == [["title_wrong"]]
    assert set(record["excluded"]) == {r.value for r in Exclusion}


# --- the command -------------------------------------------------------------------------------


def test_the_command_reads_scores_prints_and_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    a = ids(1)[0]
    path = tmp_path / "labels.json"
    save_label_file(path, label_file([entry(a)]))
    read: list[list[uuid.UUID]] = []

    async def fake_read(database_url: str, job_ids: list[uuid.UUID]) -> dict[uuid.UUID, StoredJob]:
        read.append(job_ids)
        return {a: job()}

    monkeypatch.setattr(eval_run, "read_stored_jobs", fake_read)
    monkeypatch.setattr(eval_run, "RUN_DIR", tmp_path / "runs")
    settings = settings_with(database_url="postgresql+asyncpg://u:p@db:5432/jobradar")

    assert run_eval(path, settings) == 0

    out = capsys.readouterr().out
    assert "Reading (read-only) from database db:5432/jobradar" in out
    assert "PASS (on this sample)" in out
    assert read == [[a]]
    assert len(list((tmp_path / "runs").glob("*.json"))) == 1


def test_the_command_refuses_the_wrong_database_before_reading_anything(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file([entry(ids(1)[0])]))
    settings = settings_with(database_url="postgresql+asyncpg://u:secret@db:5432/jobradar_test")

    with pytest.raises(ScriptError, match="refusing to read database"):
        run_eval(path, settings)
    assert not (tmp_path / "runs").exists()


def test_a_bad_labels_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ScriptError, match="does not match the format"):
        run_eval(path, settings_with(database_url="postgresql+asyncpg://u:p@db:5432/jobradar"))


# --- no provider, ever --------------------------------------------------------------------------


def test_eval_refuses_every_provider_option() -> None:
    for extra in (["--allow-provider"], ["--max-requests", "1"], ["--source-id", str(SOURCE)]):
        with pytest.raises(SystemExit):
            try_extraction.parse_args(["--eval", "labels.json", *extra])
    with pytest.raises(SystemExit):
        try_extraction.parse_args(
            ["--eval", "labels.json", "--allow-provider", "--max-requests", "3"]
        )


def test_the_eval_command_builds_no_client_and_asks_for_no_settings_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("a provider client must not be built for --eval")

    monkeypatch.setattr(try_extraction, "make_client", boom)
    monkeypatch.setattr(try_extraction, "build_client", boom)
    monkeypatch.setattr(try_extraction, "run_eval", lambda path, settings: calls.append(path) or 0)
    monkeypatch.setattr(try_extraction, "get_settings", lambda: settings_with())

    assert try_extraction.main(["--eval", str(tmp_path / "labels.json")]) == 0
    assert calls == [tmp_path / "labels.json"]


@pytest.mark.parametrize("name", ["eval_run", "eval_scoring", "eval_labelfile", "eval_label"])
def test_the_evaluation_modules_cannot_reach_a_provider(name: str) -> None:
    tree = ast.parse((ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)

    assert not {i for i in imported if i.startswith(("app.llm", "httpx", "requests", "aiohttp"))}
    assert not {i for i in imported if "build_client" in i}
