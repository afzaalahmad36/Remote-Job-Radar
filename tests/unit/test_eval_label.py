"""The labelling script and the label file. No network and no database: the terminal is a script of
answers and the database functions are replaced."""

import random
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.extraction.text_hash import raw_text_hash
from app.models import RemotePolicy, Seniority
from scripts import eval_label
from scripts.db_guard import ScriptError
from scripts.eval_label import (
    DEFAULT_N,
    PAGE_LINES,
    StopLabelling,
    ask_labels,
    build_parser,
    label_post,
    parse_amount,
    parse_choice,
    parse_currency,
    parse_stack,
    pick_posts,
    run_labelling,
    selection_query,
    start_or_resume,
)
from scripts.eval_labelfile import (
    LabelEntry,
    LabelFile,
    Labels,
    load_label_file,
    save_label_file,
)
from tests.settings_helpers import settings_with

SOURCE = uuid.UUID("11111111-1111-1111-1111-111111111111")
ROOT = Path(__file__).resolve().parent.parent.parent

# The answers for one post: title, company, stack, remote, min, max, currency, period, seniority,
# and Enter at "save this post?".
ONE_POST = [
    "Backend Engineer", "Acme", "python, Django , python", "remote", "90k", "120,000", "usd",
    "year", "senior", "",
]  # fmt: skip


class ScriptedPrompter:
    """Answers in order; running out of answers is the labeller closing the terminal."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.shown: list[str] = []

    def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise StopLabelling
        return self.answers.pop(0)

    def show(self, text: str) -> None:
        self.shown.append(text)


def ids(count: int) -> list[uuid.UUID]:
    return [uuid.UUID(int=i + 1) for i in range(count)]


# --- reading answers ---------------------------------------------------------------------------


def test_a_stack_is_comma_separated_trimmed_and_deduplicated() -> None:
    parsed = parse_stack(" Python, Django ,, Python ,PostgreSQL")
    assert parsed == ["Python", "Django", "PostgreSQL"]
    assert parse_stack("") == []


@pytest.mark.parametrize(
    ("answer", "amount"),
    [("90000", 90000), ("90,000", 90000), ("90 000", 90000), ("90k", 90000), ("1.5K", 1500),
     ("", None), ("  ", None)],
)  # fmt: skip
def test_amounts(answer: str, amount: int | None) -> None:
    assert parse_amount(answer) == amount


@pytest.mark.parametrize("answer", ["lots", "-5", "9x", "nan", "inf"])
def test_bad_amounts_are_rejected(answer: str) -> None:
    with pytest.raises(ValueError):
        parse_amount(answer)


def test_currency_is_a_three_letter_code_or_blank() -> None:
    assert parse_currency("usd") == "USD"
    assert parse_currency("") is None
    for bad in ("dollars", "US", "U$D"):
        with pytest.raises(ValueError):
            parse_currency(bad)


def test_choices_come_from_the_enum() -> None:
    assert parse_choice("Remote", RemotePolicy) is RemotePolicy.REMOTE
    assert parse_choice("on-site", RemotePolicy) is RemotePolicy.ONSITE
    assert parse_choice("", Seniority) is None
    with pytest.raises(ValueError, match="intern / junior"):
        parse_choice("boss", Seniority)


# --- asking ------------------------------------------------------------------------------------


def test_every_field_is_asked_in_order_and_blank_means_none() -> None:
    prompter = ScriptedPrompter(["Backend Engineer"] + [""] * 8)

    labels = ask_labels(prompter)

    assert labels == Labels(title="Backend Engineer")
    assert [p.split("[")[0].split(":")[0].strip() for p in prompter.prompts] == [
        "title", "company", "stack (a, b, c)", "remote policy", "salary min", "salary max",
        "salary currency", "salary period", "seniority",
    ]  # fmt: skip


def test_a_bad_answer_is_explained_and_asked_again() -> None:
    answers = ["T", "", "", "sometimes", "hybrid"] + [""] * 5
    prompter = ScriptedPrompter(answers)

    labels = ask_labels(prompter)

    assert labels.remote_policy is RemotePolicy.HYBRID
    assert any("choose one of" in line for line in prompter.shown)


def test_a_labelled_post_keeps_what_was_typed() -> None:
    text = "Some post text"

    entry = label_post(ScriptedPrompter(ONE_POST), ids(1)[0], text)

    assert entry.status == "labelled"
    assert entry.raw_text_hash == raw_text_hash(text)
    assert entry.labels == Labels(
        title="Backend Engineer", company="Acme", stack=["python", "Django"],
        remote_policy=RemotePolicy.REMOTE, seniority=Seniority.SENIOR, salary_min=90000,
        salary_max=120000, salary_currency="USD", salary_period="year",
    )  # fmt: skip


def test_skip_works_at_any_prompt() -> None:
    entry = label_post(ScriptedPrompter(["Title", "Acme", "/skip"]), ids(1)[0], "text")

    assert entry.status == "skipped" and entry.labels is None


def test_skip_works_at_the_final_question_too() -> None:
    entry = label_post(ScriptedPrompter(ONE_POST[:-1] + ["/skip"]), ids(1)[0], "text")

    assert entry.status == "skipped"


def test_quit_stops_without_an_entry() -> None:
    with pytest.raises(StopLabelling):
        label_post(ScriptedPrompter(["Title", "/quit"]), ids(1)[0], "text")


def test_redo_asks_the_whole_post_again() -> None:
    answers = ONE_POST[:-1] + ["r"] + ["Other Title"] + [""] * 8 + [""]

    entry = label_post(ScriptedPrompter(answers), ids(1)[0], "text")

    assert entry.labels is not None and entry.labels.title == "Other Title"


# --- the whole run: paging, saving after every post, resuming -----------------------------------


def make_file(count: int) -> LabelFile:
    return LabelFile(source_id=SOURCE, seed=1, selected=ids(count))


def test_a_long_post_is_shown_a_page_at_a_time(tmp_path: Path) -> None:
    text = "\n".join(f"line {i}" for i in range(PAGE_LINES * 2 + 5))
    prompter = ScriptedPrompter(["", "", *ONE_POST])  # Enter at the two "more" prompts

    run_labelling(prompter, make_file(1), {ids(1)[0]: text}, tmp_path / "labels.json")

    more = [prompt for prompt in prompter.prompts if prompt.startswith("-- more")]
    assert more == ["-- more (40/85 lines), Enter --", "-- more (80/85 lines), Enter --"]
    pages = [page for page in prompter.shown if page.startswith("line 0")]
    assert len(pages) == 1 and "line 39" in pages[0] and "line 40" not in pages[0]
    assert any("line 84" in page and "line 80" in page for page in prompter.shown)  # last page


def test_each_post_is_saved_before_the_next_is_asked(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    labels = make_file(3)
    texts = {job_id: f"post {n}" for n, job_id in enumerate(ids(3))}
    prompter = ScriptedPrompter([*ONE_POST, "/skip", "Title", "/quit"])

    handled = run_labelling(prompter, labels, texts, path)

    saved = load_label_file(path)
    assert handled == 2
    assert [entry.status for entry in saved.posts] == ["labelled", "skipped"]
    assert saved.pending() == [ids(3)[2]]  # the third post was quit in the middle: not recorded


def test_resuming_asks_only_the_posts_that_are_left(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    texts = {job_id: f"post number {n}" for n, job_id in enumerate(ids(3))}
    run_labelling(ScriptedPrompter([*ONE_POST, "/quit"]), make_file(3), texts, path)

    resumed = load_label_file(path)
    second = ScriptedPrompter(["/skip", "/skip"])
    left = {k: v for k, v in texts.items() if k in resumed.pending()}
    handled = run_labelling(second, resumed, left, path)

    shown = "\n".join(second.shown)
    assert handled == 2
    assert "post number 0" not in shown and "post number 1" in shown
    assert "Post 2 of 3" in shown and "Post 3 of 3" in shown
    assert load_label_file(path).pending() == []


def test_a_post_that_no_longer_exists_is_left_pending(tmp_path: Path) -> None:
    labels = make_file(1)

    handled = run_labelling(ScriptedPrompter([]), labels, {}, tmp_path / "labels.json")

    assert handled == 0 and labels.pending() == ids(1)


# --- the pick ----------------------------------------------------------------------------------


def rows_with_text(count: int) -> list[tuple[uuid.UUID, str]]:
    return [(job_id, f"distinct post {n}") for n, job_id in enumerate(ids(count))]


def test_the_same_rows_and_seed_give_the_same_pick_in_any_row_order() -> None:
    rows = rows_with_text(60)
    shuffled = rows[:]
    random.Random(5).shuffle(shuffled)

    assert pick_posts(rows, 30, seed=7) == pick_posts(shuffled, 30, seed=7)
    assert len(set(pick_posts(rows, 30, seed=7))) == 30


def test_another_seed_picks_other_posts() -> None:
    rows = rows_with_text(60)

    assert pick_posts(rows, 10, seed=1) != pick_posts(rows, 10, seed=2)


def test_posts_with_the_same_text_are_picked_once() -> None:
    rows = [(job_id, "same text") for job_id in ids(5)] + [(ids(6)[5], "  same\n text ")]
    rows += [(ids(7)[6], "other text")]

    picked = pick_posts(rows, 30, seed=1)

    assert len(picked) == 2  # whitespace differences do not make a different post
    assert ids(7)[6] in picked


def test_asking_for_more_than_exist_returns_what_exists() -> None:
    assert len(pick_posts(rows_with_text(4), 30, seed=1)) == 4
    assert pick_posts([], 30, seed=1) == []


# --- blindness ---------------------------------------------------------------------------------


def test_the_labelling_queries_select_only_the_id_and_the_text() -> None:
    columns = [column.name for column in selection_query(SOURCE).selected_columns]

    assert columns == ["id", "raw_text"]


async def test_posts_by_id_also_read_only_the_id_and_the_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []

    async def fake_fetch(database_url: str, query: object) -> list[tuple[uuid.UUID, str]]:
        seen.append([column.name for column in query.selected_columns])  # type: ignore[attr-defined]
        return []

    monkeypatch.setattr(eval_label, "_fetch_rows", fake_fetch)

    await eval_label.read_posts_by_id("postgresql+asyncpg://u:p@h/jobradar", ids(2))
    await eval_label.read_source_posts("postgresql+asyncpg://u:p@h/jobradar", SOURCE)

    assert seen == [["id", "raw_text"], ["id", "raw_text"]]


def test_the_script_never_mentions_the_extracted_columns() -> None:
    source = (ROOT / "scripts" / "eval_label.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("#", '"'))
    )

    names = ("Job.title", "Job.stack", "Job.company", "extraction_", "extract_with_rules")
    for forbidden in names:
        assert forbidden not in code


# --- starting and resuming ---------------------------------------------------------------------


def test_a_new_run_makes_the_selection(tmp_path: Path) -> None:
    labels = start_or_resume(tmp_path / "labels.json", SOURCE, 3, rows_with_text(10), 4)

    assert len(labels.selected) == 4 and labels.seed == 3 and labels.source_id == SOURCE


def test_resuming_uses_the_saved_selection_even_if_new_posts_arrived(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    first = start_or_resume(path, SOURCE, 3, rows_with_text(10), 4)
    save_label_file(path, first)

    again = start_or_resume(path, SOURCE, 3, rows_with_text(50), 4)

    assert again.selected == first.selected


def test_resuming_with_another_seed_or_source_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, start_or_resume(path, SOURCE, 3, rows_with_text(10), 4))

    with pytest.raises(ScriptError, match="same --source-id and --seed"):
        start_or_resume(path, SOURCE, 4, [], 4)
    with pytest.raises(ScriptError, match="same --source-id and --seed"):
        start_or_resume(path, uuid.uuid4(), 3, [], 4)


def test_a_source_without_posts_cannot_be_labelled(tmp_path: Path) -> None:
    with pytest.raises(ScriptError, match="no posts"):
        start_or_resume(tmp_path / "labels.json", SOURCE, 1, [], 30)


def test_the_defaults_are_30_posts_and_a_fixed_seed() -> None:
    args = build_parser().parse_args(["--source-id", str(SOURCE)])

    assert args.n == DEFAULT_N == 30
    assert args.seed == 1


def test_a_source_must_be_named_exactly_one_way() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--source-id", str(SOURCE), "--source-name", "x"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--source-id", str(SOURCE), "--n", "0"])


# --- the command, with the database replaced ----------------------------------------------------


def patch_database(monkeypatch: pytest.MonkeyPatch, texts: dict[uuid.UUID, str], url: str) -> None:
    async def source_posts(database_url: str, source_id: uuid.UUID) -> list[tuple[uuid.UUID, str]]:
        return list(texts.items())

    async def posts_by_id(
        database_url: str, job_ids: list[uuid.UUID]
    ) -> list[tuple[uuid.UUID, str]]:
        return [(k, v) for k, v in texts.items() if k in job_ids]

    monkeypatch.setattr(eval_label, "get_settings", lambda: settings_with(database_url=url))
    monkeypatch.setattr(eval_label, "read_source_posts", source_posts)
    monkeypatch.setattr(eval_label, "read_posts_by_id", posts_by_id)


def test_the_command_labels_resumes_and_never_stores_post_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    texts = {job_id: f"SECRET-POST-TEXT {n}" for n, job_id in enumerate(ids(3))}
    patch_database(monkeypatch, texts, "postgresql+asyncpg://u:p@db:5432/jobradar")
    out = tmp_path / "labels.json"
    argv = ["--source-id", str(SOURCE), "--n", "3", "--out", str(out)]

    assert eval_label.main(argv, ScriptedPrompter([*ONE_POST, "/quit"])) == 0
    assert eval_label.main(argv, ScriptedPrompter(["/skip", "/skip"])) == 0

    saved = load_label_file(out)
    assert [entry.status for entry in saved.posts].count("labelled") == 1
    assert saved.pending() == []
    assert "SECRET-POST-TEXT" not in out.read_text(encoding="utf-8")
    assert "1 of 3 done, 2 to go" in capsys.readouterr().out


def test_the_wrong_database_is_refused_before_anything_is_read(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    patch_database(monkeypatch, {}, "postgresql+asyncpg://u:secret@db:5432/jobradar_test")

    code = eval_label.main(
        ["--source-id", str(SOURCE), "--out", str(tmp_path / "labels.json")], ScriptedPrompter([])
    )

    assert code == 2
    error = capsys.readouterr().err
    assert "refusing to read database" in error and "secret" not in error
    assert not (tmp_path / "labels.json").exists()


# --- the file ----------------------------------------------------------------------------------


def entry(**changes: object) -> LabelEntry:
    values: dict[str, object] = {
        "job_id": ids(1)[0], "raw_text_hash": "a" * 64, "status": "labelled",
        "labels": Labels(title="T"),
    }  # fmt: skip
    return LabelEntry(**{**values, **changes})  # type: ignore[arg-type]


def test_labels_are_present_exactly_when_the_post_is_labelled() -> None:
    with pytest.raises(ValidationError):
        entry(labels=None)
    with pytest.raises(ValidationError):
        entry(status="skipped")  # labels given for a skipped post
    assert entry(status="skipped", labels=None).labels is None


@pytest.mark.parametrize("bad", ["", "xyz", "A" * 64, "a" * 63])
def test_the_hash_must_be_a_sha256_digest(bad: str) -> None:
    with pytest.raises(ValidationError):
        entry(raw_text_hash=bad)


def test_label_values_are_validated() -> None:
    wrong = (
        {"salary_min": -1},
        {"salary_currency": "usd"},
        {"remote_policy": "moon"},
        {"extra": 1},
    )
    for bad in wrong:
        with pytest.raises(ValidationError):
            Labels(**bad)  # type: ignore[arg-type]


def test_posts_must_come_from_the_selection_once() -> None:
    with pytest.raises(ValidationError, match="not in the selection"):
        LabelFile(source_id=SOURCE, seed=1, selected=ids(1), posts=[entry(job_id=ids(2)[1])])
    with pytest.raises(ValidationError, match="twice"):
        LabelFile(source_id=SOURCE, seed=1, selected=ids(1), posts=[entry(), entry()])


def test_the_file_round_trips_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "labels.json"
    labels = LabelFile(source_id=SOURCE, seed=9, selected=ids(2), posts=[entry()])

    save_label_file(path, labels)

    assert load_label_file(path) == labels
    assert [p.name for p in path.parent.iterdir()] == ["labels.json"]


def test_a_broken_or_missing_file_is_reported_without_a_traceback(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"version": 2}', encoding="utf-8")

    with pytest.raises(ScriptError, match="does not match the format"):
        load_label_file(bad)
    with pytest.raises(ScriptError, match="cannot read"):
        load_label_file(tmp_path / "missing.json")


def test_labels_become_the_values_the_scoring_compares() -> None:
    values = Labels(
        title="T", stack=["Python"], remote_policy=RemotePolicy.HYBRID, salary_min=1
    ).values()

    assert values.title == "T" and values.stack == ("Python",)
    assert values.remote_policy == "hybrid" and values.salary_min == 1 and values.seniority is None


def test_the_labels_folder_stays_git_ignored() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".local/" in ignored
    from scripts.eval_labelfile import DEFAULT_LABELS_PATH

    assert DEFAULT_LABELS_PATH.relative_to(ROOT).parts[0] == ".local"
