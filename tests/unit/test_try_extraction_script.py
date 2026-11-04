"""The manual script's argument parsing and request-budget logic. No network, no database: the
provider is a FakeLLMClient and the script's database and terminal code is not exercised here."""

import json
import uuid
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from app.llm.base import (
    LLMConfigError,
    LLMRateLimited,
    LLMTransientError,
    Purpose,
)
from app.llm.client import GuardedClient
from app.llm.prompts import EXTRACT_JOB_V1_SCHEMA
from app.llm.providers.fake import DEFAULT_RESPONSE, FakeLLMClient
from app.llm.providers.gemini import GeminiClient
from scripts.try_extraction import (
    HARD_MAX_REQUESTS,
    PostAnalysis,
    PostRow,
    RequestBudget,
    ScriptError,
    analyze,
    build_ping_request,
    database_target,
    drop_duplicate_texts,
    ensure_jobs_table,
    field_coverage,
    parse_args,
    provider_settings,
    request_fields,
    run_ping,
    run_ping_command,
    run_requests,
    save_record,
    select_candidates,
    settings_summary,
    timeout_hint,
)
from tests.settings_helpers import settings_with

MIN_CHARS = 500
# Long enough to pass the length gate, with no title or technology the rules could find.
VAGUE_POST = "We are a growing team and we would love to hear from you. " * 12
COMPLETE_POST = (
    "Senior Python Developer at Acme Corp\n\nStack: Python, Django, PostgreSQL. " + "x " * 300
)
STUB = "Acme Is Hiring a Backend Engineer"  # a headline only


def post(text: str = VAGUE_POST) -> PostAnalysis:
    return analyze(uuid.uuid4(), text, MIN_CHARS)


# --- arguments ---------------------------------------------------------------------------------


def test_the_default_is_a_dry_run_without_a_provider() -> None:
    args = parse_args([])

    assert args.allow_provider is False
    assert args.max_requests is None
    assert args.eval is None


def test_a_provider_run_needs_a_budget() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--allow-provider"])


def test_a_budget_alone_does_not_enable_the_provider() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--max-requests", "3"])


@pytest.mark.parametrize("value", ["0", "-1", str(HARD_MAX_REQUESTS + 1), "100", "many", "2.5"])
def test_a_budget_outside_the_hard_cap_is_refused(value: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--allow-provider", "--max-requests", value])


def test_the_hard_cap_itself_is_allowed() -> None:
    args = parse_args(["--allow-provider", "--max-requests", str(HARD_MAX_REQUESTS)])

    assert args.max_requests == HARD_MAX_REQUESTS == 10


def test_other_options_are_checked() -> None:
    source = uuid.uuid4()
    assert parse_args(["--source-id", str(source), "--limit", "5", "--yes"]).source_id == source
    for bad in (["--source-id", "nope"], ["--limit", "0"], ["--limit", "501"]):
        with pytest.raises(SystemExit):
            parse_args(bad)


def test_eval_cannot_be_combined_with_a_provider_run() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--eval", "x.json", "--allow-provider", "--max-requests", "1"])


# --- the budget --------------------------------------------------------------------------------


@pytest.mark.parametrize("size", [0, -1, HARD_MAX_REQUESTS + 1])
def test_a_budget_cannot_exceed_the_hard_cap_even_from_code(size: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 10"):
        RequestBudget(size)


async def test_the_run_stops_when_the_budget_is_used() -> None:
    client = FakeLLMClient()
    budget = RequestBudget(2)

    records = await run_requests(client, [post() for _ in range(5)], budget, max_chars=12_000)

    assert len(client.requests) == len(records) == budget.used == 2
    assert budget.can_send is False


async def test_the_first_429_stops_the_run_and_nothing_is_retried() -> None:
    client = FakeLLMClient([DEFAULT_RESPONSE, LLMRateLimited(30.0), DEFAULT_RESPONSE])
    budget = RequestBudget(5)

    records = await run_requests(client, [post() for _ in range(5)], budget, max_chars=12_000)

    assert len(client.requests) == 2  # the third post was never sent, the 429 never repeated
    assert [r.outcome for r in records] == ["ok", "rate_limited"]
    assert budget.stop_reason is not None and "429" in budget.stop_reason


async def test_a_rejected_request_such_as_a_quota_error_also_stops_the_run() -> None:
    client = FakeLLMClient([LLMConfigError("request rejected (403)")])
    budget = RequestBudget(5)

    records = await run_requests(client, [post() for _ in range(3)], budget, max_chars=12_000)

    assert len(client.requests) == 1
    assert records[0].outcome == "rejected"
    assert budget.stop_reason is not None and "rejected" in budget.stop_reason


async def test_a_timeout_costs_a_request_but_does_not_stop_the_run() -> None:
    client = FakeLLMClient([LLMTransientError("timeout"), DEFAULT_RESPONSE])
    budget = RequestBudget(5)

    records = await run_requests(client, [post(), post()], budget, max_chars=12_000)

    assert [r.outcome for r in records] == ["transient", "ok"]
    assert budget.used == 2 and budget.stop_reason is None


async def test_an_answer_that_fails_validation_is_reported_not_trusted() -> None:
    client = FakeLLMClient(['{"is_job_posting": true, "salary_min": "lots"}'])

    [record] = await run_requests(client, [post()], RequestBudget(1), max_chars=12_000)

    assert record.outcome == "ok" and record.validated is False
    assert record.validation_error is not None
    assert "lots" not in record.validation_error  # field names and error kinds only


async def test_a_valid_answer_is_merged_with_the_rules() -> None:
    answer = json.dumps({**json.loads(DEFAULT_RESPONSE), "title": "Backend Engineer"})

    [record] = await run_requests(
        FakeLLMClient([answer]), [post()], RequestBudget(1), max_chars=12_000
    )

    assert record.validated is True
    assert record.merged_values["title"] == "Backend Engineer"
    assert record.tokens["input"] == 10


async def test_contact_details_are_masked_in_what_is_sent() -> None:
    text = VAGUE_POST + " Write to jobs@acme.com or call +44 20 7946 0958."
    client = FakeLLMClient()

    await run_requests(client, [post(text)], RequestBudget(1), max_chars=12_000)

    sent = client.requests[0].user
    assert "jobs@acme.com" not in sent and "7946" not in sent


# --- which posts are sent ----------------------------------------------------------------------


def test_only_posts_that_need_a_request_and_pass_the_length_gate_are_selected() -> None:
    complete, vague, stub = post(COMPLETE_POST), post(VAGUE_POST), post(STUB)

    assert complete.needs_request is False  # title and stack found
    assert stub.needs_request is False  # missing fields, but too short to send
    assert vague.needs_request is True
    assert select_candidates([complete, stub, vague], 5) == [vague]


def test_selection_keeps_the_given_newest_first_order_and_the_budget() -> None:
    newest, middle, oldest = post(VAGUE_POST), post(VAGUE_POST + "!"), post(VAGUE_POST + "?")

    assert select_candidates([newest, middle, oldest], 2) == [newest, middle]


def test_repeated_texts_are_dropped_keeping_the_first() -> None:
    first, second = uuid.uuid4(), uuid.uuid4()
    rows = [PostRow(first, "same"), PostRow(second, "same"), PostRow(uuid.uuid4(), "other")]

    unique, skipped = drop_duplicate_texts(rows)

    assert [row.job_id for row in unique][0] == first
    assert len(unique) == 2 and skipped == 1


def test_coverage_counts_fields_per_post() -> None:
    coverage = field_coverage([post(COMPLETE_POST), post(VAGUE_POST)])

    assert coverage["title"] == 1 and coverage["stack"] == 1
    assert set(coverage) >= {"company", "remote_policy", "apply_email", "apply_url", "salary"}


# --- provider settings -------------------------------------------------------------------------


def test_a_provider_run_refuses_without_a_provider_a_model_or_a_key() -> None:
    with pytest.raises(ScriptError, match="LLM_PROVIDER"):
        provider_settings(settings_with())
    with pytest.raises(ScriptError, match="LLM_MODEL_EXTRACTION"):
        provider_settings(settings_with(llm_provider="gemini"))
    with pytest.raises(ScriptError, match="GEMINI_API_KEY"):
        provider_settings(settings_with(llm_provider="gemini", llm_model_extraction="m"))


def test_the_kill_switch_is_turned_on_in_a_copy_only() -> None:
    original = settings_with(llm_provider="fake", llm_model_extraction="m")

    copy = provider_settings(original)

    assert copy.llm_extraction_enabled is True
    assert original.llm_extraction_enabled is False


# --- files -------------------------------------------------------------------------------------


async def test_the_run_record_has_no_post_text(tmp_path: Path) -> None:
    client = FakeLLMClient()
    budget = RequestBudget(1)
    secret_post = post(VAGUE_POST + " PRIVATE-MARKER")

    records = await run_requests(client, [secret_post], budget, max_chars=12_000)

    path = save_record(tmp_path / "runs", client=client, budget=budget, records=records)

    saved = path.read_text(encoding="utf-8")
    assert "PRIVATE-MARKER" not in saved
    assert json.loads(saved)["budget"] == {"max": 1, "used": 1, "stop": None}


# --- which database may be read ----------------------------------------------------------------


def test_the_target_is_shown_without_the_password() -> None:
    url = "postgresql+asyncpg://postgres:hunter2@localhost:5434/jobradar"

    shown = database_target(url)

    assert shown == "localhost:5434/jobradar"
    assert "hunter2" not in shown and "postgres" not in shown


def test_the_default_port_is_shown_when_the_url_has_none() -> None:
    assert database_target("postgresql+asyncpg://u:p@db/jobradar") == "db:5432/jobradar"


@pytest.mark.parametrize("name", ["jobradar_test", "postgres", "otherproject", "JobRadar"])
def test_any_other_database_is_refused_before_connecting(name: str) -> None:
    with pytest.raises(ScriptError, match="refusing") as caught:
        database_target(f"postgresql+asyncpg://u:secret@localhost:5432/{name}")

    assert "secret" not in str(caught.value)


def test_an_unparsable_or_nameless_url_is_refused() -> None:
    for bad in ("not a url", "postgresql+asyncpg://u:p@localhost:5432"):
        with pytest.raises(ScriptError):
            database_target(bad)


class StubConnection:
    def __init__(self, table: str | None) -> None:
        self._table = table

    async def scalar(self, statement: object) -> str | None:
        return self._table


async def test_a_database_without_a_jobs_table_is_refused() -> None:
    with pytest.raises(ScriptError, match="no 'jobs' table"):
        await ensure_jobs_table(StubConnection(None))  # type: ignore[arg-type]

    await ensure_jobs_table(StubConnection("jobs"))  # type: ignore[arg-type]


# --- explaining a timeout ----------------------------------------------------------------------


async def test_a_response_timeout_says_how_long_it_waited_and_that_it_may_have_counted() -> None:
    client = FakeLLMClient([LLMTransientError("no response in time")])

    [record] = await run_requests(client, [post()], RequestBudget(1), max_chars=12_000)
    record.latency_ms = 25_400

    hint = timeout_hint(record, 25.0)

    assert hint is not None
    assert "waited 25.4 s of the 25 s limit" in hint and "may still have counted" in hint


async def test_a_connect_timeout_says_nothing_reached_the_provider() -> None:
    client = FakeLLMClient([LLMTransientError("connect timeout")])

    [record] = await run_requests(client, [post()], RequestBudget(1), max_chars=12_000)

    hint = timeout_hint(record, 25.0)

    assert hint is not None and "never made" in hint


async def test_other_failures_and_successes_get_no_timeout_hint() -> None:
    client = FakeLLMClient([LLMTransientError("server error 503"), DEFAULT_RESPONSE])

    records = await run_requests(client, [post(), post()], RequestBudget(2), max_chars=12_000)

    assert [timeout_hint(record, 25.0) for record in records] == [None, None]


def test_the_settings_summary_shows_the_thinking_level_timeout_and_output_budget() -> None:
    summary = settings_summary(settings_with(llm_request_timeout_seconds=25))

    assert "thinking level: low" in summary
    assert "request timeout: 25 s" in summary
    assert "max output tokens: 2048" in summary
    assert "model default" in settings_summary(settings_with(gemini_thinking_level=""))


# --- the connectivity ping ---------------------------------------------------------------------

PING_PARSE = ["--ping", "--allow-provider"]


def test_ping_needs_the_provider_flag_and_defaults_to_level_one() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--ping"])

    args = parse_args(PING_PARSE)

    assert args.ping is True and args.ping_level is None


@pytest.mark.parametrize("level", ["1", "2", "3"])
def test_the_ping_levels_are_accepted(level: str) -> None:
    assert parse_args([*PING_PARSE, "--ping-level", level]).ping_level == int(level)


@pytest.mark.parametrize(
    "bad",
    [
        [*PING_PARSE, "--ping-level", "4"],
        [*PING_PARSE, "--ping-level", "0"],
        ["--ping-level", "2"],  # a level without --ping
        [*PING_PARSE, "--max-requests", "1"],  # a ping is always exactly one request
        [*PING_PARSE, "--eval", "x.json"],
        [*PING_PARSE, "--source-id", str(uuid.uuid4())],
    ],
)
def test_bad_ping_options_are_refused(bad: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(bad)


def test_level_one_is_the_smallest_request() -> None:
    request = build_ping_request(1, max_chars=12_000)

    assert request.purpose is Purpose.CONNECTIVITY_CHECK
    assert request.system == "" and request.json_schema is None and request.store is None


def test_level_two_adds_a_system_instruction_and_store_false() -> None:
    request = build_ping_request(2, max_chars=12_000)

    assert request.system != "" and request.store is False and request.json_schema is None
    assert request.user == build_ping_request(1, max_chars=12_000).user  # the same fixed input


def test_level_three_sends_the_real_schema_with_a_fictional_post_only() -> None:
    request = build_ping_request(3, max_chars=12_000)

    assert request.purpose is Purpose.JOB_POST_EXTRACTION
    assert request.json_schema == EXTRACT_JOB_V1_SCHEMA and request.store is False
    assert "Example Corp" in request.user and "fictional" in request.user


async def body_fields_sent(level: int) -> tuple[set[str], list[str]]:
    """What the real Gemini adapter puts in the body for a level, next to what the script says."""
    seen: list[httpx.Request] = []
    answer = {
        "status": "completed",
        "steps": [{"type": "model_output", "content": [{"text": "x"}]}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=answer)

    client = GeminiClient(
        api_key=SecretStr("k"),
        model="gemini-3.8-flash",
        base_url="https://generativelanguage.googleapis.com",
        timeout_seconds=5,
        max_output_tokens=100,
        thinking_level="low",
        transport=httpx.MockTransport(handler),
    )
    request = build_ping_request(level, max_chars=12_000)
    await client.complete(request)
    claimed = request_fields(request, settings_with(gemini_thinking_level="low"))
    return set(json.loads(seen[0].content)), claimed


@pytest.mark.parametrize("level", [1, 2, 3])
async def test_the_printed_field_list_matches_what_the_adapter_really_sends(level: int) -> None:
    sent, claimed = await body_fields_sent(level)

    names = {field.split("(")[0].split("=")[0] for field in claimed}

    assert names == sent


async def test_each_level_adds_exactly_the_fields_it_is_documented_to_add() -> None:
    fields = {level: (await body_fields_sent(level))[0] for level in (1, 2, 3)}

    assert fields[1] == {"model", "input", "generation_config"}
    assert fields[2] == fields[1] | {"system_instruction", "store"}
    assert fields[3] == fields[2] | {"response_format"}


async def test_every_ping_level_is_allowed_through_the_guard_on_a_hosted_provider() -> None:
    for level in (1, 2, 3):
        inner = FakeLLMClient(local=False)

        await GuardedClient(inner).complete(build_ping_request(level, max_chars=12_000))

        assert len(inner.requests) == 1


async def test_a_ping_is_exactly_one_request_and_records_the_reply() -> None:
    client = FakeLLMClient(["pong"])
    budget = RequestBudget(1)

    record = await run_ping(client, build_ping_request(1, max_chars=12_000), 1, budget)

    assert len(client.requests) == budget.used == 1
    assert record.outcome == "ok" and record.model_text == "pong"
    assert record.job_id == "ping-level-1" and record.validated is False


async def test_level_three_checks_the_answer_against_the_extraction_schema() -> None:
    good = json.dumps({**json.loads(DEFAULT_RESPONSE), "title": "Senior Python Developer"})
    request = build_ping_request(3, max_chars=12_000)

    ok = await run_ping(FakeLLMClient([good]), request, 3, RequestBudget(1))
    bad = await run_ping(FakeLLMClient(["not json"]), request, 3, RequestBudget(1))

    assert ok.validated is True and ok.merged_values["title"] == "Senior Python Developer"
    assert bad.validated is False and bad.validation_error is not None


async def test_a_rate_limited_ping_is_recorded_and_stops() -> None:
    budget = RequestBudget(1)

    record = await run_ping(
        FakeLLMClient([LLMRateLimited(5.0)]), build_ping_request(2, max_chars=12_000), 2, budget
    )

    assert record.outcome == "rate_limited"
    assert budget.stop_reason is not None and "429" in budget.stop_reason


def test_the_ping_command_confirms_sends_one_request_and_saves_a_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = FakeLLMClient(["pong"])
    monkeypatch.setattr("scripts.try_extraction.make_client", lambda settings: client)
    monkeypatch.setattr("scripts.try_extraction.RUN_RECORD_DIR", tmp_path)
    monkeypatch.setattr("scripts.try_extraction.confirm", lambda question: True)
    args = parse_args([*PING_PARSE, "--ping-level", "2"])

    assert run_ping_command(args, settings_with(llm_request_timeout_seconds=25)) == 0

    shown = capsys.readouterr().out
    assert "level 2" in shown and "system_instruction" in shown and "store=false" in shown
    assert "request timeout: 25 s" in shown
    assert len(client.requests) == 1
    [saved] = tmp_path.glob("*.json")
    assert json.loads(saved.read_text(encoding="utf-8"))["mode"] == "ping level 2"


def test_declining_the_confirmation_sends_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeLLMClient(["pong"])
    monkeypatch.setattr("scripts.try_extraction.make_client", lambda settings: client)
    monkeypatch.setattr("scripts.try_extraction.RUN_RECORD_DIR", tmp_path)
    monkeypatch.setattr("scripts.try_extraction.confirm", lambda question: False)

    assert run_ping_command(parse_args(PING_PARSE), settings_with()) == 0

    assert client.requests == [] and list(tmp_path.iterdir()) == []
