"""The manual script's argument parsing and request-budget logic. No network, no database: the
provider is a FakeLLMClient and the script's database and terminal code is not exercised here."""

import json
import uuid
from pathlib import Path

import pytest

from app.llm.base import LLMConfigError, LLMRateLimited, LLMTransientError
from app.llm.providers.fake import DEFAULT_RESPONSE, FakeLLMClient
from scripts.try_extraction import (
    HARD_MAX_REQUESTS,
    PostAnalysis,
    PostRow,
    RequestBudget,
    ScriptError,
    analyze,
    drop_duplicate_texts,
    field_coverage,
    load_labels,
    parse_args,
    provider_settings,
    run_requests,
    save_record,
    select_candidates,
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


def test_a_labels_file_is_checked_against_the_documented_format(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"posts": [{"text": "t", "expected": {"title": "x"}}]}))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"posts": [{"text": "t", "expected": {"salary": 1}}]}))

    assert len(load_labels(good).posts) == 1
    with pytest.raises(ScriptError, match="does not match"):
        load_labels(bad)
    with pytest.raises(ScriptError, match="cannot read"):
        load_labels(tmp_path / "missing.json")
