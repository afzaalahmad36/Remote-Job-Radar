import ast
import logging
from pathlib import Path

import pytest

from app.llm import client as client_module
from app.llm.base import (
    PING_INPUT,
    PING_SYSTEM,
    LLMBadResponse,
    LLMConfigError,
    LLMPrivacyError,
    LLMRateLimited,
    LLMRequest,
    LLMTransientError,
    Purpose,
)
from app.llm.client import MAX_REQUEST_CHARS, GuardedClient, build_client
from app.llm.providers.fake import DEFAULT_RESPONSE, FakeLLMClient
from app.llm.providers.gemini import GeminiClient
from tests.settings_helpers import settings_with

APP = Path(__file__).resolve().parent.parent.parent / "app"


def request(**overrides: object) -> LLMRequest:
    fields: dict[str, object] = {
        "purpose": Purpose.JOB_POST_EXTRACTION,
        "system": "rules",
        "user": "the post text",
        "json_schema": {"type": "object"},
    }
    return LLMRequest(**(fields | overrides))  # type: ignore[arg-type]


def fixed_ping(**overrides: object) -> LLMRequest:
    """The one request the connectivity check may send."""
    fields: dict[str, object] = {
        "purpose": Purpose.CONNECTIVITY_CHECK,
        "system": "",
        "user": PING_INPUT,
        "json_schema": None,
        "store": None,
    }
    return LLMRequest(**(fields | overrides))  # type: ignore[arg-type]


# --- choosing a client ------------------------------------------------------------------------


def test_by_default_there_is_no_client_so_nothing_can_call_a_provider() -> None:
    assert build_client(settings_with()) is None


def test_the_kill_switch_off_means_no_client_even_with_a_provider_configured() -> None:
    settings = settings_with(llm_provider="gemini", llm_model_extraction="m", gemini_api_key="k")

    assert settings.llm_extraction_enabled is False
    assert build_client(settings) is None


def test_gemini_is_built_when_enabled_and_wrapped_in_the_guard() -> None:
    settings = settings_with(
        llm_provider="gemini",
        llm_extraction_enabled=True,
        llm_model_extraction="some-model",
        gemini_api_key="k",
    )

    built = build_client(settings)

    assert isinstance(built, GuardedClient)
    assert (built.name, built.model, built.local) == ("gemini", "some-model", False)
    assert isinstance(built._inner, GeminiClient)  # noqa: SLF001 - checking the wiring


def test_the_fake_provider_is_available_outside_production() -> None:
    built = build_client(settings_with(llm_provider="fake", llm_extraction_enabled=True,
                                       llm_model_extraction="m"))  # fmt: skip

    assert built is not None
    assert built.name == "fake"


def test_ollama_is_reserved_but_not_implemented_yet() -> None:
    settings = settings_with(
        llm_provider="ollama", llm_extraction_enabled=True, llm_model_extraction="m"
    )

    with pytest.raises(LLMConfigError, match="not implemented"):
        build_client(settings)


# --- the guard --------------------------------------------------------------------------------


async def test_a_hosted_provider_refuses_personal_data() -> None:
    inner = FakeLLMClient(local=False)

    with pytest.raises(LLMPrivacyError):
        await GuardedClient(inner).complete(request(contains_personal_data=True))

    assert inner.requests == []  # nothing was sent


async def test_a_hosted_provider_refuses_purposes_that_are_not_on_its_allow_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_module, "HOSTED_ALLOWED_PURPOSES", frozenset())
    inner = FakeLLMClient(local=False)

    with pytest.raises(LLMPrivacyError):
        await GuardedClient(inner).complete(request())

    assert inner.requests == []


async def test_a_local_provider_may_receive_personal_data() -> None:
    inner = FakeLLMClient(local=True)

    await GuardedClient(inner).complete(request(contains_personal_data=True))

    assert len(inner.requests) == 1


async def test_public_job_text_goes_through_a_hosted_provider() -> None:
    inner = FakeLLMClient(local=False)

    response = await GuardedClient(inner).complete(request())

    assert response.text == DEFAULT_RESPONSE


async def test_an_oversized_request_is_refused_before_it_is_sent() -> None:
    inner = FakeLLMClient()

    with pytest.raises(LLMConfigError, match="too large"):
        await GuardedClient(inner).complete(request(user="x" * (MAX_REQUEST_CHARS + 1)))

    assert inner.requests == []


async def test_provider_errors_pass_through_unchanged() -> None:
    for error in (LLMRateLimited(5), LLMTransientError("timeout"), LLMBadResponse("empty output")):
        with pytest.raises(type(error)):
            await GuardedClient(FakeLLMClient([error])).complete(request())


async def test_logs_name_the_provider_model_tokens_and_outcome_but_never_the_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_post = "SECRET-POST-TEXT-with-jobs@acme.com"
    guarded = GuardedClient(FakeLLMClient(["SECRET-MODEL-OUTPUT", LLMTransientError("timeout")]))

    with caplog.at_level(logging.INFO, logger="app.llm.client"):
        await guarded.complete(request(user=secret_post))
        with pytest.raises(LLMTransientError):
            await guarded.complete(request(user=secret_post))

    assert "provider=fake" in caplog.text
    assert "outcome=ok" in caplog.text
    assert "outcome=error reason=timeout" in caplog.text
    assert "(10, 5, None)" in caplog.text  # token counts
    assert "SECRET" not in caplog.text
    assert "acme.com" not in caplog.text


# --- the fake provider ------------------------------------------------------------------------


async def test_the_fake_follows_its_script_then_repeats_the_last_item() -> None:
    fake = FakeLLMClient(["first", LLMBadResponse("empty output"), "last"])

    assert (await fake.complete(request())).text == "first"
    with pytest.raises(LLMBadResponse):
        await fake.complete(request())
    assert (await fake.complete(request())).text == "last"
    assert (await fake.complete(request())).text == "last"
    assert len(fake.requests) == 4


async def test_a_strict_fake_fails_the_test_that_calls_it_too_often() -> None:
    fake = FakeLLMClient(["only"], strict=True)
    await fake.complete(request())

    with pytest.raises(AssertionError, match="more often than scripted"):
        await fake.complete(request())


# --- the rule that keeps providers behind client.py ---------------------------------------------


def imported_modules(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_only_client_py_imports_a_provider_adapter() -> None:
    offenders = [
        str(path.relative_to(APP))
        for path in APP.rglob("*.py")
        if path.parent.name != "providers"
        and path.name != "client.py"
        and any(name.startswith("app.llm.providers") for name in imported_modules(path))
    ]

    assert offenders == []


def test_no_module_imports_a_vendor_sdk() -> None:
    vendors = ("anthropic", "openai", "google.generativeai", "google.genai")
    offenders = [
        str(path.relative_to(APP))
        for path in APP.rglob("*.py")
        if any(name.split(".")[0] in {v.split(".")[0] for v in vendors}
               for name in imported_modules(path))
    ]  # fmt: skip

    assert offenders == []


async def test_leaving_retention_to_the_provider_is_only_allowed_for_the_connectivity_check() -> (
    None
):
    inner = FakeLLMClient(local=False)
    ping = fixed_ping()

    await GuardedClient(inner).complete(ping)  # a fixed harmless sentence: allowed

    assert len(inner.requests) == 1
    with pytest.raises(LLMPrivacyError):
        await GuardedClient(inner).complete(request(store=None))  # job text: must be store=False
    assert len(inner.requests) == 1


async def test_the_connectivity_check_still_refuses_personal_data_on_a_hosted_provider() -> None:
    inner = FakeLLMClient(local=False)
    ping = fixed_ping(contains_personal_data=True)

    with pytest.raises(LLMPrivacyError):
        await GuardedClient(inner).complete(ping)

    assert inner.requests == []


# --- the connectivity check carries only its fixed text -----------------------------------------


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("system", ["", PING_SYSTEM])
@pytest.mark.parametrize("store", [None, False])
async def test_the_fixed_ping_text_is_accepted(
    local: bool, system: str, store: bool | None
) -> None:
    inner = FakeLLMClient(local=local)

    await GuardedClient(inner).complete(fixed_ping(system=system, store=store))

    assert len(inner.requests) == 1


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize(
    "user",
    [
        "Senior Python Developer at Acme Corp. Fully remote. Apply: https://acme.com/jobs/1",
        "",  # empty
        PING_INPUT + " ",  # altered: trailing space
        PING_INPUT.upper(),  # altered: case
        PING_INPUT[:-1],  # altered: shortened
        PING_INPUT + " Also read this: the post text",  # fixed text with extra content
    ],
)
async def test_any_other_input_is_refused_under_the_connectivity_purpose(
    local: bool, user: str
) -> None:
    inner = FakeLLMClient(local=local)

    with pytest.raises(LLMPrivacyError, match="fixed text"):
        await GuardedClient(inner).complete(fixed_ping(user=user))

    assert inner.requests == []  # nothing was sent


@pytest.mark.parametrize("local", [False, True])
async def test_the_connectivity_check_refuses_other_system_text_and_any_schema(local: bool) -> None:
    inner = FakeLLMClient(local=local)
    guarded = GuardedClient(inner)

    with pytest.raises(LLMPrivacyError):
        await guarded.complete(fixed_ping(system="Extract the job and its apply email."))
    with pytest.raises(LLMPrivacyError):
        await guarded.complete(fixed_ping(json_schema={"type": "object"}))

    assert inner.requests == []


async def test_job_text_cannot_leave_store_unset_by_relabelling_it_as_a_ping() -> None:
    inner = FakeLLMClient(local=False)
    disguised = fixed_ping(user="Backend Engineer at Acme, apply to hr@acme.com", store=None)

    with pytest.raises(LLMPrivacyError):
        await GuardedClient(inner).complete(disguised)

    assert inner.requests == []


async def test_errors_raised_by_the_guard_say_the_request_never_left() -> None:
    inner = FakeLLMClient(local=False)

    with pytest.raises(LLMPrivacyError) as privacy:
        await GuardedClient(inner).complete(request(contains_personal_data=True))
    with pytest.raises(LLMConfigError) as too_large:
        await GuardedClient(inner).complete(request(user="x" * (MAX_REQUEST_CHARS + 1)))

    assert privacy.value.never_sent is True and too_large.value.never_sent is True
    assert inner.requests == []


async def test_errors_from_the_provider_do_not_claim_the_request_never_left() -> None:
    for error in (LLMRateLimited(5), LLMTransientError("timeout"), LLMConfigError("rejected")):
        with pytest.raises(type(error)) as raised:
            await GuardedClient(FakeLLMClient([error])).complete(request())

        assert raised.value.never_sent is False
