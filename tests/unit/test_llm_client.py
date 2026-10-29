import ast
import logging
from pathlib import Path

import pytest

from app.llm import client as client_module
from app.llm.base import (
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
