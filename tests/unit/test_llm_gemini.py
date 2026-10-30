import json
import logging
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from app.llm.base import (
    LLMBadResponse,
    LLMConfigError,
    LLMRateLimited,
    LLMRequest,
    LLMTransientError,
    Purpose,
)
from app.llm.providers.gemini import MAX_RESPONSE_BYTES, GeminiClient

API_KEY = "AIza-test-key-0123456789"
ANSWER = '{"is_job_posting": true, "title": "Engineer"}'
REQUEST = LLMRequest(
    Purpose.JOB_POST_EXTRACTION, "system rules", "the post", {"type": "object", "properties": {}}
)


def reply(
    text: str = ANSWER,
    *,
    status: str | None = "completed",
    usage: object = None,
    steps: object = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "id": "interaction-1",
        "steps": steps
        if steps is not None
        else [{"type": "model_output", "content": [{"type": "text", "text": text}]}],
        "usage": usage
        if usage is not None
        else {"total_input_tokens": 120, "total_output_tokens": 40, "total_thought_tokens": 5},
    }
    if status:
        body["status"] = status
    return body


class Server:
    """Records requests and answers with whatever `respond` returns."""

    def __init__(self, respond: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._respond = respond
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._respond(request)


def json_server(body: object, status: int = 200, headers: dict[str, str] | None = None) -> Server:
    return Server(lambda request: httpx.Response(status, json=body, headers=headers))


def client_for(server: Server, **overrides: object) -> GeminiClient:
    options: dict[str, object] = {
        "api_key": SecretStr(API_KEY),
        "model": "test-model",
        "base_url": "https://generativelanguage.googleapis.com",
        "timeout_seconds": 20.0,
        "max_output_tokens": 1024,
    }
    return GeminiClient(**(options | overrides), transport=server.transport)  # type: ignore[arg-type]


# --- the request ------------------------------------------------------------------------------


async def test_the_request_goes_to_the_interactions_endpoint_with_the_documented_fields() -> None:
    server = json_server(reply())

    await client_for(server).complete(REQUEST)

    [sent] = server.requests
    body = json.loads(sent.content)
    assert sent.method == "POST"
    assert str(sent.url) == "https://generativelanguage.googleapis.com/v1beta/interactions"
    assert body == {
        "model": "test-model",
        "system_instruction": "system rules",
        "input": "the post",
        "generation_config": {"max_output_tokens": 1024},
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": {"type": "object", "properties": {}},
        },
        "store": False,
    }


async def test_the_api_key_travels_in_a_header_and_never_in_the_url_or_body() -> None:
    server = json_server(reply())

    await client_for(server).complete(REQUEST)

    [sent] = server.requests
    assert sent.headers["x-goog-api-key"] == API_KEY
    assert API_KEY not in str(sent.url)
    assert API_KEY.encode() not in sent.content


async def test_the_thinking_level_is_sent_only_when_configured() -> None:
    plain, leveled = json_server(reply()), json_server(reply())

    await client_for(plain).complete(REQUEST)
    await client_for(leveled, thinking_level="low").complete(REQUEST)

    assert "thinking_level" not in json.loads(plain.requests[0].content)["generation_config"]
    assert json.loads(leveled.requests[0].content)["generation_config"]["thinking_level"] == "low"


async def test_the_client_knows_it_is_hosted() -> None:
    assert client_for(json_server(reply())).local is False


# --- reading the answer -----------------------------------------------------------------------


async def test_the_text_usage_and_request_id_are_read() -> None:
    response = await client_for(json_server(reply())).complete(REQUEST)

    assert response.text == ANSWER
    assert response.request_id == "interaction-1"
    assert (
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.usage.thought_tokens,
    ) == (120, 40, 5)


async def test_text_from_several_output_items_is_joined_and_other_steps_are_ignored() -> None:
    steps = [
        {"type": "thought", "content": [{"type": "text", "text": "thinking..."}]},
        {"type": "model_output", "content": [{"type": "text", "text": '{"is_job'}]},
        {"type": "model_output", "content": [{"type": "text", "text": '_posting": true}'}]},
    ]

    response = await client_for(json_server(reply(steps=steps))).complete(REQUEST)

    assert response.text == '{"is_job_posting": true}'


async def test_missing_or_odd_usage_gives_unknown_token_counts() -> None:
    body = reply(usage={"total_input_tokens": True, "total_output_tokens": -1})

    response = await client_for(json_server(body)).complete(REQUEST)

    assert (response.usage.input_tokens, response.usage.output_tokens) == (None, None)


async def test_usage_that_is_not_an_object_gives_unknown_token_counts() -> None:
    response = await client_for(json_server(reply(usage="nonsense"))).complete(REQUEST)

    assert response.usage.input_tokens is None


async def test_a_missing_status_is_fine_when_there_is_text() -> None:
    assert (await client_for(json_server(reply(status=None))).complete(REQUEST)).text == ANSWER


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (reply(status="incomplete"), "output was cut off"),
        (reply(status="in_progress"), "response is not final"),
        (reply(status="requires_action"), "response is not final"),
        (reply(steps=[]), "empty output"),
        (reply(steps=[{"type": "model_output", "content": []}]), "empty output"),
        (reply(steps="nonsense"), "empty output"),
        (["not", "an", "object"], "unexpected response shape"),
    ],
)
async def test_unusable_answers_are_bad_responses(body: object, reason: str) -> None:
    with pytest.raises(LLMBadResponse, match=reason):
        await client_for(json_server(body)).complete(REQUEST)


@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_a_failed_interaction_is_a_transient_error(status: str) -> None:
    with pytest.raises(LLMTransientError):
        await client_for(json_server(reply(status=status))).complete(REQUEST)


async def test_a_body_that_is_not_json_is_a_bad_response() -> None:
    server = Server(lambda request: httpx.Response(200, content=b"<html>oops</html>"))

    with pytest.raises(LLMBadResponse, match="not JSON"):
        await client_for(server).complete(REQUEST)


async def test_a_huge_body_is_refused_while_streaming() -> None:
    server = Server(lambda request: httpx.Response(200, content=b"x" * (MAX_RESPONSE_BYTES + 1)))

    with pytest.raises(LLMBadResponse, match="too large"):
        await client_for(server).complete(REQUEST)


# --- failures ---------------------------------------------------------------------------------


async def test_429_is_rate_limited_and_carries_retry_after() -> None:
    server = json_server({"error": "quota"}, status=429, headers={"Retry-After": "30"})

    with pytest.raises(LLMRateLimited) as error:
        await client_for(server).complete(REQUEST)

    assert error.value.retry_after_seconds == 30


@pytest.mark.parametrize(
    ("header", "expected"),
    [(None, None), ("Wed, 21 Oct 2026 07:28:00 GMT", None), ("999999", 3600.0), ("abc", None)],
)
async def test_retry_after_is_bounded_and_optional(header: str | None, expected: float) -> None:
    headers = {"Retry-After": header} if header else None
    server = json_server({}, status=429, headers=headers)

    with pytest.raises(LLMRateLimited) as error:
        await client_for(server).complete(REQUEST)

    assert error.value.retry_after_seconds == expected


@pytest.mark.parametrize("status", [408, 500, 502, 503, 504])
async def test_timeouts_and_server_errors_are_transient(status: int) -> None:
    with pytest.raises(LLMTransientError):
        await client_for(json_server({}, status=status)).complete(REQUEST)


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_client_errors_are_config_errors_that_retrying_cannot_fix(status: int) -> None:
    with pytest.raises(LLMConfigError):
        await client_for(json_server({}, status=status)).complete(REQUEST)


async def test_a_redirect_is_refused_and_never_followed() -> None:
    server = Server(
        lambda request: httpx.Response(302, headers={"Location": "https://evil.example/steal"})
    )

    with pytest.raises(LLMConfigError, match="redirect"):
        await client_for(server).complete(REQUEST)

    assert len(server.requests) == 1  # the key was never sent anywhere else


@pytest.mark.parametrize(
    ("raised", "reason"),
    [
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.ConnectTimeout("slow"), "timeout"),
        (httpx.ConnectError("down"), "connection failed"),
        (httpx.RemoteProtocolError("garbled"), "connection failed"),
    ],
)
async def test_network_failures_are_transient(raised: Exception, reason: str) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise raised

    with pytest.raises(LLMTransientError, match=reason):
        await client_for(Server(fail)).complete(REQUEST)


async def test_there_are_no_hidden_retries() -> None:
    server = json_server({}, status=503)

    with pytest.raises(LLMTransientError):
        await client_for(server).complete(REQUEST)

    assert len(server.requests) == 1  # one call = one request, so quotas can be counted exactly


async def test_errors_and_logs_never_contain_the_key_or_the_provider_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    echo = "the-post-text-echoed-by-the-provider"
    server = json_server({"error": {"message": f"bad input: {echo} {API_KEY}"}}, status=400)

    with caplog.at_level(logging.DEBUG), pytest.raises(LLMConfigError) as error:
        await client_for(server).complete(REQUEST)

    everything = str(error.value) + repr(error.value) + caplog.text
    assert API_KEY not in everything
    assert echo not in everything
