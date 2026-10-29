"""Google Gemini over its REST API (the Interactions endpoint), through httpx.

Everything specific to this endpoint stays in this file. It was chosen because its structured
JSON output, system instruction, generation settings, usage and status fields are all documented
(https://ai.google.dev/api/interactions-api and /gemini-api/docs/structured-output). Two things
the docs do not show are handled defensively: the shape of an HTTP error body (only the status
code is used) and the exact shape of the content items (any `text` item of a `model_output` step
is read).

One call is one HTTP request. There are no hidden retries: the caller counts requests against a
quota and decides what to retry.
"""

import json
from typing import Any

import httpx
from pydantic import SecretStr

from app.llm.base import (
    LLMBadResponse,
    LLMConfigError,
    LLMRateLimited,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    LLMUsage,
)

CONNECT_TIMEOUT_SECONDS = 5.0
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_RETRY_AFTER_SECONDS = 3600.0
USER_AGENT = "RemoteJobRadar/0.1"


class GeminiClient:
    name = "gemini"
    local = False  # a hosted service: only public text may be sent (see Purpose)

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str,
        base_url: str,
        timeout_seconds: float,
        max_output_tokens: int,
        thinking_level: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/v1beta/interactions"
        self._timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT_SECONDS,
            read=timeout_seconds,
            write=timeout_seconds,
            pool=CONNECT_TIMEOUT_SECONDS,
        )
        self._max_output_tokens = max_output_tokens
        self._thinking_level = thinking_level
        self._transport = transport  # tests pass a MockTransport; None means the real network

    async def complete(self, request: LLMRequest) -> LLMResponse:
        body = await self._post(self._payload(request))
        return _parse(body)

    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        config: dict[str, Any] = {"max_output_tokens": self._max_output_tokens}
        if self._thinking_level:
            config["thinking_level"] = self._thinking_level
        return {
            "model": self.model,
            "system_instruction": request.system,
            "input": request.user,
            "generation_config": config,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": request.json_schema,
            },
            "store": False,  # do not keep the interaction on the provider's side
        }

    async def _post(self, payload: dict[str, Any]) -> bytes:
        # The key travels in a header, never in the URL, so it cannot end up in logs or errors.
        headers = {
            "x-goog-api-key": self._api_key.get_secret_value(),
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        try:
            # follow_redirects=False: a redirect must never carry the key to another host.
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=self._timeout,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                async with client.stream("POST", self._url, json=payload, headers=headers) as reply:
                    _raise_for_status(reply)
                    return await _read_capped(reply)
        except httpx.TimeoutException:
            raise LLMTransientError("timeout") from None
        except httpx.HTTPError:
            raise LLMTransientError("connection failed") from None


def _raise_for_status(response: httpx.Response) -> None:
    status = response.status_code
    if status == 429:
        raise LLMRateLimited(_retry_after(response))
    if status == 408 or status >= 500:
        raise LLMTransientError(f"server error {status}")
    if 300 <= status < 400:
        raise LLMConfigError("unexpected redirect")
    if status >= 400:
        # The body is not read or kept: provider error messages can repeat what we sent.
        raise LLMConfigError(f"request rejected ({status})")


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after", "").strip()
    if value.isdigit():
        return min(float(value), MAX_RETRY_AFTER_SECONDS)
    return None  # an HTTP-date form is not worth parsing: the caller backs off on its own


async def _read_capped(response: httpx.Response) -> bytes:
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            raise LLMBadResponse("response too large")
        chunks.append(chunk)
    return b"".join(chunks)


def _parse(body: bytes) -> LLMResponse:
    try:
        data = json.loads(body)
    except ValueError:
        raise LLMBadResponse("response is not JSON") from None
    if not isinstance(data, dict):
        raise LLMBadResponse("unexpected response shape")
    status = data.get("status")
    if status == "incomplete":
        raise LLMBadResponse("output was cut off")
    if status in ("failed", "cancelled"):
        raise LLMTransientError("provider reported a failure")
    if status in ("in_progress", "requires_action"):
        raise LLMBadResponse("response is not final")
    text = _output_text(data.get("steps"))
    if not text:
        raise LLMBadResponse("empty output")
    request_id = data.get("id")
    return LLMResponse(
        text=text,
        usage=_usage(data.get("usage")),
        request_id=request_id if isinstance(request_id, str) else None,
    )


def _output_text(steps: object) -> str:
    """Join the text of every `model_output` step."""
    if not isinstance(steps, list):
        return ""
    parts: list[str] = []
    for step in steps:
        if isinstance(step, dict) and step.get("type") == "model_output":
            for item in step.get("content") or []:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
    return "".join(parts).strip()


def _usage(usage: object) -> LLMUsage:
    if not isinstance(usage, dict):
        return LLMUsage()
    return LLMUsage(
        input_tokens=_count(usage.get("total_input_tokens")),
        output_tokens=_count(usage.get("total_output_tokens")),
        thought_tokens=_count(usage.get("total_thought_tokens")),
    )


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
