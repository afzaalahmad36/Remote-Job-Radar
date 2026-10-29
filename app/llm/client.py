"""The only door to a language-model provider.

Code elsewhere builds a client with `build_client(settings)` and calls `complete()`; it never
imports an adapter (a test enforces that). Adding a provider = one adapter file in
`app/llm/providers/` plus one entry in `_BUILDERS` below.
"""

import logging
import time
from collections.abc import Callable

import httpx

from app.core.config import Settings
from app.llm.base import (
    HOSTED_ALLOWED_PURPOSES,
    LLMClient,
    LLMConfigError,
    LLMError,
    LLMPrivacyError,
    LLMRateLimited,
    LLMRequest,
    LLMResponse,
)
from app.llm.providers.fake import FakeLLMClient
from app.llm.providers.gemini import GeminiClient

logger = logging.getLogger(__name__)

MAX_REQUEST_CHARS = 40_000  # system + user text: far above what extraction sends


class GuardedClient:
    """Wraps any provider client with the rules that apply to all of them.

    * privacy: a hosted provider only receives purposes in HOSTED_ALLOWED_PURPOSES and never
      anything marked as personal data;
    * a size limit on what is sent;
    * logging that never contains request or response text: only provider, model, token counts,
      timing and the outcome.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.name = inner.name
        self.model = inner.model
        self.local = inner.local

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self._check_allowed(request)
        started = time.perf_counter()
        try:
            response = await self._inner.complete(request)
        except LLMRateLimited as error:
            self._log("rate_limited", started, error=error)
            raise
        except LLMError as error:
            self._log("error", started, error=error)
            raise
        usage = response.usage
        self._log(
            "ok",
            started,
            tokens=(usage.input_tokens, usage.output_tokens, usage.thought_tokens),
            request_id=response.request_id,
        )
        return response

    def _check_allowed(self, request: LLMRequest) -> None:
        if not self.local and (
            request.contains_personal_data or request.purpose not in HOSTED_ALLOWED_PURPOSES
        ):
            raise LLMPrivacyError("this data may not be sent to a hosted provider")
        if len(request.system) + len(request.user) > MAX_REQUEST_CHARS:
            raise LLMConfigError("request too large")

    def _log(
        self,
        outcome: str,
        started: float,
        *,
        error: LLMError | None = None,
        tokens: tuple[int | None, int | None, int | None] | None = None,
        request_id: str | None = None,
    ) -> None:
        logger.info(
            "llm request provider=%s model=%s outcome=%s reason=%s tokens(in,out,thought)=%s "
            "ms=%d request_id=%s",
            self.name,
            self.model,
            outcome,
            error.reason if error else "-",
            tokens,
            (time.perf_counter() - started) * 1000,
            request_id,
        )


Builder = Callable[[Settings, httpx.AsyncBaseTransport | None], LLMClient]


def _build_gemini(settings: Settings, transport: httpx.AsyncBaseTransport | None) -> LLMClient:
    return GeminiClient(
        api_key=settings.gemini_api_key,
        model=settings.llm_model_extraction,
        base_url=settings.gemini_base_url,
        timeout_seconds=settings.llm_request_timeout_seconds,
        max_output_tokens=settings.llm_max_output_tokens,
        thinking_level=settings.gemini_thinking_level,
        transport=transport,
    )


def _build_fake(settings: Settings, transport: httpx.AsyncBaseTransport | None) -> LLMClient:
    return FakeLLMClient(model=settings.llm_model_extraction or "fake-model")


def _build_ollama(settings: Settings, transport: httpx.AsyncBaseTransport | None) -> LLMClient:
    # The interface is ready for it (`local=True` clients may receive personal data); the adapter
    # itself is added when needed.
    raise LLMConfigError("the ollama provider is not implemented yet")


_BUILDERS: dict[str, Builder] = {
    "gemini": _build_gemini,
    "fake": _build_fake,
    "ollama": _build_ollama,
}


def build_client(
    settings: Settings, transport: httpx.AsyncBaseTransport | None = None
) -> LLMClient | None:
    """The configured client, or None when no model may be asked (rules-only mode).

    None means: LLM_PROVIDER is "none", or the kill switch LLM_EXTRACTION_ENABLED is off.
    `transport` lets tests route the provider's HTTP calls to an in-memory server.
    """
    if settings.llm_provider == "none" or not settings.llm_extraction_enabled:
        return None
    return GuardedClient(_BUILDERS[settings.llm_provider](settings, transport))
