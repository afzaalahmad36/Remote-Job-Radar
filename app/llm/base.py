"""The provider-neutral vocabulary of the LLM layer: requests, responses, errors, the interface.

Nothing here talks to a provider. Adapters (app/llm/providers/) implement `LLMClient`; callers
only ever see this module and `app/llm/client.py`.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class Purpose(StrEnum):
    """Why a request is made. Each purpose says what kind of data its text may contain."""

    JOB_POST_EXTRACTION = "job_post_extraction"  # public job post text, contact details removed


# Hosted providers (a free tier may use what it is sent to improve products, and people may read
# it) accept only these. Anything touching a CV or other personal data is never listed here: it
# may only go to a provider running on your own machine.
HOSTED_ALLOWED_PURPOSES = frozenset({Purpose.JOB_POST_EXTRACTION})


@dataclass(frozen=True)
class LLMRequest:
    purpose: Purpose
    system: str
    user: str
    json_schema: dict[str, Any]
    contains_personal_data: bool = False


@dataclass(frozen=True)
class LLMUsage:
    """Token counts, when the provider reports them (None when it does not)."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    thought_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    text: str  # the model's JSON text; untrusted until validated
    usage: LLMUsage = field(default_factory=LLMUsage)
    request_id: str | None = None


class LLMClient(Protocol):
    """What every provider adapter provides. One call = exactly one provider request."""

    name: str  # "gemini", "fake", ...
    model: str
    local: bool  # True when the provider runs on this machine (nothing leaves it)

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


class LLMError(Exception):
    """Base of every LLM failure. `reason` is a short fixed phrase: safe to log and to store."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class LLMTransientError(LLMError):
    """Worth trying again later: timeout, connection failure, 5xx."""


class LLMRateLimited(LLMError):
    """HTTP 429: the provider's quota is used up. Never retried at once."""

    def __init__(self, retry_after_seconds: float | None = None) -> None:
        super().__init__("rate limited")
        self.retry_after_seconds = retry_after_seconds


class LLMConfigError(LLMError):
    """Retrying cannot help: bad key, unknown model, rejected request, unsupported provider."""


class LLMBadResponse(LLMError):
    """The provider answered, but not with usable output (empty, truncated, malformed)."""


class LLMPrivacyError(LLMError):
    """The request would send data to a provider that is not allowed to receive it."""
