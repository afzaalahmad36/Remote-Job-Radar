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
    CONNECTIVITY_CHECK = "connectivity_check"  # a fixed harmless sentence, never stored data


# Hosted providers (a free tier may use what it is sent to improve products, and people may read
# it) accept only these. Anything touching a CV or other personal data is never listed here: it
# may only go to a provider running on your own machine.
HOSTED_ALLOWED_PURPOSES = frozenset({Purpose.JOB_POST_EXTRACTION, Purpose.CONNECTIVITY_CHECK})


# The only text the connectivity check may carry. GuardedClient compares against these exactly, so
# job text can never travel under that purpose (which is the one allowed to leave `store` unset).
PING_INPUT = "Reply with the single word: pong"
PING_SYSTEM = "You are a connectivity check. Answer with the single word the user asks for."


@dataclass(frozen=True)
class LLMRequest:
    purpose: Purpose
    system: str  # "" sends no system instruction
    user: str
    json_schema: dict[str, Any] | None  # None asks for plain text instead of JSON
    contains_personal_data: bool = False
    # False asks the provider not to keep the interaction. None leaves the provider's default and
    # is only allowed for the connectivity check (see GuardedClient).
    store: bool | None = False


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


def is_fixed_ping(request: LLMRequest) -> bool:
    """True for exactly the fixed ping: its input, no system text or its own, and no schema."""
    return (
        request.user == PING_INPUT
        and request.system in ("", PING_SYSTEM)
        and request.json_schema is None
        and not request.contains_personal_data
    )


class LLMClient(Protocol):
    """What every provider adapter provides. One call = exactly one provider request."""

    name: str  # "gemini", "fake", ...
    model: str
    local: bool  # True when the provider runs on this machine (nothing leaves it)

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


class LLMError(Exception):
    """Base of every LLM failure. `reason` is a short fixed phrase: safe to log and to store."""

    def __init__(self, reason: str, *, never_sent: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        # True when the request was refused inside this process, before anything was sent. The
        # caller uses it to tell "the provider may have counted this" from "it never left".
        self.never_sent = never_sent


class LLMTransientError(LLMError):
    """Worth trying again later: timeout, connection failure, 5xx."""


class LLMRateLimited(LLMError):
    """HTTP 429: the provider's quota is used up. Never retried at once."""

    def __init__(self, retry_after_seconds: float | None = None) -> None:
        super().__init__("rate limited")
        self.retry_after_seconds = retry_after_seconds


class LLMConfigError(LLMError):
    """Retrying cannot help: bad key, unknown model, rejected request, unsupported provider."""


class LLMRejected(LLMConfigError):
    """The provider answered with an HTTP 4xx status (other than 429). The status is a number from
    the protocol, not provider text, so it is safe to log and to branch on."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"request rejected ({status_code})")
        self.status_code = status_code


class LLMBadResponse(LLMError):
    """The provider answered, but not with usable output (empty, truncated, malformed)."""


class LLMPrivacyError(LLMError):
    """The request would send data to a provider that is not allowed to receive it."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason, never_sent=True)  # always refused locally
