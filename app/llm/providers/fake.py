"""A provider that never touches a network. Two uses: scripted answers in tests, and
`LLM_PROVIDER=fake` for trying the pipeline locally (it is refused in production)."""

import json
from collections.abc import Sequence

from app.llm.base import LLMError, LLMRequest, LLMResponse, LLMUsage

# A valid, empty extraction: every field "not stated".
DEFAULT_RESPONSE = json.dumps(
    {
        "is_job_posting": True, "title": None, "company": None, "stack": [], "seniority": None,
        "salary_min": None, "salary_max": None, "salary_currency": None, "salary_period": None,
        "remote_policy": None, "apply_url": None,
    }
)  # fmt: skip


class FakeLLMClient:
    """Answers from a script, one item per call: text becomes the response, an LLMError is raised.

    The last item repeats when the script runs out, unless `strict` (then it fails the test).
    Every request is recorded in `requests`.
    """

    name = "fake"

    def __init__(
        self,
        script: Sequence[str | LLMError] = (DEFAULT_RESPONSE,),
        *,
        model: str = "fake-model",
        local: bool = True,
        strict: bool = False,
    ) -> None:
        self.model = model
        self.local = local
        self.requests: list[LLMRequest] = []
        self._script = list(script)
        self._strict = strict

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        index = len(self.requests) - 1
        if index >= len(self._script):
            if self._strict:
                raise AssertionError("FakeLLMClient was called more often than scripted")
            index = len(self._script) - 1
        item = self._script[index]
        if isinstance(item, LLMError):
            raise item
        return LLMResponse(text=item, usage=LLMUsage(input_tokens=10, output_tokens=5))
