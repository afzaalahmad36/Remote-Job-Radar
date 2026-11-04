"""The labels file written by `eval_label.py` and read by `try_extraction.py --eval`.

It lives under `.local/eval/` (git-ignored) and holds the labeller's own words about public job
posts: no post text, only job ids, the hash of the text that was labelled, and the labels.
"""

import json
import os
import re
import uuid
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.models import RemotePolicy, SalaryPeriod, Seniority
from scripts.db_guard import ScriptError
from scripts.eval_scoring import FieldValues

EVAL_DIR = Path(__file__).resolve().parent.parent / ".local" / "eval"
DEFAULT_LABELS_PATH = EVAL_DIR / "labels.json"
RUN_DIR = EVAL_DIR / "runs"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CURRENCY = re.compile(r"[A-Z]{3}")


class Labels(BaseModel):
    """What one post says. None / empty means "the post does not say"."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    company: str | None = None
    stack: list[str] = []
    remote_policy: RemotePolicy | None = None
    seniority: Seniority | None = None
    salary_min: int | None = Field(default=None, ge=0)
    salary_max: int | None = Field(default=None, ge=0)
    salary_currency: str | None = None
    salary_period: SalaryPeriod | None = None

    @field_validator("salary_currency")
    @classmethod
    def _currency(cls, value: str | None) -> str | None:
        if value is not None and not _CURRENCY.fullmatch(value):
            raise ValueError("must be a three-letter upper-case code")
        return value

    def values(self) -> FieldValues:
        return FieldValues(
            title=self.title,
            company=self.company,
            stack=tuple(self.stack),
            remote_policy=self.remote_policy.value if self.remote_policy else None,
            seniority=self.seniority.value if self.seniority else None,
            salary_min=self.salary_min,
            salary_max=self.salary_max,
            salary_currency=self.salary_currency,
            salary_period=self.salary_period.value if self.salary_period else None,
        )


class LabelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: uuid.UUID
    raw_text_hash: str  # of the text that was shown; a different hash later means "stale"
    status: Literal["labelled", "skipped"]
    labels: Labels | None = None

    @field_validator("raw_text_hash")
    @classmethod
    def _hash(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("must be a lower-case SHA-256 hex digest")
        return value

    @model_validator(mode="after")
    def _labels_match_status(self) -> Self:
        if (self.status == "labelled") != (self.labels is not None):
            raise ValueError("labels are present exactly when the status is 'labelled'")
        return self


class LabelFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    source_id: uuid.UUID
    seed: int
    selected: list[uuid.UUID]  # the whole selection, in the order it is shown
    posts: list[LabelEntry] = []

    @model_validator(mode="after")
    def _posts_come_from_the_selection(self) -> Self:
        ids = [entry.job_id for entry in self.posts]
        if len(set(ids)) != len(ids):
            raise ValueError("a post is labelled twice")
        if not set(ids) <= set(self.selected):
            raise ValueError("a labelled post is not in the selection")
        return self

    def pending(self) -> list[uuid.UUID]:
        """Selected posts that have neither a label nor a skip yet, in order."""
        done = {entry.job_id for entry in self.posts}
        return [job_id for job_id in self.selected if job_id not in done]


def load_label_file(path: Path) -> LabelFile:
    try:
        return LabelFile.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ScriptError(f"cannot read {path}: {type(error).__name__}") from None
    except ValidationError as error:
        problems = sorted({".".join(str(p) for p in e["loc"]) or "(file)" for e in error.errors()})
        raise ScriptError(f"{path} does not match the format: {', '.join(problems[:6])}") from None


def save_label_file(path: Path, labels: LabelFile) -> None:
    """Write the whole file, then swap it in, so an interruption never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(labels.model_dump(mode="json"), indent=2, ensure_ascii=False)
    temporary.write_text(payload + "\n", encoding="utf-8")
    os.replace(temporary, path)
