from dataclasses import dataclass
from enum import StrEnum


class Confidence(StrEnum):
    HIGH = "high"  # explicit evidence in the text: a labelled field, an unambiguous phrase
    LOW = "low"  # a plausible reading that another source may improve


@dataclass(frozen=True)
class Found[T]:
    """A value the rules extracted, and how sure they are of it."""

    value: T
    confidence: Confidence

    @property
    def is_high(self) -> bool:
        return self.confidence is Confidence.HIGH
