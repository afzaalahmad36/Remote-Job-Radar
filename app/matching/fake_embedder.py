"""A deterministic embedder for tests and development: no model, no files, no network.

Each word switches a few fixed dimensions on or off (chosen by hashing the word), and the sum is
scaled to length 1. So the same text always gives the same vector, and two texts that share words
are closer than two that do not, which is enough to test ranking later. It is refused in
production (`EMBEDDING_PROVIDER=fake`, see config).
"""

import hashlib
import math
import re
from collections.abc import Sequence

from app.models.job import EMBEDDING_DIM

_WORD = re.compile(r"[a-z0-9+#.]+")
_DIMS_PER_WORD = 6


class FakeEmbedder:
    dimension = EMBEDDING_DIM

    def __init__(self, model_id: str = "fake-hash@v1") -> None:
        self.model_id = model_id
        self.calls: list[list[str]] = []  # every batch of texts it was asked to embed

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * EMBEDDING_DIM
        for word in _WORD.findall(text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            for step in range(_DIMS_PER_WORD):
                index = int.from_bytes(digest[step * 2 : step * 2 + 2], "big") % EMBEDDING_DIM
                values[index] += 1.0 if digest[16 + step] % 2 == 0 else -1.0
        length = math.sqrt(sum(x * x for x in values))
        if length == 0.0:  # no words: a fixed unit vector, so the result is still valid
            values[0], length = 1.0, 1.0
        return [x / length for x in values]
