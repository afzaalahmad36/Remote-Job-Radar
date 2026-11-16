"""Embeddings: what is embedded, how a text becomes a vector's identity, and the interface.

A vector is a list of 384 numbers that places a text in a space where similar texts are close.
This module is pure: it builds the strings that are embedded, hashes them, and combines vectors.
The model itself sits behind `Embedder` (`fastembed_embedder.py` runs the real one on this
machine, `fake_embedder.py` gives deterministic vectors for tests). Nothing here reaches a network,
a language model or an HTTP library (a test checks it): the CV text goes through this code.
"""

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.models.job import EMBEDDING_DIM  # fixed by the `vector(384)` columns and by the model

# bge-small reads at most 512 tokens, roughly 2,000-2,500 characters of English.
JOB_BODY_CHARS = 1_800  # of the post's text that follow the structured fields
CV_CHUNK_CHARS = 2_500
CV_MAX_CHUNKS = 6  # a longer CV is embedded from its first 6 chunks (about 15,000 characters)


class EmbeddingError(Exception):
    """An embedding could not be made. The message is a fixed phrase: it never holds a text."""


class Embedder(Protocol):
    """What every embedding provider offers. One call embeds the texts it is given, one vector
    each, in order. It blocks the CPU, so async callers run it with `asyncio.to_thread`."""

    model_id: str  # "name@revision": stored with every vector it makes
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def text_hash(model_id: str, text: str) -> str:
    """Identity of one embedding input: the same text under the same model gives the same hash,
    so an unchanged job or CV is never embedded twice."""
    return hashlib.sha256(f"{model_id}\n{text}".encode()).hexdigest()


def job_embedding_text(
    *,
    title: str | None,
    company: str | None,
    seniority: str | None,
    remote_policy: str | None,
    stack: Sequence[str],
    raw_text: str,
) -> str:
    """The text that stands for a job: its structured fields first (they are short and exact),
    then the start of the post. Only extracted jobs are embedded, so there is a title."""
    lines = [
        f"Title: {title or ''}",
        f"Company: {company or ''}",
        f"Seniority: {seniority or ''}",
        f"Remote: {remote_policy or ''}",
        f"Stack: {', '.join(stack)}",
    ]
    body = " ".join(raw_text.split())[:JOB_BODY_CHARS]
    return "\n".join([*lines, body])


@dataclass(frozen=True)
class CvChunks:
    chunks: list[str]
    total: int  # how many chunks the whole CV makes; more than len(chunks) when it was cut

    @property
    def truncated(self) -> bool:
        return self.total > len(self.chunks)


def cv_chunks(text: str) -> CvChunks:
    """Cut a CV into pieces of at most `CV_CHUNK_CHARS`, preferably at a line break, and keep the
    first `CV_MAX_CHUNKS`. The model reads about 512 tokens, so one piece is one embedding."""
    pieces: list[str] = []
    rest = text.strip()
    while rest:
        if len(rest) <= CV_CHUNK_CHARS:
            pieces.append(rest)
            break
        cut = rest.rfind("\n", CV_CHUNK_CHARS // 2, CV_CHUNK_CHARS)
        cut = cut if cut > 0 else CV_CHUNK_CHARS
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    return CvChunks(chunks=pieces[:CV_MAX_CHUNKS], total=len(pieces))


def check_vector(vector: Sequence[float]) -> list[float]:
    """The vector as a list of floats, or EmbeddingError if it is not a usable 384-dim vector."""
    if len(vector) != EMBEDDING_DIM:
        raise EmbeddingError(
            f"The embedding has {len(vector)} dimensions, expected {EMBEDDING_DIM}"
        )
    values = [float(x) for x in vector]
    if not all(math.isfinite(x) for x in values):
        raise EmbeddingError("The embedding contains a value that is not a number")
    return values


def normalised_mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    """The average of the vectors, scaled to length 1 (so cosine distance is a dot product).

    One CV is several chunks; their mean is the CV's single vector.
    """
    if not vectors:
        raise EmbeddingError("There is nothing to average")
    total = [0.0] * EMBEDDING_DIM
    for vector in vectors:
        for index, value in enumerate(check_vector(vector)):
            total[index] += value
    length = math.sqrt(sum(x * x for x in total))
    if length == 0.0:
        raise EmbeddingError("The embedding has no direction")
    return [x / length for x in total]
