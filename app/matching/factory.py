"""Builds the configured embedder. The real one is imported only here, only when asked for, so a
process that never calls this (the API, the main worker) never loads fastembed or onnxruntime."""

from app.core.config import Settings
from app.matching.embeddings import Embedder


def build_embedder(settings: Settings) -> Embedder | None:
    """The embedder for `EMBEDDING_PROVIDER`, or None when it is `none` (nothing is embedded)."""
    if settings.embedding_provider == "fake":
        from app.matching.fake_embedder import FakeEmbedder

        return FakeEmbedder(settings.embedding_model_id)
    if settings.embedding_provider == "fastembed":
        from app.matching.fastembed_embedder import FastEmbedder

        return FastEmbedder(settings)
    return None
