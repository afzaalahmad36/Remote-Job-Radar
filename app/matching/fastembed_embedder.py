"""The real embedder: fastembed (onnxruntime) running the pinned model from local files.

This is the ONLY module that imports fastembed (a test checks it), and only the `worker-embed`
service ever imports this module: the API and the main worker never load the model. Importing
fastembed also imports Pillow, requests and huggingface_hub, so the runtime is kept off the
network three ways: the weights are read from a folder (never downloaded), the image sets
HF_HUB_OFFLINE=1, and `worker-embed` sits on a Docker network with no route out.
"""

import os
from collections.abc import Sequence
from pathlib import Path

import onnxruntime
from fastembed import TextEmbedding

from app.core.config import Settings
from app.matching.embeddings import EmbeddingError, check_vector
from app.models.job import EMBEDDING_DIM

BATCH_SIZE = 1  # measured: one text at a time costs no speed on one thread and the least memory
THREADS = 1


def _snapshot_dir(settings: Settings) -> Path:
    """The folder of the pinned revision inside the cache (Hugging Face's own layout)."""
    description = TextEmbedding._get_model_description(settings.embedding_model)
    repo = description.sources.hf
    if not repo:
        raise EmbeddingError("The embedding model has no Hugging Face source")
    folder = Path(settings.embedding_cache_dir) / f"models--{repo.replace('/', '--')}"
    return folder / "snapshots" / settings.embedding_model_revision


class FastEmbedder:
    dimension = EMBEDDING_DIM

    def __init__(self, settings: Settings) -> None:
        # Offline by construction, whatever the environment says.
        os.environ["HF_HUB_OFFLINE"] = "1"
        onnxruntime.disable_telemetry_events()
        snapshot = _snapshot_dir(settings)
        if not snapshot.is_dir():
            raise EmbeddingError(
                "The embedding model files are missing: build the image with "
                "EMBEDDING_MODEL_DOWNLOAD=1"
            )
        self.model_id = settings.embedding_model_id
        self._model = TextEmbedding(
            settings.embedding_model,
            cache_dir=settings.embedding_cache_dir,
            specific_model_path=str(snapshot),
            threads=THREADS,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.embed(list(texts), batch_size=BATCH_SIZE)
        return [check_vector(vector.tolist()) for vector in vectors]
