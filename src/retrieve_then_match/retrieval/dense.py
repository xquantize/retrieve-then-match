"""Dense (bi-encoder) retrieval with exact cosine search.

Corpus embeddings are computed once at index time. Each query is embedded on
its own at search time, so the measured latency is what a live service would
see. Search is exact (a matrix-vector product): at ~1k records an ANN index
would add complexity without saving meaningful time.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from retrieve_then_match.config import DenseConfig
from retrieve_then_match.data import Record
from retrieve_then_match.retrieval.base import Hit, top_k

Embeddings = npt.NDArray[np.float32]


class Encoder(Protocol):
    """Anything that turns texts into L2-normalised embedding rows."""

    def encode(self, texts: Sequence[str]) -> Embeddings: ...


class SentenceTransformerEncoder:
    """Thin wrapper so the retriever does not depend on the library directly."""

    def __init__(self, cfg: DenseConfig) -> None:
        from sentence_transformers import SentenceTransformer

        # device=None lets sentence-transformers pick CUDA, then MPS, then CPU.
        device = None if cfg.device == "auto" else cfg.device
        self._model: Any = SentenceTransformer(cfg.model, device=device)
        self._batch_size = cfg.batch_size
        self._normalize = cfg.normalize

    def encode(self, texts: Sequence[str]) -> Embeddings:
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class DenseRetriever:
    def __init__(self, cfg: DenseConfig, encoder: Encoder | None = None) -> None:
        self.name = f"dense-{cfg.model.rsplit('/', 1)[-1]}"
        self._query_prefix = cfg.query_prefix
        # Model loading happens here, so it is not counted as index time.
        self._encoder: Encoder = encoder or SentenceTransformerEncoder(cfg)
        self._ids: list[str] = []
        self._matrix: Embeddings = np.zeros((0, 0), dtype=np.float32)

    def index(self, corpus: Sequence[Record]) -> None:
        self._ids = [r.id for r in corpus]
        self._matrix = self._encoder.encode([r.text for r in corpus])

    def search(self, query: Record, k: int) -> list[Hit]:
        if not self._ids:
            raise RuntimeError("DenseRetriever.search called before index()")
        vector = self._encoder.encode([self._query_prefix + query.text])[0]
        scores = (self._matrix @ vector).astype(np.float64)
        return top_k(scores, self._ids, k)
