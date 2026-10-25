"""Dense and hybrid retrieval tests, using a deterministic fake encoder (no model download)."""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np
import pytest

from retrieve_then_match.config import BM25Config, DenseConfig
from retrieve_then_match.data import Record
from retrieve_then_match.retrieval import BM25Retriever, DenseRetriever, Hit, HybridRetriever
from retrieve_then_match.retrieval.dense import Embeddings
from retrieve_then_match.retrieval.hybrid import reciprocal_rank_fusion

DENSE_CFG = DenseConfig(model="fake/char-trigrams", batch_size=8)
CORPUS = [
    Record("b1", "Bose 161 Speaker"),
    Record("b2", "Sony Stereo Earphones MDRJ10"),
    Record("b3", "Panasonic Automatic Bread Maker"),
]


class TrigramEncoder:
    """Hashed character-trigram vectors: typo-tolerant, like a (very) small dense model."""

    def __init__(self, dims: int = 512) -> None:
        self.dims = dims
        self.calls: list[int] = []

    def encode(self, texts: Sequence[str]) -> Embeddings:
        self.calls.append(len(texts))
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            padded = f"  {text.lower()} "
            for i in range(len(padded) - 2):
                out[row, zlib.crc32(padded[i : i + 3].encode()) % self.dims] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def _dense(encoder: TrigramEncoder | None = None) -> DenseRetriever:
    retriever = DenseRetriever(DENSE_CFG, encoder=encoder or TrigramEncoder())
    retriever.index(CORPUS)
    return retriever


def test_dense_tolerates_typos_bm25_misses() -> None:
    query = Record("q", "Breadmaker")  # no exact token overlap with "Bread Maker"
    bm25 = BM25Retriever(BM25Config(k1=1.5, b=0.75))
    bm25.index(CORPUS)
    assert bm25.search(query, k=3) == []
    assert _dense().search(query, k=3)[0].id == "b3"


def test_dense_encodes_corpus_once_and_one_query_per_search() -> None:
    encoder = TrigramEncoder()
    retriever = _dense(encoder)
    retriever.search(Record("q", "Bose speaker"), k=2)
    assert encoder.calls == [len(CORPUS), 1]


def test_dense_returns_all_docs_ranked() -> None:
    hits = _dense().search(Record("q", "Bose 161"), k=10)
    assert hits[0].id == "b1"
    assert len(hits) == len(CORPUS)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_dense_name_and_search_before_index() -> None:
    retriever = DenseRetriever(DENSE_CFG, encoder=TrigramEncoder())
    assert retriever.name == "dense-char-trigrams"
    with pytest.raises(RuntimeError):
        retriever.search(Record("q", "x"), k=1)


def test_rrf_matches_hand_computation() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], rrf_k=1)
    # b: 1/3 + 1/2, c: 1/4 + 1/3, a: 1/2, d: 1/4
    assert [h.id for h in fused] == ["b", "c", "a", "d"]
    assert fused[0].score == pytest.approx(1 / 3 + 1 / 2)


def test_rrf_ties_are_deterministic() -> None:
    fused = reciprocal_rank_fusion([["x", "y"], ["y", "x"]], rrf_k=60)
    assert [h.id for h in fused] == ["x", "y"]  # equal scores: first seen wins


class _Fixed:
    def __init__(self, name: str, ids: list[str]) -> None:
        self.name = name
        self._ids = ids
        self.indexed = False

    def index(self, corpus: Sequence[Record]) -> None:
        self.indexed = True

    def search(self, query: Record, k: int) -> list[Hit]:
        return [Hit(d, 1.0) for d in self._ids[:k]]


def test_hybrid_fuses_and_indexes_all_components() -> None:
    a, b = _Fixed("a", ["d1", "d2", "d3"]), _Fixed("b", ["d3", "d1", "d4"])
    hybrid = HybridRetriever([a, b], rrf_k=60, depth=10)
    hybrid.index(CORPUS)
    assert a.indexed and b.indexed
    assert hybrid.name == "hybrid-rrf"
    assert [h.id for h in hybrid.search(Record("q", "x"), k=2)] == ["d1", "d3"]


def test_hybrid_needs_two_retrievers() -> None:
    with pytest.raises(ValueError):
        HybridRetriever([_Fixed("a", [])], rrf_k=60, depth=10)
