"""Retrievers: candidate generation for the matching stage."""

from __future__ import annotations

from retrieve_then_match.config import RetrievalConfig
from retrieve_then_match.retrieval.base import Hit, Retriever
from retrieve_then_match.retrieval.bm25 import BM25Retriever
from retrieve_then_match.retrieval.dense import DenseRetriever
from retrieve_then_match.retrieval.hybrid import HybridRetriever

METHODS = ("bm25", "dense", "hybrid")

__all__ = [
    "METHODS",
    "BM25Retriever",
    "DenseRetriever",
    "Hit",
    "HybridRetriever",
    "Retriever",
    "build_retriever",
]


def build_retriever(method: str, cfg: RetrievalConfig) -> Retriever:
    if method == "bm25":
        return BM25Retriever(cfg.bm25)
    if method == "dense":
        return DenseRetriever(cfg.dense)
    if method == "hybrid":
        return HybridRetriever(
            [BM25Retriever(cfg.bm25), DenseRetriever(cfg.dense)],
            rrf_k=cfg.hybrid.rrf_k,
            depth=cfg.hybrid.depth,
        )
    raise ValueError(f"unknown retrieval method {method!r}; choose from {METHODS}")
