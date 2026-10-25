"""Retrievers: candidate generation for the matching stage."""

from __future__ import annotations

from retrieve_then_match.config import RetrievalConfig
from retrieve_then_match.retrieval.base import Hit, Retriever
from retrieve_then_match.retrieval.bm25 import BM25Retriever

METHODS = ("bm25",)

__all__ = ["METHODS", "BM25Retriever", "Hit", "Retriever", "build_retriever"]


def build_retriever(method: str, cfg: RetrievalConfig) -> Retriever:
    if method == "bm25":
        return BM25Retriever(cfg.bm25)
    raise ValueError(f"unknown retrieval method {method!r}; choose from {METHODS}")
