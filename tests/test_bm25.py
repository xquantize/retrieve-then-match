"""BM25 and tokeniser tests."""

from __future__ import annotations

import pytest

from retrieve_then_match.config import BM25Config
from retrieve_then_match.data import Record
from retrieve_then_match.retrieval.bm25 import (
    BM25Retriever,
    code_aware_tokenize,
    simple_tokenize,
)

CORPUS = [
    Record("b1", "Bose 161 Speaker | Magnetically Shielded"),
    Record("b2", "Sony Stereo Earphones - MDRJ10/BLUE"),
    Record("b3", "Panasonic SDYD250 Automatic Bread Maker"),
]


def _bm25(tokenizer: str) -> BM25Retriever:
    retriever = BM25Retriever(BM25Config(k1=1.5, b=0.75, tokenizer=tokenizer))  # type: ignore[arg-type]
    retriever.index(CORPUS)
    return retriever


def test_simple_tokenize() -> None:
    assert simple_tokenize("MDR-J10/BLUE 161WH") == ["mdr", "j10", "blue", "161wh"]


def test_code_aware_joins_and_splits_codes() -> None:
    hyphenated = code_aware_tokenize("Sony MDR-J10")
    slashed = code_aware_tokenize("MDRJ10/BLUE")
    assert "mdrj10" in hyphenated and "mdrj10" in slashed
    assert {"161wh", "161", "wh"} <= set(code_aware_tokenize("161WH"))


@pytest.mark.parametrize("tokenizer", ["simple", "code_aware"])
def test_exact_code_ranks_first(tokenizer: str) -> None:
    hits = _bm25(tokenizer).search(Record("q", "Panasonic Breadmaker SDYD250"), k=3)
    assert hits[0].id == "b3"


def test_code_aware_matches_code_variants_simple_misses() -> None:
    query = Record("q", "161WH")
    assert _bm25("simple").search(query, k=3) == []  # no shared token
    assert _bm25("code_aware").search(query, k=3)[0].id == "b1"


def test_scores_descending_and_k_capped() -> None:
    hits = _bm25("code_aware").search(Record("q", "Sony Bose speaker earphones"), k=10)
    assert len(hits) <= len(CORPUS)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_before_index_raises() -> None:
    retriever = BM25Retriever(BM25Config(k1=1.5, b=0.75))
    with pytest.raises(RuntimeError):
        retriever.search(Record("q", "x"), k=1)
