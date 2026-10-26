"""Training tests.

Pair construction is tested everywhere. The training loop is tested only
where torch and transformers are installed, using a tiny randomly initialised
BERT built on the fly, so no model download or network access is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from retrieve_then_match.config import Config, load_config
from retrieve_then_match.data import Dataset, Record
from retrieve_then_match.train import (
    META_FILE,
    TrainingPair,
    build_training_pairs,
    default_model_dir,
    train_cross_encoder,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _dataset() -> Dataset:
    queries = (
        Record("t1", "sony mdrj10 headphones"),
        Record("t2", "bose 161 speakers"),
        Record("v1", "canon sd1000 camera"),
    )
    corpus = (
        Record("c1", "sony mdrj10 headphones blue"),
        Record("c2", "sony mdrj20 headphones black"),  # hard negative for t1
        Record("c3", "bose 161 speakers white"),
        Record("c4", "bose 301 speakers"),  # hard negative for t2
        Record("c5", "canon powershot sd1000"),
    )
    gold = {"t1": frozenset({"c1"}), "t2": frozenset({"c3"}), "v1": frozenset({"c5"})}
    return Dataset(queries, corpus, gold, train_ids=("t1", "t2"), val_ids=("v1",), test_ids=())


def _cfg(*overrides: str) -> Config:
    return load_config(DEFAULT_CONFIG, ["training.negatives_per_query=2", *overrides])


def test_pairs_use_train_queries_only_with_hard_negatives() -> None:
    pairs = build_training_pairs(_dataset(), _cfg())
    queries = {p.query for p in pairs}
    assert queries == {"sony mdrj10 headphones", "bose 161 speakers"}  # no val query
    positives = [p for p in pairs if p.label == 1.0]
    negatives = [p for p in pairs if p.label == 0.0]
    assert {p.candidate for p in positives} == {
        "sony mdrj10 headphones blue",
        "bose 161 speakers white",
    }
    assert "sony mdrj20 headphones black" in {p.candidate for p in negatives}
    assert all(
        (p.query, p.candidate) not in {(q.query, q.candidate) for q in positives} for p in negatives
    )
    per_query = {q: sum(1 for p in negatives if p.query == q) for q in queries}
    assert all(n <= 2 for n in per_query.values())


def test_model_dir_ignores_non_training_settings() -> None:
    a = _cfg()
    b = _cfg("output.runs_dir=elsewhere", "matching.candidate_k=5", "training.device=cpu")
    c = _cfg("training.epochs=5")
    assert default_model_dir(a) == default_model_dir(b)
    assert default_model_dir(a) != default_model_dir(c)


def test_training_requires_pairs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        train_cross_encoder([], _cfg(), tmp_path / "model")


def _tiny_model(path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    words = [
        "sony",
        "mdrj10",
        "mdrj20",
        "headphones",
        "blue",
        "black",
        "bose",
        "161",
        "301",
        "speakers",
        "white",
    ]
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]
    path.mkdir(parents=True)
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(path / "vocab.txt"))
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        num_labels=1,
    )
    transformers.BertForSequenceClassification(config).save_pretrained(path)
    tokenizer.save_pretrained(path)


def test_training_loop_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    base = tmp_path / "tiny-base"
    _tiny_model(base)
    cfg = _cfg(
        f"training.base_model={base}",
        "training.epochs=2",
        "training.batch_size=2",
        "training.max_length=32",
        "training.device=cpu",
    )
    pairs = [
        TrainingPair("sony mdrj10 headphones", "sony mdrj10 headphones blue", 1.0),
        TrainingPair("sony mdrj10 headphones", "sony mdrj20 headphones black", 0.0),
        TrainingPair("bose 161 speakers", "bose 161 speakers white", 1.0),
        TrainingPair("bose 161 speakers", "bose 301 speakers", 0.0),
    ]
    out = tmp_path / "trained"
    meta = train_cross_encoder(pairs, cfg, out)
    assert len(meta["epoch_losses"]) == 2
    assert meta["pairs"] == {"total": 4, "positive": 2, "negative": 2}
    assert (out / META_FILE).is_file()

    # The saved checkpoint must load as a CrossEncoder, exactly as `rtm match` uses it.
    st = pytest.importorskip("sentence_transformers")
    scores = st.CrossEncoder(str(out), device="cpu").predict([["sony mdrj10", "sony mdrj10 blue"]])
    assert len(scores) == 1
