"""Typed, validated experiment configuration.

The YAML config is parsed into frozen Pydantic models. Unknown keys and
out-of-range values raise immediately, so a typo can never silently change
an experiment. ``Config.fingerprint()`` gives a short, stable hash of the
fully resolved config, which run reports record for traceability.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Source = Literal["abt", "buy"]
Device = Literal["auto", "cpu", "mps", "cuda"]
Tokenizer = Literal["simple", "code_aware"]


class _Strict(BaseModel):
    """Base model: reject unknown keys and forbid mutation after loading."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(_Strict):
    name: str
    url: str
    raw_dir: Path
    encoding: str = "latin-1"
    query_source: Source
    corpus_source: Source
    query_fields: tuple[str, ...] = Field(min_length=1)
    corpus_fields: tuple[str, ...] = Field(min_length=1)
    max_chars: int = Field(gt=0)

    @model_validator(mode="after")
    def _distinct_sources(self) -> DatasetConfig:
        if self.query_source == self.corpus_source:
            raise ValueError("query_source and corpus_source must differ")
        return self


class SplitConfig(_Strict):
    seed: int
    # 0 reproduces the earlier two-way (val/test) split exactly.
    train_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    val_fraction: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _leaves_a_test_split(self) -> SplitConfig:
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError("train_fraction + val_fraction must leave room for a test split")
        return self


class BM25Config(_Strict):
    k1: float = Field(gt=0.0)
    b: float = Field(ge=0.0, le=1.0)
    tokenizer: Tokenizer = "code_aware"


class DenseConfig(_Strict):
    model: str
    batch_size: int = Field(gt=0)
    normalize: bool = True
    query_prefix: str = ""
    device: Device = "auto"


class HybridConfig(_Strict):
    rrf_k: int = Field(gt=0)
    depth: int = Field(default=100, gt=0)


class RetrievalConfig(_Strict):
    top_k: int = Field(gt=0)
    eval_ks: tuple[int, ...] = Field(min_length=1)
    bm25: BM25Config
    dense: DenseConfig
    hybrid: HybridConfig

    @field_validator("eval_ks")
    @classmethod
    def _ks_positive_increasing(cls, ks: tuple[int, ...]) -> tuple[int, ...]:
        if any(k <= 0 for k in ks):
            raise ValueError("eval_ks must all be positive")
        if list(ks) != sorted(set(ks)):
            raise ValueError("eval_ks must be strictly increasing with no duplicates")
        return ks

    @model_validator(mode="after")
    def _ks_within_top_k(self) -> RetrievalConfig:
        if max(self.eval_ks) > self.top_k:
            raise ValueError(
                f"max(eval_ks)={max(self.eval_ks)} exceeds top_k={self.top_k}; "
                "recall cannot be measured beyond the retrieved candidates"
            )
        if self.hybrid.depth < self.top_k:
            raise ValueError(
                f"hybrid.depth={self.hybrid.depth} is below top_k={self.top_k}; "
                "fusion needs at least top_k candidates from each retriever"
            )
        return self


class MatchingConfig(_Strict):
    retriever: Literal["bm25", "dense", "hybrid"] = "bm25"
    scorer: Literal["cross_encoder", "retriever"] = "cross_encoder"
    candidate_k: int = Field(default=20, gt=0)
    model: str
    batch_size: int = Field(gt=0)
    device: Device = "auto"
    policy: Literal["top1", "threshold"] = "top1"
    threshold_metric: Literal["f1"] = "f1"


class TrainingConfig(_Strict):
    base_model: str
    negatives_per_query: int = Field(default=4, gt=0)
    epochs: int = Field(default=3, gt=0)
    batch_size: int = Field(default=16, gt=0)
    learning_rate: float = Field(default=2e-5, gt=0.0)
    warmup_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    max_length: int = Field(default=128, gt=0)
    seed: int = 42
    device: Device = "auto"
    output_dir: Path = Path("models")


class OutputConfig(_Strict):
    runs_dir: Path


class Config(_Strict):
    dataset: DatasetConfig
    split: SplitConfig
    retrieval: RetrievalConfig
    matching: MatchingConfig
    training: TrainingConfig
    output: OutputConfig

    @model_validator(mode="after")
    def _candidates_within_top_k(self) -> Config:
        if self.matching.candidate_k > self.retrieval.top_k:
            raise ValueError(
                f"matching.candidate_k={self.matching.candidate_k} exceeds "
                f"retrieval.top_k={self.retrieval.top_k}"
            )
        return self

    def fingerprint(self, length: int = 12) -> str:
        """Short, stable SHA-256 hash of the resolved config."""
        return _hash(self.model_dump(mode="json"), length)

    def training_fingerprint(self, length: int = 12) -> str:
        """Hash of only the settings that change a trained model.

        Output folders and evaluation settings are excluded, so the same
        training setup always maps to the same model folder.
        """
        relevant = {
            "dataset": self.dataset.model_dump(mode="json"),
            "split": self.split.model_dump(mode="json"),
            "bm25": self.retrieval.bm25.model_dump(mode="json"),
            "training": self.training.model_dump(mode="json", exclude={"output_dir", "device"}),
        }
        return _hash(relevant, length)


def _hash(obj: Any, length: int) -> str:
    canonical = json.dumps(obj, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def apply_overrides(raw: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply dotted ``section.key=value`` overrides; values are parsed as YAML.

    Example: ``retrieval.bm25.tokenizer=simple`` or ``dataset.query_fields=[name]``.
    Only existing keys can be overridden, so typos fail loudly.
    """
    result = copy.deepcopy(raw)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"override {item!r} must look like section.key=value")
        *parents, leaf = key.split(".")
        node: Any = result
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
            if not isinstance(node, dict):
                raise KeyError(f"override {item!r}: no section {part!r}")
        if leaf not in node:
            raise KeyError(f"override {item!r}: unknown key {leaf!r}")
        node[leaf] = yaml.safe_load(value)
    return result


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> Config:
    """Load a YAML config file, apply any overrides, and validate it."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return Config.model_validate(apply_overrides(raw, overrides))
