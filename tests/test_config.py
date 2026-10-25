"""Tests for config loading and validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from retrieve_then_match.config import Config, apply_overrides, load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def raw() -> dict[str, Any]:
    with DEFAULT_CONFIG.open(encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


def test_default_config_loads() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.dataset.name == "abt-buy"
    assert max(cfg.retrieval.eval_ks) <= cfg.retrieval.top_k


def test_unknown_key_rejected(raw: dict[str, Any]) -> None:
    raw["retrieval"]["topk"] = 10  # typo for top_k
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_eval_k_beyond_top_k_rejected(raw: dict[str, Any]) -> None:
    raw["retrieval"]["eval_ks"] = [1, 5, 50]
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_same_query_and_corpus_source_rejected(raw: dict[str, Any]) -> None:
    raw["dataset"]["corpus_source"] = raw["dataset"]["query_source"]
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_fingerprint_is_stable_and_sensitive(raw: dict[str, Any]) -> None:
    a = Config.model_validate(raw)
    b = Config.model_validate(copy.deepcopy(raw))
    assert a.fingerprint() == b.fingerprint()

    raw["split"]["seed"] = raw["split"]["seed"] + 1
    c = Config.model_validate(raw)
    assert a.fingerprint() != c.fingerprint()


def test_overrides_apply_and_parse_yaml_values() -> None:
    cfg = load_config(
        DEFAULT_CONFIG,
        ["retrieval.bm25.tokenizer=simple", "dataset.query_fields=[name]", "split.seed=7"],
    )
    assert cfg.retrieval.bm25.tokenizer == "simple"
    assert cfg.dataset.query_fields == ("name",)
    assert cfg.split.seed == 7
    assert cfg.fingerprint() != load_config(DEFAULT_CONFIG).fingerprint()


@pytest.mark.parametrize(
    "override",
    ["retrieval.bm25.tokeniser=simple", "retreival.top_k=5", "no_equals_sign"],
)
def test_bad_overrides_rejected(raw: dict[str, Any], override: str) -> None:
    with pytest.raises((KeyError, ValueError)):
        apply_overrides(raw, [override])
