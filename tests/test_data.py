"""Tests for the data layer, using tiny synthetic Abt-Buy-shaped CSVs."""

from __future__ import annotations

from pathlib import Path

import pytest

from retrieve_then_match.config import DatasetConfig, SplitConfig
from retrieve_then_match.data import build_text, load_dataset, split_query_ids


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="latin-1")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    _write(
        tmp_path / "Abt.csv",
        "id,name,description,price\n"
        "1,Sony TV 40in,Big   screen TV,$500\n"
        "2,Canon Camera,Compact camera,$200\n"
        "3,Bose Speaker,Portable speaker,$150\n"
        "4,No Match Item,Nothing pairs with this,$10\n",
    )
    _write(
        tmp_path / "Buy.csv",
        "id,name,description,manufacturer,price\n"
        "10,Sony 40 inch TV,LCD television,Sony,$499\n"
        "11,Canon PowerShot,Digital camera,Canon,$199\n"
        "12,Canon PowerShot (bundle),Camera with case,Canon,$219\n"
        "13,Bose SoundLink,Bluetooth speaker,Bose,$149\n",
    )
    _write(
        tmp_path / "abt_buy_perfectMapping.csv",
        "idAbt,idBuy\n1,10\n2,11\n2,12\n3,13\n",
    )
    return tmp_path


def _dataset_cfg(raw_dir: Path, max_chars: int = 200) -> DatasetConfig:
    return DatasetConfig(
        name="tiny",
        url="unused",
        raw_dir=raw_dir,
        query_source="abt",
        corpus_source="buy",
        query_fields=("name", "description"),
        corpus_fields=("name", "description"),
        max_chars=max_chars,
    )


def test_build_text_normalises_and_truncates() -> None:
    row = {"name": "  Sony   TV ", "description": "Big\n screen"}
    assert build_text(row, ("name", "description"), 200) == "Sony TV | Big screen"
    assert build_text(row, ("name", "description"), 7) == "Sony TV"
    assert build_text({"name": "X"}, ("name", "missing"), 200) == "X"


def test_load_dataset_gold_and_counts(raw_dir: Path) -> None:
    ds = load_dataset(_dataset_cfg(raw_dir), SplitConfig(seed=0, val_fraction=0.5))
    assert len(ds.queries) == 4
    assert len(ds.corpus) == 4
    assert ds.gold["2"] == frozenset({"11", "12"})  # multi-match query
    assert ds.gold["4"] == frozenset()  # unmatched query is kept
    assert set(ds.val_ids) | set(ds.test_ids) == {"1", "2", "3", "4"}
    assert not set(ds.val_ids) & set(ds.test_ids)


def test_split_is_deterministic_and_seed_sensitive(raw_dir: Path) -> None:
    ds = load_dataset(_dataset_cfg(raw_dir), SplitConfig(seed=0, val_fraction=0.5))
    a = split_query_ids(ds.gold, SplitConfig(seed=0, val_fraction=0.5))
    b = split_query_ids(ds.gold, SplitConfig(seed=0, val_fraction=0.5))
    assert a == b
    seeds = {split_query_ids(ds.gold, SplitConfig(seed=s, val_fraction=0.5)) for s in range(20)}
    assert len(seeds) > 1


def test_unknown_mapping_id_raises(raw_dir: Path) -> None:
    _write(raw_dir / "abt_buy_perfectMapping.csv", "idAbt,idBuy\n1,999\n")
    with pytest.raises(ValueError, match="unknown ids"):
        load_dataset(_dataset_cfg(raw_dir), SplitConfig(seed=0, val_fraction=0.5))


def test_missing_text_field_raises(raw_dir: Path) -> None:
    cfg = _dataset_cfg(raw_dir).model_copy(update={"query_fields": ("name", "manufacturer")})
    with pytest.raises(ValueError, match="manufacturer"):
        load_dataset(cfg, SplitConfig(seed=0, val_fraction=0.5))
