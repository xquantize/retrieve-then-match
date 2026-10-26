"""Fine-tune a cross-encoder for entity matching.

Training data comes only from the train split. For each train query:
  - every gold match is a positive pair (label 1);
  - the highest-ranked *wrong* BM25 candidates are hard negatives (label 0).
Hard negatives are near-identical products (same brand, similar model
numbers): exactly the cases the zero-shot model gets wrong.

The model is trained with binary cross-entropy in a plain PyTorch loop and
saved as a standard Hugging Face sequence-classification checkpoint, which
`sentence_transformers.CrossEncoder` loads directly for `rtm match`.

torch/transformers are imported lazily, so the rest of the package (and its
tests) does not need them.
"""

from __future__ import annotations

import importlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from retrieve_then_match.config import Config
from retrieve_then_match.data import Dataset
from retrieve_then_match.report import environment
from retrieve_then_match.retrieval import BM25Retriever

META_FILE = "training_meta.json"


@dataclass(frozen=True)
class TrainingPair:
    query: str
    candidate: str
    label: float


def build_training_pairs(ds: Dataset, cfg: Config) -> list[TrainingPair]:
    """Positives plus BM25 hard negatives, from train-split queries only."""
    n_neg = cfg.training.negatives_per_query
    retriever = BM25Retriever(cfg.retrieval.bm25)
    retriever.index(ds.corpus)
    corpus_text = {c.id: c.text for c in ds.corpus}

    pairs: list[TrainingPair] = []
    for query in ds.split_queries("train"):
        gold = ds.gold[query.id]
        pairs.extend(TrainingPair(query.text, corpus_text[c], 1.0) for c in sorted(gold))
        hits = retriever.search(query, n_neg + len(gold))
        negatives = [h.id for h in hits if h.id not in gold][:n_neg]
        pairs.extend(TrainingPair(query.text, corpus_text[c], 0.0) for c in negatives)
    return pairs


def default_model_dir(cfg: Config) -> Path:
    base = cfg.training.base_model.rsplit("/", 1)[-1]
    return cfg.training.output_dir / f"{cfg.dataset.name}-{base}-ft-{cfg.training_fingerprint()}"


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _pick_device(requested: str, torch: Any) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_cross_encoder(pairs: list[TrainingPair], cfg: Config, out_dir: Path) -> dict[str, Any]:
    """Train, save the checkpoint and metadata to out_dir, return the metadata."""
    if not pairs:
        raise ValueError("no training pairs; is train_fraction > 0?")
    torch: Any = importlib.import_module("torch")
    transformers: Any = importlib.import_module("transformers")
    t = cfg.training
    _seed_everything(t.seed, torch)
    device = _pick_device(t.device, torch)

    tokenizer = transformers.AutoTokenizer.from_pretrained(t.base_model)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        t.base_model, num_labels=1
    )
    model.to(device)
    model.train()

    steps_per_epoch = math.ceil(len(pairs) / t.batch_size)
    total_steps = steps_per_epoch * t.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.learning_rate, weight_decay=0.01)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, int(total_steps * t.warmup_ratio), total_steps
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(t.seed)

    history: list[float] = []
    start = time.perf_counter()
    for _ in range(t.epochs):
        order = rng.permutation(len(pairs))
        epoch_loss = 0.0
        for offset in range(0, len(pairs), t.batch_size):
            batch = [pairs[int(i)] for i in order[offset : offset + t.batch_size]]
            encoded = tokenizer(
                [p.query for p in batch],
                [p.candidate for p in batch],
                padding=True,
                truncation=True,
                max_length=t.max_length,
                return_tensors="pt",
            ).to(device)
            labels = torch.tensor([p.label for p in batch], dtype=torch.float32, device=device)
            logits = model(**encoded).logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.item()) * len(batch)
        history.append(epoch_loss / len(pairs))
    seconds = time.perf_counter() - start

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    positives = sum(1 for p in pairs if p.label == 1.0)
    meta: dict[str, Any] = {
        "base_model": t.base_model,
        "training_fingerprint": cfg.training_fingerprint(),
        "config_fingerprint": cfg.fingerprint(),
        "train_queries": len({p.query for p in pairs}),
        "pairs": {"total": len(pairs), "positive": positives, "negative": len(pairs) - positives},
        "epoch_losses": history,
        "train_seconds": seconds,
        "device": device,
        "training_config": t.model_dump(mode="json"),
        "environment": environment(),
    }
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta
