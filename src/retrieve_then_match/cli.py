"""Command-line interface: `rtm download`, `rtm inspect`, `rtm retrieve`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.table import Table

from retrieve_then_match.config import Config, load_config
from retrieve_then_match.data import Split, dataset_stats, download, load_dataset
from retrieve_then_match.evaluate import run_retrieval
from retrieve_then_match.retrieval import METHODS, build_retriever

app = typer.Typer(no_args_is_help=True, add_completion=False, help="retrieve-then-match")
console = Console()

DEFAULT_CONFIG = Path("configs/default.yaml")
ConfigOption = Annotated[
    Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML config.")
]
SetOption = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        "-s",
        help="Override a config value, e.g. --set retrieval.bm25.tokenizer=simple (repeatable).",
    ),
]
SPLITS = ("val", "test")


def _load(config: Path, overrides: list[str] | None) -> Config:
    return load_config(config, overrides or [])


@app.command("download")
def download_cmd(
    config: ConfigOption = DEFAULT_CONFIG,
    overrides: SetOption = None,
    force: Annotated[bool, typer.Option(help="Re-download even if present.")] = False,
) -> None:
    """Download and extract the dataset."""
    cfg = _load(config, overrides)
    path = download(cfg.dataset, force=force)
    console.print(f"[green]Dataset ready[/green] in {path}")


@app.command("inspect")
def inspect_cmd(
    config: ConfigOption = DEFAULT_CONFIG,
    overrides: SetOption = None,
    examples: Annotated[int, typer.Option("--examples", "-n", min=0)] = 3,
) -> None:
    """Print dataset statistics and a few gold matching pairs."""
    cfg = _load(config, overrides)
    ds = load_dataset(cfg.dataset, cfg.split)

    stats = Table(title=f"{cfg.dataset.name} (config {cfg.fingerprint()})")
    stats.add_column("statistic")
    stats.add_column("value", justify="right")
    for key, value in dataset_stats(ds).items():
        stats.add_row(key, f"{value:,.1f}" if isinstance(value, float) else f"{value:,}")
    console.print(stats)

    corpus_by_id = {c.id: c for c in ds.corpus}
    shown = 0
    for query in ds.split_queries("val"):
        if shown >= examples:
            break
        if not ds.gold[query.id]:
            continue
        console.rule(f"query {query.id}")
        console.print(f"[bold]{cfg.dataset.query_source}:[/bold] {query.text}")
        for match_id in sorted(ds.gold[query.id]):
            source = cfg.dataset.corpus_source
            console.print(f"[bold]{source}:[/bold] {corpus_by_id[match_id].text}")
        shown += 1


@app.command("retrieve")
def retrieve_cmd(
    method: Annotated[str, typer.Option("--method", "-m", help=f"One of {METHODS}.")] = "bm25",
    split: Annotated[str, typer.Option(help="val or test.")] = "val",
    config: ConfigOption = DEFAULT_CONFIG,
    overrides: SetOption = None,
) -> None:
    """Evaluate a retriever: recall@k, hit@k, MRR and per-query latency."""
    if split not in SPLITS:
        raise typer.BadParameter(f"split must be one of {SPLITS}", param_hint="--split")
    cfg = _load(config, overrides)
    ds = load_dataset(cfg.dataset, cfg.split)
    retriever = build_retriever(method, cfg.retrieval)
    run = run_retrieval(
        retriever, ds, cast(Split, split), cfg.retrieval.top_k, cfg.retrieval.eval_ks
    )

    table = Table(title=f"{run.method} on {run.split} (config {cfg.fingerprint()})")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for name, value in run.metrics.items():
        table.add_row(name, f"{int(value):,}" if name == "evaluated_queries" else f"{value:.3f}")
    for name, value in run.latency_ms.items():
        table.add_row(f"latency {name} (ms)", f"{value:.2f}")
    table.add_row("index time (s)", f"{run.index_seconds:.2f}")
    console.print(table)


if __name__ == "__main__":
    app()
