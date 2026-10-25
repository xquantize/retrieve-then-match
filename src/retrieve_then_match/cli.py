"""Command-line interface: `rtm download`, `rtm inspect` (more commands to come)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from retrieve_then_match.config import load_config
from retrieve_then_match.data import dataset_stats, download, load_dataset

app = typer.Typer(no_args_is_help=True, add_completion=False, help="retrieve-then-match")
console = Console()

ConfigOption = Annotated[
    Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML config.")
]
DEFAULT_CONFIG = Path("configs/default.yaml")


@app.command("download")
def download_cmd(
    config: ConfigOption = DEFAULT_CONFIG,
    force: Annotated[bool, typer.Option(help="Re-download even if present.")] = False,
) -> None:
    """Download and extract the dataset."""
    cfg = load_config(config)
    path = download(cfg.dataset, force=force)
    console.print(f"[green]Dataset ready[/green] in {path}")


@app.command("inspect")
def inspect_cmd(
    config: ConfigOption = DEFAULT_CONFIG,
    examples: Annotated[int, typer.Option("--examples", "-n", min=0)] = 3,
) -> None:
    """Print dataset statistics and a few gold matching pairs."""
    cfg = load_config(config)
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


if __name__ == "__main__":
    app()
