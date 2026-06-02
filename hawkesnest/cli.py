"""Command line interface for HawkesNest."""
from __future__ import annotations

from pathlib import Path

import click

from hawkesnest.suites import EntanglementSuite, HeterogeneitySuite


SUITES = {
    "entanglement": EntanglementSuite,
    "heterogeneity": HeterogeneitySuite,
}


def _suite(name: str):
    try:
        return SUITES[name]()
    except KeyError as exc:
        raise click.ClickException(f"Unknown suite {name!r}; expected one of {sorted(SUITES)}") from exc


@click.group()
def cli():
    """hawkesnest: spatio-temporal Hawkes-process corpus generator."""


@cli.command("generate")
@click.argument("suite_name", type=click.Choice(sorted(SUITES)))
@click.option("--level", required=True, help="Suite level, e.g. L2 or H3.")
@click.option("--n-events", type=int, default=500, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--debug", is_flag=True)
def generate(suite_name: str, level: str, n_events: int, seed: int, out: Path, debug: bool):
    """Generate one event sequence."""
    suite = _suite(suite_name)
    result = suite.generate(level=level, n_events=n_events, seed=seed, debug=debug, out_dir=out)
    click.echo(
        f"Generated {result.n_events} events for {result.suite_name}/{result.level} "
        f"seed={result.seed} with {result.simulator_class_name}"
    )
    click.echo(f"Wrote {out}")


def _parse_corpus_tokens(tokens: tuple[str, ...]) -> dict:
    parsed: dict[str, object] = {
        "levels": [],
        "seeds": [],
        "n_events": 500,
        "out": None,
        "debug": False,
    }
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "--levels":
            i += 1
            while i < len(tokens) and not tokens[i].startswith("--"):
                parsed["levels"].append(tokens[i])  # type: ignore[index, union-attr]
                i += 1
            continue
        if token == "--seeds":
            i += 1
            while i < len(tokens) and not tokens[i].startswith("--"):
                parsed["seeds"].append(int(tokens[i]))  # type: ignore[index, union-attr]
                i += 1
            continue
        if token == "--n-events":
            parsed["n_events"] = int(tokens[i + 1])
            i += 2
            continue
        if token == "--out":
            parsed["out"] = Path(tokens[i + 1])
            i += 2
            continue
        if token == "--debug":
            parsed["debug"] = True
            i += 1
            continue
        raise click.ClickException(f"Unexpected argument {token!r}")

    if not parsed["levels"]:
        raise click.ClickException("Missing --levels")
    if not parsed["seeds"]:
        raise click.ClickException("Missing --seeds")
    if parsed["out"] is None:
        raise click.ClickException("Missing --out")
    return parsed


@cli.command(
    "generate-corpus",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("suite_name", type=click.Choice(sorted(SUITES)))
@click.argument("tokens", nargs=-1, type=click.UNPROCESSED)
def generate_corpus(suite_name: str, tokens: tuple[str, ...]):
    """Generate a small corpus for one suite."""
    parsed = _parse_corpus_tokens(tokens)
    suite = _suite(suite_name)
    results = suite.generate_corpus(
        levels=parsed["levels"],  # type: ignore[arg-type]
        seeds=parsed["seeds"],  # type: ignore[arg-type]
        n_events=parsed["n_events"],  # type: ignore[arg-type]
        out_dir=parsed["out"],  # type: ignore[arg-type]
        debug=bool(parsed["debug"]),
    )
    click.echo(f"Generated {len(results)} sequences under {parsed['out']}")


@cli.command("visualize")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--kind", default="space-time", show_default=True)
@click.option("--out", type=click.Path(path_type=Path), default=None)
@click.option("--show", is_flag=True)
def visualize(path: Path, kind: str, out: Path | None, show: bool):
    """Visualize an exported event file."""
    from hawkesnest.viz import visualize_events_file

    out_path = visualize_events_file(path, kind=kind, out=out, show=show)
    click.echo(f"Wrote {out_path}")


@cli.command("simulate-entanglement")
@click.option("--level", type=click.Choice(["low", "mid", "high", "L0", "L1", "L2", "L3"]), default="L0")
@click.option("--n-events", type=int, default=500, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), default=Path("entanglement.csv"), show_default=True)
def simulate_entanglement(level: str, n_events: int, seed: int, out: Path):
    """Backward-compatible entanglement command writing a CSV file."""
    level_map = {"low": "L0", "mid": "L1", "high": "L3"}
    public_level = level_map.get(level.lower(), level)
    result = EntanglementSuite().generate(level=public_level, n_events=n_events, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.events.to_csv(out, index=False)
    click.echo(f"Wrote CSV with {result.n_events} rows to {out}")


if __name__ == "__main__":
    cli()
