import click
import pandas as pd
import torch
from hawkesnest.datasets.entanglement import EntanglementDataset

@click.group()
def cli():
    """hawkesnest: Spatio‐Temporal event data generator."""
    pass

@cli.command("simulate-entanglement")
@click.option(
    "--level",
    type=click.Choice(["low", "mid", "high"], case_sensitive=False),
    default="low",
    help="Complexity level: low, mid (medium), or high",
)
@click.option(
    "--n-events",
    type=int,
    default=500,
    help="Number of events to generate",
)
@click.option(
    "--out",
    default="entanglement.csv",
    help="Output CSV file path",
)
def simulate_entanglement(level, n_events, out):
    """
    Generate an entanglement dataset at the chosen complexity level,
    with a given number of events, and write to OUT as CSV.
    """
    api_level = level.lower()
    ds = EntanglementDataset(level=api_level, n_events=n_events)

    # report computed entanglement
    comp = ds.complexity("ent")
    click.echo(f"EntanglementDataset(level={api_level!r}, n_events={n_events}) → complexity(ent)={comp:.2f}")

    sample = ds[0]
    # handle Tensor → DataFrame
    if isinstance(sample, torch.Tensor):
        arr = sample.cpu().numpy()
        # name columns generically
        df = pd.DataFrame(arr, columns=["t", "x", "y", "m"])

    # write CSV only
    df.to_csv(out, index=False)
    click.echo(f"Wrote CSV with {len(df)} rows to {out}")
