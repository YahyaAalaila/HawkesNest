import click
from hawkesnest.simulator.hawkes import HawkesSimulator

@click.group()
def cli():
    """hawkesnest: Spatio‐Temporal event data generator with different levels of complexity based on Hawkes process."""
    pass

@cli.command()
@click.option("--domain",                default="CartoonCity")
@click.option("--background",            default="separable")
@click.option("--kernel",                default="expgau")
@click.option("--graph",                 default="block_modularity")
@click.option("--topo/--no-topo",        default=False)
@click.option("--alpha-ent", type=float, default=0.5)
@click.option("--alpha-het", type=float, default=0.5)
@click.option("--alpha-ker", type=float, default=0.5)
@click.option("--alpha-graph", type=float,default=0.5)
@click.option("--alpha-topo", type=float, default=0.0)
@click.option("--n-events",  type=int,   default=10000)
@click.option("--seed",      type=int,   default=None)
@click.option("--out",       default="events.parquet")
def simulate(domain, background, kernel, graph, topo,
             alpha_ent, alpha_het, alpha_ker, alpha_graph, alpha_topo,
             n_events, seed, out):
    """
    Simulate a dataset at the specified complexity vector and write to OUT.
    """
    gen = HawkesSimulator(
        domain=domain,
        background=background,
        kernel=kernel,
        graph=graph,
        topology=(alpha_topo > 0),
    )
    events, labels = gen.simulate(
        alpha=(alpha_ent, alpha_het, alpha_ker, alpha_graph, alpha_topo),
        n=n_events,
        seed=seed,
    )
    events.to_parquet(out)
    click.echo(f"Wrote {len(events)} events to {out}")
