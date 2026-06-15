<h1 align="center">HawkesNest</h1>

<p align="center">
  <img src="assets/logo.png" alt="HawkesNest logo" width="220"/>
</p>

HawkesNest is a configurable synthetic benchmark and DGP workbench for spatio-temporal point-process models. It provides controlled Hawkes/STPP generators, benchmark suites, export utilities, and visual diagnostics for studying complexity axes such as entanglement, heterogeneity, topology, and cross-type interaction.

<table align="center">
  <tr>
    <td align="center">
      <img src="assets/gifs/ent_evolution_evolution.gif" width="250"/>
      <br/>
      <strong>Entanglement ladder</strong>
    </td>
    <td align="center">
      <img src="assets/gifs/hetero_evolution_evolution.gif" width="250"/>
      <br/>
      <strong>Heterogeneity ladder</strong>
    </td>
    <td align="center">
      <img src="assets/gifs/topo_evolution_evolution.gif" width="250"/>
      <br/>
      <strong>Topology ladder</strong>
    </td>
  </tr>
</table>

## Why HawkesNest

Real-world event datasets often entangle spatial structure, temporal dynamics, marks, and network effects in ways that are hard to isolate. HawkesNest makes those factors explicit: define the data-generating process, inspect its intensity and event geometry, then generate event streams with known simulator settings.

The benchmark suites expose controlled complexity ladders:

- `EntanglementSuite`: space-time coupling through traveling-wave triggering kernels.
- `HeterogeneitySuite`: background-intensity variation across static, clustered, moving, and time-varying fields.

The same configurable DGP layer supports custom domains, background intensities, triggering kernels, marks, adjacency matrices, simulator settings, export formats, and visualization diagnostics.

## Install

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

## Suite Quickstart

```bash
python -m hawkesnest.cli generate entanglement \
  --level L2 \
  --n-events 50 \
  --seed 123 \
  --out outputs/entanglement_l2
```

```bash
python -m hawkesnest.cli generate heterogeneity \
  --level H3 \
  --n-events 50 \
  --seed 123 \
  --out outputs/heterogeneity_h3
```

See [docs/quickstart.md](docs/quickstart.md) and [docs/benchmark_suites.md](docs/benchmark_suites.md).

## Custom-DGP Quickstart

```python
from hawkesnest.config import SimulatorConfig

config = {
    "domain": {"type": "rectangle", "x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1},
    "backgrounds": [{"type": "constant", "rate": 1.2}],
    "kernels": [{"type": "separable", "temporal_decay": 0.4, "spatial_sigma": 0.12}],
    "adjacency": [[0.20]],
    "lambda_max": 25.0,
}

cfg = SimulatorConfig.model_validate(config)
events, parents = cfg.build().simulate(n=100, seed=7, tau_max=5.0, debug=False)
print(events.head())
```

See [docs/custom_dgp.md](docs/custom_dgp.md).

## Visual Diagnostic Snippet

```python
from hawkesnest.suites import EntanglementSuite
from hawkesnest.viz import plot_kde_panels

results = [
    EntanglementSuite().generate(level=lv, n_events=300, seed=0)
    for lv in ("L0", "L1", "L2", "L3")
]
fig = plot_kde_panels(results, cmap="magma", title="Entanglement suite - KDE spatial density")
fig.savefig("outputs/ent_kde.png", dpi=160, bbox_inches="tight")
```

See [docs/visualization.md](docs/visualization.md).

## Documentation

- [Quickstart](docs/quickstart.md)
- [Custom DGPs](docs/custom_dgp.md)
- [Benchmark suites](docs/benchmark_suites.md)
- [Visualization](docs/visualization.md)
- [Output schema](docs/output_schema.md)
- [Paper reproduction](docs/paper_reproduction.md)

## Citation

Citation information will be added before archival release.

```bibtex
@software{hawkesnest,
  title = {HawkesNest: A Multi-Axis Synthetic Benchmark for Spatiotemporal Pattern Complexity},
  author = {TBD},
  year = {2026},
  note = {Citation details forthcoming}
}
```
