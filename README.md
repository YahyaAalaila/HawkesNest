# HawkesNest

<p align="center">
  <img src="assets/logo.png" alt="HawkesNest logo" width="220"/>
</p>

HawkesNest is a configurable synthetic STPP / Hawkes-process DGP workbench. It lets you build synthetic spatio-temporal event streams by composing domains, background intensity families, triggering kernels, mark structure, adjacency matrices, simulator settings, export utilities, and visualization diagnostics.

HawkesNest also ships validated benchmark recipes for users who want reproducible stress-test data immediately:

- `EntanglementSuite`: the ready-made suite3 recipe for space-time entanglement stress tests.
- `HeterogeneitySuite`: the ready-made suite4 recipe for background heterogeneity stress tests.

The suites are important, but they are not the boundary of HawkesNest. They are the fastest path when you do not want to design your own DGP yet.

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
      <strong>Topology experiments</strong>
    </td>
  </tr>
</table>

## Why HawkesNest

Benchmarks for spatio-temporal point-process models often rely on opaque real-world datasets. That makes it hard to know whether a model succeeds because it learned the governing dynamics or because it latched onto dataset-specific noise.

HawkesNest flips that workflow: you can paint the data-generating process directly, inspect it visually, and then generate event streams from a known simulator path.

Core configurable DGP concepts include:

- spatial domains, currently including rectangular Euclidean domains;
- background intensity families, including constant rates, spatial clusters, moving hotspots, and time-varying surfaces;
- triggering kernels, including separable kernels and traveling-wave kernels;
- spatial/temporal entanglement;
- mark and adjacency structure;
- Ogata thinning simulation;
- CSV/JSONL/metadata export;
- event-cloud and intensity-surface visualization.

## Two Ways To Use HawkesNest

### A. Recipe Mode

Use validated recipes when you want reproducible benchmark data immediately.

```python
from hawkesnest.suites import EntanglementSuite, HeterogeneitySuite

ent = EntanglementSuite().generate(level="L2", n_events=50, seed=123)
het = HeterogeneitySuite().generate(level="H3", n_events=50, seed=123)

print(ent.events.head())
print(het.metadata["simulator_class"])
```

Recipe mode is the recommended path for Seahorse reproduction and paper-aligned stress tests.

### B. Custom DGP Mode

Compose your own process when you want to design a synthetic world directly.

```python
from hawkesnest.config import SimulatorConfig

config = {
    "domain": {
        "type": "rectangle",
        "x_min": 0.0,
        "x_max": 1.0,
        "y_min": 0.0,
        "y_max": 1.0,
    },
    "backgrounds": [
        {
            "type": "function",
            "name": "cluster_mix",
            "centers": [[0.25, 0.25], [0.75, 0.65]],
            "sigma": 0.12,
            "a0": 0.2,
            "amp": 1.5,
        }
    ],
    "kernels": [
        {
            "type": "separable",
            "temporal_decay": 0.4,
            "spatial_sigma": 0.12,
        }
    ],
    "adjacency": [[0.20]],
    "lambda_max": 25.0,
}

cfg = SimulatorConfig.model_validate(config)
simulator = cfg.build()
events, parents = simulator.simulate(n=100, seed=7, tau_max=5.0, debug=False)
print(events.head())
```

That call path is the core HawkesNest DGP layer:

```text
SimulatorConfig -> SimulatorConfig.build() -> HawkesSimulator -> thinning
```

## Installation

From a local checkout:

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

## Recipe Quickstart

Generate one entanglement sequence:

```bash
python -m hawkesnest.cli generate entanglement \
  --level L2 \
  --n-events 50 \
  --seed 123 \
  --out outputs/entanglement_demo
```

Generate one heterogeneity sequence:

```bash
python -m hawkesnest.cli generate heterogeneity \
  --level H3 \
  --n-events 50 \
  --seed 123 \
  --out outputs/heterogeneity_demo
```

Generate a small corpus:

```bash
python -m hawkesnest.cli generate-corpus entanglement \
  --levels L0 L1 L2 L3 \
  --seeds 0 1 \
  --n-events 50 \
  --out outputs/entanglement_corpus
```

## Custom DGP Example With Marks

Marks and triggering structure are controlled through `backgrounds`, `kernels`, and `adjacency`.

```python
from hawkesnest.config import SimulatorConfig

marked_config = {
    "domain": {"type": "rectangle", "x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1},
    "backgrounds": [
        {"type": "constant", "rate": 1.2},
        {
            "type": "function",
            "name": "moving_hotspots",
            "start": [0.45, 0.45],
            "v": [0.002, 0.001],
            "sigma": 0.08,
            "a0": 0.4,
            "amp": 0.8,
        },
    ],
    "kernels": [{"type": "separable", "temporal_decay": 0.35, "spatial_sigma": 0.10}],
    "adjacency": [[0.10, 0.05], [0.25, 0.10]],
    "lambda_max": 35.0,
}

cfg = SimulatorConfig.model_validate(marked_config)
events, parents = cfg.build().simulate(n=120, seed=11, tau_max=5.0, debug=False)
print(events["m"].value_counts())
```

The single kernel config broadcasts across the two-by-two mark interaction grid. The adjacency matrix controls how strongly each parent mark triggers each child mark.

## Visual Diagnostics

HawkesNest is meant to be inspected visually, not only exported as CSV. Useful diagnostics include event clouds, intensity surfaces, temporal evolution, complexity ladders, and optional animations.

Plot a 2D event cloud:

```python
from hawkesnest.viz import plot_events_2d

fig, ax = plot_events_2d(events, color_by="t", title="Events colored by time")
fig.savefig("outputs/custom_events_2d.png", dpi=160)
```

Plot a 3D `(x, y, t)` cloud:

```python
from hawkesnest.viz import plot_events_3d

fig, ax = plot_events_3d(events, color_by="m", title="Marked event cloud")
fig.savefig("outputs/custom_events_3d.png", dpi=160)
```

Plot intensity heatmaps at multiple timestamps:

```python
from hawkesnest.viz import plot_intensity_snapshots

background = cfg.backgrounds[0].build()
fig, axes = plot_intensity_snapshots(
    lambda s, t: background(s, t),
    times=[0.0, 1.0, 2.0],
    title="Background intensity over time",
)
fig.savefig("outputs/intensity_snapshots.png", dpi=160)
```

The CLI can also visualize exported recipe outputs:

```bash
python -m hawkesnest.cli visualize \
  outputs/entanglement_demo/events.jsonl \
  --kind space-time \
  --out outputs/entanglement_demo/space_time.png
```

GIF or animation workflows can be built on top of the same intensity snapshot functions. HawkesNest does not force extra animation dependencies for the lightweight install.

## Stable Validated Recipes

### EntanglementSuite

`EntanglementSuite` is the validated suite3 recipe for space-time entanglement stress tests.

```python
from hawkesnest.suites import EntanglementSuite

suite = EntanglementSuite()
for level in suite.levels():
    result = suite.generate(level=level, n_events=25, seed=0)
    print(level, result.events.shape)
```

Supported levels: `L0`, `L1`, `L2`, `L3`.

### HeterogeneitySuite

`HeterogeneitySuite` is the validated suite4 recipe for background heterogeneity stress tests.

```python
from hawkesnest.suites import HeterogeneitySuite

suite = HeterogeneitySuite()
for level in suite.levels():
    result = suite.generate(level=level, n_events=25, seed=0)
    print(level, result.events.shape)
```

Supported levels: `H0`, `H1`, `H2`, `H3`.

## Output Schema

CSV exports use one row per event:

```text
t,x,y,m,is_triggered
```

- `t`: event time.
- `x`, `y`: spatial coordinates.
- `m`: mark label.
- `is_triggered`: whether the accepted event had nonzero triggering intensity.

JSONL exports currently store one sequence per line:

```json
{"times": [0.1, 0.3], "locations": [[0.2, 0.4], [0.8, 0.1]]}
```

Metadata exports include suite, level, seed, event counts, tau window, simulator class, configuration, and export paths.

## Relation To Seahorse

Seahorse uses synthetic corpora generated from HawkesNest. HawkesNest should be cited when Seahorse experiments depend on these synthetic suite3 or suite4 corpora.

The validated recipes preserve the working paper-generation path:

```text
EntanglementSuite / HeterogeneitySuite
-> hawkesnest.config.SimulatorConfig
-> SimulatorConfig.build()
-> hawkesnest.simulator.hawkes.HawkesSimulator
-> hawkesnest.utils.thinning.thinning
```

## Project Areas

### Core Configurable DGP Layer

This is the heart of HawkesNest: domains, backgrounds, kernels, marks, adjacency structure, simulator config, thinning, export, and visualization.

### Validated Recipes

These are reproducible presets built on top of the configurable DGP layer:

- `EntanglementSuite`
- `HeterogeneitySuite`

### Legacy Or Experimental

These are preserved but not yet the stable public recipe layer:

- old scripts used for exploratory sweeps and paper artifact generation;
- generated corpora and transfer artifacts;
- legacy dataset/template loader paths;
- branching and topology recipes that still need validation.

Non-suite generation is not legacy. It is the core DGP layer.

## Examples And Notebooks

Small runnable examples:

```bash
python examples/generate_entanglement.py
python examples/generate_heterogeneity.py
```

Notebook demos:

- `notebooks/01_quickstart.ipynb`
- `notebooks/02_entanglement_suite.ipynb`
- `notebooks/03_heterogeneity_suite.ipynb`
- `notebooks/04_custom_dgp_design.ipynb`

All generated files should be written under `outputs/`, which is ignored by Git.

## Citation

Citation information will be added before archival release.

```bibtex
@software{hawkesnest,
  title = {HawkesNest: Synthetic Hawkes-Process Corpora for Spatio-Temporal Point-Process Evaluation},
  author = {TBD},
  year = {2026},
  note = {Citation details forthcoming}
}
```

## Roadmap And Limitations

- Keep the core configurable DGP path stable and documented.
- Keep suite3 entanglement and suite4 heterogeneity reproducible as validated recipes.
- Expand visual diagnostics around intensity surfaces and temporal evolution.
- Recover branching and topology recipes only after their working paths are verified.
- Keep generated corpora, sweep outputs, submissions, and paper artifacts out of normal commits.
