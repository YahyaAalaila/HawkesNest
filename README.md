# HawkesNest

HawkesNest is a synthetic spatio-temporal point-process corpus generator for controlled Hawkes-process experiments. It is intended for model stress tests where the data-generating process is known, reproducible, and configurable.

The public launch surface is intentionally small. The stable supported suites are:

- `EntanglementSuite`: suite3, a space-time entanglement ladder with levels `L0` through `L3`.
- `HeterogeneitySuite`: suite4, a background heterogeneity ladder with levels `H0` through `H3`.

These suites preserve the working paper-generation path while exposing it through package APIs and CLI commands.

## Relation To Seahorse

Seahorse uses synthetic corpora generated from HawkesNest. HawkesNest should be cited when Seahorse experiments depend on these synthetic suite3 or suite4 corpora.

The public API keeps the same simulator stack used by the accepted paper-generation scripts:

```text
suite -> SimulatorConfig -> HawkesSimulator -> thinning
```

The suite wrappers orchestrate configurations. They do not replace the Hawkes simulator or introduce an ad hoc generator.

## Installation

From a local checkout:

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

## Python Quickstart

```python
from hawkesnest.suites import EntanglementSuite, HeterogeneitySuite

ent = EntanglementSuite().generate(level="L2", n_events=50, seed=123)
print(ent.events.head())
print(ent.metadata["simulator_class"])

het = HeterogeneitySuite().generate(
    level="H3",
    n_events=50,
    seed=123,
    out_dir="outputs/heterogeneity_demo",
)
print(het.export_paths)
```

`generate()` returns a `GenerationResult` with:

- `events`: a pandas DataFrame.
- `config`: the suite-level simulator configuration.
- `metadata`: suite name, level, seed, requested event count, actual event count, and simulator class.
- `simulator_class_name`: expected to be `hawkesnest.simulator.hawkes.HawkesSimulator`.
- `export_paths`: paths written when `out_dir` is provided.

## CLI Quickstart

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

Visualize an exported sequence:

```bash
python -m hawkesnest.cli visualize \
  outputs/entanglement_demo/events.jsonl \
  --kind space-time \
  --out outputs/entanglement_demo/space_time.png
```

## Stable Suites

### EntanglementSuite

`EntanglementSuite` exposes the suite3 entanglement ladder:

```python
from hawkesnest.suites import EntanglementSuite

suite = EntanglementSuite()
for level in suite.levels():
    result = suite.generate(level=level, n_events=25, seed=0)
    print(level, result.events.shape)
```

Supported levels: `L0`, `L1`, `L2`, `L3`.

### HeterogeneitySuite

`HeterogeneitySuite` exposes the suite4 heterogeneity ladder:

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

## Visualization

Use the package visualizer on an exported JSONL file:

```bash
python -m hawkesnest.cli visualize \
  outputs/entanglement_demo/events.jsonl \
  --kind space-time \
  --out outputs/entanglement_demo/plot.png
```

The lightweight notebooks in `notebooks/` show equivalent Python usage.

## Experimental And Legacy Areas

The repository still contains older and experimental code paths. They are not the stable public launch surface yet:

- Branching and cross-event graph logic.
- Topology and non-Euclidean domain experiments.
- Legacy dataset/template loaders under `hawkesnest/datasets/templates`.
- Older scripts used for exploratory sweeps and paper artifact generation.

These are preserved for now, but users should treat `EntanglementSuite` and `HeterogeneitySuite` as the supported public entry points.

## Reproducibility

The stable suites are deterministic for a fixed level, event count, seed, and installed dependency set. The public suite wrappers call the HawkesNest DGP path:

```text
EntanglementSuite / HeterogeneitySuite
-> hawkesnest.config.SimulatorConfig
-> SimulatorConfig.build()
-> hawkesnest.simulator.hawkes.HawkesSimulator
-> hawkesnest.utils.thinning.thinning
```

This path preserves the working suite3 and suite4 paper-generation behavior while making it available through a public API and CLI.

## Examples

Small runnable examples are available in `examples/`:

```bash
python examples/generate_entanglement.py
python examples/generate_heterogeneity.py
```

Both write tiny outputs under `outputs/`, which is ignored by Git.

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

- Keep suite3 entanglement and suite4 heterogeneity stable and reproducible.
- Add public documentation and Colab-ready examples around the supported suites.
- Recover branching and topology only after their working paths are verified.
- Do not treat legacy scripts, generated corpora, sweep outputs, or paper artifacts as part of the public API.
