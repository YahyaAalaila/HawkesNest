# Benchmark Suites

HawkesNest includes benchmark suites for controlled complexity axes:

- `EntanglementSuite`: space-time coupling through traveling-wave triggering kernels.
- `HeterogeneitySuite`: background-intensity variation across static, clustered, moving, and time-varying fields.

Each suite exposes named levels and returns event data, simulator metadata, and optional CSV/JSONL/metadata exports. Custom DGPs use the same simulator configuration layer.

## EntanglementSuite

`EntanglementSuite` varies space-time coupling while keeping the surrounding generator configuration fixed.

```python
from hawkesnest.suites import EntanglementSuite

suite = EntanglementSuite()
for level in suite.levels():
    result = suite.generate(level=level, n_events=25, seed=0)
    print(level, result.events.shape)
```

Supported levels: `L0`, `L1`, `L2`, `L3`.

Generate one sequence from the CLI:

```bash
python -m hawkesnest.cli generate entanglement \
  --level L2 \
  --n-events 50 \
  --seed 123 \
  --out outputs/entanglement_l2
```

Generate a small corpus:

```bash
python -m hawkesnest.cli generate-corpus entanglement \
  --levels L0 L1 L2 L3 \
  --seeds 0 1 \
  --n-events 50 \
  --out outputs/entanglement_corpus
```

## HeterogeneitySuite

`HeterogeneitySuite` varies the background intensity field while keeping the triggering kernel fixed.

```python
from hawkesnest.suites import HeterogeneitySuite

suite = HeterogeneitySuite()
for level in suite.levels():
    result = suite.generate(level=level, n_events=25, seed=0)
    print(level, result.events.shape)
```

Supported levels: `H0`, `H1`, `H2`, `H3`.

Generate one sequence from the CLI:

```bash
python -m hawkesnest.cli generate heterogeneity \
  --level H3 \
  --n-events 50 \
  --seed 123 \
  --out outputs/heterogeneity_h3
```

## Topology Axis

Topology appears in the visual ladder and in the 4D complexity sweep documented in [paper_reproduction.md](paper_reproduction.md). The public quickstart focuses on the two suite classes exported by `hawkesnest.suites`: `EntanglementSuite` and `HeterogeneitySuite`.
