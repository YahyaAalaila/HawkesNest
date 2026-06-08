# Recipes

HawkesNest ships validated benchmark recipes for users who want reproducible stress-test data immediately:

- `EntanglementSuite`: the ready-made suite3 recipe for space-time entanglement stress tests.
- `HeterogeneitySuite`: the ready-made suite4 recipe for background heterogeneity stress tests.

The suites are important, but they are not the boundary of HawkesNest. They are the fastest path when you do not want to design your own DGP yet.

## EntanglementSuite

`EntanglementSuite` is the validated suite3 recipe for space-time entanglement stress tests.

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
  --out outputs/entanglement_demo
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

`HeterogeneitySuite` is the validated suite4 recipe for background heterogeneity stress tests.

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
  --out outputs/heterogeneity_demo
```

## Topology Recipes

Topology-oriented recipes and GIFs are experimental and under validation. They are useful for exploration, but they are not yet part of the stable validated public recipe layer.
