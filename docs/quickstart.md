# Quickstart

HawkesNest has two main entry points:

- recipe mode for validated benchmark data you can generate immediately;
- custom-DGP mode for composing your own synthetic spatio-temporal point process.

The validated fast-path recipes are `EntanglementSuite` and `HeterogeneitySuite`. They are useful for reproducible stress tests, but they are not the boundary of HawkesNest. The configurable DGP layer is the core tool.

## Install

From a local checkout:

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

## Recipe Mode

Use validated recipes when you want reproducible benchmark data immediately.

```python
from hawkesnest.suites import EntanglementSuite, HeterogeneitySuite

ent = EntanglementSuite().generate(level="L2", n_events=50, seed=123)
het = HeterogeneitySuite().generate(level="H3", n_events=50, seed=123)

print(ent.events.head())
print(het.metadata["simulator_class"])
```

Recipe mode is the recommended path for Seahorse reproduction and paper-aligned stress tests.

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

## Custom DGP Mode

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

## More

- [Custom DGPs](custom_dgp.md)
- [Recipes](recipes.md)
- [Visualization](visualization.md)
- [Output schema](output_schema.md)
