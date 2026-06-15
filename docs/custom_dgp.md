# Custom DGPs

HawkesNest is a configurable STPP/Hawkes DGP workbench. The benchmark suites provide standard generator configurations, while the same DGP layer can be composed directly for custom experiments.

The core configurable DGP concepts include:

- spatial domains, currently including rectangular Euclidean domains;
- background intensity families, including constant rates, spatial clusters, moving hotspots, and time-varying surfaces;
- triggering kernels, including separable kernels and traveling-wave kernels;
- spatial/temporal entanglement;
- mark and adjacency structure;
- Ogata thinning simulation;
- CSV/JSONL/metadata export;
- event-cloud and intensity-surface visualization.

## Minimal Custom Process

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

The main call path is:

```text
SimulatorConfig -> SimulatorConfig.build() -> HawkesSimulator -> thinning
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
