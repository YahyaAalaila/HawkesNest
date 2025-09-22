<!-- ────────────────────────── Hero ─────────────────────────── -->
<p align="center">
  <!-- replace with an actual logo asset if you have one -->
  <img src="assets/logo.png" alt="logo" width="100"/>
</p>

<h1 align="center">HawkesNest</h1>

<p align="center">
  <b>Composable generators &amp; metrics for <i>synthetic</i> spatio-temporal
     point-process data.</b><br/>
  Stress-test your next STPP model on data whose ground-truth complexity is
  known in advance.
</p>

<!-- Badges -->
<p align="center">
  <img src="https://img.shields.io/pypi/v/hawkesnest?color=blue" alt="PyPI"/>
  <img src="https://img.shields.io/github/last-commit/YahyaAalaila/HawkesNest" alt="last commit"/>
  <img src="https://img.shields.io/github/license/your-org/hawkesnest" alt="license"/>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue?logo=python"/>
</p>

---

<!-- Pillar GIF overview -->
<table align="center">
  <tr>
    <td align="center">
      <img src="assets/gifs/ent_evolution_evolution.gif" width="250"/>
      <br/>
      <strong>Spatial–temporal<br/>heterogeneity&nbsp;(α<sub>het</sub>)</strong>
    </td>
    <td align="center">
      <img src="assets/gifs/hetero_evolution_evolution.gif" width="250"/>
      <br/>
      <strong>Space–time<br/>entanglement&nbsp;(α<sub>ent</sub>)</strong>
    </td>
    <td align="center">
      <img src="assets/gifs/topo_evolution_evolution.gif" width="250"/>
      <br/>
      <strong>Network topology<br/>distortion&nbsp;(α<sub>topo</sub>)</strong>
    </td>
  </tr>
</table>


---
## Why HawkesNest ?

Benchmarks for spatio-temporal point-process (STPP) models often rely on a
single opaque real-world dataset.  
That makes it impossible to know whether a model succeeds because it truly
captures the governing dynamics or because it latches onto idiosyncratic noise.

**HawkesNest flips the script**: we generate event streams from configurable
Hawkes processes whose *latent* patterns are precisely controlled along four
orthogonal *complexity pillars*:

| Pillar | What changes | Typical knob |
| ------ | ------------ | ------------ |
| **Heterogeneity** | non-uniform background rate | spatial clusters, travelling Gaussian |
| **Entanglement**  | space–time coupling | cross-term polynomials, travelling waves |
| **Topology**      | non-Euclidean support | random-geometric graphs, street grids |
| **Interaction Graph** | cross-type triggers | dense vs. sparse adjacency |

Researchers can therefore dial complexity up or down and observe how a new
method copes — from “easy” homogeneous Poisson streams to highly entangled,
network-constrained cascades.

---

## Key Features

* **Lego-style generator** – mix-and-match *domain*, *background*, *kernel* &
  *interaction graph* in a few YAML lines.
* **Ogata thinning simulator** – with both Euclidean and geodesic distances.
* **Complexity meters** – heterogeneity $\alpha_{\text{het}}$, entanglement
  $\alpha_{\text{ent}}$, topology $\alpha_{\text{topo}}$, graph density
  $\alpha_{\text{graph}}$.
* **Preset data classes** – one-liners that instantiate high-complexity samples
  for each pillar.
* **Extensible** – register new background surfaces, kernels, even
  non-Hawkes DGPs or alternative thinning routines.
* **CLI** – `hawkesnest simulate …` makes dataset creation reproducible.

---

## Get started  <!-- still under active development -->

> ⚠️ **Development status:** HawkesNest is a work-in-progress.  
> The API may change without notice and some modules are still experimental.  
> If you hit issues, please open an issue or PR—feedback is welcome!

### Instalation

```bash
git clone https://github.com/your-org/hawkesnest.git
cd hawkesnest
pip install -e .        # Python 3.9+
```

### Generate out-the-box datasets
To generate shipped synthetic spatio-temporal datasets. The hawkesnest console script provides a sub-command simulate-entanglement to generate pre-baked entanglement data at three complexity levels (`low`, `mid`, or `high`).
⚠️ **Development status:** Only entangelement is pre-baked now, future updates (soon), will include the rest of the pillars.

```bash
hawkesnest simulate-entanglement --level <low|mid|high> --n-events <N> --out <path.csv>
```

 - `--level` selects complexity: `low`, `mid`, or `high`.
 - `--n-events` is the number of events to simulate (default: `500`).
 - `--out` is the path to write the resulting CSV (default: `entanglement.csv`).

 ### Example 
 ```bash
 hawkesnest simulate-entanglement --level mid --n-events 1000 --out ent_mid.csv
```

## Direct simulation with flexibility
Other than read-to-use datasets, the user can specify the building blocks for HawkesNest to simulate data with specific settings. 

### YAML‑driven experiments (recommended for large sweeps)

```bash
# Simulation space
domain:
  type: rectangle
  x_min: 0
  x_max: 1000
  y_min: 0
  y_max: 1000

lambda_max: 20            # safety bound (> maximum row‑sum of branching)

# Background intensity (one entry per mark)
backgrounds:
  - {type: function, name: sine, amp: 1.0, freq: 0.5}

# Triggering kernel (same kernel reused for all i→j pairs)
kernels:
  - [{type: separable}]   # uses default branching_ratio=1, sigma=100, decay=1.0

# 1×1 branching matrix
adjacency: [[0.0]]

meta: {n_events: 500}
```
Unspecified `kwargs` fall back to the component defaults—e.g. a separable kernel defaults to `temporal_decay=1.0` and `spatial_sigma=10`

```bash
import yaml
from hawkesnest.simulator.config import SimulatorConfig
from hawkesnest.simulator.hawkes import HawkesSimulator

cfg_yaml = yaml.safe_load(open("experiment.yml"))
cfg       = SimulatorConfig.model_validate(cfg_yaml)

# Build simulator from configuration (each block is therefor initialised and built)
sim    = cfg.build()

# Once simulator is built, generate data with .simulate() method
events_df, parent = sim.simulate(n=cfg.meta.n_events, seed=42)
events_df.to_csv("sine_sep.csv", index=False)
```
## Roadmap / Coming Soon  🚀

- **More pre-baked datasets** – we’ll ship the remaining HawkesNest benchmark suites (heterogeneity, graph structure, topology, …).

- **Richer dataset APIs** – every `*Dataset` class will get convenience helpers such as `.plot()`, `.summary()`, `.gif_kde()`, `.gif_intensity()`, etc.

- **Extensibility guides** – step-by-step walkthroughs that show you how to:
  - integrate **your own simulator** and expose it on the CLI,
  - register custom building blocks (*kernels*, *backgrounds*, *domains*, …) so they work inside YAML configs,
  - implement a **new metric** for an existing complexity pillar – or define an entirely new pillar.

## License

HawkesNest is licensed under the **Apache License 2.0**.  
See the [LICENSE](./LICENSE) file for full text and the NOTICE file for third-party attributions.

