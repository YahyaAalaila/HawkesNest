<!-- ────────────────────────── Hero ─────────────────────────── -->
<p align="center">
  <!-- replace with an actual logo asset if you have one -->
  <img src="assets/logo.png" alt="logo" width="150"/>
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
  <img src="https://img.shields.io/github/last-commit/your-org/hawkesnest" alt="last commit"/>
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

## Installation  <!-- still under active development -->

> ⚠️ **Development status:** HawkesNest is a work-in-progress.  
> The API may change without notice and some modules are still experimental.  
> If you hit issues, please open an issue or PR—feedback is welcome!

```bash
git clone https://github.com/your-org/hawkesnest.git
cd hawkesnest
pip install -e .        # Python 3.9+

