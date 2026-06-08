# Development Notes

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

Topology-oriented material is experimental and under validation. Non-suite generation is not legacy. It is the core DGP layer.

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

## Roadmap And Limitations

- Keep the core configurable DGP path stable and documented.
- Keep suite3 entanglement and suite4 heterogeneity reproducible as validated recipes.
- Expand visual diagnostics around intensity surfaces and temporal evolution.
- Recover branching and topology recipes only after their working paths are verified.
- Keep generated corpora, sweep outputs, submissions, and paper artifacts out of normal commits.
