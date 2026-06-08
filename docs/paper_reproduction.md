# Paper Reproduction

This page documents the public path for reproducing the paper's 4D complexity sweep and index-validation artifacts.

The full 4D sweep is expensive. It was not rerun during the patch that added this reproduction path; only a tiny smoke test was run.

## 4D Complexity Sweep

Generate the paper-aligned `results_4d.csv`:

```bash
python scripts/sweeps_complexity.py \
  --mode full \
  --out results/paper_complexity/results_4d.csv \
  --replicates 5 \
  --T 50 \
  --lambda0 40 \
  --Marks 3 \
  --theta-het-grid 0.0,0.25,0.5,0.75,1.0 \
  --theta-ent-grid 0.0,0.25,0.5,0.75,0.95 \
  --theta-topo-grid 0.0,0.25,0.5,0.75,1.0 \
  --theta-graph-grid 0.0,0.25,0.5,0.75,0.95 \
  --tau-max 0.5 \
  --field-name-bg moving_gauss_slow \
  --ent-option rt \
  --seed-offset 1234 \
  --parallel
```

The CLI writes mode-specific output under:

```text
results/paper_complexity/results_4d_full/results_4d.csv
```

It also writes per-configuration event CSVs next to the result table.

## Monotonicity Plots

Run the factorial analysis script on the completed `results_4d.csv`:

```bash
python scripts/analysis/exp2_factorial_analyze.py \
  --results-csv results/paper_complexity/results_4d_full/results_4d.csv \
  --out-dir results/paper_complexity/analysis_factorial \
  --replicates-col replicate \
  --theta-cols theta_het,theta_ent,theta_topo,theta_graph \
  --baseline 0.0 \
  --controls n_events,lambda_hat_global \
  --do-1d \
  --do-4d
```

Monotonicity checks are written to:

```text
results/paper_complexity/analysis_factorial/analysis_4d/monotonicity_raw.csv
results/paper_complexity/analysis_factorial/analysis_4d/monotonicity_resid.csv
```

The 1D monotonicity-oriented plots are written under:

```text
results/paper_complexity/analysis_factorial/plots_1d/
```

## ANOVA Variance-Decomposition Table

Produce the deterministic balanced-factorial ANOVA decomposition:

```bash
python scripts/analysis/exp2_factorial_anova.py \
  --results-csv results/paper_complexity/results_4d_full/results_4d.csv \
  --out-dir results/paper_complexity/anova \
  --theta-cols theta_het,theta_ent,theta_topo,theta_graph \
  --alpha-cols alpha_het_bg,alpha_ent_mi,alpha_topo,alpha_graph \
  --rep-col replicate
```

The main table is:

```text
results/paper_complexity/anova/anova_decomp.csv
```

Per-effect fractions are written to:

```text
results/paper_complexity/anova/anova_terms_by_effect.csv
```

## Interaction-Mass Table

Produce the paper artifact bundle and interaction-mass table:

```bash
python scripts/analysis/exp_2_paper_artifacts_4d.py \
  --results-csv results/paper_complexity/results_4d_full/results_4d.csv \
  --out-root results/paper_complexity/paper_artifacts \
  --theta-cols theta_het,theta_ent,theta_topo,theta_graph \
  --metric-cols alpha_het_bg,alpha_ent_mi,alpha_topo,alpha_graph \
  --baseline 0.0 \
  --min-rows-interaction 20 \
  --repo-root .
```

The interaction-mass outputs are:

```text
results/paper_complexity/paper_artifacts/figs_4d/tables/interaction_mass.csv
results/paper_complexity/paper_artifacts/figs_4d/tables/interaction_mass.md
```

## Tiny Smoke Test

For operational checks, use a tiny grid instead of the full paper sweep:

```bash
python scripts/sweeps_complexity.py \
  --mode full \
  --out /tmp/hawkesnest_paper_smoke/results_4d.csv \
  --replicates 1 \
  --T 0.5 \
  --lambda0 2 \
  --Marks 2 \
  --theta-het-grid 0.0,0.1 \
  --theta-ent-grid 0.0,0.1 \
  --theta-topo-grid 0.0,0.1 \
  --theta-graph-grid 0.0,0.1 \
  --tau-max 0.5 \
  --field-name-bg moving_gauss_slow \
  --ent-option rt \
  --seed-offset 1234 \
  --progress-every 1 \
  --fail-fast
```

Then run the analysis smoke checks:

```bash
python scripts/analysis/exp2_factorial_analyze.py \
  --results-csv /tmp/hawkesnest_paper_smoke/results_4d_full/results_4d.csv \
  --out-dir /tmp/hawkesnest_paper_smoke/analysis_factorial \
  --replicates-col replicate \
  --theta-cols theta_het,theta_ent,theta_topo,theta_graph \
  --baseline 0.0 \
  --controls n_events,lambda_hat_global \
  --do-1d \
  --do-4d \
  --cv-splits 2
```

```bash
python scripts/analysis/exp2_factorial_anova.py \
  --results-csv /tmp/hawkesnest_paper_smoke/results_4d_full/results_4d.csv \
  --out-dir /tmp/hawkesnest_paper_smoke/anova \
  --theta-cols theta_het,theta_ent,theta_topo,theta_graph \
  --alpha-cols alpha_het_bg,alpha_ent_mi,alpha_topo,alpha_graph \
  --rep-col replicate
```

```bash
python scripts/analysis/exp_2_paper_artifacts_4d.py \
  --results-csv /tmp/hawkesnest_paper_smoke/results_4d_full/results_4d.csv \
  --out-root /tmp/hawkesnest_paper_smoke/paper_artifacts \
  --theta-cols theta_het,theta_ent,theta_topo,theta_graph \
  --metric-cols alpha_het_bg,alpha_ent_mi,alpha_topo,alpha_graph \
  --baseline 0.0 \
  --min-rows-interaction 4 \
  --repo-root .
```
