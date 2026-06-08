# Visualization

HawkesNest is designed to be inspected visually. Use these diagnostics to check event clouds, suite-level differences, and background intensity surfaces before treating generated data as benchmark material.

<table align="center">
  <tr>
    <td align="center">
      <img src="../assets/gifs/ent_evolution_evolution.gif" width="250"/>
      <br/>
      <strong>Entanglement ladder</strong>
    </td>
    <td align="center">
      <img src="../assets/gifs/hetero_evolution_evolution.gif" width="250"/>
      <br/>
      <strong>Heterogeneity ladder</strong>
    </td>
    <td align="center">
      <img src="../assets/gifs/topo_evolution_evolution.gif" width="250"/>
      <br/>
      <strong>Topology experiments under validation</strong>
    </td>
  </tr>
</table>

The recommended hierarchy is:

| Priority | View | Function |
|----------|------|----------|
| Primary | Smooth KDE density over space | `plot_kde_panels` |
| Primary | Background intensity surface snapshots | `plot_intensity_snapshots` |
| Secondary | 2D event scatter coloured by time or mark | `plot_events_2d` |
| Secondary | Suite-level comparison grid | `plot_suite_event_grid` |
| Tertiary | 3D (x, y, t) event cloud | `plot_events_3d` |

## Smooth KDE Density

Primary view, with no scatter artefacts:

```python
from hawkesnest.suites import EntanglementSuite
from hawkesnest.viz import plot_kde_panels

results = [
    EntanglementSuite().generate(level=lv, n_events=300, seed=0)
    for lv in ("L0", "L1", "L2", "L3")
]
fig = plot_kde_panels(results, cmap="magma", title="Entanglement suite - KDE spatial density")
fig.savefig("outputs/ent_kde.png", dpi=160, bbox_inches="tight")
```

## Background Intensity Surfaces

```python
from hawkesnest.viz import plot_intensity_snapshots

sim = cfg.build()
fig, axes = plot_intensity_snapshots(
    lambda s, t: sim.background(s, t, 1),
    times=[0.0, 0.5, 1.0],
    cmap="magma",
    title="Background intensity mu(s,t)",
)
fig.savefig("outputs/intensity_snapshots.png", dpi=160)
```

## 2D Event Scatter

```python
from hawkesnest.viz import plot_events_2d

fig, ax = plot_events_2d(events, color_by="t", title="Events coloured by time")
fig.savefig("outputs/events_2d.png", dpi=160)
```

## Suite Comparison Panel

```python
from hawkesnest.viz import plot_suite_event_grid

fig = plot_suite_event_grid(results, color_by="t", title="Suite levels - event scatter")
fig.savefig("outputs/suite_grid.png", dpi=160, bbox_inches="tight")
```

## 3D Event Cloud

```python
from hawkesnest.viz import plot_events_3d

fig, ax = plot_events_3d(events, color_by="m", title="Marked event cloud")
fig.savefig("outputs/events_3d.png", dpi=160)
```

## CLI Visualization

The CLI can also visualize exported recipe outputs:

```bash
python -m hawkesnest.cli visualize \
  outputs/entanglement_demo/events.jsonl \
  --kind space-time \
  --out outputs/entanglement_demo/space_time.png
```

GIF animation support (`plot_kde_gif`) is available in `hawkesnest.utils.viz` and requires `imageio`. The complexity-ladder GIFs above were generated with this utility.
