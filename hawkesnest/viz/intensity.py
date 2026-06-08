"""Intensity-surface visualization helpers."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _evaluate_on_grid(
    intensity_fn: Callable[[np.ndarray, float], float],
    *,
    t: float,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    values = np.empty((len(ys), len(xs)), dtype=float)
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            values[row, col] = float(intensity_fn(np.array([x, y], dtype=float), float(t)))
    return values


def plot_intensity_snapshots(
    intensity_fn: Callable[[np.ndarray, float], float],
    *,
    times: Sequence[float] = (0.0, 1.0, 2.0),
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0)),
    n_grid: int = 60,
    cmap: str = "magma",
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot heatmap snapshots for a callable intensity surface."""
    if n_grid <= 1:
        raise ValueError("n_grid must be greater than 1")

    (x_min, x_max), (y_min, y_max) = bounds
    xs = np.linspace(x_min, x_max, n_grid)
    ys = np.linspace(y_min, y_max, n_grid)
    grids = [_evaluate_on_grid(intensity_fn, t=float(t), xs=xs, ys=ys) for t in times]

    if vmin is None:
        vmin = min(float(np.nanmin(grid)) for grid in grids)
    if vmax is None:
        vmax = max(float(np.nanmax(grid)) for grid in grids)

    fig, axes = plt.subplots(1, len(times), figsize=(3.6 * len(times), 3.4), squeeze=False)
    flat_axes = axes.ravel()
    image = None
    for ax, t, grid in zip(flat_axes, times, grids):
        image = ax.imshow(
            grid,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        ax.set_title(f"t={float(t):.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    if title:
        fig.suptitle(title)
    if image is not None:
        fig.colorbar(image, ax=flat_axes.tolist(), shrink=0.85, label="intensity")
    return fig, flat_axes
