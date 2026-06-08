"""Suite-level diagnostic visualisation helpers."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from hawkesnest.viz.events import plot_events_2d


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def plot_suite_event_grid(
    results,
    *,
    color_by: str = "t",
    cmap: str = "viridis",
    title: str | None = None,
    out: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Spatial event scatter for each suite level in a single row.

    Parameters
    ----------
    results:
        Iterable of ``GenerationResult`` objects, one per level.
    color_by:
        Event DataFrame column used for point colour (``"t"``, ``"m"``…).
    cmap:
        Matplotlib colormap name.
    title:
        Optional figure super-title.
    out:
        If given, save the figure to this path.
    show:
        Call ``plt.show()`` after rendering.
    """
    results = list(results)
    n = len(results)
    if n == 0:
        raise ValueError("results is empty")

    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.6), squeeze=False)
    for ax, result in zip(axes[0], results):
        plot_events_2d(result.events, ax=ax, color_by=color_by, cmap=cmap)
        ax.set_title(f"{result.level}  (n={result.n_events})", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()

    if out is not None:
        from pathlib import Path
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_kde_panels(
    results,
    *,
    domain: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    grid_size: int = 80,
    levels: int = 14,
    cmap: str = "magma",
    title: str | None = None,
    out: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Smooth KDE density panel for each suite level in a single row.

    This is the *primary* diagnostic view: it shows the smooth spatial
    density of event locations without discretisation artefacts, making
    background structure and self-excitation clustering clearly visible.

    Parameters
    ----------
    results:
        Iterable of ``GenerationResult`` objects, one per level.
    domain:
        ``(x_min, x_max, y_min, y_max)`` bounding box for the KDE grid.
    grid_size:
        Resolution of the evaluation grid (``grid_size × grid_size``).
    levels:
        Number of contour fill levels.
    cmap:
        Matplotlib colormap name.  ``"magma"`` works well for intensity.
    title:
        Optional figure super-title.
    out:
        If given, save the figure to this path.
    show:
        Call ``plt.show()`` after rendering.
    """
    results = list(results)
    n = len(results)
    if n == 0:
        raise ValueError("results is empty")

    x_min, x_max, y_min, y_max = domain
    X, Y = np.mgrid[x_min:x_max:grid_size * 1j, y_min:y_max:grid_size * 1j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), squeeze=False)
    for ax, result in zip(axes[0], results):
        df = result.events
        xs = df["x"].to_numpy(dtype=float)
        ys = df["y"].to_numpy(dtype=float)

        if len(xs) > 3:
            kde = gaussian_kde(np.vstack([xs, ys]))
            Z = kde(positions).reshape(X.shape)
        else:
            Z = np.zeros_like(X)

        ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(f"{result.level}  (n={result.n_events})", fontsize=9)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.tick_params(labelsize=7)

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()

    if out is not None:
        from pathlib import Path
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig


__all__ = ["plot_kde_panels", "plot_suite_event_grid"]
