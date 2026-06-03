"""Suite-level diagnostic visualisation helpers."""
from __future__ import annotations

from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkesnest.viz.events import plot_events_2d


def plot_suite_event_grid(
    results,
    *,
    color_by: str = "t",
    title: str | None = None,
    out: str | None = None,
    show: bool = False,
) -> "plt.Figure":
    """Plot one spatial event scatter per suite level in a single row.

    Parameters
    ----------
    results:
        Iterable of ``GenerationResult`` objects (from ``BaseSuite.generate``),
        one per level, in the order they should appear left-to-right.
    color_by:
        Column to use for point colour (``"t"``, ``"m"``, or ``"is_triggered"``).
    title:
        Optional super-title for the figure.
    out:
        If given, save the figure to this path (PNG, PDF, …).
    show:
        If True, call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    results = list(results)
    n = len(results)
    if n == 0:
        raise ValueError("results is empty")

    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.6), squeeze=False)
    for ax, result in zip(axes[0], results):
        plot_events_2d(result.events, ax=ax, color_by=color_by)
        ax.set_title(f"{result.suite_name}\n{result.level}  n={result.n_events}", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    if out is not None:
        from pathlib import Path
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160)
    if show:
        plt.show()
    return fig


__all__ = ["plot_suite_event_grid"]
