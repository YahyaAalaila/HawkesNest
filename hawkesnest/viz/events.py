"""Event visualization helpers."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_events(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            row = json.loads(handle.readline())
        times = row.get("times", [])
        locations = row.get("locations", [])
        return pd.DataFrame(
            {
                "t": times,
                "x": [xy[0] for xy in locations],
                "y": [xy[1] for xy in locations],
            }
        )
    return pd.read_csv(path)


def _coerce_events(events: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        return events
    return _load_events(Path(events))


def plot_events_2d(
    events: pd.DataFrame | str | Path,
    *,
    ax=None,
    color_by: str = "t",
    title: str | None = None,
    cmap: str = "viridis",
    s: float = 18.0,
    alpha: float = 0.85,
):
    """Plot event locations in the spatial domain."""
    df = _coerce_events(events)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.0))
    else:
        fig = ax.figure

    color = df[color_by] if color_by in df.columns else None
    scatter = ax.scatter(df["x"], df["y"], c=color, cmap=cmap, s=s, alpha=alpha)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "Event locations")
    ax.set_aspect("equal", adjustable="box")
    if color is not None:
        fig.colorbar(scatter, ax=ax, label=color_by)
    return fig, ax


def plot_events_3d(
    events: pd.DataFrame | str | Path,
    *,
    ax=None,
    color_by: str = "t",
    title: str | None = None,
    cmap: str = "viridis",
    s: float = 18.0,
    alpha: float = 0.85,
):
    """Plot a 3D event cloud with axes (x, y, t)."""
    df = _coerce_events(events)
    if ax is None:
        fig = plt.figure(figsize=(5.2, 4.4))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    color = df[color_by] if color_by in df.columns else None
    scatter = ax.scatter(df["x"], df["y"], df["t"], c=color, cmap=cmap, s=s, alpha=alpha)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.set_title(title or "Event cloud")
    if color is not None:
        fig.colorbar(scatter, ax=ax, shrink=0.75, label=color_by)
    return fig, ax


def visualize_events_file(
    path: str | Path,
    *,
    kind: str = "space-time",
    out: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """Create a basic generated-event visualization from CSV or JSONL."""
    event_path = Path(path)
    events = _load_events(event_path)

    if kind == "3d":
        fig, _ = plot_events_3d(events)
    elif kind == "space-time":
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        plot_events_2d(events, ax=axes[0], color_by="t", title="Space")
        axes[1].plot(range(len(events)), events["t"], linewidth=1.2)
        axes[1].set_xlabel("event index")
        axes[1].set_ylabel("t")
        axes[1].set_title("Time")
        fig.tight_layout()
    else:
        raise ValueError("kind must be 'space-time' or '3d'")

    out_path = Path(out) if out is not None else event_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return out_path
