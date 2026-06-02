"""Event-file visualization helpers."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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


def visualize_events_file(
    path: str | Path,
    *,
    kind: str = "space-time",
    out: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """Create a basic generated-event visualization."""
    event_path = Path(path)
    events = _load_events(event_path)
    if kind != "space-time":
        raise ValueError("Only kind='space-time' is implemented in Phase 1.")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].scatter(events["x"], events["y"], s=10, alpha=0.75)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Space")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].plot(range(len(events)), events["t"], linewidth=1.2)
    axes[1].set_xlabel("event index")
    axes[1].set_ylabel("t")
    axes[1].set_title("Time")
    fig.tight_layout()

    out_path = Path(out) if out is not None else event_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return out_path
