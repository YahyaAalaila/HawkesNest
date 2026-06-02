"""JSONL export utilities."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_events_jsonl(events: pd.DataFrame, path: str | Path) -> Path:
    """Write one sequence as one JSONL row with times and locations."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "times": [float(t) for t in events["t"].tolist()] if "t" in events else [],
        "locations": [
            [float(x), float(y)]
            for x, y in zip(events["x"].tolist(), events["y"].tolist(), strict=False)
        ]
        if {"x", "y"}.issubset(events.columns)
        else [],
    }
    with out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return out
