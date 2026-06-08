"""CSV export utilities."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_events_csv(events: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(out, index=False)
    return out
