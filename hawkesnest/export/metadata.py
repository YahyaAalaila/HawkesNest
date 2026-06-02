"""Metadata export utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def write_metadata(result: Any, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **dict(result.metadata),
        "suite_name": result.suite_name,
        "level": result.level,
        "seed": result.seed,
        "simulator_class_name": result.simulator_class_name,
        "export_paths": dict(result.export_paths),
        "config": result.config,
    }
    out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return out
