"""Future recipe object for composing suite dimensions."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HawkesNestRecipe:
    """Design placeholder; not wired to generation in Phase 1."""

    entanglement: str | None = None
    heterogeneity: str | None = None
    branching: str | None = None
    topology: str | None = None
