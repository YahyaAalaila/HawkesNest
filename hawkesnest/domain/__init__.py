"""Background intensity components.

A *background* represents the exogenous or baseline arrival rate `μ(s, t)` of a
spatio‑temporal point process.  It is evaluated independently of any
self‑exciting interaction kernels.
"""

from __future__ import annotations

from hawkesnest.domain.standard import RectangleDomain  # noqa: F401
from hawkesnest.domain.base import SpatialDomain  # noqa: F401

__all__ = [
    "RectangleDomain",
    "SpatialDomain",
]