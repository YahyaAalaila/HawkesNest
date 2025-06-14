"""Background intensity components.

A *background* represents the exogenous or baseline arrival rate `μ(s, t)` of a
spatio‑temporal point process.  It is evaluated independently of any
self‑exciting interaction kernels.
"""

from __future__ import annotations

from hawkesnest.background.constant import ConstantBackground  # noqa: F401
from hawkesnest.background.spatial import SpatialBackground  # noqa: F401
from hawkesnest.background.spatiotemporal import (
    SeparableBackground,  # noqa: F401
    EntangledBackground,  # noqa: F401
)
from hawkesnest.background.temporal      import TemporalProfile, SinusoidProfile, StepProfile, SplineProfile, RandomGPProfile  # noqa: F401


__all__ = [
    "ConstantBackground",
    "SpatialBackground",
    "SeparableBackground",
    "EntangledBackground",
    "TemporalProfile",
    "SinusoidProfile",
    "StepProfile",
    "SplineProfile",
]