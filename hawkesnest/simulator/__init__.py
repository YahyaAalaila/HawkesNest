"""Simulator function components.

A *simulator* represents a spatio-temporal point process simulator that
generates synthetic event data based on a specified background intensity and
self-exciting interaction kernel. It is evaluated independently of any
background intensity components or interaction kernels.
"""

from __future__ import annotations

from hawkesnest.simulator.base import SimulatorBase  # noqa: F401
from hawkesnest.simulator.hawkes import HawkesSimulator  # noqa: F401

__all__ = [
    "SimulatorBase",
    "HawkesSimulator",
]
