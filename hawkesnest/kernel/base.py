"""
Abstract base class and protocols for triggering kernels.

Each kernel \phi_{mn}(s, tau) must be non-negative, integrate to a finite
branching ratio, and support vectorized evaluation over spatial lags and
temporal lags.
"""
from __future__ import annotations

import abc
import numpy as np
from typing import Protocol, Tuple

# Type alias for spatial and temporal arrays
tArray = np.ndarray

class KernelBase(Protocol):
    """
    Protocol for a spatio-temporal triggering kernel phi(s, tau).
    s: either a tuple of (dx, dy) arrays or precomputed distances.
    tau: array of time lags > 0.
    Returns an array of intensities >= 0.
    """
    def __call__(self, s: tArray, tau: tArray) -> tArray:
        ...
    
    def integrate(self) -> float:
        """
        Return the total mass \int phi(s, tau) ds dt = branching ratio.
        """
        ...