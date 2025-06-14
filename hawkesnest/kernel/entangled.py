"""
Non-separable space-time kernel implementation.
"""
from __future__ import annotations

import numpy as np
from hawkesnest.kernel.base import KernelBase

class SpaceTimeKernel(KernelBase):
    def __init__(self, func: callable) -> None:
        """
        func: user-provided function f(s, tau) -> non-negative float/array
        """
        self.func = func

    def __call__(self, s: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return self.func(s, tau)

    def integrate(self) -> float:
        # Integrate via Monte Carlo or analytic method as needed
        raise NotImplementedError("Entangled kernel integration must be implemented")
 