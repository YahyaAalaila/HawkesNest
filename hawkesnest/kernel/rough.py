"""
Rough kernel via mixture of basis functions.
"""
from __future__ import annotations

import numpy as np
from hawkesnest.kernel.base import KernelBase

class MixtureKernel:
    def __init__(
        self,
        components: list[KernelBase],
        weights: np.ndarray
    ) -> None:
        """Mixture of other KernelBase objects with given weights."""
        self.components = components
        self.weights = np.array(weights)

    def __call__(self, s: np.ndarray, tau: np.ndarray) -> np.ndarray:
        out = 0
        for w, comp in zip(self.weights, self.components):
            out = out + w * comp(s, tau)
        return out

    def integrate(self) -> float:
        return float(np.sum(self.weights * np.array([c.integrate() for c in self.components])))
