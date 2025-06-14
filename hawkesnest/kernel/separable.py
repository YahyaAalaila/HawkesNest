"""
Separable exponential-Gaussian kernel implementation.
phi(s, tau) = eta * g(tau) * h(||s||)
"""
from __future__ import annotations

import numpy as np
from hawkesnest.kernel.base import KernelBase

class ExponentialGaussianKernel:
    def __init__(
        self,
        branching_ratio: float,
        temporal_scale: float,
        spatial_scale: float
    ) -> None:
        """
        branching_ratio: total mass eta
        temporal_scale: decay parameter for exponential in time
        spatial_scale: bandwidth for Gaussian in space
        """
        self.eta = branching_ratio
        self.t_scale = temporal_scale
        self.s_scale = spatial_scale
    
    def __call__(self, s: np.ndarray, tau: np.ndarray) -> np.ndarray:
        # assume s = Euclidean distance array
        
        g = np.exp(-tau / self.t_scale)
        h = np.exp(-0.5 * (s / self.s_scale)**2)
        return self.eta * (g * h)

    def integrate(self) -> float:
        # For separable: eta * (integral g dt) * (integral h ds) normalized
        # Here we assume g and h normalized so integrate=eta
        return float(self.eta)
