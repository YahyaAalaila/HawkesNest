"""Separable spatio-temporal kernels."""

from __future__ import annotations

import numpy as np

from hawkesnest.kernel.base import KernelBase


class ExponentialGaussianKernel(KernelBase):
    """Product kernel with exponential time decay and Gaussian spatial decay."""

    def __init__(self, spatial_scale: float = 1.0, temporal_scale: float = 1.0, branching_ratio: float = 1.0):
        self.s_scale = spatial_scale
        self.t_scale = temporal_scale
        self.branching_ratio = branching_ratio

    def __call__(self, s: np.ndarray | float, tau: np.ndarray | float) -> np.ndarray:
        g = np.exp(-np.asarray(tau) / self.t_scale)
        h = np.exp(-0.5 * (np.asarray(s) / self.s_scale) ** 2)
        return g * h

    def max_value(self) -> float:
        return 1.0

    def integrate(self) -> float:
        return 1.0
