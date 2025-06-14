"""
Network-constrained kernel: replaces Euclidean distance with geodesic.
"""
from __future__ import annotations

import numpy as np
from hawkesnest.kernel.base import KernelBase

class NetworkKernel(KernelBase):
    def __init__(
        self,
        base_kernel: KernelBase,
        distance_func: callable
    ) -> None:
        """
        base_kernel: any KernelBase using Euclidean s
        distance_func: function mapping spatial coords to geodesic distances
        """
        self.base = base_kernel
        self.dist_fn = distance_func

    def __call__(self, s: np.ndarray, tau: np.ndarray) -> np.ndarray:
        # s here is raw vector offsets; convert to geodesic distances
        geo_dist = self.dist_fn(s)
        return self.base(geo_dist, tau)

    def integrate(self) -> float:
        return self.base.integrate()
