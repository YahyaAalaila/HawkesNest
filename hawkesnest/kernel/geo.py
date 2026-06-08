# hawkesnest/kernel/geodesic.py
from __future__ import annotations
import math

from hawkesnest.kernel.base import KernelBase

class ExponentialGeodesicKernel(KernelBase):
    def __init__(self, temporal_scale: float, geodesic_scale: float):
        self.temporal_scale = temporal_scale
        self.geodesic_scale = geodesic_scale

    # NB: called by thinning(ds, dt)
    def __call__(self, ds_G: float, dt: float) -> float:
        """
        k(ds_G, dt) = β · (1/τ) e^{-dt/τ} · (1/σ) e^{-ds_G/σ}
        """
        tau = self.temporal_scale
        sigma = self.geodesic_scale
        return (1.0 / tau) * math.exp(-dt / tau) * (1.0 / sigma) * math.exp(-ds_G / sigma)