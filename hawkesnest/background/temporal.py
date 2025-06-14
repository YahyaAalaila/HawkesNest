# hawkesnest/background/temporal.py
"""
TemporalProfile implementations for background intensity.

A TemporalProfile maps a time t to a non-negative multiplier g(t).
"""
from __future__ import annotations
from typing import Protocol, Callable
import math
import numpy as np

class TemporalProfile(Protocol):
    """
    Protocol for temporal background intensity profiles.
    """
    def __call__(self, t: float) -> float:
        """Return the multiplier g(t) >= 0 at time t."""
        ...

class SinusoidProfile:
    """
    Periodic temporal profile: g(t) = 1 + amplitude * sin(2*pi*(t - phase)/period)
    Ensures positivity when amplitude < 1.
    """
    def __init__(self, period: float, amplitude: float = 0.5, phase: float = 0.0):
        self.period = period
        self.amplitude = amplitude
        self.phase = phase

    def __call__(self, t: float) -> float:
        omega = 2 * math.pi / self.period
        return max(0.0, 1.0 + self.amplitude * math.sin(omega * (t - self.phase)))
class StepProfile:
    """
    Piecewise-constant temporal profile: g(t) = value_i whenever
    t (mod 24h) ∈ [start_i, end_i), for the provided list of steps.
    """

    def __init__(self,
                 steps: list[tuple[float,float,float]]):
        # steps: [(start1,end1,val1), (start2,end2,val2), …]
        self.steps = steps

    def __call__(self, t: float) -> float:
        t_mod = t % 24.0
        for start, end, val in self.steps:
            if start <= t_mod < end:
                return val
        # if no interval matches, you could return 0 or last value
        return 0.0

class SplineProfile:
    """
    Smooth spline-based profile fitted to control points.
    """
    def __init__(self, times: list[float], values: list[float]):
        from scipy.interpolate import UnivariateSpline
        self.spline = UnivariateSpline(times, values, s=0, k=3)

    def __call__(self, t: float) -> float:
        val = self.spline(t)
        return max(0.0, float(val))

class RandomGPProfile:
    """
    Log-Gaussian process temporal profile: g(t) = exp(Z(t)),
    Z ~ GP(0, kernel).
    """
    def __init__(self, kernel: Callable[[float, float], float], rng: np.random.Generator | None = None):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Kernel
        self._gp = GaussianProcessRegressor(kernel=kernel if isinstance(kernel, Kernel) else Kernel(),
                                           random_state=rng)
        self._is_fitted = False
        self._rng = rng or np.random.default_rng()

    def fit(self, times: np.ndarray):
        # sample GP at given times
        X = times.reshape(-1, 1)
        y = self._rng.normal(size=len(times))  # draw latent values
        self._gp.fit(X, y)
        self._is_fitted = True

    def __call__(self, t: float) -> float:
        if not self._is_fitted:
            raise RuntimeError("RandomGPProfile must be fitted before use")
        val, _ = self._gp.predict(np.array([[t]]), return_std=True)
        return float(math.exp(val[0]))
