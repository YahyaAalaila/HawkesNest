"""
Traveling-wave triggering kernel.

  φ(s_vec, τ) = exp(-τ/β) × exp(-‖s_vec − v·ê·τ‖² / (2σ²))

where
  s_vec  : 2-D spatial offset vector (x_child − x_parent, y_child − y_parent)
  τ      : time lag  t_child − t_parent  > 0
  v      : wave speed  (domain units / time unit)
  ê      : unit propagation direction  [cos(θ_wave), sin(θ_wave)]
  β      : temporal decay scale
  σ      : spatial bandwidth

The kernel is **not separable**: its spatial mode shifts by v·ê·τ as τ grows.
Any model that factors the conditional intensity into independent spatial and
temporal heads will systematically misfit this DGP.

Max value: 1.0, achieved at s_vec = v·ê·τ for any τ (the exact wave-front
location), and in particular at (s_vec=0, τ=0).

The thinning algorithm must pass the **vector** offset, not the scalar
distance.  Set ``uses_vector_s = True`` to signal this to the thinning loop.
"""
from __future__ import annotations

import math
import numpy as np

from hawkesnest.kernel.base import KernelBase


class TravelingWaveKernel(KernelBase):
    """
    Traveling-wave spatio-temporal triggering kernel.

    Parameters
    ----------
    v : float
        Wave speed (spatial units per time unit).  ``v = 0`` recovers an
        isotropic separable kernel.
    theta_wave : float
        Propagation direction in radians (0 = positive-x axis).
    sigma : float
        Spatial bandwidth of the Gaussian triggering cloud.
    temporal_scale : float
        Temporal decay parameter β.
    """

    #: Signal to thinning that this kernel needs the 2-D offset vector,
    #: not the scalar Euclidean distance.
    uses_vector_s: bool = True

    def __init__(
        self,
        v: float,
        theta_wave: float,
        sigma: float,
        temporal_scale: float,
    ) -> None:
        self.v = float(v)
        self.theta_wave = float(theta_wave)
        self.sigma = float(sigma)
        self.t_scale = float(temporal_scale)

        # Pre-compute unit direction vector once
        self._ex = math.cos(theta_wave)
        self._ey = math.sin(theta_wave)

    # ------------------------------------------------------------------
    def __call__(self, s: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        Evaluate φ(s_vec, τ).

        Parameters
        ----------
        s : array-like, shape (2,) or (N, 2)
            2-D spatial offset vector(s).
        tau : scalar or array-like
            Time lag(s) > 0.
        """
        s = np.asarray(s, dtype=float)
        tau = np.asarray(tau, dtype=float)

        # Scalar 0 is accepted as the zero-vector (used by thinning debug check)
        if s.ndim == 0:
            s = np.zeros(2, dtype=float)

        # Temporal decay
        g = np.exp(-tau / self.t_scale)

        # Centre of the triggering cloud at lag τ: shift = v·ê·τ
        shift_x = self.v * self._ex * tau
        shift_y = self.v * self._ey * tau

        if s.ndim == 1 and s.shape == (2,):
            # Single evaluation: s = [dx, dy]
            dx = s[0] - shift_x
            dy = s[1] - shift_y
        elif s.ndim == 2 and s.shape[-1] == 2:
            # Batch: s = (N, 2)
            dx = s[:, 0] - shift_x
            dy = s[:, 1] - shift_y
        else:
            raise ValueError(
                f"TravelingWaveKernel expects s of shape (2,) or (N,2); got {s.shape}"
            )

        r2 = dx**2 + dy**2
        h = np.exp(-0.5 * r2 / self.sigma**2)
        return g * h

    def max_value(self) -> float:
        """
        Upper bound on φ over all (s_vec, τ).

        The maximum is exactly 1, attained on the wave-front
        (s_vec = v·ê·τ) as τ → 0+.
        """
        return 1.0

    def integrate(self) -> float:
        """
        Approximate total mass ∫∫ φ(s, τ) ds dτ.

        For a traveling-wave kernel, the spatial integral at any fixed τ
        is  2π σ²  (the Gaussian integrates to 2πσ² over ℝ²), and the
        temporal integral of exp(-τ/β) is β.

        So the total mass is 2π σ² β.  The adjacency matrix scales this
        to the correct branching ratio, so we return 1 here (same
        convention as ``ExponentialGaussianKernel``).
        """
        return 1.0
