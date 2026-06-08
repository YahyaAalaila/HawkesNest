"""
Two-scale temporal triggering kernel.

  φ(r, τ) = [α_fast · exp(-τ/β_fast) + α_slow · exp(-τ/β_slow)]
            · exp(-r² / (2σ²))

The two temporal components create a fast burst and a slow echo.
Any model with a single temporal decay rate β cannot fit both components
simultaneously: it will either over-smooth the fast burst or under-weight
the slow echo.

Complexity axis: scale ratio ρ = β_slow / β_fast ∈ {1, 3, 10, 30, 100}.
At ρ = 1, the two components coincide and this reduces to a standard
ExponentialGaussianKernel.
"""
from __future__ import annotations

import numpy as np
from hawkesnest.kernel.base import KernelBase


class TwoScaleKernel(KernelBase):
    """
    Temporal two-component triggering kernel.

    Parameters
    ----------
    alpha_fast : float
        Weight of the fast (short-lag) component. Must be in (0, 1).
    beta_fast : float
        Temporal decay of the fast component.
    beta_slow : float
        Temporal decay of the slow component.
    sigma : float
        Isotropic spatial bandwidth (shared by both components).

    Notes
    -----
    ``alpha_slow = 1 - alpha_fast`` so the weights sum to 1.  The
    adjacency matrix provides the overall branching ratio.
    """

    def __init__(
        self,
        alpha_fast: float,
        beta_fast: float,
        beta_slow: float,
        sigma: float,
    ) -> None:
        if not (0.0 < alpha_fast < 1.0):
            raise ValueError(f"alpha_fast must be in (0,1), got {alpha_fast}")
        if beta_fast <= 0.0 or beta_slow <= 0.0:
            raise ValueError("Both beta_fast and beta_slow must be positive.")

        self.alpha_fast = float(alpha_fast)
        self.alpha_slow = 1.0 - self.alpha_fast
        self.beta_fast = float(beta_fast)
        self.beta_slow = float(beta_slow)
        self.sigma = float(sigma)

    # ------------------------------------------------------------------
    def __call__(self, s: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        Evaluate φ(r, τ) where r = ‖s‖ (scalar Euclidean distance).

        Parameters
        ----------
        s : scalar or array-like
            Euclidean distance between parent and candidate event.
        tau : scalar or array-like
            Time lag > 0.
        """
        s = np.asarray(s, dtype=float)
        tau = np.asarray(tau, dtype=float)

        # Temporal mixture
        g = (
            self.alpha_fast * np.exp(-tau / self.beta_fast)
            + self.alpha_slow * np.exp(-tau / self.beta_slow)
        )

        # Isotropic spatial Gaussian (r = s if already scalar distance)
        h = np.exp(-0.5 * (s / self.sigma) ** 2)

        return g * h

    def max_value(self) -> float:
        """Upper bound: at r = 0, τ = 0, φ = alpha_fast + alpha_slow = 1."""
        return 1.0

    def integrate(self) -> float:
        return 1.0
