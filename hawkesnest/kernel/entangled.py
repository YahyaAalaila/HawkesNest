"""
Non-separable space-time kernel implementation.
"""

from __future__ import annotations

import numpy as np
from hawkesnest.kernel.base import KernelBase
from typing import Callable

from hawkesnest.kernel.separable import ExponentialGaussianKernel

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
 

class EntangledExponentialGaussianKernel(KernelBase):
    """
    Wrapper that adds a controllable entanglement parameter θ_ent to a
    base ExponentialGaussianKernel.

    Parameters
    ----------
    base_kernel : ExponentialGaussianKernel
        The underlying separable kernel φ_base(r, τ).
    shape_fn : callable
        Entangling shape function shape_fn(r, τ). Must accept NumPy
        arrays r, τ (broadcastable) and return an array of the same
        broadcasted shape. Typical choice: a wrapped function from
        _FUNCTION_REGISTRY like 'poly_entangled_kernel' or
        'gabor_travel_kernel'.
    theta_ent : float
        Entanglement strength parameter. θ_ent = 0 reproduces the
        base kernel. Increasing θ_ent increases non-separability.
    r_max : float
        Maximum spatial lag used to define the normalisation grid
        [0, r_max]. For a unit-square domain, r_max ≈ sqrt(2) is
        reasonable; you can also just use 1.0.
    tau_max : float
        Maximum temporal lag used to define the normalisation grid
        [0, tau_max]. Should cover the effective support of the kernel.
    n_r : int, default 64
        Number of grid points in the r-direction.
    n_tau : int, default 64
        Number of grid points in the τ-direction.
    renormalize : bool, default True
        If True, rescales the modulated kernel so that its total mass
        on the (r, τ) grid matches that of the base kernel.
    """

    def __init__(
        self,
        base_kernel: ExponentialGaussianKernel,
        shape_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        theta_ent: float,
        r_max: float,
        tau_max: float,
        n_r: int = 64,
        n_tau: int = 64,
        renormalize: bool = True,
    ) -> None:
        self.base_kernel = base_kernel
        self.shape_fn = shape_fn
        self.theta_ent = float(theta_ent)
        self.renormalize = bool(renormalize)

        # --- 1. Build normalisation grid --------------------------------
        rs = np.linspace(0.0, float(r_max), int(n_r))
        taus = np.linspace(0.0, float(tau_max), int(n_tau))
        R, T = np.meshgrid(rs, taus, indexing="ij")  # shapes (n_r, n_tau)
        dr = float(rs[1] - rs[0]) if len(rs) > 1 else 1.0
        dt = float(taus[1] - taus[0]) if len(taus) > 1 else 1.0

        # --- 2. Evaluate shape and base kernel on grid -------------------
        shape_vals = np.asarray(shape_fn(R, T), dtype=float)
        if not np.all(np.isfinite(shape_vals)):
            raise ValueError("shape_fn produced non-finite values on normalisation grid.")

        base_vals = np.asarray(base_kernel(R, T), dtype=float)
        if not np.all(np.isfinite(base_vals)) or np.any(base_vals < 0.0):
            raise ValueError("base_kernel must be finite and non-negative on normalisation grid.")

        # --- 3. Centre / scale shape to get z(r, τ) ----------------------
        mu = float(shape_vals.mean())
        sigma = float(shape_vals.std())
        if sigma <= 0.0:
            # Degenerate shape: fall back to zero interaction (z == 0)
            sigma = 1.0

        self._shape_mean = mu
        self._shape_std = sigma

        # --- 4. Precompute renormalisation factor c_θ --------------------
        # z on grid
        z_grid = (shape_vals - mu) / sigma
        mod_vals = base_vals * np.exp(self.theta_ent * z_grid)

        base_mass = float((base_vals * dr * dt).sum())
        mod_mass = float((mod_vals * dr * dt).sum())
        self._max_kernel_val = float(np.max(mod_vals))

        if self.renormalize and mod_mass > 0.0:
            self._scale = base_mass / mod_mass
        else:
            self._scale = 1.0

    # ------------------------------------------------------------------ #
    # Call interface used by thinning()
    # ------------------------------------------------------------------ #
    def __call__(self, r: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        Evaluate the entangled kernel at given spatial distance r and
        time lag tau. Supports scalar or array inputs via NumPy
        broadcasting.
        """
        r_arr = np.asarray(r, dtype=float)
        tau_arr = np.asarray(tau, dtype=float)

        base_val = self.base_kernel(r_arr, tau_arr)

        shape_val = self.shape_fn(r_arr, tau_arr)
        z = (shape_val - self._shape_mean) / self._shape_std

        return self._scale * base_val * np.exp(self.theta_ent * z)
    def max_value(self) -> float:
        """
        Upper bound on φ_θ(r, τ) over the normalisation grid.

        Used by HawkesSimulator._estimate_lambda_max. It is a grid-based
        bound, not a closed-form supremum, but that is enough for Ogata
        thinning as long as it is finite and conservative.
        """
        return self._max_kernel_val
    # ------------------------------------------------------------------ #
    # Optional: approximate total mass (not used by simulator)
    # ------------------------------------------------------------------ #
    def integrate(self) -> float:
        """
        Approximate total mass of the kernel. For now we simply return
        1.0 because adjacency is treated as a free branching parameter
        and the simulator does not depend on this method.
        """
        return 1.0
