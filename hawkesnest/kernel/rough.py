"""
Rough kernel via mixture of basis functions.
"""
from __future__ import annotations

import numpy as np
from numpy.fft import ifft2, fftshift
from pathlib import Path

from hawkesnest.kernel.base import KernelBase

class MixtureKernel(KernelBase):
    def __init__(
        self,
        sigmas: list[KernelBase],
        weights: np.ndarray,
        branching_ratio: float = 1.0,
        temporal_scale: float = 1.0,
    ) -> None:
        """Mixture of other KernelBase objects with given weights."""
        self.t_scale = temporal_scale 
        self.s2 = sigmas
        self.w = np.array(weights)

    def __call__(self, ds, dt):
        g = np.exp(-dt / self.t_scale)               # temporal part
        # spatial mixture of Gaussians
        h = ( self.w * np.exp(-0.5 * ds**2 / self.s2) ).sum(axis=-1)
        return g * h

    def integrate(self) -> float:
        return float(np.sum(self.weights * np.array([c.integrate() for c in self.components])))
    

class RoughKernel(KernelBase):
    """Frozen realisation of a (1 + 2)-D power-law Gaussian random field.

    Parameters
    ----------
    branching_ratio : float
        Total mass (η).  `integrate()` returns exactly this value.
    temporal_scale  : float
        Exponential decay parameter τ in g(dt)=exp(-dt/τ).
    length_scale    : float
        Overall spatial bandwidth ℓ; sets the *mean* bump size.
    hurst           : float in (0, 1)
        Controls roughness.  H→1 ⇒ very smooth, H→0 ⇒ spiky.
    n_fourier       : int
        Grid resolution used to sample the random surface.
    rng             : np.random.Generator | None
        Random generator (defaults to `np.random.default_rng()`).
    """

    def __init__(
        self,
        branching_ratio: float,
        temporal_scale: float,
        length_scale: float = 0.1,
        hurst: float = 0.5,
        n_fourier: int = 256,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.eta = float(branching_ratio)
        self.tau = float(temporal_scale)
        self.ℓ   = float(length_scale)
        self.H   = float(hurst)
        self.N   = int(n_fourier)
        self.rng = rng or np.random.default_rng()

        kx = np.fft.fftfreq(self.N, d=1 / self.N)
        ky = np.fft.fftfreq(self.N, d=1 / self.N)
        kx, ky = np.meshgrid(kx, ky, indexing="xy")
        k2 = kx**2 + ky**2 + 1e-12            # avoid 0 division at DC
        scale = k2 ** (-(self.H + 1.0) / 2)

        phase = self.rng.uniform(0.0, 2 * np.pi, size=(self.N, self.N))
        spec  = scale * np.exp(1j * phase)

        # make spectrum Hermitian so that ifft2 ⇒ real field
        spec = fftshift(spec)                 # AC  at centre for prettier field
        spec[0, 0] = 0.0                      # remove DC component

        field = np.real(ifft2(spec))
        field -= field.min()
        field /= field.sum()                  # normalise ∑ h = 1

        # store for fast look-ups
        self._field = field                   # (N,N) on [0,1)×[0,1)

    def __call__(self, s: np.ndarray | float, dt: np.ndarray | float) -> np.ndarray:
        """
        Evaluate φ(ds,dt) for an array of spatial offsets `s` (Euclidean
        distance) and temporal lags `dt`.

        `s` and `dt` must be broadcastable to the same shape.
        """
        s = np.asarray(s, dtype=float)
        dt = np.asarray(dt, dtype=float)

        # g(dt) part (exponential)
        g = np.exp(-dt / self.tau)

        # h(ds) part via table look-up
        # map distance s to a periodic coordinate on the [0,1) grid
        r = (s / self.ℓ) % 1.0
        ix = (r * self.N).astype(int) % self.N
        iy = np.zeros_like(ix)                # isotropic ⇒ use x index twice
        h = self._field[ix, iy]

        return self.eta * g * h

    # ------------------------------------------------------------------
    def integrate(self) -> float:
        """Return total mass η (by construction)."""
        return self.eta
    
    @classmethod
    def from_config(cls, cfg) -> RoughKernel:

        """
        Build a RoughKernel from the Pydantic KernelConfig.
        """
        return cls(
            branching_ratio = cfg.branching_ratio,
            temporal_scale  = cfg.temporal.decay,
            length_scale    = cfg.spatial.length_scale,
            hurst           = cfg.spatial.hurst,
            n_fourier       = getattr(cfg.spatial, "n_fourier", 256),
            rng             = None,  # or pass cfg.seed if you add one
        )
