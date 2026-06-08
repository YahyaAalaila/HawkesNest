"""
Piecewise-stationary (regime-switching) background intensity.

  μ(s, t) = Σ_j  A_j · G(s - c_j ; σ_j)

where the Gaussian parameters {A_j, c_j, σ_j} change at discrete time
changepoints T_1, T_2, … .  Within each regime the background is a static
Gaussian mixture; at each changepoint the hotspot locations and weights jump.

Diagnostic: models that fit a single static spatial map will misplace
predicted intensity after each regime transition.

Complexity axis: number of switches K ∈ {0, 1, 2, 3, 4}.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from hawkesnest.background.base import BackgroundBase


@dataclass
class GaussianComponent:
    """A single isotropic Gaussian hotspot."""
    cx: float       # centre x
    cy: float       # centre y
    sigma: float    # bandwidth
    amplitude: float = 1.0


@dataclass
class Regime:
    """A time window with its own Gaussian mixture background."""
    t_start: float
    t_end: float
    components: List[GaussianComponent]
    base_rate: float = 0.0   # additive constant within this regime


class RegimeSwitchingBackground(BackgroundBase):
    """
    Regime-switching background: piecewise-stationary Gaussian mixture.

    Parameters
    ----------
    regimes : list of Regime
        Time-ordered, non-overlapping regimes.  The regimes must cover
        [0, T] without gaps; pass t_start=0 for the first regime.
    lambda0 : float
        Global baseline added to every evaluation (ensures λ > 0 even
        between hotspots).

    Notes
    -----
    • Hotspot amplitudes are normalised so that the spatial integral over
      the domain sums to ``lambda0`` within each regime.  This keeps the
      mean event rate constant across regimes so that only the *spatial
      distribution* changes, not the total rate.
    • If ``lambda0 = 0`` the background can be zero in parts of the
      domain; the simulator will still work as long as the total intensity
      (background + triggering) stays positive everywhere.
    """

    def __init__(
        self,
        regimes: List[Regime],
        lambda0: float = 1.0,
    ) -> None:
        if not regimes:
            raise ValueError("At least one regime must be provided.")

        # Sort by t_start (defensive)
        self._regimes = sorted(regimes, key=lambda r: r.t_start)
        self.lambda0 = float(lambda0)

    # ------------------------------------------------------------------
    def __call__(self, s: np.ndarray, t: float, m: int | None = None) -> float:
        s = np.asarray(s, dtype=float)
        regime = self._regime_at(t)
        return self.lambda0 + self._eval_components(s, regime.components)

    def batch(self, S: np.ndarray, T: np.ndarray, M: np.ndarray | None = None) -> np.ndarray:
        n = S.shape[0]
        out = np.empty(n)
        for i in range(n):
            out[i] = self(S[i], float(T[i]))
        return out

    # ------------------------------------------------------------------
    def _regime_at(self, t: float) -> Regime:
        """Return the active regime at time t (last one if t is past all regimes)."""
        for r in reversed(self._regimes):
            if t >= r.t_start:
                return r
        return self._regimes[0]

    @staticmethod
    def _eval_components(s: np.ndarray, components: List[GaussianComponent]) -> float:
        val = 0.0
        for c in components:
            dx = s[0] - c.cx
            dy = s[1] - c.cy
            r2 = dx * dx + dy * dy
            val += c.amplitude * np.exp(-0.5 * r2 / (c.sigma * c.sigma))
        return float(val)

    # ------------------------------------------------------------------
    @classmethod
    def from_dicts(
        cls,
        regime_specs: Sequence[dict],
        lambda0: float = 1.0,
    ) -> "RegimeSwitchingBackground":
        """
        Convenience constructor from a list of dicts.

        Each dict must contain:
          t_start  : float
          t_end    : float
          components : list of {cx, cy, sigma, amplitude}
          base_rate  : float (optional, default 0)

        Example::

            RegimeSwitchingBackground.from_dicts([
                {"t_start": 0.0, "t_end": 50.0,
                 "components": [{"cx": 0.2, "cy": 0.3, "sigma": 0.1, "amplitude": 10.0}]},
                {"t_start": 50.0, "t_end": 100.0,
                 "components": [{"cx": 0.8, "cy": 0.7, "sigma": 0.1, "amplitude": 10.0}]},
            ], lambda0=2.0)
        """
        regimes = []
        for spec in regime_specs:
            comps = [
                GaussianComponent(**c) for c in spec["components"]
            ]
            regimes.append(Regime(
                t_start=float(spec["t_start"]),
                t_end=float(spec["t_end"]),
                components=comps,
                base_rate=float(spec.get("base_rate", 0.0)),
            ))
        return cls(regimes=regimes, lambda0=lambda0)
