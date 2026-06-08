from __future__ import annotations

from typing import Callable

import numpy as np

from hawkesnest.background.base import BackgroundBase
from hawkesnest.domain import RectangleDomain


class HeteroLadderBackground(BackgroundBase):
    """
    Heterogeneity ladder:

        μ_θ(s,t) = λ0 * ((1 − θ_het) + θ_het * g_norm(s,t)),

    where g_norm is a structured field normalised to mean 1 over
    domain × [0, T].

    This ensures:
      • Mean intensity is always λ0 for all θ_het.
      • θ_het controls ONLY heterogeneity, not global scaling.
    """

    def __init__(
        self,
        lambda0: float,
        g_fn: Callable[[np.ndarray, float], float],
        domain: RectangleDomain,
        T: float,
        theta_het: float,
        n_grid_space: int = 64,
        n_grid_time: int = 64,
    ) -> None:
        self.lambda0 = float(lambda0)
        self.g_fn = g_fn
        self.theta = float(theta_het)

        xs = np.linspace(domain.x_min, domain.x_max, n_grid_space)
        ys = np.linspace(domain.y_min, domain.y_max, n_grid_space)
        ts = np.linspace(0.0, T, n_grid_time)

        vals = []
        for x in xs:
            for y in ys:
                for t in ts:
                    vals.append(float(g_fn(np.array([x, y], dtype=float), float(t))))
        vals = np.asarray(vals, dtype=float)

        mean_val = float(np.mean(vals))
        if mean_val <= 0.0 or not np.isfinite(mean_val):
            raise ValueError(
                f"[HeteroLadderBackground] g_fn must have positive finite mean; got {mean_val}"
            )

        self._norm_const = mean_val

    def __call__(self, s, t, m=None) -> float:
        g_raw = float(self.g_fn(np.asarray(s, float), float(t)))
        g_norm = g_raw / self._norm_const  # mean-normalised
        return self.lambda0 * ((1.0 - self.theta) + self.theta * g_norm)
