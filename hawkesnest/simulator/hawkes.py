"""
HawkesSimulator: implements Ogata's thinning algorithm for multivariate
space-time Hawkes processes using injected background and kernels.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

from hawkesnest.background import BackgroundBase
from hawkesnest.domain import RectangleDomain, SpatialDomain
from hawkesnest.kernel import KernelBase
from hawkesnest.simulator import SimulatorBase
from hawkesnest.utils.thinning import thinning

_DEFAULT_HORIZON = 10_000.0


class HawkesSimulator(SimulatorBase):
    """Simulator for multivariate space-time Hawkes processes."""

    def __init__(self, domain: SpatialDomain = RectangleDomain(0.0, 1.0, 0.0, 1.0), background: BackgroundBase = None, kernels: Dict[Tuple[int, int], KernelBase] = None, adjacency: np.ndarray = None, lambda_max: float | None = 1.0) -> None:
        super().__init__()
        self.domain = domain
        self.background = background or (lambda s, t, m=None: 1.0)
        self.kernels = kernels
        self.adjacency = adjacency if adjacency is not None else np.array([[0.5, 0], [0, 0.2]])
        self.lambda_max = lambda_max

    def simulate(self, n: Optional[int] = None, horizon: Optional[float] = None, seed: Optional[int] = None, tau_max: float = 5.0, *, debug: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(seed)
        n_acc, time_horizon = self._resolve_n_horizon(n, horizon)
        if time_horizon is None:
            time_horizon = _DEFAULT_HORIZON
        lam_max = self.lambda_max
        if lam_max is None:
            lam_max = self._estimate_lambda_max(time_horizon=time_horizon, rng=rng)
        events = thinning(background=self.background, kernel=self.kernels, time_horizon=time_horizon, domain=self.domain, adjacency=self.adjacency, Lambda=lam_max, n_acc=n_acc, rng=rng, debug=debug, tau_max=tau_max)
        if not events:
            return pd.DataFrame(columns=["t", "x", "y", "m", "is_triggered"]), pd.Series(dtype=object)
        df = pd.DataFrame(events).rename(columns=lambda c: "m" if c == "type" else c).assign(parent=lambda d: d.get("parent", pd.Series([None] * len(d), index=d.index))).loc[:, ["t", "x", "y", "m", "is_triggered", "parent"]].sort_values("t").reset_index(drop=True)
        labels = df.pop("parent")
        return df, labels

    @staticmethod
    def _resolve_n_horizon(n: Optional[int], horizon: Optional[float], *, default_horizon: float = _DEFAULT_HORIZON) -> Tuple[Optional[int], Optional[float]]:
        if (n is not None) and (horizon is not None):
            warnings.warn("Both 'n' and 'horizon' were provided; ignoring both and using default horizon.", UserWarning, stacklevel=2)
            return None, default_horizon
        if (n is None) and (horizon is None):
            return None, default_horizon
        return (n, None) if n is not None else (None, horizon)

    def _estimate_lambda_max(self, time_horizon: float, rng: np.random.Generator, n_grid_space: int = 64, n_grid_time: int = 64, safety_factor: float = 1.5) -> float:
        xs = np.linspace(self.domain.x_min, self.domain.x_max, n_grid_space)
        ys = np.linspace(self.domain.y_min, self.domain.y_max, n_grid_space)
        ts = np.linspace(0.0, time_horizon, n_grid_time)
        m = self.adjacency.shape[0]
        lam_bg_max = 0.0
        for x in xs:
            for y in ys:
                for t in ts:
                    for mark in range(1, m + 1):
                        lam_bg_max = max(lam_bg_max, self.background(np.array([x, y]), t, mark))
        lam_trig_max = 0.0
        if self.kernels is not None:
            for kobj in self.kernels.values():
                if hasattr(kobj, "max_value"):
                    lam_trig_max = max(lam_trig_max, float(kobj.max_value()))
            lam_trig_max *= float(np.max(self.adjacency))
        lam_max = (lam_bg_max + lam_trig_max) * safety_factor
        if lam_max <= 0.0:
            raise ValueError(f"Estimated lambda_max non-positive: {lam_max}")
        return lam_max
