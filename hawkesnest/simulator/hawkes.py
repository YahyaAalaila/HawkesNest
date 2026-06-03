"""
HawkesSimulator: implements Ogata's thinning algorithm for multivariate
space–time Hawkes processes using injected background and kernels.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from hawkesnest.background import BackgroundBase
from hawkesnest.domain import SpatialDomain
from hawkesnest.domain import RectangleDomain
from hawkesnest.kernel import KernelBase
from hawkesnest.simulator import SimulatorBase
from hawkesnest.utils.thinning import thinning
import warnings

_DEFAULT_HORIZON = 10_000.0     # large enough for n-mode to reach target count

class HawkesSimulator(SimulatorBase):
    """
    Simulator for multivariate space–time Hawkes processes using Ogata thinning.
    All parameters have sensible defaults to allow smoke-testing with no args.
    """

    def __init__(
        self,
        domain: SpatialDomain = RectangleDomain(0.0, 1.0, 0.0, 1.0),
        background: BackgroundBase = None,
        kernels: Dict[Tuple[int, int], KernelBase] = None,
        adjacency: np.ndarray = None,
        lambda_max: float | None = None) -> None:
        
        super().__init__()
        # Defaults: homogeneous Poisson
        self.domain = domain
        self.background = background #or (lambda s, t, m=None: 1.0)
        self.kernels = kernels  # or {} # Default to single type with self-loop
        self.adjacency = (
            adjacency if adjacency is not None else np.array([[0.5, 0], [0, 0.2]])
        )
        self.lambda_max = lambda_max

    def simulate(
        self,
        n: Optional[int] = None,
        horizon: Optional[float] = None,
        seed: Optional[int] = None,
        tau_max : float = 5.0,
        *,
        debug: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Simulate n events. Returns DataFrame of events and Series of parent labels.
        """
        
        rng = np.random.default_rng(seed)
        n_acc, time_horizon = self._resolve_n_horizon(n, horizon)
        # thinning always needs a finite time horizon even in n_acc mode
        if time_horizon is None:
            time_horizon = _DEFAULT_HORIZON

        lam_max = self.lambda_max
        if lam_max is None:
            lam_max = self._estimate_lambda_max(time_horizon=time_horizon, rng=rng)
        
        # Thinning returns list of dicts with keys 't','x','y','m','parent'
        events = thinning(
            background=self.background,
            kernel=self.kernels,
            time_horizon=time_horizon,  # No fixed time horizon
            domain=self.domain,
            adjacency=self.adjacency,
            Lambda=lam_max,
            n_acc=n_acc,
            rng=rng,
            debug=debug,
            tau_max=tau_max,
        )
        
        if not events:
            if debug:
                print("[HawkesSimulator] thinning returned zero events; emitting empty frame.")
            empty_df = pd.DataFrame(columns=["t", "x", "y", "m", "is_triggered", "parent"])
            empty_labels = pd.Series(dtype=float)
            return empty_df, empty_labels

        df = (
            pd.DataFrame(events)
            .rename(columns=lambda c: "m" if c == "type" else c)
            .assign(parent=lambda d: d.get("parent", pd.Series([None] * len(d), index=d.index)))
            .loc[:, ["t", "x", "y", "m", "is_triggered", "parent"]]
            .sort_values("t")
            .reset_index(drop=True)
        )

        # Now pull off the labels
        labels = df.pop("parent")
        if debug:
            print(
                "[HawkesSimulator] simulation complete: "
                f"{len(df)} events (n={n_acc}, horizon={time_horizon})"
            )
        return df, labels
    
    @staticmethod
    def _resolve_n_horizon(
        n: Optional[int],
        horizon: Optional[float],
        *,
        default_horizon: float = _DEFAULT_HORIZON,
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Return a pair (n_acc, time_horizon) suitable for the `thinning` call.

        Exactly one of ``n`` or ``horizon`` should be provided.
        """
        if (n is not None) and (horizon is not None):
            warnings.warn(
                "Both 'n' and 'horizon' were provided; "
                "ignoring both and using default horizon.",
                UserWarning,
                stacklevel=2,
            )
            return None, default_horizon

        if (n is None) and (horizon is None):
            return None, default_horizon

        # Exactly one is set – map to the names expected by `thinning`
        return (n, None) if n is not None else (None, horizon)
    
    def _estimate_lambda_max(
        self,
        time_horizon: float,
        rng: np.random.Generator,
        n_grid_space: int = 64,
        n_grid_time: int = 64,
        safety_factor: float = 1.5,
    ) -> float:
        """
        Crude upper bound for λ(s,t) = μ(s,t) + max triggering term,
        scanning a space–time grid and adding a conservative margin.
        """
        # 1. space grid in domain
        xs = np.linspace(self.domain.x_min, self.domain.x_max, n_grid_space)
        ys = np.linspace(self.domain.y_min, self.domain.y_max, n_grid_space)
        ts = np.linspace(0.0, time_horizon, n_grid_time)

        # 2. background upper bound across marks
        m = self.adjacency.shape[0]
        lam_bg_max = 0.0
        for x in xs:
            for y in ys:
                for t in ts:
                    for mark in range(1, m + 1):
                        lam_bg_max = max(
                            lam_bg_max, self.background(np.array([x, y]), t, mark)
                        )

        # 3. conservative kernel bound if kernels provided
        lam_trig_max = 0.0
        if self.kernels is not None:
            # assume each kernel[(i,j)] attains its maximum at ds=0, dt→0+
            for (i, j), kobj in self.kernels.items():
                #print(f"kernel max for marks ({i},{j}): {kobj}")
                # you may need to expose a method on KernelBase if not present
                lam_trig_max = max(lam_trig_max, kobj.max_value())

            # effective branching contribution: sum over parents into any mark
            # adjacency[i,j] is mean offspring count; we just take max over matrix
            eta_max = float(np.max(self.adjacency))
            lam_trig_max *= eta_max

        lam_max = (lam_bg_max + lam_trig_max) * safety_factor
        if lam_max <= 0.0:
            raise ValueError(f"Estimated lambda_max non-positive: {lam_max}")
        return lam_max

