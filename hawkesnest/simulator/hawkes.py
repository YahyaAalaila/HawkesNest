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
from yaml import warnings

_DEFAULT_HORIZON = 100.0        # module-level constant
_UNSET = object()               # sentinel if you ever need it

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
        lambda_max: float = 1.0,
    ) -> None:
        super().__init__()
        # Defaults: homogeneous Poisson
        self.domain = domain
        self.background = background or (lambda s, t, m=None: 1.0)
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
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Simulate n events. Returns DataFrame of events and Series of parent labels.
        """
        
        rng = np.random.default_rng(seed)
        n_acc, time_horizon = self._resolve_n_horizon(n, horizon)
        
        # Thinning returns list of dicts with keys 't','x','y','m','parent'
        events = thinning(
            background=self.background,
            kernel=self.kernels,
            time_horizon=time_horizon,  # No fixed time horizon
            domain=self.domain,
            adjacency=self.adjacency,
            Lambda=self.lambda_max,
            n_acc=n_acc,
            rng=rng,
        )
        
        df = (
            pd.DataFrame(events)
            .rename(columns=lambda c: "m" if c == "type" else c)
            .assign(parent=lambda d: d.get("parent", pd.Series([None]*len(d), index=d.index)))
            .loc[:, ["t", "x", "y", "m", "is_triggered", "parent"]]
            .sort_values("t")
            .reset_index(drop=True)
        )

        # Now pull off the labels
        labels = df.pop("parent")
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
