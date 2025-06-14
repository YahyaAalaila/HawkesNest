# hawkesnest/simulator/hawkes.py
"""
HawkesSimulator: implements Ogata's thinning algorithm for multivariate
space–time Hawkes processes using injected background and kernels.
"""
from __future__ import annotations
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd

from hawkesnest.simulator.base import SimulatorBase
from hawkesnest.background.base import BackgroundBase
from hawkesnest.kernel.base import KernelBase
from hawkesnest.utils.thinning import thinning
from hawkesnest.domain.base import SpatialDomain
from hawkesnest.domain.standard import RectangleDomain
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
        self.kernels = kernels #or {} # Default to single type with self-loop
        self.adjacency = adjacency if adjacency is not None else np.array([[0.5, 0], [0, 0.2]])
        self.lambda_max = lambda_max

    def simulate(
        self,
        n: int,
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Simulate n events. Returns DataFrame of events and Series of parent labels.
        """
        rng = np.random.default_rng(seed)
        # Thinning returns list of dicts with keys 't','x','y','m','parent'
        events = thinning(
            background=self.background,
            kernel=self.kernels,
            time_horizon=None,  # No fixed time horizon
            domain=self.domain,
            adjacency=self.adjacency,
            Lambda=self.lambda_max,
            n_acc=n,
            rng=rng,
        )
        df = pd.DataFrame(events)
        # Rename 'type' to 'm' if needed
        if 'type' in df.columns:
            df = df.rename(columns={'type': 'm'})
        # Extract labels if available
        labels = df['parent'] if 'parent' in df.columns else pd.Series([None]*len(df))
        # Finalize events DataFrame
        df = df[['t', 'x', 'y', 'm']]
        df = df.sort_values('t').reset_index(drop=True)
        return df, labels