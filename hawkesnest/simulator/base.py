# hawkesnest/simulator/base.py
"""
Abstract base class for all simulators in hawkesnest.

A simulator generates spatio-temporal point events given a specified
complexity configuration (background, kernels, graph, topology) and returns
both the event data and, optionally, ground-truth labels.
"""
from __future__ import annotations

import abc
from typing import Optional, Tuple
import pandas as pd

class SimulatorBase(abc.ABC):
    """
    Defines the interface for spatio-temporal point process simulators.
    """
    @abc.abstractmethod
    def simulate(
        self,
        n: int,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate n events according to the configured generative model.

        Parameters:
            n: int
                The number of events to simulate (or approximate target).
            seed: Optional[int]
                Random seed for reproducibility.

        Returns:
            pd.DataFrame
                A DataFrame with columns ['s_x', 's_y', 't', 'm'], sorted by 't'.
        """
        pass

    def __call__(
        self,
        n: int,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Syntactic sugar to allow the simulator object to be called
        directly like a function.
        """
        return self.simulate(n=n, seed=seed)
