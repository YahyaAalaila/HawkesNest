from __future__ import annotations
import copy
from typing import Sequence, Dict, Any, Optional


import torch
import numpy as np

from hawkesnest.datasets.base import SpatioTemporalDataset, StdScaler          # your base class
from hawkesnest.utils.simulate import simulate_pillar                # the helper above
from hawkesnest.metrics import alpha_ent_kl

class EntanglementDataset(SpatioTemporalDataset):
    """
    Pre-baked synthetic datasets that primarily vary **spatio-temporal
    entanglement**.  The base template `data_templates/entanglement.yml`
    encodes the *low* setting; we override only the few knobs that need to
    change for *mid* and *high*.
    """

    _LEVEL_OVERRIDES: dict[str, Dict[str, Any]] = {
        "low":  {},                                     # just use the template
        "mid": {
            # moderate cross-term in the background …
            "backgrounds": [
                {"type":"function","name":"moving_gauss", "start":[[0.2,0.2],[0.5, 0.5], [0.3,0.3]],
                 "v":[[2,1],[5,9],[4,4]],
                 "sigma":70,"base":[1, 0.5, 0.5]}
            ]
        },
        "high": {
            "backgrounds": [
                {
         "type": "function", "name": "gabor_travel",
         "a0": -2, "amp": 0.05, "fx": 0.2, "fy": 0, "ft": 0.2, "sigma": 0.5}
            ]
        },
    }

    def __init__(
        self,
        level: str = "low",
        *,
        overrides: Optional[Dict[str, Any]] = None,
        n_events: int = 600,
        horizon: Optional[float] = None,
        n_realisations: int = 0,
        seed: int = 0,
        fit_scaler: bool = True,
        normalise: bool = True,
        scaler: "StdScaler | None" = None,
    ):
        if level not in self._LEVEL_OVERRIDES:
            raise ValueError(f"level must be one of {list(self._LEVEL_OVERRIDES)}")

        pres = copy.deepcopy(self._LEVEL_OVERRIDES[level])
        if overrides:
            self._deep_merge(pres, overrides)

        
        self.seqs = simulate_pillar(
            pillar="entanglement",
            overrides=pres,
            n_realisations=n_realisations,
            n_events=n_events,
            horizon=horizon,
            seed=seed,
        )

        super().__init__(
            sequences=self.seqs,
            scaler=scaler,
            fit_scaler=fit_scaler,
            normalise=normalise,
        )
        
    def complexity(self, which = "ent") -> float:
        """
        Return a measure of the dataset's complexity.
        Here, we use the number of events as a proxy for complexity.
        """
        if len(self.seqs) == 1: events = self.seqs[0] 
        else: events = torch.cat(self.seqs, dim=0)
        
        if which == "ent":
            # entanglement complexity: number of events
            comp = alpha_ent_kl(events, bw_joint=0.4, bw_space=0.5, bw_time=0.25)
        # TODO: Add the rest metric
        return comp
        
    @staticmethod
    def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        """Recursively merge *src* into *dst* (in-place)."""
        for k, v in src.items():
            if isinstance(v, dict) and k in dst and isinstance(dst[k], dict):
                EntanglementDataset._deep_merge(dst[k], v)
            else:
                dst[k] = v
