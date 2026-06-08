"""Suite 4: background heterogeneity benchmark."""
from __future__ import annotations

from typing import Any

from hawkesnest.suites.base import BaseSuite, compute_adj, lambda_max_for


class HeterogeneitySuite(BaseSuite):
    """Public generator for the paper's suite4 heterogeneity ladder."""

    suite_name = "suite4_heterogeneity"
    default_n_events = 8_000

    beta = 0.3
    sigma = 0.15
    target_eta = 0.40
    tau_max = 5.0
    domain_bounds = [[0.0, 1.0], [0.0, 1.0]]

    kernel_config: dict[str, Any] = {
        "type": "separable",
        "temporal_decay": beta,
        "spatial_sigma": sigma,
    }

    level_backgrounds: dict[str, dict[str, Any]] = {
        "H0": {"type": "constant", "rate": 2.4},
        "H1": {
            "type": "function",
            "name": "cluster_mix",
            "centers": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75]],
            "sigma": 0.20,
            "a0": 0.5,
            "amp": 2.5,
        },
        "H2": {
            "type": "function",
            "name": "moving_hotspots",
            "start": [0.5, 0.5],
            "v": [0.0003, 0.0002],
            "sigma": 0.08,
            "a0": 2.0,
            "amp": 1.0,
        },
        "H3": {
            "type": "function",
            "name": "gabor_travel",
            "a0": 1.0,
            "amp": 2.0,
            "freq": 1.5,
            "freq_t": 0.003,
            "sigma": 0.40,
            "start": [0.5, 0.5],
        },
    }

    _adj_cache: float | None = None

    def levels(self) -> tuple[str, ...]:
        return tuple(self.level_backgrounds)

    def _adj(self) -> float:
        if self._adj_cache is None:
            self._adj_cache = compute_adj(self.kernel_config, self.target_eta, self.tau_max)
        return self._adj_cache

    def level_config(self, level: str) -> dict[str, Any]:
        if level not in self.level_backgrounds:
            raise ValueError(f"Unknown heterogeneity level {level!r}; expected one of {self.levels()}")

        bg = self.level_backgrounds[level]
        adj = self._adj()
        return {
            "domain": {
                "type": "rectangle",
                "x_min": 0.0,
                "x_max": 1.0,
                "y_min": 0.0,
                "y_max": 1.0,
            },
            "lambda_max": lambda_max_for(bg, adj),
            "backgrounds": [bg],
            "kernels": [self.kernel_config],
            "adjacency": [[adj]],
            "tau_max": self.tau_max,
            "_level": level,
            "_background": bg,
        }
