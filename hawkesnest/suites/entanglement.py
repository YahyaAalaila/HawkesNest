"""Suite 3: entanglement/separability benchmark."""
from __future__ import annotations

import math
from typing import Any

from hawkesnest.suites.base import BaseSuite, compute_adj, lambda_max_for


class EntanglementSuite(BaseSuite):
    """Public generator for the paper's suite3 entanglement ladder."""

    suite_name = "suite3_entanglement"
    default_n_events = 8_000

    beta = 0.3
    sigma = 0.15
    target_eta = 0.40
    tau_max = 5.0
    domain_bounds = [[0.0, 1.0], [0.0, 1.0]]

    background_config: dict[str, Any] = {
        "type": "function",
        "name": "cluster_mix",
        "centers": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75]],
        "sigma": 0.20,
        "a0": 0.5,
        "amp": 2.5,
    }

    level_speeds: dict[str, float] = {
        "L0": 0.00,
        "L1": 0.05,
        "L2": 0.15,
        "L3": 0.30,
    }

    def levels(self) -> tuple[str, ...]:
        return tuple(self.level_speeds)

    def level_config(self, level: str) -> dict[str, Any]:
        if level not in self.level_speeds:
            raise ValueError(f"Unknown entanglement level {level!r}; expected one of {self.levels()}")

        speed = self.level_speeds[level]
        if speed == 0.0:
            kernel = {
                "type": "separable",
                "temporal_decay": self.beta,
                "spatial_sigma": self.sigma,
            }
        else:
            kernel = {
                "type": "traveling_wave",
                "v": speed,
                "theta_wave": math.pi / 4,
                "sigma": self.sigma,
                "temporal_scale": self.beta,
            }

        adj = compute_adj(kernel, self.target_eta, self.tau_max)
        bg = self.background_config
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
            "kernels": [kernel],
            "adjacency": [[adj]],
            "tau_max": self.tau_max,
            "_level": level,
            "_wave_speed": speed,
        }
