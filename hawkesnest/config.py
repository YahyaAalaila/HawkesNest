"""High-level configuration layer for the Hawkes simulator.

The public entry point is :class:`SimulatorConfig.build()` which assembles
and returns a fully-wired :class:`hawkesnest.simulator.HawkesSimulator`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, model_validator

from hawkesnest.config_factory import DomainConfig, BackgroundCfg, KernelCfg
from hawkesnest.simulator import HawkesSimulator


class SimulatorConfig(BaseModel):
    domain: DomainConfig
    backgrounds: list[BackgroundCfg]
    kernels: list[list[KernelCfg]]
    adjacency: Optional[list[list[float]]] = None
    lambda_max: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _broadcast(cls, data):
        # number of marks
        A = np.asarray(data.get("adjacency", [[1.0]]), float)
        M = A.shape[0]

        # --- backgrounds ---------------------------------------------------
        bgs = data["backgrounds"]
        if len(bgs) == 1:
            data["backgrounds"] = bgs * M
        elif len(bgs) != M:
            raise ValueError(f"`backgrounds` length must be 1 or {M}")

        # --- kernels -------------------------------------------------------
        raw = data["kernels"]
        if isinstance(raw, dict):
            grid = [[raw]]
        elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
            grid = [raw]
        elif isinstance(raw, list) and raw and isinstance(raw[0], list):
            grid = raw
        else:
            raise ValueError("`kernels` must be a dict or list")

        R, C = len(grid), len(grid[0])
        if   R == 1 and C == 1: grid = [[grid[0][0]] * M for _ in range(M)]
        elif R == 1 and C == M: grid = [grid[0]        for _ in range(M)]
        elif R == M and C == 1: grid = [[row[0]] * M   for row in grid]
        elif R == M and C == M: pass
        else:
            raise ValueError(f"`kernels` shape {R}×{C} cannot broadcast to {M}×{M}")
        data["kernels"] = grid

        return data

    def build(self) -> HawkesSimulator:
        """Build and return a fully-wired HawkesSimulator."""
        domain = self.domain.build()

        # Build backgrounds — hetero_ladder needs the domain object
        bg_objs = []
        for i, bg_cfg in enumerate(self.backgrounds):
            if bg_cfg.type == "hetero_ladder":
                bg_objs.append(bg_cfg.build(domain=domain))
            else:
                bg_objs.append(bg_cfg.build(idx=i))

        def bg_fn(space: np.ndarray, t: float, mark: int) -> float:
            if mark < 1 or mark > len(bg_objs):
                raise ValueError(f"mark {mark} out of range [1, {len(bg_objs)}]")
            return float(bg_objs[mark - 1](space, t))

        # Build kernel dict  {(i+1, j+1): KernelBase}
        kernel_dict = {
            (i + 1, j + 1): cfg.build()
            for i, row in enumerate(self.kernels)
            for j, cfg in enumerate(row)
        }

        # Adjacency matrix
        M = len(bg_objs)
        adj = (
            np.array(self.adjacency, dtype=float)
            if self.adjacency is not None
            else np.eye(M) * 0.3
        )

        return HawkesSimulator(
            domain=domain,
            background=bg_fn,
            kernels=kernel_dict,
            adjacency=adj,
            lambda_max=self.lambda_max,  # None → simulator will estimate
        )
