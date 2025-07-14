"""High-level configuration layer for the Hawkes simulator.

The design follows **one component = one config class**.  Each config knows
how to *build* its runtime object while performing thorough validation and
applying safe fall-backs when the YAML file leaves a field unspecified.

The public entry point is :class:`SimulatorConfig.build()` which assembles
and returns a fully-wired :class:`hawkesnest.simulator.HawkesSimulator`.
"""
import numpy as np
from typing import Optional

from pydantic import BaseModel, model_validator

from hawkesnest.config_factory import DomainConfig, BackgroundCfg, KernelCfg
from hawkesnest.simulator import HawkesSimulator

from hawkesnest.utils.thinning import auto_lambda


class SimulatorConfig(BaseModel):
    domain: DomainConfig
    backgrounds: list[BackgroundCfg]
    kernels: list[list[KernelCfg]]
    adjacency: Optional[list[list[float]]]
    lambda_max: float

    @model_validator(mode="before")
    def _broadcast(cls, data):
        # 0. number of marks
        A = np.asarray(data.get("adjacency", [[1.0]]), float)
        M = A.shape[0]

        # 1. BACKGROUNDS  ----------------------------------------------------
        bgs = data["backgrounds"]
        if len(bgs) == 1:
            data["backgrounds"] = bgs * M
        elif len(bgs) != M:
            raise ValueError(f"`backgrounds` length must be 1 or {M}")

        # 2. KERNELS  --------------------------------------------------------
        raw = data["kernels"]

        # --- normalise to list-of-lists shape ------------------------------
        # a) single dict  -> [[dict]]
        if isinstance(raw, dict):
            grid = [[raw]]
        # b) flat list    -> [list]
        elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
            grid = [raw]
        # c) already nested list
        elif isinstance(raw, list) and raw and isinstance(raw[0], list):
            grid = raw
        else:
            raise ValueError("`kernels` must be dict or list")

        R, C = len(grid), len(grid[0])
        if   R == 1 and C == 1: grid = [[grid[0][0]] * M for _ in range(M)]           # 1×1
        elif R == 1 and C == M: grid = [grid[0]        for _ in range(M)]             # 1×M
        elif R == M and C == 1: grid = [[row[0]] * M   for row in grid]               # M×1
        elif R == M and C == M: pass                                                   # M×M
        else:
            raise ValueError(f"`kernels` shape {R}×{C} cannot broadcast to {M}×{M}")
        data["kernels"] = grid
        
        return data
    

    def build(self) -> HawkesSimulator:
        """Build a HawkesSimulator from this configuration."""
        self.domain = self.domain.build()
        # build and dispatch backgrounds
        bg_objs, lamdas = [], []
        for i, bg in enumerate(self.backgrounds):
            bg_i = bg.build(i)
            lamdas.append(auto_lambda(bg_i))
            bg_objs.append(bg_i)
        #bg_objs = [ bg.build(i) for i, bg in enumerate(self.backgrounds) ]
        def bg_fn(space: np.ndarray, t: float, mark: int) -> float:
            if mark < 1 or mark > len(bg_objs):
                raise ValueError(f"mark {mark} out of range [1, {len(bg_objs)}]")
            return float(bg_objs[mark-1](space, t))
        
        
        # build and flatten kernels
        kernel_dict = {
            (i+1, j+1): cfg.build()
            for i, row in enumerate(self.kernels)
            for j, cfg in enumerate(row)
        }
    
        # ker = kernel_dict[1, 1]
        # lamdas = lamdas + auto_lambda(ker)
        print(f"[DEBUG] lambda_max before = {self.lambda_max}")
        self.lambda_max = 10 * max(lamdas)
        print(f"[DEBUG] lambda_max after = {self.lambda_max}")
        # adjacency
        M = len(bg_objs)
        adj = np.array(self.adjacency, dtype=float) if self.adjacency else np.eye(M)
        return HawkesSimulator(
            domain=self.domain,
            background=bg_fn,
            kernels=kernel_dict,
            adjacency=adj,
            lambda_max=self.lambda_max,
        )
