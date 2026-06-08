"""Public suite generation primitives.

The suite layer promotes the benchmark orchestration logic into the package
without replacing the HawkesNest simulator stack.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from hawkesnest.config import SimulatorConfig


@dataclass
class GenerationResult:
    """Result returned by a public benchmark suite generator."""

    events: pd.DataFrame
    config: dict[str, Any]
    metadata: dict[str, Any]
    suite_name: str
    level: str
    seed: int
    simulator_class_name: str
    export_paths: dict[str, str] = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return int(len(self.events))


def compute_adj(kernel_cfg: Mapping[str, Any], target_eta: float, tau_max: float) -> float:
    """Match the suite-script adjacency calibration exactly."""
    ktype = kernel_cfg.get("type", "separable")
    two_pi = 2.0 * math.pi

    if ktype in ("separable", "traveling_wave"):
        beta = float(kernel_cfg.get("temporal_decay", kernel_cfg.get("temporal_scale", 1.0)))
        sigma = float(kernel_cfg.get("spatial_sigma", kernel_cfg.get("sigma", 0.08)))
        temp_int = beta * (1.0 - math.exp(-tau_max / beta))
        kern_int = temp_int * two_pi * sigma**2
    elif ktype == "two_scale":
        alpha_fast = float(kernel_cfg.get("alpha_fast", 0.65))
        alpha_slow = 1.0 - alpha_fast
        beta_fast = float(kernel_cfg.get("beta_fast", 0.3))
        beta_slow = float(kernel_cfg.get("beta_slow", 1.0))
        sigma = float(kernel_cfg.get("sigma", 0.08))
        cap = min(tau_max, 5.0 * beta_slow)
        temp_int = (
            alpha_fast * beta_fast * (1.0 - math.exp(-cap / beta_fast))
            + alpha_slow * beta_slow * (1.0 - math.exp(-cap / beta_slow))
        )
        kern_int = temp_int * two_pi * sigma**2
    else:
        raise ValueError(f"compute_adj: unsupported kernel type '{ktype}'")

    if kern_int <= 0.0:
        raise ValueError(f"compute_adj: non-positive kernel integral {kern_int}")
    return target_eta / kern_int


def lambda_max_for(bg_cfg: Mapping[str, Any], adj: float, burst: int = 20) -> float:
    """Match the suite-script envelope heuristic exactly."""
    btype = bg_cfg.get("type", "constant")
    if btype == "constant":
        peak_bg = float(bg_cfg.get("rate", 1.0))
    elif btype == "regime_switching":
        lam0 = float(bg_cfg.get("lambda0", 1.0))
        max_amp = (
            max(
                float(c["amplitude"])
                for r in bg_cfg.get("regimes", [])
                for c in r.get("components", [])
            )
            if bg_cfg.get("regimes")
            else 0.0
        )
        peak_bg = lam0 + max_amp
    elif btype == "function":
        fname = bg_cfg.get("name", "")
        if fname == "cluster_mix":
            a0_bg = float(bg_cfg.get("a0", 0.0))
            amp_bg = float(bg_cfg.get("amp", 1.0))
            sigma_bg = float(bg_cfg.get("sigma", 0.1))
            centers = bg_cfg.get("centers", [])
            max_sum = 1.0
            if centers:
                grid = np.linspace(0.0, 1.0, 200)
                xx, yy = np.meshgrid(grid, grid)
                z = np.zeros_like(xx)
                inv2s2 = 1.0 / (2.0 * sigma_bg**2)
                for center in centers:
                    dx = xx - float(center[0])
                    dy = yy - float(center[1])
                    z += np.exp(-(dx**2 + dy**2) * inv2s2)
                max_sum = float(z.max())
            peak_bg = math.exp(a0_bg + amp_bg * max_sum) * 1.05
        elif fname in ("moving_gauss", "moving_gauss_slow", "moving_hotspots"):
            peak_bg = float(bg_cfg.get("a0", 0.05)) + float(bg_cfg.get("amp", 1.0)) * 10.0
        elif fname == "gabor_travel":
            peak_bg = math.exp(float(bg_cfg.get("a0", 0.0)) + float(bg_cfg.get("amp", 3.0)))
        else:
            peak_bg = 20.0
    elif btype == "hetero_ladder":
        peak_bg = float(bg_cfg.get("lambda0", 1.0)) * 5.0
    else:
        peak_bg = 20.0
    return peak_bg + burst * adj


class BaseSuite:
    """Base class for suite-specific public generators."""

    suite_name: str = "base"
    default_n_events: int = 8_000

    def levels(self) -> tuple[str, ...]:
        raise NotImplementedError

    def level_config(self, level: str) -> dict[str, Any]:
        raise NotImplementedError

    def generate(
        self,
        *,
        level: str,
        n_events: int | None = None,
        seed: int = 0,
        horizon: float | None = None,
        debug: bool = False,
        out_dir: str | Path | None = None,
    ) -> GenerationResult:
        cfg_dict = self.level_config(level)
        clean = {k: v for k, v in cfg_dict.items() if not k.startswith("_")}
        tau_max = float(clean.pop("tau_max", 5.0))
        cfg = SimulatorConfig.model_validate(clean)
        simulator = cfg.build()
        events, _ = simulator.simulate(
            horizon=horizon,
            n=self.default_n_events if n_events is None else int(n_events),
            seed=int(seed),
            debug=debug,
            tau_max=tau_max,
        )

        result = GenerationResult(
            events=events,
            config=cfg_dict,
            metadata={
                "suite": self.suite_name,
                "level": level,
                "seed": int(seed),
                "n_events": int(len(events)),
                "requested_n_events": self.default_n_events if n_events is None else int(n_events),
                "tau_max": tau_max,
                "simulator_class": f"{type(simulator).__module__}.{type(simulator).__name__}",
            },
            suite_name=self.suite_name,
            level=level,
            seed=int(seed),
            simulator_class_name=f"{type(simulator).__module__}.{type(simulator).__name__}",
        )
        if out_dir is not None:
            self.export_result(result, out_dir)
        return result

    def generate_corpus(
        self,
        *,
        levels: Iterable[str],
        seeds: Iterable[int],
        n_events: int | None = None,
        out_dir: str | Path,
        debug: bool = False,
    ) -> list[GenerationResult]:
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        results: list[GenerationResult] = []
        index: list[dict[str, Any]] = []

        for level in levels:
            for seed in seeds:
                result = self.generate(
                    level=level,
                    n_events=n_events,
                    seed=int(seed),
                    debug=debug,
                    out_dir=root / level / f"seed_{int(seed)}",
                )
                results.append(result)
                index.append({**result.metadata, "export_paths": result.export_paths})

        (root / "corpus_metadata.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        return results

    def export_result(self, result: GenerationResult, out_dir: str | Path) -> GenerationResult:
        from hawkesnest.export.csv import write_events_csv
        from hawkesnest.export.jsonl import write_events_jsonl
        from hawkesnest.export.metadata import write_metadata

        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        jsonl_path = root / "events.jsonl"
        csv_path = root / "events.csv"
        metadata_path = root / "metadata.json"

        write_events_jsonl(result.events, jsonl_path)
        write_events_csv(result.events, csv_path)
        write_metadata(result, metadata_path)

        result.export_paths.update(
            {
                "jsonl": str(jsonl_path),
                "csv": str(csv_path),
                "metadata": str(metadata_path),
            }
        )
        write_metadata(result, metadata_path)
        return result
