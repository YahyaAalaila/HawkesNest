from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hawkesnest.experiments.complexity import run_complexity_sweep, run_complexity_sweep_4d


def _parse_grid(arg: str) -> np.ndarray:
    return np.array([float(x) for x in arg.split(",")]) if arg else np.array([])


def main():
    parser = argparse.ArgumentParser(description="Run complexity pillar sweeps.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["het", "ent", "joint", "topo", "graph", "full"],
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--T", type=float, default=50.0)
    parser.add_argument("--lambda0", type=float, default=40)

    parser.add_argument(
        "--Marks",
        type=int,
        default=3,
        help="Number of event types/marks used when theta_graph > 0 (graph pillar).",
    )

    parser.add_argument("--theta-het-grid", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--theta-ent-grid", type=str, default="0.0,0.25,0.5,0.75,0.95")
    parser.add_argument("--theta-topo-grid", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--theta-graph-grid", type=str, default="0.0,0.25,0.5,0.75,0.95")

    # Parallelization controls (only used in mode=full)
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run full 4D sweep in parallel (ProcessPoolExecutor).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max workers for parallel full sweep (default: None -> executor default).",
    )

    # Diagnostics: MUST be handled inside run_complexity_sweep_4d for real effect,
    # but we define CLI knobs here.
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Print progress every k completed configs (mode=full).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug prints per config (recommended only with --mode full and WITHOUT --parallel).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on first worker error (mode=full).",
    )
    parser.add_argument(
        "--worker-timeout",
        type=float,
        default=None,
        help="Optional per-config timeout in seconds (mode=full, parallel only).",
    )

    args = parser.parse_args()
    print("[INFO] starting complexity sweeps")
    print(f"[INFO] parsed args: {args}")

    M = args.Marks
    theta_het_grid = _parse_grid(args.theta_het_grid)
    theta_ent_grid = _parse_grid(args.theta_ent_grid)
    theta_topo_grid = _parse_grid(args.theta_topo_grid)
    theta_graph_grid = _parse_grid(args.theta_graph_grid)

    # Place outputs under a mode-specific folder for clarity
    out_dir = args.out.parent / f"{args.out.stem}_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out.name

    if args.mode == "full":
        run_complexity_sweep_4d(
            out_dir=out_dir,
            replicates=args.replicates,
            T=args.T,
            lambda0=args.lambda0,
            theta_het_grid=theta_het_grid,
            theta_ent_grid=theta_ent_grid,
            theta_topo_grid=theta_topo_grid,
            theta_graph_grid=theta_graph_grid,
            results_filename=args.out.name,
            num_marks=M,
            use_parallel=args.parallel,
            max_workers=args.max_workers,

            # Diagnostics / safety (must be implemented in hawkesnest/experiments/complexity.py)
            debug=args.debug,
            progress_every=args.progress_every,
            fail_fast=args.fail_fast,
            worker_timeout=args.worker_timeout,
        )
        print(f"[OK] completed 4D sweeps -> {out_dir}")
    else:
        run_complexity_sweep(
            mode=args.mode,
            out=out_path,
            replicates=args.replicates,
            T=args.T,
            lambda0=args.lambda0,
            theta_het_grid=theta_het_grid,
            theta_ent_grid=theta_ent_grid,
            theta_topo_grid=theta_topo_grid,
            theta_graph_grid=theta_graph_grid,
            num_marks=M,
        )
        print(f"[OK] completed sweeps for mode={args.mode} -> {out_path}")


if __name__ == "__main__":
    main()
