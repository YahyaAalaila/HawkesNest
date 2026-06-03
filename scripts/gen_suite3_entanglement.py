#!/usr/bin/env python3
"""
gen_suite3_entanglement.py
==========================
Suite 3 — Entanglement (separability) pillar sweep.

θ = wave speed v ∈ {0.00, 0.05, 0.15, 0.30}
  L0 (v=0.00): separable baseline
  L1 (v=0.05): mild non-separability
  L2 (v=0.15): visible directional streak
  L3 (v=0.30): strong traveling wave

Fixed: theta_wave=π/4, sigma=0.15, beta=0.3, cluster_mix bg, effective_eta=0.4.
Each level: 5 seeds, n=8000 events, time-split 70/10/20.
Intensity grid stored for every level (seed 0 only).

Output:
  <out_dir>/suite3_entanglement/
    sequences/
      L<k>_r<seed>.npz
    jsonl/
      L<k>/  {train,val,test}.jsonl
    ground_truth/
      L<k>_intensity_grid_r0.npz
    metadata.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.suite_utils import (  # noqa: E402
    compute_adj, lambda_max_for,
    simulate_sequence, time_split,
    save_npz, write_jsonl_splits, seq_to_jsonl_dict,
    count_stats, print_stats_table,
    compute_intensity_grid, save_intensity_video,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BETA       = 0.3
SIGMA      = 0.15  # wider clusters → visible elongation in wave direction; was 0.08
TARGET_ETA = 0.40
TAU_MAX    = 5.0
N_SEEDS    = 5
N_EVENTS   = 8_000
DOMAIN_BOUNDS = [[0.0, 1.0], [0.0, 1.0]]

# Structured background so spatial pattern is readable at all levels
BG_CFG = {
    "type": "function", "name": "cluster_mix",
    "centers": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75]],
    "sigma": 0.20, "a0": 0.5, "amp": 2.5,
}

LEVELS = {
    "L0": 0.00,   # separable: round blobs at hotspot locations
    "L1": 0.05,   # slight elongation in θ=π/4 direction
    "L2": 0.15,   # visible directional streak
    "L3": 0.30,   # strong NE-elongation; was 1.00 (exited domain)
}


def _level_cfg(v: float) -> dict:
    if v == 0.0:
        k = {"type": "separable", "temporal_decay": BETA, "spatial_sigma": SIGMA}
    else:
        k = {
            "type": "traveling_wave", "v": v,
            "theta_wave": math.pi / 4, "sigma": SIGMA,
            "temporal_scale": BETA,
        }
    adj  = compute_adj(k, TARGET_ETA, TAU_MAX)
    bg   = BG_CFG
    lmax = lambda_max_for(bg, adj)
    return {
        "domain":     {"type": "rectangle", "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
        "lambda_max": lmax,
        "backgrounds": [bg],
        "kernels":    [k],
        "adjacency":  [[adj]],
        "tau_max":    TAU_MAX,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "hawkesnest_suites")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug",   action="store_true")
    p.add_argument("--no-intensity-video", action="store_true",
                   help="Skip intensity grid and video (saves time during testing).")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    suite_dir = args.out_dir / "suite3_entanglement"
    n_seeds   = 2 if args.dry_run else N_SEEDS

    if not args.dry_run:
        suite_dir.mkdir(parents=True, exist_ok=True)

    summary      = []
    meta_entries = []
    grand_t0     = time.perf_counter()

    for label, v in LEVELS.items():
        cfg = _level_cfg(v)
        adj = cfg["adjacency"][0][0]
        print(f"\n{'='*55}")
        print(f"  {label}  v={v}  adj={adj:.2f}  lambda_max={cfg['lambda_max']:.1f}")
        print(f"{'='*55}")

        lengths: List[int] = []
        train_seqs, val_seqs, test_seqs = [], [], []

        for seed in range(n_seeds):
            times, locs, _, _ = simulate_sequence(cfg, seed=seed, n=N_EVENTS, debug=args.debug)
            lengths.append(len(times))
            print(f"  seed={seed} N={len(times)}", flush=True)

            if not args.dry_run:
                T_window = float(times[-1]) if len(times) else 1.0
                tri, vli, tei = time_split(times)

                (suite_dir / "sequences").mkdir(parents=True, exist_ok=True)
                save_npz(
                    suite_dir / "sequences" / f"{label}_r{seed}.npz",
                    times, locs, tri, vli, tei, T_window, DOMAIN_BOUNDS,
                )

                train_seqs.append(seq_to_jsonl_dict(times[tri], locs[tri]))
                val_seqs.append(seq_to_jsonl_dict(times[vli], locs[vli]))
                test_seqs.append(seq_to_jsonl_dict(times[tei], locs[tei]))

                # Intensity grid + video for every level, first seed only
                if seed == 0 and not args.no_intensity_video:
                    print(f"  Computing intensity grid for {label} ...", flush=True)
                    try:
                        grid = compute_intensity_grid(cfg, times, locs)
                        gt_dir = suite_dir / "ground_truth"
                        gt_dir.mkdir(parents=True, exist_ok=True)
                        np.savez(gt_dir / f"{label}_intensity_grid_r0.npz", **grid)
                        print(f"  Grid shape: {grid['lambda_'].shape}")
                        vid = save_intensity_video(
                            grid, gt_dir / f"{label}_intensity_r0", label=label,
                        )
                        print(f"  Video saved: {vid.name}")
                    except Exception as e:
                        print(f"  [WARN] Grid/video failed: {e}")

        stats = count_stats(lengths)
        print(f"  Stats: mean={stats['mean']:.0f}  p95={stats['p95']:.0f}")
        summary.append({"label": label, "stats": stats})
        meta_entries.append({"label": label, "v": v, "adj": adj, "n_seeds": n_seeds, "stats": stats})

        if not args.dry_run and train_seqs:
            write_jsonl_splits(suite_dir / "jsonl" / label, train_seqs, val_seqs, test_seqs)

    if not args.dry_run:
        (suite_dir / "metadata.json").write_text(json.dumps({
            "suite": "suite3_entanglement",
            "theta": "wave_speed_v",
            "levels": meta_entries,
        }, indent=2))

    elapsed = time.perf_counter() - grand_t0
    print_stats_table(summary)
    print(f"Total elapsed: {elapsed:.1f}s")
    if args.dry_run:
        print("[dry-run] No files written.")
    else:
        print(f"[suite3] Written to: {suite_dir.resolve()}")


if __name__ == "__main__":
    main()
