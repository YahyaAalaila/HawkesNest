#!/usr/bin/env python3
"""
gen_suite4_heterogeneity.py
===========================
Suite 4 — Heterogeneity (background structure) pillar sweep.

θ = background complexity level ∈ {H0, H1, H2, H3}
  H0: constant background (homogeneous baseline)
  H1: stationary 3-component GMM hotspots
  H2: single traveling Gaussian hotspot
  H3: Gabor traveling-wave spatial pattern

Fixed: separable kernel, beta=0.3, sigma=0.08, effective_eta=0.4.
Each level: 5 seeds, n=8000 events, time-split 70/10/20.

Output:
  <out_dir>/suite4_heterogeneity/
    sequences/  H<k>_r<seed>.npz
    jsonl/      H<k>/  {train,val,test}.jsonl
    metadata.json
"""
from __future__ import annotations

import argparse
import json
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
BETA       = 0.3
SIGMA      = 0.15  # wider clusters; was 0.08
TARGET_ETA = 0.40
TAU_MAX    = 5.0
N_SEEDS    = 5
N_EVENTS   = 8_000
DOMAIN_BOUNDS = [[0.0, 1.0], [0.0, 1.0]]

KERNEL_CFG = {"type": "separable", "temporal_decay": BETA, "spatial_sigma": SIGMA}
_ADJ = None


def _adj() -> float:
    global _ADJ
    if _ADJ is None:
        _ADJ = compute_adj(KERNEL_CFG, TARGET_ETA, TAU_MAX)
    return _ADJ


LEVEL_BGS = {
    # H0: flat background → pure Hawkes clustering (random blobs, no spatial structure)
    "H0": {"type": "constant", "rate": 2.4},

    # H1: 3 fixed hotspots with high contrast (12× peak/floor)
    "H1": {
        "type": "function", "name": "cluster_mix",
        "centers": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75]],
        "sigma": 0.20, "a0": 0.5, "amp": 2.5,
    },

    # H2: moving hotspots — high floor (a0=2.0) ensures ≥8000 events within horizon;
    #     sharp peaks (amp×10=10 on top of floor) give 6× contrast (clearly visible drift)
    "H2": {
        "type": "function", "name": "moving_hotspots",
        "start": [0.5, 0.5], "v": [0.0003, 0.0002],
        "sigma": 0.08, "a0": 2.0, "amp": 1.0,
    },

    # H3: diagonal Gabor stripes (static spatial pattern; ft very low → slow temporal sweep)
    "H3": {
        "type": "function", "name": "gabor_travel",
        "a0": 1.0, "amp": 2.0, "freq": 1.5,
        "freq_t": 0.003,   # ~6 temporal cycles over T=2000; slow enough to see structure
        "sigma": 0.40, "start": [0.5, 0.5],
    },
}


def _level_cfg(bg: dict) -> dict:
    adj  = _adj()
    lmax = lambda_max_for(bg, adj)
    return {
        "domain":     {"type": "rectangle", "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
        "lambda_max": lmax,
        "backgrounds": [bg],
        "kernels":    [KERNEL_CFG],
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
    suite_dir = args.out_dir / "suite4_heterogeneity"
    n_seeds   = 2 if args.dry_run else N_SEEDS

    if not args.dry_run:
        suite_dir.mkdir(parents=True, exist_ok=True)

    summary      = []
    meta_entries = []
    grand_t0     = time.perf_counter()

    for label, bg in LEVEL_BGS.items():
        cfg = _level_cfg(bg)
        adj = cfg["adjacency"][0][0]
        print(f"\n{'='*55}")
        print(f"  {label}  bg_type={bg['type']}  adj={adj:.2f}  lmax={cfg['lambda_max']:.1f}")
        print(f"{'='*55}")

        lengths: List[int] = []
        train_seqs, val_seqs, test_seqs = [], [], []
        seqs_by_seed: dict = {}

        for seed in range(n_seeds):
            times, locs, _, _ = simulate_sequence(cfg, seed=seed, n=N_EVENTS, debug=args.debug)
            lengths.append(len(times))
            seqs_by_seed[seed] = (times, locs)
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

        stats = count_stats(lengths)
        print(f"  Stats: mean={stats['mean']:.0f}  p95={stats['p95']:.0f}")
        summary.append({"label": label, "stats": stats})
        meta_entries.append({"label": label, "bg": bg, "adj": adj, "n_seeds": n_seeds, "stats": stats})

        if not args.dry_run and train_seqs:
            write_jsonl_splits(suite_dir / "jsonl" / label, train_seqs, val_seqs, test_seqs)

        # Intensity grid + video for seed=0
        if not args.dry_run and not args.no_intensity_video:
            times0, locs0 = seqs_by_seed[0]
            print(f"  Computing intensity grid for {label} seed=0 ...", flush=True)
            try:
                grid = compute_intensity_grid(cfg, times0, locs0)
                gt_dir = suite_dir / "ground_truth"
                gt_dir.mkdir(parents=True, exist_ok=True)
                np.savez(gt_dir / f"{label}_intensity_grid_r0.npz", **grid)
                vid = save_intensity_video(
                    grid, gt_dir / f"{label}_intensity_r0", label=label,
                )
                print(f"  Video saved: {vid.name}")
            except Exception as e:
                print(f"  [WARN] Grid/video failed: {e}")

    if not args.dry_run:
        (suite_dir / "metadata.json").write_text(json.dumps({
            "suite": "suite4_heterogeneity",
            "theta": "background_complexity",
            "kernel": KERNEL_CFG,
            "levels": meta_entries,
        }, indent=2))

    elapsed = time.perf_counter() - grand_t0
    print_stats_table(summary)
    print(f"Total elapsed: {elapsed:.1f}s")
    if args.dry_run:
        print("[dry-run] No files written.")
    else:
        print(f"[suite4] Written to: {suite_dir.resolve()}")


if __name__ == "__main__":
    main()
