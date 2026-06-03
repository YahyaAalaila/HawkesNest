#!/usr/bin/env python3
"""
validate_suites.py
==================
Validation script for all generated HawkesNest benchmark suites.

For each suite:
  1. Loads every .npz file → checks shapes, dtypes, time ordering, location bounds, no NaN
  2. Checks split indices are disjoint and cover [0, N)
  3. Loads train.jsonl files → checks format, event count, location bounds
  4. Prints per-suite count statistics
  5. Saves a spatial scatter figure (one panel per level, training events)

Usage:
  python scripts/validate_suites.py --suites-dir hawkesnest_suites/
  python scripts/validate_suites.py --suites-dir hawkesnest_suites/ --plot-dir validation_plots/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Checkers
# ---------------------------------------------------------------------------

def _check_npz(path: Path) -> Tuple[bool, List[str]]:
    """Return (ok, errors)."""
    errors = []
    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        return False, [f"Cannot load: {e}"]

    # Required fields
    for field in ("times", "locations", "train_idx", "val_idx", "test_idx", "T_window", "domain_bounds"):
        if field not in d:
            errors.append(f"Missing field: {field}")

    if errors:
        return False, errors

    times = d["times"]
    locs  = d["locations"]
    N     = len(times)

    if locs.shape != (N, 2):
        errors.append(f"locations shape {locs.shape} != ({N},2)")
    if not np.all(np.isfinite(times)):
        errors.append("times contains non-finite values")
    if not np.all(np.isfinite(locs)):
        errors.append("locations contains non-finite values")
    if not np.all(np.diff(times) >= 0):
        errors.append("times not sorted")
    if np.any(locs < -0.01) or np.any(locs > 1.01):
        errors.append(f"locations out of [0,1]^2: min={locs.min():.4f} max={locs.max():.4f}")

    # Split indices
    tri = d["train_idx"].astype(int)
    vli = d["val_idx"].astype(int)
    tei = d["test_idx"].astype(int)
    all_idx = np.concatenate([tri, vli, tei])
    if len(all_idx) != len(np.unique(all_idx)):
        errors.append("Split indices not disjoint")
    if len(all_idx) != N:
        errors.append(f"Split indices cover {len(all_idx)} events, expected {N}")

    return len(errors) == 0, errors


def _check_jsonl_split(path: Path) -> Tuple[bool, List[str], List[int]]:
    """Return (ok, errors, lengths)."""
    errors  = []
    lengths = []
    try:
        with path.open() as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                if "times" not in obj or "locations" not in obj:
                    errors.append(f"Line {i}: missing times or locations")
                    continue
                t = obj["times"]
                l = obj["locations"]
                if len(t) != len(l):
                    errors.append(f"Line {i}: len(times)={len(t)} != len(locations)={len(l)}")
                lengths.append(len(t))
                for xy in l:
                    if len(xy) != 2:
                        errors.append(f"Line {i}: location has {len(xy)} coords, expected 2")
                        break
    except Exception as e:
        errors.append(f"Cannot read: {e}")
    return len(errors) == 0, errors, lengths


# ---------------------------------------------------------------------------
# Per-suite validation
# ---------------------------------------------------------------------------

def _validate_suite(suite_dir: Path, plot_dir: Optional[Path]) -> Dict[str, Any]:
    suite_name = suite_dir.name
    report     = {"suite": suite_name, "npz": [], "jsonl": [], "ok": True}

    npz_files = sorted((suite_dir / "sequences").glob("*.npz"))
    if not npz_files:
        print(f"  [WARN] No NPZ files in {suite_dir/'sequences'}")
        report["ok"] = False
        return report

    # NPZ validation
    all_lengths = []
    for p in npz_files:
        ok, errs = _check_npz(p)
        all_lengths.append(int(np.load(p)["times"].shape[0]))
        entry = {"file": p.name, "ok": ok, "errors": errs}
        report["npz"].append(entry)
        if not ok:
            report["ok"] = False
            for e in errs:
                print(f"  [ERROR] {p.name}: {e}")

    arr = np.array(all_lengths, dtype=float)
    report["count_stats"] = {
        "n": len(all_lengths), "mean": float(arr.mean()), "std": float(arr.std()),
        "min": int(arr.min()), "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)), "max": int(arr.max()),
    }
    s = report["count_stats"]
    print(f"  NPZ: {len(npz_files)} files | N mean={s['mean']:.0f}  p95={s['p95']:.0f}  "
          f"min={s['min']}  max={s['max']}")

    # JSONL validation
    jsonl_root = suite_dir / "jsonl"
    if jsonl_root.exists():
        for split_file in sorted(jsonl_root.rglob("train.jsonl")):
            ok, errs, lens = _check_jsonl_split(split_file)
            rel = split_file.relative_to(suite_dir)
            entry = {"file": str(rel), "ok": ok, "errors": errs, "n_seqs": len(lens)}
            report["jsonl"].append(entry)
            if not ok:
                report["ok"] = False
                for e in errs:
                    print(f"  [ERROR] {rel}: {e}")

        n_checked = len(report["jsonl"])
        n_ok      = sum(1 for e in report["jsonl"] if e["ok"])
        print(f"  JSONL: {n_ok}/{n_checked} train.jsonl files OK")
    else:
        print(f"  [INFO] No jsonl/ directory in {suite_dir.name}")

    # Scatter plot
    if plot_dir is not None:
        try:
            _plot_scatter(suite_dir, npz_files, plot_dir)
        except Exception as e:
            print(f"  [WARN] Plot failed: {e}")

    status = "OK" if report["ok"] else "FAIL"
    print(f"  --> {status}")
    return report


def _plot_scatter(suite_dir: Path, npz_files: List[Path], plot_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)

    # Group by level prefix (everything before last _r<N>)
    from collections import defaultdict
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in npz_files:
        stem = p.stem
        # strip _r<digit>
        prefix = stem
        parts  = stem.rsplit("_r", 1)
        if len(parts) == 2 and parts[1].isdigit():
            prefix = parts[0]
        groups[prefix].append(p)

    n_groups = len(groups)
    if n_groups == 0:
        return

    ncols = min(n_groups, 4)
    nrows = (n_groups + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for ax_flat, (prefix, paths) in zip(axes.flat, groups.items()):
        # First realization, training events
        d        = np.load(paths[0], allow_pickle=True)
        locs     = d["locations"]
        train_id = d["train_idx"].astype(int)
        locs_tr  = locs[train_id]
        ax_flat.scatter(locs_tr[:, 0], locs_tr[:, 1], s=2, alpha=0.3, rasterized=True)
        ax_flat.set_xlim(0, 1); ax_flat.set_ylim(0, 1)
        ax_flat.set_title(prefix, fontsize=9)
        ax_flat.set_aspect("equal")

    # Hide unused axes
    for ax in axes.flat[n_groups:]:
        ax.set_visible(False)

    fig.suptitle(suite_dir.name, fontsize=11)
    fig.tight_layout()
    out = plot_dir / f"{suite_dir.name}_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suites-dir", type=Path, default=_REPO_ROOT / "hawkesnest_suites")
    p.add_argument("--plot-dir",   type=Path, default=None,
                   help="Directory for scatter plots. If omitted, no plots are saved.")
    p.add_argument("--suite",      nargs="+", default=None,
                   help="Validate only these suite directories (e.g. suite1_branching)")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    suites_dir = Path(args.suites_dir)

    if not suites_dir.exists():
        print(f"[ERROR] Suites directory not found: {suites_dir}")
        sys.exit(1)

    suite_dirs = sorted(d for d in suites_dir.iterdir() if d.is_dir() and d.name.startswith("suite"))
    if args.suite:
        suite_dirs = [d for d in suite_dirs if d.name in args.suite]

    if not suite_dirs:
        print(f"[ERROR] No suite directories found in {suites_dir}")
        sys.exit(1)

    all_reports = []
    for sd in suite_dirs:
        print(f"\n{'='*55}")
        print(f"  Validating: {sd.name}")
        print(f"{'='*55}")
        r = _validate_suite(sd, args.plot_dir)
        all_reports.append(r)

    # Final summary
    print(f"\n{'='*55}")
    print("VALIDATION SUMMARY")
    print(f"{'='*55}")
    n_ok   = sum(1 for r in all_reports if r["ok"])
    n_fail = len(all_reports) - n_ok
    for r in all_reports:
        status = "OK  " if r["ok"] else "FAIL"
        s = r.get("count_stats", {})
        stats_str = (
            f"N mean={s.get('mean',0):.0f} p95={s.get('p95',0):.0f}"
            if s else "no data"
        )
        print(f"  {status}  {r['suite']:<35} {stats_str}")

    print(f"\n{n_ok}/{len(all_reports)} suites passed.")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
