#!/usr/bin/env python3
"""
exp_2_paper_artifacts_4d.py

Paper artifacts for HawkesNest 4D sweep results.

What it produces (self-contained, reproducible artifacts):
  exp_2_paper_Artifacts/
    README.md
    manifest.json
    input/
      results_4d.csv          (verbatim copy)
      schema.json             (columns/dtypes/missingness/theta grids)
    figs_4d/
      heatmaps/               (2D slice heatmaps)
      tables/
        interaction_mass.csv  (R^2 additive vs full pairwise)
        interaction_mass.md   (markdown version)
    logs/
      run_args.json
      run_stdout.txt          (optional; only if you redirect stdout)

No project imports. Only uses: numpy, pandas, matplotlib, scikit-learn.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


# -------------------------
# I/O helpers
# -------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_json_dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
def args_to_jsonable(args):
    out = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x) if isinstance(x, Path) else x for x in v]
        else:
            out[k] = v
    return out

def try_git_commit(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def try_git_is_dirty(repo_root: Path) -> Optional[bool]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return bool(out.decode("utf-8").strip())
    except Exception:
        return None


def get_versions() -> Dict[str, str]:
    vers = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": plt.matplotlib.__version__,
    }
    try:
        import sklearn
        vers["scikit_learn"] = sklearn.__version__
    except Exception:
        vers["scikit_learn"] = "missing"
    return vers


# -------------------------
# Schema capture
# -------------------------

def build_schema(df: pd.DataFrame, theta_cols: Sequence[str]) -> dict:
    schema = {
        "n_rows": int(len(df)),
        "columns": [],
        "missing_fraction": {},
        "theta_grids": {},
    }
    for c in df.columns:
        schema["columns"].append({"name": c, "dtype": str(df[c].dtype)})
    miss = df.isna().mean().sort_values(ascending=False)
    schema["missing_fraction"] = {k: float(v) for k, v in miss.items()}

    for t in theta_cols:
        if t in df.columns:
            vals = df[t].dropna().unique()
            try:
                vals = np.sort(vals.astype(float))
                schema["theta_grids"][t] = [float(x) for x in vals.tolist()]
            except Exception:
                schema["theta_grids"][t] = sorted(df[t].dropna().astype(str).unique().tolist())
    return schema


# -------------------------
# Heatmaps (2D slices)
# -------------------------

def pivot_mean_std(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    val_col: str,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Returns a pivot table with index=y, columns=x for val_col aggregated.
    """
    if agg not in {"mean", "std"}:
        raise ValueError(f"agg must be mean|std, got {agg}")
    g = df.groupby([y_col, x_col])[val_col].agg(agg).reset_index()
    piv = g.pivot(index=y_col, columns=x_col, values=val_col)
    return piv.sort_index(axis=0).sort_index(axis=1)


def plot_heatmap(
    Z: pd.DataFrame,
    out_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    cbar_label: str,
) -> None:
    # Z index=y, columns=x
    x_vals = Z.columns.to_numpy(dtype=float)
    y_vals = Z.index.to_numpy(dtype=float)
    z = Z.to_numpy(dtype=float)

    plt.figure(figsize=(6.2, 5.0))
    im = plt.imshow(
        z,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()),
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_2d_slice_heatmaps(
    df: pd.DataFrame,
    theta_cols: Sequence[str],
    metric_cols: Sequence[str],
    baseline: float,
    out_dir: Path,
    pairs: Optional[List[Tuple[str, str]]] = None,
    min_points: int = 4,
) -> List[dict]:
    """
    For each metric and each theta-pair, fix the other two thetas at baseline,
    then plot mean heatmap over replicates.

    Returns: list of dict records describing each saved figure.
    """
    ensure_dir(out_dir)
    saved = []

    if pairs is None:
        # default: all unordered pairs among theta cols
        pairs = []
        for i in range(len(theta_cols)):
            for j in range(i + 1, len(theta_cols)):
                pairs.append((theta_cols[i], theta_cols[j]))

    for metric in metric_cols:
        if metric not in df.columns:
            continue

        for (tx, ty) in pairs:
            if tx not in df.columns or ty not in df.columns:
                continue

            others = [t for t in theta_cols if t not in (tx, ty)]
            sub = df.copy()
            for o in others:
                # keep only rows close to baseline
                if o in sub.columns and np.issubdtype(sub[o].dtype, np.number):
                    sub = sub[np.isclose(sub[o].to_numpy(dtype=float), baseline)]
                else:
                    # if a theta col is non-numeric, cannot slice reliably
                    sub = sub[sub[o].astype(str) == str(baseline)]

            sub = sub.dropna(subset=[tx, ty, metric])
            if len(sub) < min_points:
                continue

            # pivot mean; if grid sparse, still works
            Zm = pivot_mean_std(sub, x_col=tx, y_col=ty, val_col=metric, agg="mean")

            # file naming
            fixed_str = "__".join([f"{o}{baseline:g}" for o in others])
            out_name = f"{metric}__{tx.replace('theta_','')}_vs_{ty.replace('theta_','')}__{fixed_str}.png"
            out_path = out_dir / out_name

            title = f"{metric} | {tx} vs {ty} (others={baseline:g})"
            plot_heatmap(
                Zm,
                out_path=out_path,
                title=title,
                x_label=tx,
                y_label=ty,
                cbar_label=metric,
            )

            saved.append(
                {
                    "metric": metric,
                    "x": tx,
                    "y": ty,
                    "baseline": float(baseline),
                    "n_rows_used": int(len(sub)),
                    "path": str(out_path),
                }
            )

    return saved


# -------------------------
# Interaction mass table (ΔR²)
# -------------------------

def build_design_additive(df: pd.DataFrame, theta_cols: Sequence[str]) -> np.ndarray:
    X = df[list(theta_cols)].to_numpy(dtype=float)
    # standardize columns to comparable scale (important for numerical stability)
    mu = np.nanmean(X, axis=0)
    sig = np.nanstd(X, axis=0)
    sig[sig == 0] = 1.0
    Xz = (X - mu) / sig
    return Xz


def build_design_full_pairwise(Xz: np.ndarray) -> np.ndarray:
    # X_full = [main | all pairwise products]
    n, d = Xz.shape
    inter = []
    for i in range(d):
        for j in range(i + 1, d):
            inter.append(Xz[:, i] * Xz[:, j])
    if inter:
        return np.column_stack([Xz] + inter)
    return Xz


def interaction_mass_table(
    df: pd.DataFrame,
    theta_cols: Sequence[str],
    metric_cols: Sequence[str],
    out_csv: Path,
    out_md: Path,
    min_rows: int = 20,
) -> pd.DataFrame:
    rows = []
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        sub = df.dropna(subset=list(theta_cols) + [metric]).copy()
        if len(sub) < min_rows:
            rows.append(
                {
                    "metric": metric,
                    "n": int(len(sub)),
                    "r2_add": np.nan,
                    "r2_full": np.nan,
                    "delta_r2": np.nan,
                }
            )
            continue

        y = sub[metric].to_numpy(dtype=float)
        Xz = build_design_additive(sub, theta_cols)
        X_full = build_design_full_pairwise(Xz)

        lr_add = LinearRegression()
        lr_full = LinearRegression()
        lr_add.fit(Xz, y)
        lr_full.fit(X_full, y)

        r2_add = float(lr_add.score(Xz, y))
        r2_full = float(lr_full.score(X_full, y))

        rows.append(
            {
                "metric": metric,
                "n": int(len(sub)),
                "r2_add": r2_add,
                "r2_full": r2_full,
                "delta_r2": float(r2_full - r2_add),
            }
        )

    out = pd.DataFrame(rows).sort_values("delta_r2", ascending=False)
    out.to_csv(out_csv, index=False)
    try:
        out_md.write_text(out.to_markdown(index=False), encoding="utf-8")
    except Exception:
        # markdown is convenience; do not fail the run
        pass
    return out

from pathlib import Path
import argparse
import numpy as np

def to_jsonable(obj):
    """Recursively convert common non-JSON types to JSON-serializable ones."""
    # Paths
    if isinstance(obj, Path):
        return str(obj)

    # argparse.Namespace
    if isinstance(obj, argparse.Namespace):
        return {k: to_jsonable(v) for k, v in vars(obj).items()}

    # dict
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # fallback: leave as-is if JSON can handle it
    return obj

# -------------------------
# Main
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build paper artifacts for 4D sweep results (heatmaps + interaction table).")
    p.add_argument("--results-csv", type=Path, required=True, help="Path to results_4d.csv from run_complexity_sweep_4d")
    p.add_argument("--out-root", type=Path, default=Path("exp_2_paper_Artifacts"))
    p.add_argument("--theta-cols", type=str, default="theta_het,theta_ent,theta_topo,theta_graph")
    p.add_argument("--metric-cols", type=str, default="alpha_het_bg,alpha_ent_mi,alpha_topo,alpha_graph")
    p.add_argument("--baseline", type=float, default=0.0, help="Fix other thetas to this value for 2D slice heatmaps")
    p.add_argument("--pairs", type=str, default="auto",
                   help="Comma-separated theta pairs like 'theta_het:theta_topo,theta_ent:theta_graph' or 'auto'")
    p.add_argument("--min-rows-interaction", type=int, default=20)
    p.add_argument("--repo-root", type=Path, default=Path("."), help="Used for optional git commit capture")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(args.out_root)

    # Structure
    input_dir = ensure_dir(out_root / "input")
    figs_dir = ensure_dir(out_root / "figs_4d")
    heatmaps_dir = ensure_dir(figs_dir / "heatmaps")
    tables_dir = ensure_dir(figs_dir / "tables")
    logs_dir = ensure_dir(out_root / "logs")

    theta_cols = [c.strip() for c in args.theta_cols.split(",") if c.strip()]
    metric_cols = [c.strip() for c in args.metric_cols.split(",") if c.strip()]

    # Read input
    df = pd.read_csv(args.results_csv)

    # Save verbatim copy
    copied_csv = input_dir / "results_4d.csv"
    shutil.copyfile(args.results_csv, copied_csv)

    # Schema
    schema = build_schema(df, theta_cols)
    safe_json_dump(schema, input_dir / "schema.json")

    # Parse pairs
    if args.pairs.strip().lower() == "auto":
        pairs = None
    else:
        pairs = []
        for tok in args.pairs.split(","):
            tok = tok.strip()
            if not tok:
                continue
            a, b = tok.split(":")
            pairs.append((a.strip(), b.strip()))

    # Make heatmaps
    saved_heatmaps = make_2d_slice_heatmaps(
        df=df,
        theta_cols=theta_cols,
        metric_cols=metric_cols,
        baseline=float(args.baseline),
        out_dir=heatmaps_dir,
        pairs=pairs,
    )

    # Interaction table
    interaction_csv = tables_dir / "interaction_mass.csv"
    interaction_md = tables_dir / "interaction_mass.md"
    inter_df = interaction_mass_table(
        df=df,
        theta_cols=theta_cols,
        metric_cols=metric_cols,
        out_csv=interaction_csv,
        out_md=interaction_md,
        min_rows=int(args.min_rows_interaction),
    )

    # Manifest
    manifest = {
        "created_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "command": " ".join([shlex_quote(x) for x in sys.argv]),
        "results_csv_in": str(args.results_csv),
        "results_csv_sha256": sha256_file(copied_csv),
        "out_root": str(out_root),
        "theta_cols": theta_cols,
        "metric_cols": metric_cols,
        "baseline": float(args.baseline),
        "pairs": "auto" if pairs is None else pairs,
        "versions": get_versions(),
        "git": {
            "commit": try_git_commit(args.repo_root),
            "dirty": try_git_is_dirty(args.repo_root),
            "repo_root": str(args.repo_root.resolve()),
        },
        "artifacts": {
            "schema_json": str((input_dir / "schema.json").resolve()),
            "interaction_csv": str(interaction_csv.resolve()),
            "interaction_md": str(interaction_md.resolve()),
            "heatmaps": [r["path"] for r in saved_heatmaps],
        },
        "counts": {
            "n_rows_input": int(len(df)),
            "n_heatmaps_saved": int(len(saved_heatmaps)),
            "interaction_rows": int(len(inter_df)),
        },
    }
    safe_json_dump(to_jsonable(manifest), out_root / "manifest.json")

    # Run args
    safe_json_dump(to_jsonable(args), logs_dir / "run_args.json")

    # README
    readme = []
    readme.append("# exp_2_paper_Artifacts")
    readme.append("")
    readme.append("Artifacts generated by `scripts/exp_2_paper_artifacts_4d.py`.")
    readme.append("")
    readme.append("## Contents")
    readme.append("- `input/results_4d.csv`: verbatim input used for analysis")
    readme.append("- `input/schema.json`: columns, dtypes, missingness, theta grids")
    readme.append("- `figs_4d/heatmaps/`: 2D slice heatmaps (mean over replicates)")
    readme.append("- `figs_4d/tables/interaction_mass.*`: R^2 additive vs full pairwise (delta_R^2)")
    readme.append("- `manifest.json`: command, hashes, versions, optional git commit")
    readme.append("")
    readme.append("## Regenerate")
    readme.append("Example:")
    readme.append("```bash")
    readme.append("python3 scripts/exp_2_paper_artifacts_4d.py \\")
    readme.append("  --results-csv path/to/results_4d.csv \\")
    readme.append("  --out-root exp_2_paper_Artifacts")
    readme.append("```")
    readme.append("")
    (out_root / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"[OK] wrote artifacts to: {out_root.resolve()}")
    print(f"[OK] heatmaps saved: {len(saved_heatmaps)}")
    print(f"[OK] interaction table: {interaction_csv.resolve()}")


def shlex_quote(s: str) -> str:
    # minimal quoting for manifest readability
    if any(ch in s for ch in " \t\n\"'\\$"):
        return "'" + s.replace("'", "'\"'\"'") + "'"
    return s


if __name__ == "__main__":
    main()
