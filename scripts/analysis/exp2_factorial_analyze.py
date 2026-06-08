#!/usr/bin/env python3
"""
exp2_factorial_analyze.py

Self-contained post-hoc analysis for a FULL 4D factorial sweep.

What it does (robust + audit-heavy):
1) Loads the 4D results CSV and performs strict hygiene.
2) AUDIT: verifies factorial coverage (missing theta-tuples, replicate counts).
   - Writes out_dir/audit_factorial.txt and out_dir/theta_tuple_counts.csv
   - Prints a short audit summary to stdout.
3) 1D plots (two variants, both saved):
   (A) BASELINE-SLICE 1D: hold other thetas at baseline (default 0.0) and vary one theta.
   (B) MAIN-EFFECT 1D: average over all other thetas (groupby theta only).
   - Produces plots for alpha_* and for selected observable summaries.
4) 4D analysis (no heatmaps):
   - PCA scatter colored by each theta (raw + residualized).
   - Spearman table theta vs summary (raw + residualized).
   - Monotonicity check table (percent monotone + violation magnitude).
   - Crosstalk table (linear R^2 gain from interactions + partial drops).
5) Predictability sanity check:
   - Predict each theta level from summary stats using cross-validated classifiers.
   - Saves confusion matrices + a JSON score summary.

Dependencies:
  pandas, numpy, matplotlib, scikit-learn
Optional:
  umap-learn (if --do-umap)

Usage:
  python exp2_factorial_analyze.py \
    --results-csv path/to/results.csv \
    --out-dir exp2_analysis \
    --replicates-col replicate \
    --theta-cols theta_het,theta_ent,theta_topo,theta_graph \
    --baseline 0.0 \
    --controls n_events,lambda_hat_global \
    --do-1d --do-4d

Notes:
- You said baseline must be theta_graph_baseline=0 too. Default baseline=0.0.
- This script will *prove* whether the baseline slice exists as you think:
  it prints the slice sizes per theta and writes them to audit_factorial.txt.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Sequence, Optional, Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze FULL 4D factorial sweep results (audit + 1D + 4D + predictability).")
    p.add_argument("--results-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)

    p.add_argument("--replicates-col", type=str, default="replicate")
    p.add_argument("--events-root", type=Path, default=Path("."))  # unused but kept for compatibility
    p.add_argument("--theta-cols", type=str, default="theta_het,theta_ent,theta_topo,theta_graph")
    p.add_argument("--summary-cols", type=str, default="auto")
    p.add_argument("--controls", type=str, default="n_events,lambda_hat_global")
    p.add_argument("--baseline", type=float, default=0.0, help="Baseline value used for BASELINE-SLICE 1D plots.")

    p.add_argument("--do-1d", action="store_true")
    p.add_argument("--do-4d", action="store_true")
    p.add_argument("--do-umap", action="store_true")

    # misc hygiene knobs
    p.add_argument("--nan-thresh", type=float, default=0.30, help="Drop summary cols with > this fraction NaN.")
    p.add_argument("--round-theta", type=int, default=6, help="Round thetas to stabilize equality checks.")
    p.add_argument("--cv-splits", type=int, default=5, help="Max CV splits for predictability.")
    return p.parse_args()


# -----------------------------------------------------------------------------
# IO / utils
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_float_series(s: pd.Series) -> pd.Series:
    # Convert numeric-like strings safely; keep NaN where conversion fails.
    return pd.to_numeric(s, errors="coerce")


def select_summary_cols(df: pd.DataFrame, theta_cols: Sequence[str], summary_cols_arg: str) -> list[str]:
    if summary_cols_arg != "auto":
        out = [c.strip() for c in summary_cols_arg.split(",") if c.strip() and c.strip() in df.columns]
        return out

    drop = set(theta_cols)
    drop.update([
        "seed", "events_path", "wall_time_sec", "num_marks_used",
        "row_id", "config_id", "config", "config_", "cfg_id"
    ])
    drop.update([c for c in df.columns if c.startswith("alpha_")])  # keep alphas separate
    drop.update([c for c in df.columns if c.endswith("_id")])

    # numeric summaries only
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def residualize(df: pd.DataFrame, y_cols: Sequence[str], control_cols: Sequence[str]) -> pd.DataFrame:
    if not control_cols:
        return df[list(y_cols)].copy()
    X = df[list(control_cols)].copy()
    for c in control_cols:
        X[c] = _as_float_series(X[c])
    X = X.fillna(X.mean(numeric_only=True))
    out: Dict[str, pd.Series] = {}
    for col in y_cols:
        y = df[col].copy()
        y = _as_float_series(y)
        mask = y.notna()
        if mask.sum() < 3:
            out[col] = pd.Series(np.nan, index=df.index)
            continue
        model = LinearRegression()
        model.fit(X[mask], y[mask])
        y_pred = model.predict(X)
        out[col] = y - y_pred
    return pd.DataFrame(out)


def mean_std_group(df: pd.DataFrame, group_col: str, val_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[group_col] + [f"{c}_mean" for c in val_cols] + [f"{c}_std" for c in val_cols])
    agg = df.groupby(group_col)[list(val_cols)].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([x for x in tup if x]) for tup in agg.columns.values]
    return agg


def plot_1d_panel(agg: pd.DataFrame, theta_col: str, metrics: Sequence[str], out_path: Path,
                  title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(8, 5))
    any_plotted = False
    for m in metrics:
        m_mean = f"{m}_mean"
        m_std = f"{m}_std"
        if m_mean not in agg.columns:
            continue
        y = agg[m_mean].to_numpy()
        yerr = agg[m_std].to_numpy() if m_std in agg.columns else None
        plt.errorbar(agg[theta_col], y, yerr=yerr, marker="o", linestyle="-", label=m)
        any_plotted = True
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if any_plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_confmat(cm: np.ndarray, labels: Sequence[str], out_path: Path, title: str):
    plt.figure(figsize=(5.2, 4.5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha="right")
    plt.yticks(ticks=range(len(labels)), labels=labels)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def pca_plot(X: np.ndarray, color: np.ndarray, out_path: Path, title: str, color_label: str):
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X)
    plt.figure(figsize=(6.2, 5.3))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=color, cmap="viridis", s=10)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.colorbar(sc, label=color_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def stratified_cv_scores(model, X: np.ndarray, y: np.ndarray, n_splits: int) -> Tuple[float, float, np.ndarray]:
    # must have at least 2 samples per class for stratified splits
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        # not enough samples in some class: fallback to 2-fold if possible, else no CV
        n_splits = 2

    n_splits = max(2, min(n_splits, int(counts.min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    y_pred = cross_val_predict(model, X, y, cv=skf)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")
    cm = confusion_matrix(y, y_pred, labels=np.unique(y))
    return float(acc), float(f1), cm


# -----------------------------------------------------------------------------
# AUDIT: factorial coverage
# -----------------------------------------------------------------------------

def audit_factorial(df: pd.DataFrame,
                    theta_cols: Sequence[str],
                    rep_col: str,
                    out_dir: Path,
                    round_theta: int) -> None:
    """
    Writes:
      - audit_factorial.txt
      - theta_tuple_counts.csv
      - theta_levels.json
    """
    out_dir = ensure_dir(out_dir)

    # theta levels
    levels = {t: sorted(df[t].dropna().unique().tolist()) for t in theta_cols}
    with (out_dir / "theta_levels.json").open("w") as f:
        json.dump(levels, f, indent=2)

    # tuple counts
    tup_cols = list(theta_cols)
    if rep_col in df.columns:
        # count rows per theta-tuple and replicate coverage
        g = df.groupby(tup_cols)[rep_col].nunique().reset_index(name="n_reps_present")
        g["n_rows"] = df.groupby(tup_cols).size().values
    else:
        g = df.groupby(tup_cols).size().reset_index(name="n_rows")
        g["n_reps_present"] = np.nan
    g.to_csv(out_dir / "theta_tuple_counts.csv", index=False)

    # expected combos
    n_levels = [len(levels[t]) for t in theta_cols]
    expected_tuples = int(np.prod(n_levels)) if all(n > 0 for n in n_levels) else 0

    # missing tuples check by cartesian product of observed levels (not hard-coded)
    # We avoid materializing huge products unless needed (here small).
    grids = [levels[t] for t in theta_cols]
    all_tuples = [(a, b, c, d) for a in grids[0] for b in grids[1] for c in grids[2] for d in grids[3]]
    seen = set(tuple(r) for r in df[tup_cols].to_numpy())
    missing = [tup for tup in all_tuples if tup not in seen]

    # baseline slice sizes (diagnostic)
    baseline = float(0.0)
    base_mask = np.ones(len(df), dtype=bool)
    for t in theta_cols:
        if t == "":  # defensive
            continue
        # baseline slice means other=baseline later; here we just report the corner existence
        pass

    # write report
    lines = []
    lines.append("=== FACTORIAL AUDIT ===")
    lines.append(f"rows_total_loaded = {len(df)}")
    lines.append(f"theta_cols = {list(theta_cols)}")
    lines.append(f"replicates_col = {rep_col} (present={rep_col in df.columns})")
    lines.append("")
    for t in theta_cols:
        lines.append(f"{t}: n_levels={len(levels[t])} levels={levels[t]}")
    lines.append("")
    lines.append(f"expected_theta_tuples (product of observed levels) = {expected_tuples}")
    lines.append(f"observed_theta_tuples = {len(seen)}")
    lines.append(f"missing_theta_tuples = {len(missing)}")
    if len(missing) > 0:
        lines.append("first_25_missing_tuples:")
        for tup in missing[:25]:
            lines.append(f"  {tup}")

    # replicate uniformity
    if rep_col in df.columns:
        rep_counts = g["n_reps_present"].value_counts(dropna=False).sort_index()
        lines.append("")
        lines.append("n_reps_present distribution over theta-tuples:")
        lines.append(rep_counts.to_string())

    (out_dir / "audit_factorial.txt").write_text("\n".join(lines))


def print_slice_diagnostics(df: pd.DataFrame,
                            theta_cols: Sequence[str],
                            baseline: float,
                            rep_col: str,
                            out_dir: Path) -> None:
    """
    For each theta, shows how many rows exist in the baseline slice (others fixed to baseline),
    and which levels are present in that slice.
    Writes out_dir/baseline_slice_diagnostics.txt
    """
    lines = []
    lines.append("=== BASELINE SLICE DIAGNOSTICS ===")
    lines.append(f"baseline = {baseline}")
    lines.append(f"rows_total = {len(df)}")
    lines.append("")

    for theta in theta_cols:
        others = [c for c in theta_cols if c != theta]
        sub = df.copy()
        for o in others:
            sub = sub[np.isclose(sub[o].to_numpy(dtype=float), baseline)]
        levels = sorted(sub[theta].dropna().unique().tolist())
        lines.append(f"[{theta}] slice_rows={len(sub)} unique_levels={levels}")
        if rep_col in sub.columns and len(sub) > 0:
            # expected rows = (#levels) * (#replicates)
            nrep = sub.groupby(theta)[rep_col].nunique()
            lines.append(f"  reps_per_level:\n{nrep.to_string()}")
        lines.append("")

    (out_dir / "baseline_slice_diagnostics.txt").write_text("\n".join(lines))


# -----------------------------------------------------------------------------
# 1D plots
# -----------------------------------------------------------------------------

def do_1d_plots(df: pd.DataFrame,
                theta_cols: Sequence[str],
                baseline: float,
                rep_col: str,
                out_dir: Path) -> None:
    out_dir = ensure_dir(out_dir)

    # alpha metrics (if exist)
    alpha_map = {
        "theta_het": "alpha_het_bg",
        "theta_ent": "alpha_ent_mi",
        "theta_topo": "alpha_topo",
        "theta_graph": "alpha_graph",
    }
    alpha_metrics = [alpha_map[t] for t in theta_cols if alpha_map.get(t) in df.columns]

    # observable summaries: pick common ones if present, otherwise a small auto list
    candidate_obs = [
        "n_events", "lambda_hat_global",
        "time_bin_gini", "space_bin_gini", "st_bin_gini",
        "mark_entropy", "self_exc_index", "cross_exc_index",
    ]
    obs_metrics = [c for c in candidate_obs if c in df.columns]

    # --- A) baseline-slice 1D ---
    for theta in theta_cols:
        others = [c for c in theta_cols if c != theta]
        slice_df = df.copy()
        for o in others:
            slice_df = slice_df[np.isclose(slice_df[o].to_numpy(dtype=float), baseline)]

        # alpha plot
        if alpha_metrics:
            agg = mean_std_group(slice_df, theta, alpha_metrics)
            plot_1d_panel(
                agg, theta_col=theta, metrics=alpha_metrics,
                out_path=out_dir / f"1d_baseline_alpha_{theta}.png",
                title=f"1D baseline-slice (others={baseline}): {theta} → alpha_*",
                xlabel=theta, ylabel="alpha metric"
            )

        # observables plot
        if obs_metrics:
            agg = mean_std_group(slice_df, theta, obs_metrics)
            plot_1d_panel(
                agg, theta_col=theta, metrics=obs_metrics,
                out_path=out_dir / f"1d_baseline_obs_{theta}.png",
                title=f"1D baseline-slice (others={baseline}): {theta} → observables",
                xlabel=theta, ylabel="observable"
            )

    # --- B) main-effect 1D (use ALL rows) ---
    for theta in theta_cols:
        if alpha_metrics:
            agg = mean_std_group(df, theta, alpha_metrics)
            plot_1d_panel(
                agg, theta_col=theta, metrics=alpha_metrics,
                out_path=out_dir / f"1d_maineffect_alpha_{theta}.png",
                title=f"1D main-effect (avg over other thetas): {theta} → alpha_*",
                xlabel=theta, ylabel="alpha metric"
            )
        if obs_metrics:
            agg = mean_std_group(df, theta, obs_metrics)
            plot_1d_panel(
                agg, theta_col=theta, metrics=obs_metrics,
                out_path=out_dir / f"1d_maineffect_obs_{theta}.png",
                title=f"1D main-effect (avg over other thetas): {theta} → observables",
                xlabel=theta, ylabel="observable"
            )


# -----------------------------------------------------------------------------
# 4D analysis (no heatmaps)
# -----------------------------------------------------------------------------

def spearman_table(df: pd.DataFrame, theta_cols: Sequence[str], Y: pd.DataFrame, out_path: Path):
    rows = []
    for t in theta_cols:
        for s in Y.columns:
            tmp = pd.DataFrame({"t": df[t], "s": Y[s]}).dropna()
            if len(tmp) < 5:
                continue
            corr = tmp.corr(method="spearman").iloc[0, 1]
            rows.append({"theta": t, "summary": s, "spearman": float(corr)})
    pd.DataFrame(rows).to_csv(out_path, index=False)


def monotonicity_checks(df: pd.DataFrame, theta_cols: Sequence[str], Y: pd.DataFrame, out_path: Path):
    records = []
    for theta in theta_cols:
        others = [c for c in theta_cols if c != theta]
        combos = df.groupby(others, dropna=False)
        for summary in Y.columns:
            n_combo = 0
            n_mono = 0
            viols = []
            for _, sub in combos:
                mean_by_theta = sub.assign(_y=Y.loc[sub.index, summary]).groupby(theta)["_y"].mean().sort_index()
                if len(mean_by_theta) < 2:
                    continue
                diffs = np.diff(mean_by_theta.to_numpy())
                neg = diffs[diffs < 0]
                viol_mag = float(np.abs(neg).sum()) if neg.size else 0.0
                is_mono = bool(np.all(diffs >= -1e-9))
                n_combo += 1
                n_mono += int(is_mono)
                viols.append(viol_mag)
            if n_combo == 0:
                continue
            records.append({
                "theta": theta,
                "summary": summary,
                "pct_monotone": float(n_mono / n_combo),
                "avg_violation": float(np.mean(viols)) if viols else 0.0,
            })
    pd.DataFrame(records).to_csv(out_path, index=False)


def crosstalk_tables(df: pd.DataFrame, theta_cols: Sequence[str], Y: pd.DataFrame, out_path: Path):
    theta_mat = df[list(theta_cols)].to_numpy(dtype=float)
    theta_std = (theta_mat - theta_mat.mean(axis=0)) / (theta_mat.std(axis=0) + 1e-12)

    main_cols = list(theta_cols)
    inter_vals = []
    for a, b in combinations(range(len(theta_cols)), 2):
        inter_vals.append(theta_std[:, a] * theta_std[:, b])
    X_main = theta_std
    X_full = np.column_stack([theta_std] + inter_vals)

    rows = []
    for col in Y.columns:
        y = _as_float_series(Y[col]).to_numpy(dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < 10:
            continue

        lr_main = LinearRegression()
        lr_main.fit(X_main[mask], y[mask])
        r2_main = float(lr_main.score(X_main[mask], y[mask]))

        lr_full = LinearRegression()
        lr_full.fit(X_full[mask], y[mask])
        r2_full = float(lr_full.score(X_full[mask], y[mask]))

        partials = {}
        for i, name in enumerate(main_cols):
            X_drop = np.delete(X_full, i, axis=1)
            lr_drop = LinearRegression()
            lr_drop.fit(X_drop[mask], y[mask])
            r2_drop = float(lr_drop.score(X_drop[mask], y[mask]))
            partials[name] = r2_full - r2_drop

        row = {
            "summary": col,
            "r2_main": r2_main,
            "r2_full": r2_full,
            "interaction_gain": r2_full - r2_main,
        }
        for name in main_cols:
            row[f"partial_{name}"] = float(partials.get(name, np.nan))
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)


def do_4d_analysis(df: pd.DataFrame,
                   theta_cols: Sequence[str],
                   summary_cols: Sequence[str],
                   control_cols: Sequence[str],
                   out_dir: Path,
                   do_umap: bool):
    out_dir = ensure_dir(out_dir)

    # build Y_raw and Y_resid
    Y_raw = df[list(summary_cols)].copy()
    Y_resid = residualize(df, summary_cols, [c for c in control_cols if c in df.columns])

    # PCA colored by each theta (raw + resid)
    Xr = Y_raw.fillna(Y_raw.mean(numeric_only=True)).to_numpy(dtype=float)
    Xz = Y_resid.fillna(Y_resid.mean(numeric_only=True)).to_numpy(dtype=float)
    for t in theta_cols:
        color = df[t].to_numpy(dtype=float)
        pca_plot(Xr, color, out_dir / f"pca_raw_color_{t}.png", f"PCA (raw summaries) colored by {t}", t)
        pca_plot(Xz, color, out_dir / f"pca_resid_color_{t}.png", f"PCA (residualized summaries) colored by {t}", t)

    # Optional UMAP
    if do_umap:
        try:
            import umap  # type: ignore
            reducer = umap.UMAP(random_state=0)
            coords = reducer.fit_transform(Xz)
            for t in theta_cols:
                plt.figure(figsize=(6.2, 5.3))
                sc = plt.scatter(coords[:, 0], coords[:, 1], c=df[t].to_numpy(dtype=float), cmap="viridis", s=10)
                plt.colorbar(sc, label=t)
                plt.title(f"UMAP (residualized summaries) colored by {t}")
                plt.tight_layout()
                plt.savefig(out_dir / f"umap_resid_color_{t}.png", dpi=180)
                plt.close()
        except Exception as e:
            (out_dir / "umap_error.txt").write_text(repr(e))

    # Spearman
    spearman_table(df, theta_cols, Y_raw, out_dir / "theta_summary_spearman_raw.csv")
    spearman_table(df, theta_cols, Y_resid, out_dir / "theta_summary_spearman_resid.csv")

    # Monotonicity
    monotonicity_checks(df, theta_cols, Y_raw, out_dir / "monotonicity_raw.csv")
    monotonicity_checks(df, theta_cols, Y_resid, out_dir / "monotonicity_resid.csv")

    # Crosstalk
    crosstalk_tables(df, theta_cols, Y_raw, out_dir / "crosstalk_table_raw.csv")
    crosstalk_tables(df, theta_cols, Y_resid, out_dir / "crosstalk_table_resid.csv")

    # Save which columns were used
    (out_dir / "summary_columns_used.json").write_text(json.dumps(list(summary_cols), indent=2))


# -----------------------------------------------------------------------------
# Predictability: can we predict theta from summaries?
# -----------------------------------------------------------------------------

def predict_thetas(df: pd.DataFrame,
                   theta_cols: Sequence[str],
                   Y: pd.DataFrame,
                   out_dir: Path,
                   cv_splits: int):
    out_dir = ensure_dir(out_dir)

    # standardize
    X = Y.fillna(Y.mean(numeric_only=True)).to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X)

    scores = {}
    for theta in theta_cols:
        y_raw = df[theta].astype(str).to_numpy()
        enc = LabelEncoder()
        y = enc.fit_transform(y_raw)

        # models
        models = {
            "logreg": LogisticRegression(max_iter=400),
            "gradboost": GradientBoostingClassifier(random_state=0),
        }

        scores[theta] = {}
        for name, model in models.items():
            try:
                acc, f1, cm = stratified_cv_scores(model, X, y, n_splits=cv_splits)
                scores[theta][name] = {"accuracy": acc, "macro_f1": f1}
                plot_confmat(
                    cm,
                    labels=[str(x) for x in enc.classes_],
                    out_path=out_dir / f"confmat_{theta}_{name}.png",
                    title=f"{theta} predicted from summaries ({name})"
                )
            except Exception as e:
                scores[theta][name] = {"error": repr(e)}

    (out_dir / "predictability_scores.json").write_text(json.dumps(scores, indent=2))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    theta_cols = [c.strip() for c in args.theta_cols.split(",") if c.strip()]
    control_cols = [c.strip() for c in args.controls.split(",") if c.strip()]

    out_dir = ensure_dir(args.out_dir)
    plots_1d_dir = ensure_dir(out_dir / "plots_1d")
    analysis_4d_dir = ensure_dir(out_dir / "analysis_4d")
    audit_dir = ensure_dir(out_dir / "audit")

    # --- load ---
    df0 = pd.read_csv(args.results_csv)
    (audit_dir / "rows_loaded.txt").write_text(f"{len(df0)}\n")

    # --- coerce thetas to float + round ---
    df = df0.copy()
    for t in theta_cols:
        if t not in df.columns:
            raise KeyError(f"Missing theta column: {t}. Have: {list(df.columns)}")
        df[t] = _as_float_series(df[t]).round(args.round_theta)

    # ensure replicate col exists (if not, create a dummy one)
    if args.replicates_col not in df.columns:
        df[args.replicates_col] = 0

    # minimal hygiene: require thetas + replicate col to be non-null
    df = df.dropna(subset=list(theta_cols) + [args.replicates_col]).reset_index(drop=True)

    # --- pick summary cols (numeric) ---
    summary_cols = select_summary_cols(df, theta_cols, args.summary_cols)

    # also keep alpha_* separately for 1D plotting
    alpha_cols = [c for c in df.columns if c.startswith("alpha_") and pd.api.types.is_numeric_dtype(df[c])]
    # For predictability we typically want observables, not alpha; but you can include alpha if desired.
    # Here we use summary_cols only; alpha lives as separate plots.

    # drop summary cols with too much NaN
    nan_frac = df[summary_cols].isna().mean()
    summary_cols = [c for c in summary_cols if nan_frac.get(c, 1.0) <= args.nan_thresh]

    # --- AUDIT: prove we use all rows we can ---
    used_rows = len(df)
    audit_text = []
    audit_text.append("=== ROW USAGE AUDIT ===")
    audit_text.append(f"rows_loaded_raw = {len(df0)}")
    audit_text.append(f"rows_after_theta_coerce_and_dropna = {used_rows}")
    audit_text.append(f"fraction_used = {used_rows / max(1, len(df0)):.6f}")
    audit_text.append("")
    audit_text.append(f"theta_cols = {theta_cols}")
    audit_text.append(f"replicates_col = {args.replicates_col}")
    audit_text.append(f"baseline = {args.baseline}")
    audit_text.append("")
    audit_text.append(f"n_summary_cols_used = {len(summary_cols)}")
    audit_text.append("summary_cols_used:")
    audit_text.append("\n".join([f"  - {c}" for c in summary_cols]))
    (audit_dir / "audit_rows_used.txt").write_text("\n".join(audit_text))

    # --- factorial audit ---
    audit_factorial(df, theta_cols, args.replicates_col, audit_dir, round_theta=args.round_theta)
    print_slice_diagnostics(df, theta_cols, float(args.baseline), args.replicates_col, audit_dir)

    # short stdout summary (you want fast feedback)
    print(f"[AUDIT] rows_loaded={len(df0)} rows_used={len(df)} fraction_used={len(df)/max(1,len(df0)):.4f}")
    for t in theta_cols:
        lv = sorted(df[t].dropna().unique().tolist())
        print(f"[AUDIT] {t} levels ({len(lv)}): {lv}")

    # --- 1D plots ---
    if args.do_1d:
        do_1d_plots(df, theta_cols, float(args.baseline), args.replicates_col, plots_1d_dir)
        print(f"[OK] wrote 1D plots -> {plots_1d_dir}")

    # --- 4D analysis + predictability ---
    if args.do_4d:
        do_4d_analysis(
            df=df,
            theta_cols=theta_cols,
            summary_cols=summary_cols,
            control_cols=control_cols,
            out_dir=analysis_4d_dir,
            do_umap=bool(args.do_umap),
        )
        # predictability on residualized summaries (more honest)
        Y_resid = residualize(df, summary_cols, [c for c in control_cols if c in df.columns])
        predict_thetas(df, theta_cols, Y_resid, analysis_4d_dir / "predictability", cv_splits=int(args.cv_splits))
        print(f"[OK] wrote 4D analysis -> {analysis_4d_dir}")

    # final “proof file” that ties to full row usage
    final_report = [
        "DONE",
        f"rows_loaded_raw={len(df0)}",
        f"rows_used_after_hygiene={len(df)}",
        f"outputs:",
        f"  audit: {audit_dir}",
        f"  plots_1d: {plots_1d_dir if args.do_1d else '(skipped)'}",
        f"  analysis_4d: {analysis_4d_dir if args.do_4d else '(skipped)'}",
    ]
    (out_dir / "DONE.txt").write_text("\n".join(final_report))


if __name__ == "__main__":
    main()