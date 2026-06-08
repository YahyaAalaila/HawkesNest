#!/usr/bin/env python3
"""
Deterministic variance decomposition for a balanced 4D factorial sweep.

- Aggregates replicates per cell (theta_het, theta_ent, theta_topo, theta_graph) -> mean alpha
- Computes *exact* ANOVA sum-of-squares decomposition for balanced 4-factor design:
    SS_total = SS_main + SS_2way + SS_3way + SS_4way + SS_resid
  After cell-mean aggregation, SS_resid should be ~0.

Outputs (in out-dir):
  - diagnostics.txt
  - cell_means.csv
  - anova_decomp.csv                  (fractions by order)
  - anova_terms_by_effect.csv         (fractions per specific term: A, AB, ABC, ABCD, ...)
  - fig_anova_orders_<alpha>.png      (stacked bar: main/2way/3way/4way/resid)
  - fig_anova_terms_<alpha>.png       (bar: each term's fraction)

No p-values. This is a deterministic mapping check (axis disentanglement via low interaction mass).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALPHAS_DEFAULT = ["alpha_het_bg", "alpha_ent_mi", "alpha_topo", "alpha_graph"]
THETAS_DEFAULT = ["theta_het", "theta_ent", "theta_topo", "theta_graph"]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--theta-cols", type=str, default=",".join(THETAS_DEFAULT))
    ap.add_argument("--alpha-cols", type=str, default=",".join(ALPHAS_DEFAULT))
    ap.add_argument("--rep-col", type=str, default="replicate")
    ap.add_argument("--round-thetas", type=int, default=6, help="Round theta floats to avoid 0.7500000001 issues.")
    return ap.parse_args()


def _sorted_levels(df: pd.DataFrame, col: str) -> list[float]:
    vals = df[col].dropna().unique()
    vals = [float(v) for v in vals]
    return sorted(vals)


def _balanced_cell_diagnostics(df: pd.DataFrame, thetas: list[str], rep_col: str) -> tuple[str, pd.DataFrame]:
    """
    Returns (diagnostic_text, reps_per_cell_df)
    """
    lines = []
    lines.append("=== DATASET DIAGNOSTICS ===")
    lines.append(f"rows_total = {len(df)}")
    for t in thetas:
        lv = _sorted_levels(df, t)
        lines.append(f"{t}: n_levels={len(lv)} levels={lv}")

    # reps per cell
    g = df.groupby(thetas, dropna=False)[rep_col].count().rename("n_rows").reset_index()
    lines.append("")
    lines.append("=== REPLICATES PER CELL (counts of rows) ===")
    lines.append(f"n_cells = {len(g)}")
    lines.append(g["n_rows"].describe().to_string())
    # show if balanced
    n_unique = g["n_rows"].nunique()
    lines.append(f"balanced = {n_unique == 1} (unique n_rows per cell = {n_unique})")
    return "\n".join(lines), g


def _pivot_to_4d(cell_df: pd.DataFrame, thetas: list[str], ycol: str):
    """
    Create a dense 4D array Y[a,b,c,d] following sorted theta levels.
    Returns (Y, levels_dict)
    """
    levels = {t: sorted(cell_df[t].unique().tolist()) for t in thetas}
    for t in thetas:
        levels[t] = [float(v) for v in levels[t]]
        levels[t].sort()

    A, B, C, D = [len(levels[t]) for t in thetas]
    Y = np.full((A, B, C, D), np.nan, dtype=float)

    idx_maps = {t: {v: i for i, v in enumerate(levels[t])} for t in thetas}

    for _, r in cell_df.iterrows():
        ia = idx_maps[thetas[0]][float(r[thetas[0]])]
        ib = idx_maps[thetas[1]][float(r[thetas[1]])]
        ic = idx_maps[thetas[2]][float(r[thetas[2]])]
        idd = idx_maps[thetas[3]][float(r[thetas[3]])]
        Y[ia, ib, ic, idd] = float(r[ycol])

    if not np.isfinite(Y).all():
        # identify missing
        missing = np.argwhere(~np.isfinite(Y))
        raise ValueError(
            f"Missing cells for {ycol}: found {len(missing)} NaNs in 4D grid. "
            f"This is not a full factorial (or aggregation failed)."
        )

    return Y, levels


def _mean_over_axes(Y: np.ndarray, axes_to_keep: tuple[int, ...]) -> np.ndarray:
    """
    Average over axes not in axes_to_keep, keeping specified axes.
    """
    axes_all = tuple(range(Y.ndim))
    axes_to_mean = tuple(ax for ax in axes_all if ax not in axes_to_keep)
    return Y.mean(axis=axes_to_mean, keepdims=False)


def _ss_decomp_balanced_4d(Y: np.ndarray):
    """
    Exact SS decomposition for balanced 4-factor design with one obs per cell.

    Y shape: (A,B,C,D)

    Returns:
      ss_terms: dict term_name -> SS
      ss_by_order: dict {"main":..., "2way":..., "3way":..., "4way":..., "resid":...}
    """
    A, B, C, D = Y.shape
    N = A * B * C * D

    grand = Y.mean()
    ss_total = float(((Y - grand) ** 2).sum())

    # Main effects
    mean_A = _mean_over_axes(Y, (0,))
    mean_B = _mean_over_axes(Y, (1,))
    mean_C = _mean_over_axes(Y, (2,))
    mean_D = _mean_over_axes(Y, (3,))

    SS_A = float(B * C * D * ((mean_A - grand) ** 2).sum())
    SS_B = float(A * C * D * ((mean_B - grand) ** 2).sum())
    SS_C = float(A * B * D * ((mean_C - grand) ** 2).sum())
    SS_D = float(A * B * C * ((mean_D - grand) ** 2).sum())

    # 2-way interactions
    mean_AB = _mean_over_axes(Y, (0, 1))
    mean_AC = _mean_over_axes(Y, (0, 2))
    mean_AD = _mean_over_axes(Y, (0, 3))
    mean_BC = _mean_over_axes(Y, (1, 2))
    mean_BD = _mean_over_axes(Y, (1, 3))
    mean_CD = _mean_over_axes(Y, (2, 3))

    # interaction "effects" (balanced orthogonal contrasts)
    E_AB = mean_AB - mean_A[:, None] - mean_B[None, :] + grand
    E_AC = mean_AC - mean_A[:, None] - mean_C[None, :] + grand
    E_AD = mean_AD - mean_A[:, None] - mean_D[None, :] + grand
    E_BC = mean_BC - mean_B[:, None] - mean_C[None, :] + grand
    E_BD = mean_BD - mean_B[:, None] - mean_D[None, :] + grand
    E_CD = mean_CD - mean_C[:, None] - mean_D[None, :] + grand

    SS_AB = float(C * D * (E_AB ** 2).sum())
    SS_AC = float(B * D * (E_AC ** 2).sum())
    SS_AD = float(B * C * (E_AD ** 2).sum())
    SS_BC = float(A * D * (E_BC ** 2).sum())
    SS_BD = float(A * C * (E_BD ** 2).sum())
    SS_CD = float(A * B * (E_CD ** 2).sum())

    # 3-way interactions
    mean_ABC = _mean_over_axes(Y, (0, 1, 2))
    mean_ABD = _mean_over_axes(Y, (0, 1, 3))
    mean_ACD = _mean_over_axes(Y, (0, 2, 3))
    mean_BCD = _mean_over_axes(Y, (1, 2, 3))

    # build 3-way effects
    # ABC: subtract all lower-order components included in marginal means
    # E_ABC = mean_ABC - mean_AB - mean_AC - mean_BC + mean_A + mean_B + mean_C - grand
    E_ABC = (
        mean_ABC
        - mean_AB[:, :, None]
        - mean_AC[:, None, :]
        - mean_BC[None, :, :]
        + mean_A[:, None, None]
        + mean_B[None, :, None]
        + mean_C[None, None, :]
        - grand
    )

    E_ABD = (
        mean_ABD
        - mean_AB[:, :, None]
        - mean_AD[:, None, :]
        - mean_BD[None, :, :]
        + mean_A[:, None, None]
        + mean_B[None, :, None]
        + mean_D[None, None, :]
        - grand
    )

    E_ACD = (
        mean_ACD
        - mean_AC[:, :, None]
        - mean_AD[:, None, :]
        - mean_CD[None, :, :]
        + mean_A[:, None, None]
        + mean_C[None, :, None]
        + mean_D[None, None, :]
        - grand
    )

    E_BCD = (
        mean_BCD
        - mean_BC[:, :, None]
        - mean_BD[:, None, :]
        - mean_CD[None, :, :]
        + mean_B[:, None, None]
        + mean_C[None, :, None]
        + mean_D[None, None, :]
        - grand
    )

    SS_ABC = float(D * (E_ABC ** 2).sum())
    SS_ABD = float(C * (E_ABD ** 2).sum())
    SS_ACD = float(B * (E_ACD ** 2).sum())
    SS_BCD = float(A * (E_BCD ** 2).sum())

    # 4-way interaction
    # E_ABCD = Y - (grand + mains + 2ways + 3ways) in orthogonal balanced sense
    # Construct additive prediction from all lower-order effects at each cell:

    # broadcast mains
    pred = (
        grand
        + (mean_A - grand)[:, None, None, None]
        + (mean_B - grand)[None, :, None, None]
        + (mean_C - grand)[None, None, :, None]
        + (mean_D - grand)[None, None, None, :]
    )

    # broadcast 2-way effects
    pred = pred + E_AB[:, :, None, None] + E_AC[:, None, :, None] + E_AD[:, None, None, :]
    pred = pred + E_BC[None, :, :, None] + E_BD[None, :, None, :] + E_CD[None, None, :, :]

    # broadcast 3-way effects
    pred = pred + E_ABC[:, :, :, None] + E_ABD[:, :, None, :] + E_ACD[:, None, :, :] + E_BCD[None, :, :, :]

    E_ABCD = Y - pred
    SS_ABCD = float((E_ABCD ** 2).sum())

    ss_terms = {
        "A": SS_A, "B": SS_B, "C": SS_C, "D": SS_D,
        "AB": SS_AB, "AC": SS_AC, "AD": SS_AD, "BC": SS_BC, "BD": SS_BD, "CD": SS_CD,
        "ABC": SS_ABC, "ABD": SS_ABD, "ACD": SS_ACD, "BCD": SS_BCD,
        "ABCD": SS_ABCD,
    }

    SS_main = SS_A + SS_B + SS_C + SS_D
    SS_2way = SS_AB + SS_AC + SS_AD + SS_BC + SS_BD + SS_CD
    SS_3way = SS_ABC + SS_ABD + SS_ACD + SS_BCD
    SS_4way = SS_ABCD

    # after cell means, resid should be 0 (numerical eps aside)
    SS_resid = max(0.0, ss_total - (SS_main + SS_2way + SS_3way + SS_4way))

    ss_by_order = {
        "total": ss_total,
        "main": SS_main,
        "2way": SS_2way,
        "3way": SS_3way,
        "4way": SS_4way,
        "resid": SS_resid,
    }

    return ss_terms, ss_by_order


def _plot_orders(frac: dict, out_path: Path, title: str):
    # fixed order
    keys = ["main", "2way", "3way", "4way", "resid"]
    vals = [float(frac.get(k, 0.0)) for k in keys]
    plt.figure(figsize=(6.2, 3.6))
    plt.bar(keys, vals)
    plt.ylim(0, 1.0)
    plt.ylabel("fraction of total SS")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def _plot_terms(frac_terms: pd.DataFrame, out_path: Path, title: str):
    # order by contribution
    df = frac_terms.sort_values("fraction", ascending=False).copy()
    plt.figure(figsize=(8.8, 3.8))
    plt.bar(df["term"].astype(str), df["fraction"].astype(float))
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("fraction of total SS")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    thetas = [c.strip() for c in args.theta_cols.split(",") if c.strip()]
    alphas = [c.strip() for c in args.alpha_cols.split(",") if c.strip()]

    if len(thetas) != 4:
        raise ValueError(f"Need exactly 4 theta cols. Got {thetas}")
    df = pd.read_csv(args.results_csv)

    # keep only needed columns (+ rep col if exists)
    keep = [c for c in (thetas + alphas + [args.rep_col]) if c in df.columns]
    df = df[keep].copy()

    # round theta levels for stable grouping
    for t in thetas:
        df[t] = df[t].astype(float).round(int(args.round_thetas))

    # diagnostics on raw rows
    diag_txt, reps_per_cell = _balanced_cell_diagnostics(df, thetas, args.rep_col)
    (out_dir / "diagnostics.txt").write_text(diag_txt)

    # expected cell count for full factorial from observed levels
    levels = [len(_sorted_levels(df, t)) for t in thetas]
    expected_cells = int(np.prod(levels))
    found_cells = int(reps_per_cell.shape[0])

    # aggregate to cell means (use ALL rows)
    agg_cols = {a: "mean" for a in alphas if a in df.columns}
    cell = (
        df.groupby(thetas, dropna=False)
          .agg(**{f"{a}_mean": (a, "mean") for a in alphas if a in df.columns},
               **{f"{a}_std": (a, "std") for a in alphas if a in df.columns},
               n_rows=(args.rep_col, "count"))
          .reset_index()
    )
    cell.to_csv(out_dir / "cell_means.csv", index=False)

    # extra sanity checks
    lines = []
    lines.append("\n=== FULL FACTORIAL CHECK ===")
    lines.append(f"expected_cells_from_levels = {expected_cells} (levels={levels})")
    lines.append(f"found_cells_in_data        = {found_cells}")
    if found_cells != expected_cells:
        lines.append("WARNING: not a full factorial grid w.r.t observed levels (or missing cells).")
    # determinism check
    for a in alphas:
        mcol = f"{a}_mean"
        scol = f"{a}_std"
        if mcol in cell.columns and scol in cell.columns:
            max_std = float(np.nanmax(cell[scol].to_numpy()))
            lines.append(f"max_within_cell_std({a}) = {max_std:.6g}")
    with (out_dir / "diagnostics.txt").open("a") as f:
        f.write("\n" + "\n".join(lines) + "\n")

    # ANOVA decomposition per alpha
    decomp_rows = []
    term_rows = []

    for a in alphas:
        mcol = f"{a}_mean"
        if mcol not in cell.columns:
            continue

        # build 4D tensor of cell means
        Y4, lvl = _pivot_to_4d(cell, thetas, mcol)

        ss_terms, ss_by_order = _ss_decomp_balanced_4d(Y4)
        total = max(ss_by_order["total"], 1e-30)

        frac_by_order = {
            "alpha": a,
            "main": ss_by_order["main"] / total,
            "2way": ss_by_order["2way"] / total,
            "3way": ss_by_order["3way"] / total,
            "4way": ss_by_order["4way"] / total,
            "resid": ss_by_order["resid"] / total,
        }
        frac_by_order["interaction_total"] = frac_by_order["2way"] + frac_by_order["3way"] + frac_by_order["4way"]
        frac_by_order["main_over_interaction"] = frac_by_order["main"] / max(frac_by_order["interaction_total"], 1e-12)
        decomp_rows.append(frac_by_order)

        # per-term fractions
        for term, ss in ss_terms.items():
            order = len(term)  # A=1, AB=2, ABC=3, ABCD=4
            term_rows.append({
                "alpha": a,
                "term": term,
                "order": order,
                "ss": float(ss),
                "fraction": float(ss) / total,
            })

        # plots
        _plot_orders(
            frac_by_order,
            out_dir / f"fig_anova_orders_{a}.png",
            title=f"{a}: SS fractions by interaction order",
        )
        term_df = pd.DataFrame([r for r in term_rows if r["alpha"] == a])
        _plot_terms(
            term_df,
            out_dir / f"fig_anova_terms_{a}.png",
            title=f"{a}: SS fractions per term (A, AB, ABC, ABCD)",
        )

    decomp = pd.DataFrame(decomp_rows)
    decomp.to_csv(out_dir / "anova_decomp.csv", index=False)

    terms = pd.DataFrame(term_rows)
    terms.to_csv(out_dir / "anova_terms_by_effect.csv", index=False)

    # quick console summary (paper-friendly)
    if not decomp.empty:
        print("\n=== DETERMINISTIC ANOVA VARIANCE FRACTIONS (cell means) ===")
        for _, r in decomp.iterrows():
            print(
                f"{r['alpha']}: main={r['main']:.3f}, 2-way={r['2way']:.3f}, 3-way={r['3way']:.3f}, "
                f"4-way={r['4way']:.3f}, resid={r['resid']:.3f}, "
                f"interaction_total={r['interaction_total']:.3f}, main/interaction={r['main_over_interaction']:.2f}"
            )

    print(f"\nSaved outputs to: {out_dir.resolve()}")
    print(f"- diagnostics.txt")
    print(f"- cell_means.csv")
    print(f"- anova_decomp.csv")
    print(f"- anova_terms_by_effect.csv")
    print(f"- fig_anova_orders_*.png / fig_anova_terms_*.png")


if __name__ == "__main__":
    main()