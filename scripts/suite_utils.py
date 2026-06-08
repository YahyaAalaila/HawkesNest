"""
suite_utils.py
==============
Shared utilities for all HawkesNest benchmark suite generation scripts.

Public API
----------
compute_adj(kernel_cfg, target_eta, tau_max)   -> float
lambda_max_for(bg_cfg, adj, burst=8)           -> float
simulate_sequence(cfg_dict, seed, *, horizon, n, debug) -> (times, locs, meta)
time_split(times, frac_train, frac_val)        -> (train_idx, val_idx, test_idx)
position_split(N, n_train, n_val, n_test)      -> (train_idx, val_idx, test_idx)
save_npz(path, times, locs, train_idx, val_idx, test_idx, T_window, domain_bounds, **extra)
write_jsonl(path, seqs)
write_jsonl_splits(out_dir, train_seqs, val_seqs, test_seqs)
count_stats(lengths)                           -> dict
compute_intensity_grid(cfg, times, locs, nx, ny, nt) -> dict
save_intensity_video(grid, out_path, label, fps)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hawkesnest.config import SimulatorConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Adjacency calibration
# ---------------------------------------------------------------------------

def compute_adj(
    kernel_cfg: dict,
    target_eta: float,
    tau_max: float,
) -> float:
    """
    Return the raw adjacency value so that
        effective_branching_ratio = adjacency × kernel_integral = target_eta.

    Supports kernel types: ``separable``, ``traveling_wave``, ``two_scale``.

    For separable / traveling_wave:
        integral = beta × 2π × sigma² × (1 − exp(−tau_max/beta))

    For two_scale:
        integral = [α_f β_f (1−exp(−T/β_f)) + α_s β_s (1−exp(−T/β_s))] × 2π σ²
    """
    ktype = kernel_cfg.get("type", "separable")
    two_pi = 2.0 * math.pi

    if ktype in ("separable", "traveling_wave"):
        beta  = float(kernel_cfg.get("temporal_decay", kernel_cfg.get("temporal_scale", 1.0)))
        sigma = float(kernel_cfg.get("spatial_sigma",  kernel_cfg.get("sigma", 0.08)))
        temp_int  = beta * (1.0 - math.exp(-tau_max / beta))
        kern_int  = temp_int * two_pi * sigma ** 2

    elif ktype == "two_scale":
        alpha_fast = float(kernel_cfg.get("alpha_fast", 0.65))
        alpha_slow = 1.0 - alpha_fast
        beta_fast  = float(kernel_cfg.get("beta_fast", 0.3))
        beta_slow  = float(kernel_cfg.get("beta_slow", 1.0))
        sigma      = float(kernel_cfg.get("sigma", 0.08))
        cap        = min(tau_max, 5.0 * beta_slow)
        temp_int   = (
            alpha_fast * beta_fast * (1.0 - math.exp(-cap / beta_fast))
            + alpha_slow * beta_slow * (1.0 - math.exp(-cap / beta_slow))
        )
        kern_int = temp_int * two_pi * sigma ** 2

    else:
        raise ValueError(f"compute_adj: unsupported kernel type '{ktype}'")

    if kern_int <= 0.0:
        raise ValueError(f"compute_adj: non-positive kernel integral {kern_int}")
    return target_eta / kern_int


def lambda_max_for(bg_cfg: dict, adj: float, burst: int = 20) -> float:
    """
    Upper bound on λ*(s,t):  peak_background + burst × adj × kernel_max.

    kernel_max = 1.0 for all supported kernels.
    burst=20 accounts for transient clusters; high branching ratios (η→0.8)
    can produce local bursts well above 8 simultaneous parents.
    """
    btype = bg_cfg.get("type", "constant")
    if btype == "constant":
        peak_bg = float(bg_cfg.get("rate", 1.0))
    elif btype == "regime_switching":
        lam0    = float(bg_cfg.get("lambda0", 1.0))
        max_amp = max(
            float(c["amplitude"])
            for r in bg_cfg.get("regimes", [])
            for c in r.get("components", [])
        ) if bg_cfg.get("regimes") else 0.0
        peak_bg = lam0 + max_amp
    elif btype == "function":
        fname = bg_cfg.get("name", "")
        if fname == "cluster_mix":
            # Grid-scan [0,1]² to find true peak (off-center due to neighbor gradients)
            a0_bg    = float(bg_cfg.get("a0", 0.0))
            amp_bg   = float(bg_cfg.get("amp", 1.0))
            sigma_bg = float(bg_cfg.get("sigma", 0.1))
            centers  = bg_cfg.get("centers", [])
            max_sum  = 1.0
            if centers:
                grid = np.linspace(0.0, 1.0, 200)
                XX, YY = np.meshgrid(grid, grid)
                Z = np.zeros_like(XX)
                inv2s2 = 1.0 / (2.0 * sigma_bg ** 2)
                for c in centers:
                    dx = XX - float(c[0])
                    dy = YY - float(c[1])
                    Z += np.exp(-(dx ** 2 + dy ** 2) * inv2s2)
                max_sum = float(Z.max())
            peak_bg = math.exp(a0_bg + amp_bg * max_sum) * 1.05  # 5% safety margin
        elif fname in ("moving_gauss", "moving_gauss_slow", "moving_hotspots"):
            peak_bg = float(bg_cfg.get("a0", 0.05)) + float(bg_cfg.get("amp", 1.0)) * 10.0
        elif fname == "gabor_travel":
            peak_bg = math.exp(float(bg_cfg.get("a0", 0.0)) + float(bg_cfg.get("amp", 3.0)))
        else:
            peak_bg = 20.0
    elif btype == "hetero_ladder":
        peak_bg = float(bg_cfg.get("lambda0", 1.0)) * 5.0   # conservative
    else:
        peak_bg = 20.0
    return peak_bg + burst * adj


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _repair_float32_ties(times: np.ndarray) -> np.ndarray:
    """
    Return a float64 copy of times that is strictly increasing when cast to float32.

    Strategy: snap each value to the float32 grid, then do one forward pass
    advancing any non-increasing value by one float32 ULP.  The repair is
    physically negligible (one ULP at t=2000 ≈ 1.2e-4 time units) but
    guarantees that NN models using float32 tensors see distinct timestamps.
    """
    t = times.astype(np.float32)          # snap to float32 grid
    for i in range(len(t) - 1):
        if t[i + 1] <= t[i]:
            t[i + 1] = np.nextafter(t[i], np.float32(np.inf))
    return t.astype(np.float64)           # return as float64 for storage


def simulate_sequence(
    cfg_dict: dict,
    seed: int,
    *,
    horizon: Optional[float] = None,
    n: Optional[int] = None,
    debug: bool = False,
    tau_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate one sequence.

    Returns
    -------
    times        : float64 array [N], strictly increasing and float32-distinct
    locs         : float32 array [N, 2]
    is_triggered : bool array [N]
    parents      : object array [N]  (int index or None)
    """
    clean = {k: v for k, v in cfg_dict.items() if not k.startswith("_")}
    _tau_max = float(clean.pop("tau_max", tau_max if tau_max is not None else 5.0))
    cfg = SimulatorConfig.model_validate(clean)
    sim = cfg.build()
    df, _ = sim.simulate(horizon=horizon, n=n, seed=seed, debug=debug, tau_max=_tau_max)

    if df.empty:
        return (
            np.empty(0, dtype=np.float64),
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=object),
        )

    times        = _repair_float32_ties(df["t"].to_numpy(dtype=np.float64))
    locs         = df[["x", "y"]].to_numpy(dtype=np.float32)
    is_triggered = df["is_triggered"].to_numpy(dtype=bool)
    parents      = df.get("parent", None)
    parents      = parents.to_numpy(dtype=object) if parents is not None else np.full(len(times), None, dtype=object)
    return times, locs, is_triggered, parents


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def time_split(
    times: np.ndarray,
    frac_train: float = 0.70,
    frac_val: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split event indices by time thresholds (temporal holdout).

    Returns sorted index arrays train_idx, val_idx, test_idx.
    """
    N = len(times)
    T = float(times[-1]) if N > 0 else 1.0
    t_train = T * frac_train
    t_val   = T * (frac_train + frac_val)

    train_idx = np.where(times <= t_train)[0]
    val_idx   = np.where((times > t_train) & (times <= t_val))[0]
    test_idx  = np.where(times > t_val)[0]
    return train_idx, val_idx, test_idx


def position_split(
    N: int,
    n_train: int,
    n_val: Optional[int] = None,
    n_test: int = 5_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Suite-2 positional split: test = last n_test events (fixed),
    train = first n_train, val = next n_val (default 10% of n_train).
    """
    if n_val is None:
        n_val = max(1, n_train // 10)
    test_idx  = np.arange(N - n_test, N)
    train_idx = np.arange(0, n_train)
    val_idx   = np.arange(n_train, n_train + n_val)
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# NPZ / JSONL I/O
# ---------------------------------------------------------------------------

def save_npz(
    path: Path,
    times: np.ndarray,
    locs: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    T_window: float,
    domain_bounds: List[List[float]],
    **extra,
) -> None:
    """
    Save a single sequence as NPZ.

    Required fields:
        times [N], locations [N,2], train_idx, val_idx, test_idx,
        T_window (scalar), domain_bounds [[x0,x1],[y0,y1]]
    Any extra keyword arrays are also stored.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        times=times.astype(np.float64),   # float64 preserves strict ordering; float32 caused ties
        locations=locs.astype(np.float32),
        train_idx=train_idx.astype(np.int32),
        val_idx=val_idx.astype(np.int32),
        test_idx=test_idx.astype(np.int32),
        T_window=np.float32(T_window),
        domain_bounds=np.array(domain_bounds, dtype=np.float32),
        **extra,
    )


def write_jsonl(path: Path, seqs: List[Dict[str, Any]]) -> None:
    """Write a list of {times, locations} dicts as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for seq in seqs:
            f.write(json.dumps(seq, separators=(",", ":")) + "\n")


def write_jsonl_splits(
    out_dir: Path,
    train_seqs: List[Dict],
    val_seqs: List[Dict],
    test_seqs: List[Dict],
) -> None:
    """Write train.jsonl, val.jsonl, test.jsonl to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_seqs)
    write_jsonl(out_dir / "val.jsonl",   val_seqs)
    write_jsonl(out_dir / "test.jsonl",  test_seqs)


def seq_to_jsonl_dict(times: np.ndarray, locs: np.ndarray) -> Dict:
    return {
        "times":     times.tolist(),
        "locations": locs.tolist(),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def count_stats(lengths: List[int]) -> Dict[str, Any]:
    arr = np.array(lengths, dtype=float)
    return {
        "n":    len(lengths),
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr)),
        "min":  int(np.min(arr)),
        "p50":  float(np.percentile(arr, 50)),
        "p95":  float(np.percentile(arr, 95)),
        "max":  int(np.max(arr)),
    }


def print_stats_table(rows: List[Dict]) -> None:
    hdr = f"{'Label':<22} {'Mean':>7} {'Std':>6} {'P50':>6} {'P95':>6} {'Min':>5} {'Max':>5}"
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        s = r["stats"]
        print(
            f"{r['label']:<22} "
            f"{s['mean']:>7.1f} {s['std']:>6.1f} "
            f"{s['p50']:>6.0f} {s['p95']:>6.0f} "
            f"{s['min']:>5d} {s['max']:>5d}"
        )
    print()


# ---------------------------------------------------------------------------
# Ground-truth intensity grid + video
# ---------------------------------------------------------------------------

def compute_intensity_grid(
    cfg: dict,
    times: np.ndarray,
    locs: np.ndarray,
    nx: int = 50,
    ny: int = 50,
    nt: int = 200,
    max_events: int = 2_000,
) -> Dict[str, np.ndarray]:
    """
    Compute λ*(s,t | H_t) on a spatiotemporal grid.

    λ*(s,t) = μ(s,t) + Σ_{i: t_i < t} adj × φ(s − s_i, t − t_i)

    Only the first ``max_events`` events are used as history for efficiency.
    This routine assumes a single-mark process and evaluates the actual
    background and kernel objects built from the config, so structured
    backgrounds and vector-valued kernels are visualised faithfully.

    Returns
    -------
    dict with keys: t_grid [nt], x_grid [nx], y_grid [ny], lambda_ [nt, ny, nx]
    """
    clean   = {k: v for k, v in cfg.items() if not k.startswith("_")}
    tau_max = float(clean.pop("tau_max", 5.0))
    sim_cfg = SimulatorConfig.model_validate(clean)
    sim     = sim_cfg.build()
    adj = float(cfg["adjacency"][0][0])
    kobj = sim.kernels[(1, 1)]

    # Clip history for speed
    times = times[:max_events]
    locs  = locs[:max_events]

    T  = float(times[-1]) if len(times) else 1.0
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    ts = np.linspace(0.0, T, nt)
    lam = np.zeros((nt, ny, nx), dtype=np.float32)

    XX, YY = np.meshgrid(xs, ys)   # (ny, nx)
    grid_points = np.column_stack([XX.ravel(), YY.ravel()])

    for it, t in enumerate(ts):
        bg_grid = np.fromiter(
            (sim.background(point, float(t), mark=1) for point in grid_points),
            dtype=np.float64,
            count=grid_points.shape[0],
        ).reshape(ny, nx)

        # Triggering contribution
        mask = times < t
        if mask.any() and adj > 0:
            ti_arr = times[mask]
            li_arr = locs[mask]

            trig = np.zeros((ny, nx), dtype=np.float64)
            for k in range(len(ti_arr)):
                dt = float(t - ti_arr[k])
                if dt > tau_max:
                    continue
                dx = XX - float(li_arr[k, 0])
                dy = YY - float(li_arr[k, 1])
                if getattr(kobj, "uses_vector_s", False):
                    s_arg = np.column_stack([dx.ravel(), dy.ravel()])
                    tau_arg = np.full(s_arg.shape[0], dt, dtype=float)
                    kval = np.asarray(kobj(s_arg, tau_arg), dtype=float).reshape(ny, nx)
                else:
                    ds = np.sqrt(dx ** 2 + dy ** 2)
                    tau_arg = np.full(ds.shape, dt, dtype=float)
                    kval = np.asarray(kobj(ds, tau_arg), dtype=float)
                trig += adj * kval

            bg_grid += trig

        lam[it] = bg_grid.astype(np.float32)

    return {
        "t_grid":  ts.astype(np.float32),
        "x_grid":  xs.astype(np.float32),
        "y_grid":  ys.astype(np.float32),
        "lambda_": lam,
    }


def save_intensity_video(
    grid: Dict[str, np.ndarray],
    out_path: Path,
    label: str = "",
    fps: int = 20,
) -> Path:
    """
    Render λ*(s,t) as a video and save to ``out_path`` (.mp4 or .gif fallback).

    Returns the actual path written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    t_grid = grid["t_grid"]   # [nt]
    x_grid = grid["x_grid"]   # [nx]
    y_grid = grid["y_grid"]   # [ny]
    lam    = grid["lambda_"]  # [nt, ny, nx]

    vmin = float(lam.min())
    vmax = float(np.percentile(lam, 99))  # clip top 1% for colour range
    if vmax <= vmin:
        vmax = vmin + 1e-3

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(
        lam[0],
        origin="lower",
        extent=[float(x_grid[0]), float(x_grid[-1]),
                float(y_grid[0]), float(y_grid[-1])],
        vmin=vmin, vmax=vmax,
        cmap="hot",
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("λ*(s,t)", fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title(f"{label}  t={t_grid[0]:.2f}", fontsize=10)
    fig.tight_layout()

    def _update(frame: int):
        im.set_data(lam[frame])
        title.set_text(f"{label}  t={t_grid[frame]:.2f}")
        return im, title

    anim = FuncAnimation(
        fig, _update, frames=len(t_grid),
        interval=1000 / fps, blit=True,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try MP4 first (requires ffmpeg), fall back to GIF
    try:
        mp4_path = out_path.with_suffix(".mp4")
        writer   = FFMpegWriter(fps=fps, bitrate=800,
                                extra_args=["-pix_fmt", "yuv420p"])
        anim.save(str(mp4_path), writer=writer, dpi=100)
        plt.close(fig)
        return mp4_path
    except Exception:
        gif_path = out_path.with_suffix(".gif")
        writer   = PillowWriter(fps=fps)
        anim.save(str(gif_path), writer=writer, dpi=100)
        plt.close(fig)
        return gif_path
