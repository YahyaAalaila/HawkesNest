from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

import numpy as np
import pandas as pd

from hawkesnest.config_factory.background_configs import (
    ConstantBackgroundCfg,
    HeteroLadderBackgroundCfg,
)
from hawkesnest.config_factory.domain_configs import TopologyRGGDomainCfg
from hawkesnest.config_factory.graph_configs import BranchingMatrixCfg
from hawkesnest.config_factory.kernel_configs import (
    EntangledExpGaussianKernelCfg,
    SeparableKernelCfg,
)
from hawkesnest.domain import RectangleDomain
from hawkesnest.metrics import gaussian_entanglement_index
from hawkesnest.metrics.graph import alpha_graph_new, spectral_norm
from hawkesnest.metrics.heterogeneity import background_heterogeneity_index
from hawkesnest.metrics.topology import alpha_topo_new
from hawkesnest.simulator.hawkes import HawkesSimulator


# -------------------------
# Small utilities
# -------------------------

def gini(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float).ravel()
    if vals.size == 0:
        return 0.0
    vals = vals[vals >= 0.0]
    if vals.size == 0:
        return 0.0
    mean_val = vals.mean()
    if mean_val == 0.0:
        return 0.0
    diff_sum = np.abs(vals[:, None] - vals[None, :]).sum()
    return float(diff_sum / (2.0 * vals.size * vals.size * mean_val))


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float).ravel()
    total = counts.sum()
    if total <= 0.0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0.0]
    return float(-(probs * np.log(probs)).sum())


def _format_val(val: float) -> str:
    """Safe string for filenames."""
    return f"{val:.3f}".replace(".", "p")


# -------------------------
# Summary features
# -------------------------

def compute_summary_features(
    df_events: pd.DataFrame,
    domain: Any,
    T: float,
    num_time_bins: int = 10,
    num_space_bins: int = 8,
    num_time_bins_st: int = 5,
    num_space_bins_st: int = 4,
) -> dict[str, float]:
    """
    Compute coarse summary features for identifiability experiments.
    """
    features: dict[str, float] = {}
    n_events = len(df_events)
    features["n_events"] = float(n_events)

    domain_area = (domain.x_max - domain.x_min) * (domain.y_max - domain.y_min)
    lambda_hat_global = n_events / (T * domain_area) if T > 0 and domain_area > 0 else 0.0
    features["lambda_hat_global"] = float(lambda_hat_global)

    # Inter-event times
    if n_events > 1 and "t" in df_events:
        times = np.sort(df_events["t"].to_numpy(dtype=float))
        iet = np.diff(times)
        if iet.size > 0:
            iet_mean = float(np.mean(iet))
            iet_std = float(np.std(iet))
            features["iet_mean"] = iet_mean
            features["iet_std"] = iet_std
            features["iet_cv"] = float(iet_std / iet_mean) if iet_mean > 0 else 0.0
        else:
            features["iet_mean"] = 0.0
            features["iet_std"] = 0.0
            features["iet_cv"] = 0.0
    else:
        features["iet_mean"] = 0.0
        features["iet_std"] = 0.0
        features["iet_cv"] = 0.0

    # Temporal inhomogeneity
    if n_events > 0 and "t" in df_events:
        counts_time, _ = np.histogram(
            df_events["t"].to_numpy(dtype=float),
            bins=num_time_bins,
            range=(0.0, T),
        )
        features["time_bin_gini"] = gini(counts_time)
        features["time_bin_entropy"] = entropy_from_counts(counts_time)
    else:
        features["time_bin_gini"] = 0.0
        features["time_bin_entropy"] = 0.0

    # Spatial moments + grid inhomogeneity
    if n_events > 0 and {"x", "y"}.issubset(df_events.columns):
        xs = df_events["x"].to_numpy(dtype=float)
        ys = df_events["y"].to_numpy(dtype=float)
        features["mean_x"] = float(np.mean(xs))
        features["mean_y"] = float(np.mean(ys))
        features["var_x"] = float(np.var(xs))
        features["var_y"] = float(np.var(ys))
        features["cov_xy"] = float(np.cov(xs, ys)[0, 1]) if xs.size > 1 else 0.0

        counts_space, _, _ = np.histogram2d(
            xs,
            ys,
            bins=(num_space_bins, num_space_bins),
            range=((domain.x_min, domain.x_max), (domain.y_min, domain.y_max)),
        )
        counts_space_flat = counts_space.ravel()
        features["space_bin_gini"] = gini(counts_space_flat)
        features["space_bin_entropy"] = entropy_from_counts(counts_space_flat)
    else:
        features.update(
            {
                "mean_x": 0.0,
                "mean_y": 0.0,
                "var_x": 0.0,
                "var_y": 0.0,
                "cov_xy": 0.0,
                "space_bin_gini": 0.0,
                "space_bin_entropy": 0.0,
            }
        )

    # Mark features + crude transition structure
    marks = df_events["mark"].to_numpy(dtype=int) if "mark" in df_events else np.ones(n_events, dtype=int)
    if marks.size > 0:
        unique, counts = np.unique(marks, return_counts=True)
        probs = counts / counts.sum()
        for m, p in zip(unique, probs, strict=False):
            features[f"mark_prob_{int(m)}"] = float(p)
        features["mark_entropy"] = entropy_from_counts(counts)

        order = np.argsort(df_events["t"].to_numpy(dtype=float)) if "t" in df_events else np.arange(n_events)
        marks_sorted = marks[order]
        if marks_sorted.size > 1:
            trans_counts: dict[tuple[int, int], int] = {}
            for a, b in zip(marks_sorted[:-1], marks_sorted[1:], strict=False):
                key = (int(a), int(b))
                trans_counts[key] = trans_counts.get(key, 0) + 1

            max_mark = int(marks_sorted.max())
            trans_mat = np.zeros((max_mark + 1, max_mark + 1), dtype=float)
            for (a, b), cnt in trans_counts.items():
                trans_mat[a, b] += cnt

            row_sums = trans_mat.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                trans_probs = np.divide(trans_mat, row_sums, out=np.zeros_like(trans_mat), where=row_sums > 0)

            diag_vals = []
            off_vals = []
            for idx in range(trans_probs.shape[0]):
                if row_sums[idx] > 0:
                    diag_vals.append(trans_probs[idx, idx])
                    for j in range(trans_probs.shape[1]):
                        if j != idx and trans_probs[idx, j] > 0:
                            off_vals.append(trans_probs[idx, j])

            features["self_exc_index"] = float(np.mean(diag_vals)) if diag_vals else 0.0
            features["cross_exc_index"] = float(np.mean(off_vals)) if off_vals else 0.0
        else:
            features["self_exc_index"] = 0.0
            features["cross_exc_index"] = 0.0
    else:
        features["mark_entropy"] = 0.0
        features["self_exc_index"] = 0.0
        features["cross_exc_index"] = 0.0

    # Coarse space–time binning
    if n_events > 0 and {"t", "x", "y"}.issubset(df_events.columns):
        t_arr = df_events["t"].to_numpy(dtype=float)
        x_arr = df_events["x"].to_numpy(dtype=float)
        y_arr = df_events["y"].to_numpy(dtype=float)

        if T > 0:
            t_idx = np.clip((t_arr / T * num_time_bins_st).astype(int), 0, num_time_bins_st - 1)
        else:
            t_idx = np.zeros_like(t_arr, dtype=int)

        if domain.x_max > domain.x_min:
            x_idx = np.clip(
                ((x_arr - domain.x_min) / (domain.x_max - domain.x_min) * num_space_bins_st).astype(int),
                0,
                num_space_bins_st - 1,
            )
        else:
            x_idx = np.zeros_like(x_arr, dtype=int)

        if domain.y_max > domain.y_min:
            y_idx = np.clip(
                ((y_arr - domain.y_min) / (domain.y_max - domain.y_min) * num_space_bins_st).astype(int),
                0,
                num_space_bins_st - 1,
            )
        else:
            y_idx = np.zeros_like(y_arr, dtype=int)

        counts_st = np.zeros((num_time_bins_st, num_space_bins_st, num_space_bins_st), dtype=float)
        for ti, xi, yi in zip(t_idx, x_idx, y_idx, strict=False):
            counts_st[ti, xi, yi] += 1.0

        counts_st_flat = counts_st.ravel()
        features["st_bin_gini"] = gini(counts_st_flat)
        features["st_bin_entropy"] = entropy_from_counts(counts_st_flat)
    else:
        features["st_bin_gini"] = 0.0
        features["st_bin_entropy"] = 0.0

    return features


# -------------------------
# Existing 1D / 2D helpers
# -------------------------

def simulate_and_metrics(
    mode: str,
    theta_het: float,
    theta_ent: float,
    seed: int,
    T: float,
    lambda0: float,
    adjacency: np.ndarray,
) -> Dict[str, Any]:
    domain = RectangleDomain(0.0, 1.0, 0.0, 1.0)

    if mode in {"het", "joint"} and theta_het > 0.0:
        bg_cfg = HeteroLadderBackgroundCfg(lambda0=lambda0, theta_het=theta_het, T=T)
        background = bg_cfg.build(domain=domain)
    else:
        background = ConstantBackgroundCfg(rate=lambda0).build()

    if mode in {"ent", "joint"} and theta_ent > 0.0:
        kernel_cfg = EntangledExpGaussianKernelCfg(theta_ent=theta_ent)
    else:
        kernel_cfg = SeparableKernelCfg()
    kernel_obj = kernel_cfg.build()

    alpha_ent_mi = gaussian_entanglement_index(theta_ent) if mode in {"ent", "joint"} else 0.0
    kernels = {(1, 1): kernel_obj}

    sim = HawkesSimulator(
        domain=domain,
        background=background,
        kernels=kernels,
        adjacency=np.asarray(adjacency, dtype=float),
        lambda_max=None,
    )

    df, _ = sim.simulate(n=None, horizon=T, seed=seed, debug=False)
    n_events = len(df)

    alpha_het_bg, _ = background_heterogeneity_index(
        background=background,
        domain=domain,
        T=T,
        n_grid_space=64,
        n_grid_time=1,
    )

    return {
        "theta_het": theta_het,
        "theta_ent": theta_ent,
        "seed": seed,
        "n_events": n_events,
        "alpha_het_bg": alpha_het_bg,
        "alpha_ent_mi": alpha_ent_mi,
    }


# -------------------------
# 4D sweep machinery
# -------------------------

@dataclass
class SweepConfig:
    theta_het: float
    theta_ent: float
    theta_topo: float
    theta_graph: float
    rep: int
    T: float
    lambda0: float
    num_marks: int
    seed_offset: int
    out_dir: str
    tau_max: float | None = 0.5
    debug: bool = False
    field_name_bg: str = "moving_gauss_slow"
    ent_option: str = "rt"


def _dbg(cfg: SweepConfig, *msg: object) -> None:
    if cfg.debug:
        print(
            f"[DBG cfg={cfg.theta_het},{cfg.theta_ent},{cfg.theta_topo},{cfg.theta_graph},rep={cfg.rep}]",
            *msg,
            flush=True,
        )


def _validate_domain(cfg: SweepConfig, domain: Any) -> None:
    """
    Enforce the contract:
      - theta_topo == 0 -> RectangleDomain
      - theta_topo > 0 -> not RectangleDomain (TopologyRGG domain)
    """
    if cfg.theta_topo == 0.0:
        if not isinstance(domain, RectangleDomain):
            raise ValueError(
                f"[Domain Error] theta_topo=0.0 requires RectangleDomain, got {type(domain).__name__}. "
                f"Config=({cfg.theta_het},{cfg.theta_ent},{cfg.theta_topo},{cfg.theta_graph},rep={cfg.rep})"
            )
    else:
        if isinstance(domain, RectangleDomain):
            raise ValueError(
                f"[Domain Error] theta_topo={cfg.theta_topo} > 0 requires TopologyRGG domain, got RectangleDomain. "
                f"Config=({cfg.theta_het},{cfg.theta_ent},{cfg.theta_topo},{cfg.theta_graph},rep={cfg.rep})"
            )


def run_single_config(cfg: SweepConfig) -> dict[str, Any]:
    """
    Worker for a single 4D configuration.
    """

    seed = cfg.seed_offset + cfg.rep
    t0 = time.time()
    _dbg(cfg, f"Start seed={seed}")

    # Domain pillar
    #print(f"[STEP] build_domain start θ_topo={cfg.theta_topo} seed={seed}", flush=True)

    if cfg.theta_topo > 0.0:
        domain_cfg = TopologyRGGDomainCfg(theta_topo=cfg.theta_topo, seed=seed)
        domain = domain_cfg.build()
        _dbg(cfg, f"Domain=TopologyRGG theta_topo={cfg.theta_topo}")
    else:
        domain = RectangleDomain(0.0, 1.0, 0.0, 1.0)
        _dbg(cfg, "Domain=Rectangle")
    _validate_domain(cfg, domain)
    if cfg.debug:
        dom_type = type(domain).__name__
        if hasattr(domain, "G"):
            print(
                f"[CFG DEBUG] domain={dom_type} |V|={domain.G.number_of_nodes()} |E|={domain.G.number_of_edges()} "
                f"apsp={'_apsp' in domain.__dict__} tau_max={cfg.tau_max}",
                flush=True,
            )
        else:
            print(f"[CFG DEBUG] domain={dom_type} tau_max={cfg.tau_max}", flush=True)
    #print(f"[STEP] build_domain done type={type(domain).__name__}", flush=True)

    # Background pillar
    if cfg.theta_het > 0.0:
        bg_cfg = HeteroLadderBackgroundCfg(lambda0=cfg.lambda0, theta_het=cfg.theta_het, T=cfg.T, field_name = cfg.field_name_bg)
        background = bg_cfg.build(domain=domain)

        _dbg(cfg, f"Background=HeteroLadder theta_het={cfg.theta_het}")
    else:
        background = ConstantBackgroundCfg(rate=cfg.lambda0).build()
        _dbg(cfg, "Background=Constant")

    #Kernel pillar
    if cfg.theta_ent > 0.0:
        kernel_cfg = EntangledExpGaussianKernelCfg(
            theta_ent=cfg.theta_ent,
            tau_max=float(cfg.tau_max) if cfg.tau_max is not None else 5.0,
            ent_option=cfg.ent_option,
        )
        _dbg(cfg, f"Kernel=Entangled theta_ent={cfg.theta_ent}")


    else:
        kernel_cfg = SeparableKernelCfg()
        _dbg(cfg, "Kernel=Separable")
    kernel_obj = kernel_cfg.build()

    # Graph pillar: always use cfg.num_marks and build neutral/structured A based on theta_graph
    num_marks = cfg.num_marks
    if cfg.theta_graph <= 0.0:
        #A = np.eye(num_marks, dtype=float) * 0.6
        A = np.array([[0.6]], dtype=float)
    else:
        A = BranchingMatrixCfg(theta_graph=cfg.theta_graph, M=num_marks).build()
        sA = spectral_norm(A)
        if sA >= 0.99:   # or >= norm_max, depending on your convention
            raise ValueError(f"Unstable A: ||A||_2={sA:.3f} for theta_graph={cfg.theta_graph}")

    kernels = {(i, j): kernel_obj for i in range(1, num_marks + 1) for j in range(1, num_marks + 1)}
    _dbg(cfg, f"Graph={num_marks}-mark theta_graph={cfg.theta_graph}")

    sim = HawkesSimulator(
        domain=domain,
        background=background,
        kernels=kernels,
        adjacency=A,
        lambda_max=None,
    )

    #print("[STEP] simulate start", flush=True)

    try:
        df_events, _ = sim.simulate(n=None, horizon=cfg.T, seed=seed, debug=False, tau_max=cfg.tau_max)
    except TypeError:
        df_events, _ = sim.simulate(n=None, horizon=cfg.T, seed=seed, debug=False)
    print(f"[STEP] simulate done n={len(df_events)}", flush=True)
    df_events = df_events.copy()
    df_events["theta_het"] = cfg.theta_het
    df_events["theta_ent"] = cfg.theta_ent
    df_events["theta_topo"] = cfg.theta_topo
    df_events["theta_graph"] = cfg.theta_graph
    df_events["replicate"] = cfg.rep

    alpha_het_bg, _ = background_heterogeneity_index(
        background=background,
        domain=domain,
        T=cfg.T,
        n_grid_space=64,
        n_grid_time=1,
    )
    alpha_ent_mi = gaussian_entanglement_index(cfg.theta_ent)
    print("[STEP] alpha_topo start", flush=True)
    alpha_topo = alpha_topo_new(domain)
    print(f"[STEP] alpha_topo done {alpha_topo}", flush=True)
    alpha_graph = alpha_graph_new(A)
    print(f"[STEP] alpha_het_bg={alpha_het_bg:.4f} alpha_ent_mi={alpha_ent_mi:.4f} "
          f"alpha_topo={alpha_topo:.4f} alpha_graph={alpha_graph:.4f}", flush=True)
    ev_fname = (
        f"events_het{_format_val(cfg.theta_het)}"
        f"_ent{_format_val(cfg.theta_ent)}"
        f"_topo{_format_val(cfg.theta_topo)}"
        f"_graph{_format_val(cfg.theta_graph)}"
        f"_rep{cfg.rep}.csv"
    )
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ev_path = out_dir / ev_fname
    df_events.to_csv(ev_path, index=False)

    summary = compute_summary_features(df_events=df_events, domain=domain, T=cfg.T)

    record: dict[str, Any] = {
        "theta_het": cfg.theta_het,
        "theta_ent": cfg.theta_ent,
        "theta_topo": cfg.theta_topo,
        "theta_graph": cfg.theta_graph,
        "replicate": cfg.rep,
        "seed": seed,
        "n_events": len(df_events),
        "alpha_het_bg": alpha_het_bg,
        "alpha_ent_mi": alpha_ent_mi,
        "alpha_topo": alpha_topo,
        "alpha_graph": alpha_graph,
        "events_path": str(ev_path),
        "num_marks_used": num_marks,
    }
    # Avoid collision: explicit n_events above is source of truth
    summary.pop("n_events", None)
    record.update(summary)

    dt = time.time() - t0
    record["wall_time_sec"] = float(dt)

    # Flag slow configs without spamming
    if dt > 5.0:
        print(
            f"[SLOW] θ=({cfg.theta_het},{cfg.theta_ent},{cfg.theta_topo},{cfg.theta_graph}) "
            f"rep={cfg.rep} took {dt:.1f}s n={len(df_events)} marks={num_marks}",
            flush=True,
        )

    _dbg(cfg, f"Done in {dt:.2f}s n={len(df_events)} alpha_topo={alpha_topo:.3f} alpha_graph={alpha_graph:.3f}")
    return record


def run_complexity_sweep_4d(
    out_dir: Path,
    replicates: int,
    T: float,
    lambda0: float,
    num_marks: int,
    theta_het_grid: np.ndarray,
    theta_ent_grid: np.ndarray,
    theta_topo_grid: np.ndarray,
    theta_graph_grid: np.ndarray,
    results_filename: str = "results_4d.csv",
    use_parallel: bool = False,
    max_workers: int | None = None,
    tau_max: float | None = 0.5,
    field_name_bg: str = "moving_gauss_slow",
    ent_option: str = "rt",
    seed_offset: int = 1234,
    # new diagnostics/safety knobs
    debug: bool = False,
    progress_every: int = 5,
    fail_fast: bool = False,
    worker_timeout: float | None = None,
) -> pd.DataFrame:
    """
    Run the full 4D grid and save events + metrics + identifiability features.

    Progress is reported on COMPLETION of configs (not on enumeration),
    so you can see whether it's actually stuck or just slow.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    configs: list[SweepConfig] = []
    for theta_het in theta_het_grid:
        for theta_ent in theta_ent_grid:
            for theta_topo in theta_topo_grid:
                for theta_graph in theta_graph_grid:
                    for rep in range(replicates):
                        configs.append(
                            SweepConfig(
                                theta_het=float(theta_het),
                                theta_ent=float(theta_ent),
                                theta_topo=float(theta_topo),
                                theta_graph=float(theta_graph),
                                rep=int(rep),
                                T=float(T),
                                lambda0=float(lambda0),
                                num_marks=int(num_marks) if float(theta_graph) > 0.0 else 1,
                                seed_offset=int(seed_offset),
                                out_dir=str(out_dir),
                                tau_max=float(tau_max) if tau_max is not None else None,
                                # avoid noisy interleaved output in parallel
                                debug=bool(debug) and (not use_parallel),
                                field_name_bg=str(field_name_bg),
                                ent_option=str(ent_option),
                            )
                        )

    total = len(configs)
    print(f"[INFO] 4D sweep configs: {total} (replicates={replicates})")
    print(
        "[INFO] 4D sweep settings: "
        f"tau_max={tau_max}, field_name_bg={field_name_bg}, "
        f"ent_option={ent_option}, seed_offset={seed_offset}"
    )
    if use_parallel:
        print(f"[INFO] running in parallel (max_workers={max_workers})")
        if worker_timeout is not None:
            print(f"[INFO] worker timeout enabled: {worker_timeout}s")
    else:
        print("[INFO] running sequentially")

    results: list[dict[str, Any]] = []
    t_all0 = time.time()

    if use_parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(run_single_config, cfg) for cfg in configs]
            done = 0
            for fut in as_completed(futures):
                try:
                    rec = fut.result(timeout=worker_timeout) if worker_timeout is not None else fut.result()
                    results.append(rec)
                    done += 1
                    if progress_every > 0 and (done % progress_every == 0 or done == total):
                        elapsed = time.time() - t_all0
                        print(f"[PROGRESS] {done}/{total} completed | elapsed={elapsed:.1f}s", flush=True)
                except TimeoutError as e:
                    msg = f"[WORKER TIMEOUT] A config exceeded {worker_timeout}s."
                    print(msg, flush=True)
                    if fail_fast:
                        raise TimeoutError(msg) from e
                except Exception as e:
                    print(f"[WORKER ERROR] {e}", flush=True)
                    if fail_fast:
                        raise
    else:
        for i, cfg in enumerate(configs, start=1):
            print(f"[INFO] Running config {i}/{total}: θ=({cfg.theta_het},{cfg.theta_ent},{cfg.theta_topo},{cfg.theta_graph}) rep={cfg.rep}", flush=True)
            rec = run_single_config(cfg)
            results.append(rec)
            if progress_every > 0 and (i % progress_every == 0 or i == total):
                elapsed = time.time() - t_all0
                print(f"[PROGRESS] {i}/{total} completed | elapsed={elapsed:.1f}s", flush=True)

    df_results = pd.DataFrame.from_records(results)
    results_path = out_dir / results_filename
    df_results.to_csv(results_path, index=False)
    elapsed_all = time.time() - t_all0
    print(f"[OK] wrote 4D sweep results to {results_path} | total_elapsed={elapsed_all:.1f}s")
    return df_results


# -------------------------
# Existing sweeps / plots (unchanged)
# -------------------------

def graph_sweep(theta_graph_grid: Iterable[float], replicates: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for theta_graph in theta_graph_grid:
        print(f"[INFO] theta_graph={theta_graph}")
        for rep in range(replicates):
            print(f"  [INFO] replicate {rep+1}/{replicates}")
            seed = 1234 + rep
            cfg = BranchingMatrixCfg(theta_graph=theta_graph, M=3)
            A = cfg.build()
            try:
                a_graph = alpha_graph_new(A)
            except Exception as e:  # pragma: no cover
                print(f"[WARN] alpha_graph failed for θ_graph={theta_graph}, rep={rep}: {e}")
                a_graph = np.nan
            records.append({"theta_graph": theta_graph, "seed": seed, "alpha_graph": a_graph})
    return pd.DataFrame.from_records(records)


def topo_sweep(theta_topo_grid: Iterable[float], replicates: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for theta_topo in theta_topo_grid:
        print(f"[INFO] theta_topo={theta_topo}")
        for rep in range(replicates):
            print(f"  [INFO] replicate {rep+1}/{replicates}")
            seed = 1234 + rep
            domain_cfg = TopologyRGGDomainCfg(theta_topo=theta_topo, seed=seed)
            domain = domain_cfg.build()
            try:
                a_topo = alpha_topo_new(domain)
            except Exception as e:  # pragma: no cover
                print(f"[WARN] alpha_topo failed for θ_topo={theta_topo}, rep={rep}: {e}")
                a_topo = np.nan
            records.append({"theta_topo": theta_topo, "seed": seed, "alpha_topo": a_topo})
    return pd.DataFrame.from_records(records)


def plot_metrics(
    df: pd.DataFrame,
    x_col: str,
    metric_cols: list[str],
    out_prefix: Path,
    base_name: str,
    x_label: str,
    y_label: str,
    title_prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    if x_col not in df.columns:
        print(f"[WARN] plot_metrics: x_col '{x_col}' not found in DataFrame columns {list(df.columns)}")
        return

    present_metrics = [m for m in metric_cols if m in df.columns]
    if not present_metrics:
        print(
            f"[WARN] plot_metrics: none of the requested metric_cols {metric_cols} "
            f"are present in DataFrame columns {list(df.columns)}"
        )
        return

    cols_for_na = [x_col] + present_metrics
    df_clean = df.dropna(subset=cols_for_na)
    if df_clean.empty:
        print(
            f"[WARN] plot_metrics: no valid rows after dropna for {base_name} "
            f"(x_col={x_col}, metrics={present_metrics})."
        )
        return

    grouped = df_clean.groupby(x_col)[present_metrics].agg(["mean", "std"]).reset_index()

    def _flatten_col(c):
        if not isinstance(c, tuple):
            return c
        if len(c) == 1 or c[1] in ("", None):
            return c[0]
        return f"{c[0]}_{c[1]}"

    grouped.columns = [_flatten_col(c) for c in grouped.columns]
    xs = grouped[x_col]

    plt.figure()
    for metric in present_metrics:
        m_col = f"{metric}_mean"
        s_col = f"{metric}_std"
        if m_col not in grouped or s_col not in grouped:
            continue
        plt.errorbar(xs, grouped[m_col], yerr=grouped[s_col], marker="o", linestyle="-", label=metric)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title_prefix} metrics vs {x_col}")
    plt.grid(True, linestyle="--", alpha=0.4)
    if len(present_metrics) > 1:
        plt.legend()

    out_path = out_prefix.with_suffix(f".{base_name}_all.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] saved {base_name} combined metrics plot to {out_path}")

    for metric in present_metrics:
        m_col = f"{metric}_mean"
        s_col = f"{metric}_std"
        if m_col not in grouped or s_col not in grouped:
            continue
        plt.figure()
        plt.errorbar(xs, grouped[m_col], yerr=grouped[s_col], marker="o", linestyle="-")
        plt.xlabel(x_label)
        plt.ylabel(metric)
        plt.title(f"{title_prefix}: {metric} vs {x_col}")
        plt.grid(True, linestyle="--", alpha=0.4)
        out_path_metric = out_prefix.with_suffix(f".{base_name}.{metric}.png")
        plt.tight_layout()
        plt.savefig(out_path_metric, dpi=150)
        plt.close()
        print(f"[OK] saved {base_name} metric plot for {metric} to {out_path_metric}")


def run_complexity_sweep(
    mode: str,
    out: Path,
    replicates: int,
    T: float,
    lambda0: float,
    theta_het_grid: np.ndarray,
    theta_ent_grid: np.ndarray,
    theta_topo_grid: np.ndarray,
    theta_graph_grid: np.ndarray,
    num_marks: int | None = None,
) -> pd.DataFrame:
    adjacency = np.array([[0.3]])
    records: list[dict[str, Any]] = []

    if mode == "het":
        print("[INFO] running heterogeneity sweep")
        for theta_het in theta_het_grid:
            print(f"[INFO] theta_het={theta_het}")
            for rep in range(replicates):
                print(f"  [INFO] replicate {rep+1}/{replicates}")
                seed = 1234 + rep
                rec = simulate_and_metrics(
                    mode="het",
                    theta_het=theta_het,
                    theta_ent=0.0,
                    seed=seed,
                    T=T,
                    lambda0=lambda0,
                    adjacency=adjacency,
                )
                records.append(rec)
        df = pd.DataFrame.from_records(records)

    elif mode == "ent":
        print("[INFO] running entanglement sweep")
        for theta_ent in theta_ent_grid:
            print(f"[INFO] theta_ent={theta_ent}")
            for rep in range(replicates):
                print(f"  [INFO] replicate {rep+1}/{replicates}")
                seed = 1234 + rep
                rec = simulate_and_metrics(
                    mode="ent",
                    theta_het=0.0,
                    theta_ent=theta_ent,
                    seed=seed,
                    T=T,
                    lambda0=lambda0,
                    adjacency=adjacency,
                )
                records.append(rec)
        df = pd.DataFrame.from_records(records)

    elif mode == "joint":
        print("[INFO] running joint heterogeneity–entanglement sweep")
        for theta_het in theta_het_grid:
            print(f"[INFO] theta_het={theta_het}")
            for theta_ent in theta_ent_grid:
                print(f"  [INFO] theta_ent={theta_ent}")
                for rep in range(replicates):
                    print(f"    [INFO] replicate {rep+1}/{replicates}")
                    seed = 1234 + rep
                    rec = simulate_and_metrics(
                        mode="joint",
                        theta_het=theta_het,
                        theta_ent=theta_ent,
                        seed=seed,
                        T=T,
                        lambda0=lambda0,
                        adjacency=adjacency,
                    )
                    records.append(rec)
        df = pd.DataFrame.from_records(records)

    elif mode == "topo":
        print("[INFO] running topology sweep")
        df = topo_sweep(theta_topo_grid=theta_topo_grid, replicates=replicates)

    elif mode == "graph":
        print("[INFO] running graph / mark-interaction sweep")
        df = graph_sweep(theta_graph_grid=theta_graph_grid, replicates=replicates)

    else:
        raise ValueError(f"Unknown mode '{mode}'")

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[OK] wrote results to {out}")

    out_prefix = out.with_suffix("")
    if mode == "het":
        plot_metrics(
            df=df,
            x_col="theta_het",
            metric_cols=["alpha_het_bg"],
            out_prefix=out_prefix,
            base_name="het",
            x_label=r"$\theta_{\mathrm{het}}$",
            y_label=r"$\alpha_{\mathrm{het}}$",
            title_prefix="Heterogeneity",
        )
    elif mode == "ent":
        plot_metrics(
            df=df,
            x_col="theta_ent",
            metric_cols=["alpha_ent_mi"],
            out_prefix=out_prefix,
            base_name="ent",
            x_label=r"$\theta_{\mathrm{ent}}$",
            y_label=r"$\alpha_{\mathrm{ent}}$",
            title_prefix="Entanglement",
        )
    elif mode == "joint":
        plot_metrics(
            df=df,
            x_col="theta_het",
            metric_cols=["alpha_het_bg"],
            out_prefix=out_prefix,
            base_name="joint_het",
            x_label=r"$\theta_{\mathrm{het}}$",
            y_label=r"$\alpha_{\mathrm{het}}$",
            title_prefix="Joint (heterogeneity slice)",
        )
        plot_metrics(
            df=df,
            x_col="theta_ent",
            metric_cols=["alpha_ent_mi"],
            out_prefix=out_prefix,
            base_name="joint_ent",
            x_label=r"$\theta_{\mathrm{ent}}$",
            y_label=r"$\alpha_{\mathrm{ent}}$",
            title_prefix="Joint (entanglement slice)",
        )
    elif mode == "topo":
        plot_metrics(
            df=df,
            x_col="theta_topo",
            metric_cols=["alpha_topo"],
            out_prefix=out_prefix,
            base_name="topo",
            x_label=r"$\theta_{\mathrm{topo}}$",
            y_label=r"$\alpha_{\mathrm{topo}}$",
            title_prefix="Topology",
        )
    elif mode == "graph":
        plot_metrics(
            df=df,
            x_col="theta_graph",
            metric_cols=["alpha_graph"],
            out_prefix=out_prefix,
            base_name="graph",
            x_label=r"$\theta_{\mathrm{graph}}$",
            y_label=r"$\alpha_{\mathrm{graph}}$",
            title_prefix="Graph / mark-interaction",
        )

    return df
