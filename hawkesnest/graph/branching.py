from __future__ import annotations

import numpy as np

from hawkesnest.metrics.graph import spectral_norm


def _build_branching_legacy(theta_graph: float, n_types: int, norm_max: float) -> np.ndarray:
    """Original interpolation used in the graph/mark pillar."""
    base_self = 0.3  # diagonal (self-excitation)
    base_cross = 0.15  # off-diagonal scale

    A = np.full((n_types, n_types), 0, dtype=float)
    np.fill_diagonal(A, base_self)
    return A


def _block_assignments(n_types: int, n_blocks: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic block labels z (len n_types) and indicator matrix U (n_types x n_blocks).

    Types are assigned round-robin to blocks to avoid θ-dependent randomness.
    """
    z = np.arange(n_types) % n_blocks
    U = np.zeros((n_types, n_blocks), dtype=float)
    U[np.arange(n_types), z] = 1.0
    return z, U


def _build_branching_low_rank(
    theta_graph: float,
    n_types: int,
    n_blocks: int,
    self_excitation: float,
    within_strength_base: float,
    within_strength_boost: float,
    norm_max: float,
) -> np.ndarray:
    """
    Low-rank block model: θ_graph interpolates between homogeneous and modular block couplings.

    At θ=0: S_theta is homogeneous (no modular advantage) -> near-zero modularity.
    At θ=1: within-block weights > between-block weights -> strong modularity.
    """
    _, U = _block_assignments(n_types, n_blocks)

    # Homogeneous baseline: all blocks interact with equal strength
    S_weak = within_strength_base * np.ones((n_blocks, n_blocks), dtype=float)

    # Structured target: boost within-block entries
    S_strong = within_strength_base * np.ones((n_blocks, n_blocks), dtype=float)
    np.fill_diagonal(S_strong, within_strength_base + within_strength_boost)

    # Interpolate in S-space
    S_theta = (1.0 - theta_graph) * S_weak + theta_graph * S_strong

    # Low-rank block structure in type space
    A_blocks = U @ S_theta @ U.T

    # Keep cross-structure off-diagonal only; self-excitation separate
    np.fill_diagonal(A_blocks, 0.0)

    # Add self-excitation on the diagonal
    A_raw = self_excitation * np.eye(n_types, dtype=float) + A_blocks

    np.clip(A_raw, 0.0, None, out=A_raw)

    # For graph pillar: fix spectral norm exactly at norm_max (if possible)
    s = spectral_norm(A_raw)
    if s > 0.0:
        scale = norm_max / s
        A = A_raw * scale
    else:
        A = A_raw

    return A

def _build_branching_dcsbm(
    theta_graph: float,
    n_types: int,
    n_blocks: int,
    self_excitation: float,      # IGNORED in this mode
    base_strength: float,
    delta_strength: float,
    sparsity: float | None,      # IGNORED in this mode
    norm_max: float,
    rng: np.random.Generator,    # IGNORED in this mode
) -> np.ndarray:
    """
    Monotone graph-pillar ladder: A_theta = theta * B_base.

    - B_base is a fixed block-modular matrix (within > between, diag=0).
    - theta in [0,1] scales the entire interaction structure.
    - We scale B_base once (independent of theta) so that ||A_1||_2 <= norm_max.

    For theta=0: A=0 -> no cross structure, alpha_graph=0.
    For theta>0: Q/Q_max is constant; alpha_graph(theta) is linear in theta
    up to the stability limit.
    """
    # Deterministic block assignments
    z, _ = _block_assignments(n_types, n_blocks)

    # Build unscaled B_base: symmetric, diag=0, within > between
    w_out = base_strength
    w_in  = base_strength + delta_strength

    B_base = np.zeros((n_types, n_types), dtype=float)
    for i in range(n_types):
        for j in range(i + 1, n_types):
            val = w_in if z[i] == z[j] else w_out
            B_base[i, j] = val
            B_base[j, i] = val
    np.fill_diagonal(B_base, 0.0)

    # Global, theta-independent scaling to respect norm_max at theta=1
    # (we allow a small safety margin, e.g. 0.99)
    delta = 0.3

    s_full = spectral_norm(B_base)
    budget = max(0.0, norm_max - delta)  # remaining room
    if s_full > 0.0 and budget > 0.0:
        scale = min(1.0, (budget * 0.99) / s_full)
        B_base *= scale
    elif budget <= 0.0:
        B_base[:] = 0.0

    A = theta_graph * B_base + delta * np.eye(n_types, dtype=float)
    return A





def build_branching_matrix(
    theta_graph: float,
    M: int = 3,
    *,
    mode: str = "dcsbm",
    n_types: int | None = None,
    n_blocks: int = 3,
    norm_max: float = 0.95,
    self_excitation: float = 0.3,
    within_strength_base: float = 0.05,
    within_strength_boost: float = 0.25,
    base_strength: float = 0.1,
    delta_strength: float = 0.23,
    sparsity: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Build a branching / adjacency matrix A(θ_graph) under different parameterisations.

    Modes:
      - legacy/interpolate : original linear interpolation of cross-type strength.
      - low_rank           : low-rank block coupling; θ increases within-block contrast.
      - dcsbm/assortative  : assortative DCSBM-like weights; θ increases modularity.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_types = n_types or M

    if mode == "low_rank":
        return _build_branching_low_rank(
            theta_graph=theta_graph,
            n_types=n_types,
            n_blocks=n_blocks,
            self_excitation=self_excitation,
            within_strength_base=within_strength_base,
            within_strength_boost=within_strength_boost,
            norm_max=norm_max,
        )

    if mode in {"dcsbm", "assortative"}:
        return _build_branching_dcsbm(
            theta_graph=theta_graph,
            n_types=n_types,
            n_blocks=n_blocks,
            self_excitation=self_excitation,
            base_strength=base_strength,
            delta_strength=delta_strength,
            sparsity=sparsity,
            norm_max=norm_max,
            rng=rng,
        )

    # Fallback to legacy behaviour
    return _build_branching_legacy(theta_graph=theta_graph, n_types=n_types, norm_max=norm_max)
