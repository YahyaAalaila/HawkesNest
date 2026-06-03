
import networkx as nx
import numpy as np

try:
    from community import community_louvain
except ImportError:
    # If python-louvain is not installed, fallback
    community_louvain = None


def spectral_norm(A: np.ndarray) -> float:
    """Return the spectral norm (largest singular value) of matrix A."""
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0])


def _graph_and_strengths(A: np.ndarray):
    """
    Build symmetrised weighted graph from A and return:
    - G: networkx.Graph
    - strengths: np.ndarray of node strengths (sum of incident weights)
    """
    # Symmetrise A -> undirected weighted adjacency
    B = (A + A.T) / 2.0
    G = nx.from_numpy_array(B)

    # Node strength = sum of weights of incident edges
    strengths = np.asarray(B.sum(axis=1), dtype=float).reshape(-1)
    return G, strengths


def modularity_with_Qmax(A: np.ndarray) -> tuple[float, float]:
    """
    Compute Newman modularity Q and its theoretical upper bound Q_max
    for the symmetrised branching graph.

    Q_max = 1 - sum_m p_m^2,
    where p_m is the fraction of total strength at node m.
    """
    if community_louvain is None:
        raise ImportError("Please install python-louvain to compute modularity.")

    G, strengths = _graph_and_strengths(A)

    # Total incident weight
    total_strength = float(strengths.sum())
    if total_strength <= 0.0:
        # Graph has no weight; treat as no structure
        return 0.0, 1.0

    # Node-strength distribution
    p = strengths / total_strength

    # Theoretical maximum modularity for a perfectly assortative block structure
    Q_max = 1.0 - float(np.sum(p ** 2))

    # If all mass is on one node (degenerate), Q_max ~ 0: no modular structure
    if Q_max <= 0.0:
        return 0.0, 1e-8  # avoid division by zero downstream

    # Louvain partition and empirical modularity
    partition = community_louvain.best_partition(G, weight="weight")
    Q = community_louvain.modularity(partition, G, weight="weight")

    return float(Q), float(Q_max)


def alpha_graph_new(A: np.ndarray, norm_max: float = 0.95) -> float:
    """
    Compute the multi-type relation complexity index in [0,1].

    Parameters
    ----------
    A : np.ndarray
        MxM branching matrix of expected offspring means.
    norm_max : float
        Maximum allowed spectral norm under stability (default 0.95).

    Returns
    -------
    alpha : float
        Complexity index combining spectral norm and modularity.
    """
    # Spectral norm component (clipped at 1)
    norm_val = spectral_norm(A) / norm_max
    norm_val = float(np.clip(norm_val, 0.0, 1.0))

    # Modularity component with theoretical Q_max
    Q, Q_max = modularity_with_Qmax(A)

    # Clamp negative Q to 0 before normalisation
    Q_pos = max(Q, 0.0)
    Q_ratio = Q_pos / Q_max if Q_max > 0.0 else 0.0
    Q_ratio = float(np.clip(Q_ratio, 0.0, 1.0))

    # Combine with square-root scaling on modularity
    alpha = norm_val * np.sqrt(Q_ratio)

    return float(np.clip(alpha, 0.0, 1.0)) # clamp to [0,1] to make sure, despite theoretically in [0,1]
