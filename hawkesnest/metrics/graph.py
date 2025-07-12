# hawkesnest/metrics/graph.py
"""
Multi-event-type relation complexity metric: combines the spectral norm of the
branching matrix A with the network modularity of its symmetrized graph.

Index:
    alpha_graph = (||A||_2 / norm_max) * sqrt(Q / Q_max)

Where:
- ||A||_2 is the spectral norm (largest singular value) of A.
- Q is Newman modularity on the undirected version of A.
- norm_max < 1 is the max spectral norm under stability (default 0.95).
- Q_max is the theoretical or empirically observed maximum modularity (default 1.0).
"""
import networkx as nx
import numpy as np

try:
    from community import community_louvain
except ImportError:
    # If python-louvain is not installed, fallback
    community_louvain = None


def spectral_norm(A: np.ndarray) -> float:
    """Return the spectral norm (largest singular value) of matrix A."""
    # Compute singular values
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0])


def modularity(A: np.ndarray) -> float:
    """Compute Newman modularity of the weighted undirected graph sym(A)."""
    # Symmetrize A
    B = (A + A.T) / 2
    G = nx.from_numpy_array(B)
    if community_louvain is None:
        raise ImportError("Please install python-louvain to compute modularity.")
    # Compute best partition
    partition = community_louvain.best_partition(G, weight="weight")
    # Compute modularity
    Q = community_louvain.modularity(partition, G, weight="weight")
    return float(Q)


def alpha_graph(A: np.ndarray, norm_max: float = 0.95, Q_max: float = 1.0) -> float:
    """
    Compute the multi-type relation complexity index in [0,1].

    Parameters
    ----------
    A: np.ndarray
        MxM branching matrix of expected offspring means.
    norm_max: float
        Maximum allowed spectral norm under stability (default 0.95).
    Q_max: float
        Maximum modularity for normalization (default 1.0).

    Returns
    -------
    alpha: float
        Complexity index combining spectral norm and modularity.
    """
    # Spectral norm component
    norm_val = spectral_norm(A) / norm_max
    # Modularity component
    Q_val = modularity(A) / Q_max
    # Combine with square-root scaling on modularity
    alpha = norm_val * np.sqrt(Q_val)
    # Clip to [0,1]
    return float(np.clip(alpha, 0.0, 1.0))

