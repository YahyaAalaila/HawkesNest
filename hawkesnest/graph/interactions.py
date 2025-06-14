# hawkesnest/graph/interaction.py
"""
MarkInteractionGraph: encapsulates the multi-type adjacency matrix, computes
its spectral norm, modularity, and combined complexity index.
Also supports estimation of the branching matrix from event data.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Callable, Sequence, Tuple
try:
    import community as community_louvain
except ImportError:
    community_louvain = None

class MarkInteractionGraph:
    """
    Graph of marks with weighted, directed edges capturing branching ratios.
    """
    def __init__(
        self,
        A: np.ndarray,
        norm_max: float = 0.95,
        Q_max: float = 1.0
    ):
        """
        Parameters:
        -----------
        A : (M, M) array
            Branching ratios eta_{mn} where A[m, n] is expected offspring of type m from type n.
        norm_max : float
            Maximum expected spectral norm under stability (default 0.95).
        Q_max : float
            Maximum modularity (default 1.0).
        """
        self.A = A
        self.M = A.shape[0]
        self.norm_max = norm_max
        self.Q_max = Q_max
        self._G_undirected = None

    @classmethod
    def from_branching_matrix(
        cls,
        A: np.ndarray,
        norm_max: float = 0.95,
        Q_max: float = 1.0
    ) -> MarkInteractionGraph:
        """Construct directly from known branching matrix."""
        return cls(A.copy(), norm_max=norm_max, Q_max=Q_max)

    @classmethod
    def estimate_from_events(
        cls,
        events: Sequence[Tuple[np.ndarray, float, int]],
        kernel_funcs: dict[Tuple[int,int], Callable],
        domain: any,
        max_lag: float = None,
        norm_max: float = 0.95,
        Q_max: float = 1.0
    ) -> MarkInteractionGraph:
        """
        Estimate A from event history and known kernel shapes by integrating
        trigger functions over the domain.

        events: list of (s, t, m) tuples
        kernel_funcs: mapping (m,n) -> phi_{mn}(s,tau)
        domain: provides integration bounds for spatial and temporal dims
        max_lag: optional temporal cutoff for integration
        """
        M = max(m for *_ , m in events) + 1
        A = np.zeros((M, M), dtype=float)
        # integrate phi over space-time or approximate by Monte Carlo
        # here we assume kernel_funcs returns normalized pdf times eta
        # user must supply kernels normalized to eta
        for (m, n), phi in kernel_funcs.items():
            # Monte Carlo integration: sample U uniform on domain
            # sample tau uniform in [0, max_lag]
            # approximate integral
            # Placeholder: user should implement accurate integration
            raise NotImplementedError("Estimation from events not yet implemented.")
        return cls(A, norm_max=norm_max, Q_max=Q_max)

    def to_networkx(self, directed: bool = False) -> nx.Graph:
        """
        Build a NetworkX graph from the branching matrix.

        directed: if True, return a DiGraph, else an undirected Graph.
        """
        G = nx.DiGraph() if directed else nx.Graph()
        for m in range(self.M):
            G.add_node(m)
        for m in range(self.M):
            for n in range(self.M):
                w = float(self.A[m, n])
                if w > 0:
                    if directed:
                        G.add_edge(n, m, weight=w)
                    else:
                        # symmetrize by summing both directions
                        if G.has_edge(m, n):
                            G[m][n]['weight'] += w
                        else:
                            G.add_edge(m, n, weight=w)
        self._G_undirected = G if not directed else None
        return G

    def spectral_norm(self) -> float:
        """Return the spectral norm (largest singular value) of A."""
        # SVD returns singular values sorted descending
        s = np.linalg.svd(self.A, compute_uv=False)
        return float(s[0]) if s.size > 0 else 0.0

    def modularity(self, resolution: float = 1.0) -> float:
        """Compute Newman modularity of the undirected symmetrized graph."""
        if community_louvain is None:
            raise ImportError("python-louvain is required to compute modularity")
        if self._G_undirected is None:
            self.to_networkx(directed=False)
        partition = community_louvain.best_partition(
            self._G_undirected, weight='weight', resolution=resolution
        )
        Q = community_louvain.modularity(partition, self._G_undirected, weight='weight')
        return float(Q)

    def alpha_graph(self) -> float:
        """Combine spectral norm and modularity into a [0,1] complexity index."""
        norm = self.spectral_norm() / self.norm_max
        Q = self.modularity() / self.Q_max
        return float(norm * np.sqrt(Q))
