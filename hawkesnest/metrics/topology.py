# hawkesnest/metrics/topology.py
"""
Topology complexity metric: normalised average geodesic-to-Euclidean distance ratio.

Given a spatial support (planar or network), we sample random point pairs,
compute their Euclidean and geodesic distances, and form:

    alpha_topo = (E[d_G] / E[d_E] - 1) / (D_max - 1)

where D_max is a fixed upper calibration constant (e.g. 2.5 for a Sierpiński network).
"""

import numpy as np
from typing import Protocol, Tuple

class SpatialDomain(Protocol):
    """
    Protocol for spatial domains. Must provide:
      - sample_point() -> Tuple[float, float]
      - geodesic(u: Tuple[float,float], v: Tuple[float,float]) -> float
    """
    def sample_point(self) -> Tuple[float, float]:
        ...
    def geodesic(self, u: Tuple[float, float], v: Tuple[float, float]) -> float:
        ...


def alpha_topo(domain: SpatialDomain,
               n_pairs: int = 2000,
               D_max: float = 2.5) -> float:
    """
    Compute the topology index for the given domain.

    Parameters
    ----------
    domain
        An object implementing SpatialDomain.
    n_pairs
        Number of random point pairs to sample.
    D_max
        Calibration constant for maximum expected distortion.

    Returns
    -------
    alpha
        Topology complexity in [0,1].
    """
    # Sample random pairs
    eucl_dists = np.empty(n_pairs)
    geo_dists  = np.empty(n_pairs)
    for i in range(n_pairs):
        u = domain.sample_point()
        v = domain.sample_point()
        eucl_dists[i] = np.linalg.norm(np.subtract(u, v))
        geo_dists[i]  = domain.geodesic(u, v)

    # Compute mean distances
    mean_euc = np.mean(eucl_dists)
    mean_geo = np.mean(geo_dists)
    if mean_euc == 0.0:
        D = 1.0
    else:
        D = mean_geo / mean_euc

    # Normalise to [0,1]
    alpha = (D - 1.0) / (D_max - 1.0)
    return float(np.clip(alpha, 0.0, 1.0))
