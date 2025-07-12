import numpy as np
import networkx as nx
from typing import Protocol, Tuple
from hawkesnest.domain import SpatialDomain

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
    eucl_list = []
    geo_list  = []
    rng = np.random.default_rng()

    tries = 0
    max_tries = n_pairs * 10
    while len(eucl_list) < n_pairs and tries < max_tries:
        tries += 1
        u = domain.sample_point(rng)
        v = domain.sample_point(rng)

        # Euclidean
        d_e = np.linalg.norm(np.subtract(u, v))

        # Geodesic: may raise NetworkXNoPath
        try:
            d_g = domain.distance(u, v)
        except (nx.NetworkXNoPath, ValueError):
            continue

        eucl_list.append(d_e)
        geo_list.append(d_g)

    if len(eucl_list) < 1:
        raise RuntimeError(f"Couldn’t sample any connected pairs (got {len(eucl_list)}) in {tries} tries")

    eucl_dists = np.array(eucl_list)
    geo_dists  = np.array(geo_list)

    mean_euc = float(np.mean(eucl_dists))
    mean_geo = float(np.mean(geo_dists))
    if mean_euc == 0.0:
        D = 1.0
    else:
        D = mean_geo / mean_euc

    alpha = (D - 1.0) / (D_max - 1.0)
    return float(np.clip(alpha, 0.0, 1.0))
