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



def alpha_topo_new(domain: SpatialDomain) -> float:
    """
    Deterministic topology index based on average metric distortion
    of the underlying graph vs. Euclidean geometry.

    Assumptions
    -----------
    - `domain` has an attribute `graph` which is a networkx Graph.
    - Each node in `graph` has a 2D position stored in node attribute "pos".
    - Edge weights (if present) represent geodesic length along the network.
      If not present, shortest paths are computed with unweighted edges
      and you should pre-weight edges externally.

    Returns
    -------
    alpha : float
        Topology complexity in [0, 1). 0 for flat Euclidean-like domains,
        increasing towards 1 as paths are more distorted.
    """
    # If the domain has no graph (pure Euclidean rectangle), complexity is 0
    if hasattr(domain, "graph"):
        G = domain.graph
    elif hasattr(domain, "G"):
        G = domain.G
    else:
        # No graph → flat Euclidean domain, complexity 0
        return 0.0
    if G.number_of_nodes() < 2:
        return 0.0

    # Extract positions
    pos = nx.get_node_attributes(G, "pos")
    if len(pos) != G.number_of_nodes():
        raise ValueError("All graph nodes must have a 'pos' attribute for alpha_topo.")

    nodes = list(G.nodes())
    n = len(nodes)

    # All-pairs shortest paths (deterministic)
    # Uses edge attribute 'weight' if present, else unweighted
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))

    total_geo = 0.0
    total_euc = 0.0
    count = 0

    for i in range(n):
        u = nodes[i]
        xu = np.asarray(pos[u], dtype=float)

        # distances from u to all v
        dist_u = lengths[u]

        for j in range(i + 1, n):
            v = nodes[j]
            xv = np.asarray(pos[v], dtype=float)

            # Euclidean distance
            d_e = float(np.linalg.norm(xu - xv))
            if d_e == 0.0:
                # Same position; skip, no information
                continue

            # Geodesic distance; if disconnected, skip
            d_g = dist_u.get(v, None)
            if d_g is None or np.isinf(d_g):
                continue

            total_euc += d_e
            total_geo += float(d_g)
            count += 1

    if count == 0:
        # Graph is effectively disconnected for our purposes
        # Topology is "maximally bad"; if you prefer, you can raise instead.
        return 1.0

    mean_euc = total_euc / count
    mean_geo = total_geo / count

    # Distortion ratio; enforce R >= 1 numerically
    if mean_euc == 0.0:
        R = 1.0
    else:
        R = max(1.0, mean_geo / mean_euc)

    # Map R in [1, +inf) to alpha in [0,1)
    alpha = 1.0 - 1.0 / R

    # Numerical clipping for safety
    return float(np.clip(alpha, 0.0, 1.0))
