from __future__ import annotations

import numpy as np
import networkx as nx

from hawkesnest.domain.network import NetworkDomain


def build_topology_domain(
    theta_topo: float,
    seed: int,
    n_nodes: int = 400,
    r_min: float = 0.05,
    r_max: float = 1.0,
) -> NetworkDomain:
    """
    Random geometric graph on [0,1]^2 with fixed |V| and θ_topo
    controlling the connection radius (hence metric distortion).

    θ_topo = 0   -> radius ≈ r_max  (dense, low distortion)
    θ_topo = 1   -> radius ≈ r_min  (sparse, high distortion)
    """
    rng = np.random.default_rng(seed)
    radius = r_max - theta_topo * (r_max - r_min)

    G = nx.random_geometric_graph(n_nodes, radius=radius, seed=int(seed))

    if not nx.is_connected(G):
        # keep largest connected component
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    # write x,y attributes expected by NetworkDomain
    pos = nx.get_node_attributes(G, "pos")
    for n, (x, y) in pos.items():
        G.nodes[n]["x"] = float(x)
        G.nodes[n]["y"] = float(y)

    return NetworkDomain(G)
