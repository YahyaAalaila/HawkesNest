"""
Generate an embedded graph for the topology-pillar experiments and
save it to a pickle that NetworkDomainCfg can read.

Usage:
    python make_graph.py  grid            30 30 0.03  out_grid.pkl
    python make_graph.py  rgg             1000 0.05    out_rgg.pkl
    python make_graph.py  sierpinski      4            out_sier.pkl
    python make_graph.py  1nn             500 42        out_1nn.pkl
    python make_graph.py  mst             500 123       out_mst.pkl
"""

from __future__ import annotations
import sys, math, pickle, pathlib
from typing import Tuple
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay, cKDTree


def _add_lengths(G: nx.Graph) -> None:
    """Fill edge attr 'length' and ensure node attr x,y exist."""
    for u, v, d in G.edges(data=True):
        ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
        vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
        d["length"] = math.hypot(vx - ux, vy - uy)

def make_grid(n_x: int, n_y: int, spacing: float = 1.0) -> nx.Graph:
    """n_x × n_y rectangular lattice."""
    G = nx.grid_2d_graph(n_x, n_y)
    mapping = {}
    for (i, j) in G.nodes():
        nid = len(mapping)
        mapping[(i, j)] = nid
        G.nodes[(i, j)]["x"] = i * spacing
        G.nodes[(i, j)]["y"] = j * spacing
    G = nx.relabel_nodes(G, mapping)
    _add_lengths(G)
    return G


def make_rgg(n: int, radius: float, seed: int = 0) -> nx.Graph:
    """Random geometric graph inside [0,1]²."""
    rng = np.random.default_rng(seed)
    pos = {i: rng.uniform(0, 1, size=2) for i in range(n)}
    G = nx.random_geometric_graph(n, radius, pos=pos)
    for i, (x, y) in pos.items():
        G.nodes[i]["x"] = float(x)
        G.nodes[i]["y"] = float(y)
    _add_lengths(G)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


def make_sierpinski(level: int = 3) -> nx.Graph:
    """
    Deterministic fractal carpet in [0,1]².
    """
    size = 3 ** level
    mask = np.ones((size, size), dtype=bool)
    for l in range(level):
        step = 3 ** l
        for i in range(0, size, 3 * step):
            for j in range(0, size, 3 * step):
                mask[i + step : i + 2 * step, j + step : j + 2 * step] = False

    G = nx.Graph()
    for i in range(size):
        for j in range(size):
            if not mask[i, j]:
                continue
            nid = i * size + j
            G.add_node(nid, x=i / size, y=j / size)
            for di, dj in [(1, 0), (0, 1)]:
                ii, jj = i + di, j + dj
                if ii < size and jj < size and mask[ii, jj]:
                    vid = ii * size + jj
                    G.add_edge(nid, vid)
    _add_lengths(G)
    return G


def make_1nn(n: int, seed: int = 0) -> nx.Graph:
    """
    Sample n points in [0,1]², connect each to its nearest neighbor.
    """
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    tree = cKDTree(pts)
    G = nx.Graph()
    for i, (x, y) in enumerate(pts):
        G.add_node(i, x=float(x), y=float(y))
    for i, p in enumerate(pts):
        dists, idxs = tree.query(p, k=2)
        j = idxs[1]
        G.add_edge(i, j)
    _add_lengths(G)
    return G


def make_mst_graph(n: int, seed: int = 0) -> nx.Graph:
    """
    Sample n points, build Delaunay graph, then extract MST.
    """
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    tri = Delaunay(pts)
    edges = set()
    for simplex in tri.simplices:
        for k in range(3):
            a, b = sorted((simplex[k], simplex[(k+1) % 3]))
            edges.add((a, b))
    G0 = nx.Graph()
    for i, (x, y) in enumerate(pts):
        G0.add_node(i, x=float(x), y=float(y))
    for u, v in edges:
        length = float(np.linalg.norm(pts[u] - pts[v]))
        G0.add_edge(u, v, length=length)
    T = nx.minimum_spanning_tree(G0, weight="length")
    return T


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    typ = sys.argv[1]
    out = pathlib.Path(sys.argv[-1])

    if typ == "grid":
        nx_, ny_, spacing = map(float, sys.argv[2:5])
        G = make_grid(int(nx_), int(ny_), spacing)
    elif typ == "rgg":
        n, radius = int(sys.argv[2]), float(sys.argv[3])
        G = make_rgg(n, radius)
        import networkx as nx, pickle
        
        G = nx.minimum_spanning_tree(G, weight="length")

    elif typ == "sierpinski":
        level = int(sys.argv[2])
        G = make_sierpinski(level)
    elif typ == "1nn":
        n, seed = int(sys.argv[2]), int(sys.argv[3])
        G = make_1nn(n, seed)
    elif typ == "mst":
        n, seed = int(sys.argv[2]), int(sys.argv[3])
        G = make_mst_graph(n, seed)
    else:
        raise SystemExit(f"unknown type {typ}")

    with out.open("wb") as f:
        pickle.dump(G, f)
        
    
    print(f"Saved graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges → {out}")
