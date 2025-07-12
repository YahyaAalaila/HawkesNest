# hawkesnest/domain/network.py
from __future__ import annotations
import random, math
from typing import Tuple, Sequence

import networkx as nx
import numpy as np


class NetworkDomain:
    """
    Spatial support = a (multi)graph with edge lengths in the 'length' attr.
    """

    def __init__(self, G: nx.Graph):
        xs = [float(d["x"]) for _, d in G.nodes(data=True)]
        ys = [float(d["y"]) for _, d in G.nodes(data=True)]
        self.x_min, self.x_max = min(xs), max(xs)
        self.y_min, self.y_max = min(ys), max(ys)
        # or as a tuple:
        self.bounds = [self.x_min, self.x_max, self.y_min, self.y_max]
        if not nx.get_edge_attributes(G, "length"):
            # if missing: use Euclidean edge length as default
            for u, v, d in G.edges(data=True):
                ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
                vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
                d["length"] = math.hypot(vx - ux, vy - uy)

        self.G = G
        self._edges, self._edge_cum = self._precompute_edge_table(G)

    # ------------------------------------------------------------------ API
    def sample_point(self, rng: random.Random | np.random.Generator) -> Tuple[float, float]:
        """
        Uniform on total edge length:
        1) pick an edge proportional to its 'length',
        2) pick a random fraction along that edge.
        """
        if isinstance(rng, random.Random):
            r = rng.random() * self._edge_cum[-1]
        else:
            r = float(rng.random()) * self._edge_cum[-1]

        # binary search
        idx = np.searchsorted(self._edge_cum, r, side="right")
        (u, v, length) = self._edges[idx]

        alpha = rng.random() if isinstance(rng, random.Random) else float(rng.random())
        ux, uy = self.G.nodes[u]["x"], self.G.nodes[u]["y"]
        vx, vy = self.G.nodes[v]["x"], self.G.nodes[v]["y"]
        x = ux + alpha * (vx - ux)
        y = uy + alpha * (vy - uy)
        return (x, y)

    def distance(self, u: Tuple[float, float], v: Tuple[float, float]) -> float:
        """
        Project arbitrary points back to the nearest node (fast & simple).
        For large graphs you may want a KD-tree lookup.
        """
        nu = self._nearest_node(*u)
        nv = self._nearest_node(*v)
        # Dijkstra on length attribute
        return nx.shortest_path_length(self.G, nu, nv, weight="length")

    # ---------------------------------------------------------------- internals
    def _nearest_node(self, x: float, y: float) -> int:
        G = self.G
        # brute force   (ok for <10⁴ nodes; otherwise KD-tree this)
        return min(G.nodes, key=lambda n: (G.nodes[n]["x"] - x) ** 2 + (G.nodes[n]["y"] - y) ** 2)

    @staticmethod
    def _precompute_edge_table(G: nx.Graph):
        edgelist: list[tuple[int, int, float]] = []
        cum = []
        tot = 0.0
        for u, v, d in G.edges(data=True):
            L = float(d["length"])
            edgelist.append((u, v, L))
            tot += L
            cum.append(tot)
        return edgelist, np.asarray(cum)

    # -------------- convenience for SimulatorConfig -----------------
    def to_cfg(self) -> dict:
        """return plain dict serialisable by pydantic."""
        import json, tempfile, os, pathlib, pickle
        # quickest: pickle the graph to a tmp file
        tmp = pathlib.Path(tempfile.mktemp(suffix=".pkl"))
        pickle.dump(self.G, tmp.open("wb"))
        return {"type": "network", "pickle_path": str(tmp)}
