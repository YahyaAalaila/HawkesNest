# hawkesnest/domain/network.py
from __future__ import annotations
import random, math
import time
from typing import Tuple

import networkx as nx
import numpy as np

from hawkesnest.domain.base import SpatialDomain


class NetworkDomain(SpatialDomain):
    """
    Spatial support = a (multi)graph with edge lengths in the 'length' attr.
    """

    def __init__(self, G: nx.Graph):
        xs = [float(d["x"]) for _, d in G.nodes(data=True)]
        ys = [float(d["y"]) for _, d in G.nodes(data=True)]
        self.x_min, self.x_max = min(xs), max(xs)
        self.y_min, self.y_max = min(ys), max(ys)
        super().__init__(self.x_min,  self.x_max,  self.y_min,  self.y_max)
        # or as a tuple:
        #self.bounds = [self.x_min, self.x_max, self.y_min, self.y_max]
        if not nx.get_edge_attributes(G, "length"):
            # if missing: use Euclidean edge length as default
            for u, v, d in G.edges(data=True):
                ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
                vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
                d["length"] = math.hypot(vx - ux, vy - uy)

        self.G = G
        self._edges, self._edge_cum = self._precompute_edge_table(G)
        self._node_list = list(G.nodes())
        self._node_index = {n: i for i, n in enumerate(self._node_list)}
        self._coords = np.asarray([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in self._node_list], dtype=float)
        n = len(self._node_list)
        self._apsp = np.full((n, n), np.inf, dtype=float)
        for src, lengths in nx.all_pairs_dijkstra_path_length(G, weight="length"):
            i = self._node_index[src]
            for dst, dist in lengths.items():
                j = self._node_index[dst]
                self._apsp[i, j] = float(dist)
        np.fill_diagonal(self._apsp, 0.0)
        if np.isinf(self._apsp).any():
            raise ValueError("Graph not fully connected: APSP contains inf")

    # ------------------------------------------------------------------ API
    def sample_point(self, rng: random.Random | np.random.Generator) -> Tuple[float, float]:
        """
        Uniform on total edge length:
        1) pick an edge proportional to its 'length',
        2) pick a random fraction along that edge.
        """
        pt, _ = self.sample_edgepoint(rng)
        return pt

    def sample_edgepoint(self, rng: random.Random | np.random.Generator) -> Tuple[Tuple[float, float], dict]:
        """
        Like sample_point, but also returns metadata for fast geodesic distances:
        meta = {"u": int, "v": int, "alpha": float, "edge_len": float}
        point coordinate is interpolated between u and v by alpha.
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
        meta = {"u": u, "v": v, "alpha": float(alpha), "edge_len": float(length)}
        return (x, y), meta

    def distance(self, u: Tuple[float, float], v: Tuple[float, float]) -> float:
        """
        Project arbitrary points back to the nearest node (fast & simple).
        For large graphs you may want a KD-tree lookup.
        """
        coords = self._coords
        u_arr = np.asarray(u, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        iu = int(np.argmin(np.sum((coords - u_arr) ** 2, axis=1)))
        iv = int(np.argmin(np.sum((coords - v_arr) ** 2, axis=1)))
        return float(self._apsp[iu, iv])

    def distance_edgepoints(self, a: dict, b: dict) -> float:
        """
        O(1) geodesic distance between two edgepoints using APSP.
        """
        u_a, v_a = a["u"], a["v"]
        u_b, v_b = b["u"], b["v"]
        alpha_a, alpha_b = float(a["alpha"]), float(b["alpha"])
        len_a, len_b = float(a["edge_len"]), float(b["edge_len"])

        da_u = alpha_a * len_a
        da_v = (1.0 - alpha_a) * len_a
        db_u = alpha_b * len_b
        db_v = (1.0 - alpha_b) * len_b

        iu = self._node_index[u_a]
        iv = self._node_index[v_a]
        ju = self._node_index[u_b]
        jv = self._node_index[v_b]

        cand = [
            da_u + self._apsp[iu, ju] + db_u,
            da_u + self._apsp[iu, jv] + db_v,
            da_v + self._apsp[iv, ju] + db_u,
            da_v + self._apsp[iv, jv] + db_v,
        ]
        return float(min(cand))

    # ---------------------------------------------------------------- internals
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
