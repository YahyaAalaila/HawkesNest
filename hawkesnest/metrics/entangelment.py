"""
Spatio–temporal entanglement index.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KernelDensity
from scipy.special import digamma
from typing import Union

def _ksg_mi(coords: np.ndarray, times: np.ndarray, k: int = 6) -> float:
    """
    KSG-1 MI estimator in nats.

    Parameters
    ----------
    coords : (n, d)  spatial coordinates
    times  : (n,)    event times
    k      : int     NN

    Returns
    -------
    I(S;T) in nats   (≥ 0 up to estimator noise)
    """
    n = coords.shape[0]
    xyz = np.c_[coords, times.reshape(-1, 1)]

    # ε_i = distance to k-th neighbour in joint (max-norm)
    joint_tree = NearestNeighbors(metric="chebyshev", n_neighbors=k + 1).fit(xyz)
    eps = joint_tree.kneighbors(xyz, return_distance=True)[0][:, -1] - 1e-12  # strict >

    # counts in marginal spaces 
    tree_x = NearestNeighbors(metric="chebyshev").fit(coords)
    tree_t = NearestNeighbors(metric="chebyshev").fit(times.reshape(-1, 1))

    n_x = np.array([
        tree_x.radius_neighbors([coords[i]], radius=eps[i],
                                return_distance=False)[0].size - 1
        for i in range(n)
    ])
    n_t = np.array([
        tree_t.radius_neighbors([[times[i]]], radius=eps[i],
                                return_distance=False)[0].size - 1
        for i in range(n)
    ])

    mi = (
        digamma(k)
        + digamma(n)                                   
        - (digamma(n_x + 1) + digamma(n_t + 1)).mean() 
    )
    return float(max(mi, 0.0))                         


def alpha_ent(events, k: int | None = None, debug: bool = False) -> float:
    """
    α_ent ∈ [0,1] from events DataFrame or ndarray [[x,y,t], …].
    """
    if isinstance(events, np.ndarray):
        coords = events[:, :2].astype(float)
        times  = events[:, 2].astype(float)
    else:                                            
        coords = events[["x", "y"]].to_numpy(float)
        times  = events["t"].to_numpy(float)

    
    coords = (coords - coords.mean(0)) / coords.std(0)
    times  = (times  - times.mean())  / times.std()

    n = len(times)
    if k is None:
        k = max(4, int(n ** 0.4))                   

    mi = _ksg_mi(coords, times, k=k)
    alpha = mi / np.log(n)
    return alpha


def alpha_ent_kl(
    events: Union[np.ndarray, pd.DataFrame],
    bw_joint: float = 0.4,
    bw_space: float = 0.5,
    bw_time:  float = 0.25,
) -> float:
    """
    Estimate entanglement via KL-divergence:
      D_KL(p(s,t) || p(s)p(t))
    using three Gaussian KDEs.
    Accepts either
      - events: np.ndarray of shape (N,3) with columns [x,y,t], or
      - events: pd.DataFrame with columns ['x','y','t'].
    """
    # --- 1) pull out X, Y, T as NumPy arrays ---
    if isinstance(events, pd.DataFrame):
        if not set(['x','y','t']).issubset(events.columns):
            raise ValueError("DataFrame must have ['x','y','t'] columns")
        XY  = events[['x','y']].to_numpy(dtype=float)
        T   = events[['t']].to_numpy(dtype=float)
        XY = (XY - XY.mean(0)) / XY.std(0)
        T  = (T  -  T.mean()) / T.std()
        XYT = np.hstack((XY, T))
    else:
        arr = np.asarray(events, float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("NumPy array must be shape (N,3+) with x,y,t in first three columns")
        XYT = arr[:, :3]
        XY  = arr[:, :2]
        T   = arr[:, 2:3]
    
    # --- 2) fit KDEs ---
    kde_joint = KernelDensity(bandwidth=bw_joint).fit(XYT)
    kde_space = KernelDensity(bandwidth=bw_space).fit(XY)
    kde_time  = KernelDensity(bandwidth=bw_time).fit(T)

    # --- 3) evaluate log‐densities ---
    log_p_joint = kde_joint.score_samples(XYT)
    log_p_space = kde_space.score_samples(XY)
    log_p_time  = kde_time .score_samples(T)

    # --- 4) Monte-Carlo KL estimate ---
    kl_est = np.mean(log_p_joint - (log_p_space + log_p_time))
    D = max(0.0, kl_est)  # ensure non-negativity
    ret = D / (1.0 + D)
    return ret
