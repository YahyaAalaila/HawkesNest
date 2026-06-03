"""
Spatio–temporal entanglement index.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
from typing import Union
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.spatial import distance_matrix
import math

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

def alpha_ent(
    events: Union[np.ndarray, pd.DataFrame],
    k: int | None = None,
    min_n: int = 200,
) -> float:
    """
    KSG-based entanglement index α_ent ∈ [0,1].

    Parameters
    ----------
    events : (N,3) array or DataFrame with columns ['x','y','t'].
    k      : k-NN parameter for KSG; if None, uses max(4, floor(n**0.4)).
    min_n  : minimum sample size; below this we abort.

    Returns
    -------
    α_ent  : float in [0,1].
    """
    # --- 1. extract arrays ---
    if isinstance(events, np.ndarray):
        coords = events[:, :2].astype(float)
        times  = events[:, 2].astype(float)
    else:
        coords = events[["x", "y"]].to_numpy(dtype=float)
        times  = events["t"].to_numpy(dtype=float)

    n = len(times)
    if n < min_n:
        raise ValueError(f"alpha_ent requires at least {min_n} events, got n={n}")

    # --- 2. standardise ---
    coords_std = coords.std(axis=0)
    t_std      = times.std()
    if np.any(coords_std == 0.0) or t_std == 0.0:
        # no variation in space or time => no entanglement
        return 0.0

    coords = (coords - coords.mean(axis=0)) / coords_std
    times  = (times  - times.mean()) / t_std

    # --- 3. pick k ---
    if k is None:
        k = max(4, int(n ** 0.4))

    # --- 4. KSG MI and normalisation ---
    mi = _ksg_mi(coords, times, k=k)
    if mi <= 0.0:
        return 0.0

    alpha = float(mi / np.log(n))
    if alpha < 0.0:
        alpha = 0.0
    if alpha > 1.0:
        alpha = 1.0
    return alpha




def _center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def alpha_ent_hsic(
    events: Union[np.ndarray, pd.DataFrame],
    sigma_space: float | None = None,
    sigma_time: float | None = None,
    min_n: int = 200,
) -> float:
    """
    HSIC-based entanglement index α_ent_hsic ∈ [0,1].
    Uses Gaussian kernels on space and time.
    """
    # extract
    if isinstance(events, np.ndarray):
        X = events[:, :2].astype(float)
        t = events[:, 2:3].astype(float)
    else:
        X = events[["x","y"]].to_numpy(float)
        t = events[["t"]].to_numpy(float)

    n = X.shape[0]
    if n < min_n:
        raise ValueError(f"alpha_ent_hsic requires at least {min_n} events, got n={n}")

    # bandwidths as median heuristic if not given
    if sigma_space is None:
        d_sp = distance_matrix(X, X)
        sigma_space = np.median(d_sp[d_sp > 0])
    if sigma_time is None:
        d_t = distance_matrix(t, t)
        sigma_time = np.median(d_t[d_t > 0])

    Kx = pairwise_kernels(X, X, metric="rbf", gamma=1.0 / (2 * sigma_space ** 2))
    Kt = pairwise_kernels(t, t, metric="rbf", gamma=1.0 / (2 * sigma_time ** 2))

    Kc = _center_gram(Kx)
    Lc = _center_gram(Kt)

    hsic = (Kc * Lc).sum() / ((n - 1) ** 2)

    # normalise by Frobenius norms to get something in [0,1]
    norm = np.linalg.norm(Kc, "fro") * np.linalg.norm(Lc, "fro")
    if norm == 0.0:
        return 0.0
    alpha = float(hsic / norm)
    return max(0.0, min(1.0, alpha))


def analytic_gaussian_mi(theta: float) -> float:
    """
    Analytic mutual information for a bivariate Gaussian with correlation
    rho = theta:

        I(T;X) = -0.5 * log(1 - theta^2).

    Defined for |theta| < 1.
    """
    if abs(theta) >= 1.0:
        raise ValueError(f"theta must satisfy |theta| < 1, got {theta}")
    return -0.5 * math.log(1.0 - theta * theta)


def gaussian_entanglement_index(
    theta: float,
    theta_max: float = 0.9,
) -> float:
    """
    Normalised entanglement index in [0, 1] based on the analytic MI.

    Parameters
    ----------
    theta : float
        Entanglement knob (the kernel parameter). Must satisfy |theta| < 1.
    theta_max : float, default=0.9
        Reference value for "maximal" entanglement in the synthetic study.
        We define

            alpha_ent(theta) = I(theta) / I(theta_max).

        So alpha_ent(0) = 0 and alpha_ent(theta_max) = 1.

    Returns
    -------
    alpha : float
        Entanglement index in [0, 1].
    """
    if abs(theta_max) >= 1.0:
        raise ValueError(f"theta_max must satisfy |theta_max| < 1, got {theta_max}")

    I_theta = analytic_gaussian_mi(theta)
    I_max = analytic_gaussian_mi(theta_max)

    if I_max <= 0.0:
        # Should not happen with |theta_max| < 1, but be defensive.
        return 0.0

    alpha = I_theta / I_max
    # alpha is already between 0 and 1 for |theta| < 1, but to make sure
    # clamp numerically into [0,1]
    if alpha < 0.0:
        alpha = 0.0
    elif alpha > 1.0:
        alpha = 1.0
    return alpha
