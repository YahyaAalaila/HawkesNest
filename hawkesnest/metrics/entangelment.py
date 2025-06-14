# hawkesnest/metrics/entanglement.py
"""
Spatio-temporal entanglement metric: estimates the KL-divergence
D_{KL}(p(s,t) || p(s)p(t)) via a k-nearest-neighbours mutual information
estimator (Kraskov et al. 2004) and normalises to [0,1].
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

def _kraskov_mi(x: np.ndarray, y: np.ndarray, k: int = 6) -> float:
    """
    Estimate mutual information I(X;Y) between two continuous variables
    using the Kraskov k-NN estimator.
    X: array of shape (n_samples, d)
    Y: array of shape (n_samples,)
    Returns MI in nats.
    """
    # Number of samples
    n = x.shape[0]
    # Joint space
    xy = np.hstack((x, y.reshape(-1, 1)))
    # Build k-NN graphs
    nbrs_xy = NearestNeighbors(n_neighbors=k+1).fit(xy)
    distances_xy, _ = nbrs_xy.kneighbors(xy)
    # Exclude self-distance at index 0
    eps = distances_xy[:, -1]  # distance to k-th neighbour
    # Count neighbors in marginal spaces
    nbrs_x = NearestNeighbors(radius=eps, algorithm='auto').fit(x)
    cnt_x = np.array([len(neighbors) for neighbors in nbrs_x.radius_neighbors(x, eps, return_distance=False)]) - 1
    nbrs_y = NearestNeighbors(radius=eps.reshape(-1,1), algorithm='auto').fit(y.reshape(-1,1))
    cnt_y = np.array([len(neighbors) for neighbors in nbrs_y.radius_neighbors(y.reshape(-1,1), eps, return_distance=False)]) - 1
    # Digamma approximation
    from scipy.special import digamma
    mi = (digamma(k)
          - np.mean(digamma(cnt_x + 1) + digamma(cnt_y + 1))
          + digamma(n))
    return max(0.0, mi)


def alpha_entanglement(events: np.ndarray, k: int = 6) -> float:
    """
    Compute the entanglement index for a set of events.
    events: array of shape (n,3) with columns [x, y, t]
    Returns a float in [0,1].
    """
    coords = events[:, :2]
    times  = events[:, 2]
    mi_val = _kraskov_mi(coords, times, k=k)
    # Normalise by maximum possible MI = log(n)
    n = events.shape[0]
    return float(mi_val / np.log(n))
