# hawkesnest/metrics/heterogeneity.py
"""
Spatio-temporal heterogeneity metric based on the inhomogeneous K-function.

This module provides a function to estimate the heterogeneity index
\alpha_het = \int\int |K_inhom(r,h) - K_HPP(r,h)| / K_HPP(r,h) dr dh,
normalized to [0,1].
"""

import numpy as np
from sklearn.neighbors import KernelDensity


def alpha_het(
    events: np.ndarray, r_max: float, h_max: float, n_r: int = 20, n_h: int = 20
) -> float:
    """
    Estimate the heterogeneity index from event data.

    Parameters
    ----------
    events: array_like of shape (n, 3)
        Each row is (x, y, t).
    r_max: float
        Maximum spatial lag to consider.
    h_max: float
        Maximum temporal lag to consider.
    n_r, n_h: int
        Number of bins for spatial and temporal integration.

    Returns
    -------
    alpha: float
        Heterogeneity index in [0, 1].
    """
    # 1. Fit background intensity via 3D KDE on (x, y, t)
    kde = KernelDensity(kernel="epanechnikov", bandwidth=0.1)
    kde.fit(events)

    # 2. Estimate inhomogeneous K(r,h)
    rs = np.linspace(0, r_max, n_r)
    hs = np.linspace(0, h_max, n_h)
    K_inhom = np.zeros((n_r, n_h))
    n = events.shape[0]
    # naive double loop; can be optimized
    for i in range(n):
        xi, yi, ti = events[i]
        lam_i = np.exp(kde.score_samples([[xi, yi, ti]]))[0]
        for j in range(n):
            if i == j:
                continue
            xj, yj, tj = events[j]
            lam_j = np.exp(kde.score_samples([[xj, yj, tj]]))[0]
            dr = np.hypot(xj - xi, yj - yi)
            dh = abs(tj - ti)
            # find bin indices
            ir = np.searchsorted(rs, dr) - 1
            ih = np.searchsorted(hs, dh) - 1
            if 0 <= ir < n_r and 0 <= ih < n_h:
                K_inhom[ir, ih] += 1.0 / (lam_i * lam_j)
    # normalize by domain area*time (assumed 1) and edge-correction omitted

    # 3. Compute homogeneous K = pi r^2 h
    R, H = np.meshgrid(rs, hs, indexing="ij")
    K_hpp = np.pi * R**2 * H

    # 4. Integrate absolute difference
    num = np.trapz(np.trapz(np.abs(K_inhom - K_hpp), hs, axis=1), rs)
    den = np.trapz(np.trapz(K_hpp, hs, axis=1), rs)
    alpha = np.clip(num / den, 0.0, 1.0)
    return alpha
