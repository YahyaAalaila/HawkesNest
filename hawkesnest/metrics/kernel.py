# hawkesnest/metrics/kernel.py
"""
Kernel roughness complexity metric: computes the effective degrees-of-freedom
of the empirical triggering kernels phi_{mn}(s, tau) via functional PCA.

For each mark pair (m,n), parent-child offsets (dx, dt) are binned into a 2D
histogram. We perform SVD on the centered histogram matrix to obtain eigenvalues
lambda_k, then compute d_mn = sum_k lambda_k / (lambda_k + sigma2). The overall
index is the sum of d_mn over all pairs, normalized by d_max.

If a mark pair has fewer than 2 offset observations or all offsets identical,
its roughness contribution is defined to be zero.
"""

import numpy as np
from typing import Dict, Tuple

def alpha_ker(offsets: Dict[Tuple[int,int], np.ndarray],
              sigma2: float,
              d_max: float,
              bins: Tuple[int,int] = (10,10)) -> float:
    """
    Compute the kernel roughness index alpha_ker in [0,1].

    Parameters:
        offsets: mapping (m,n) -> array of shape (K,2) of spatial-temporal lags
        sigma2: shot-noise variance for regularization
        d_max:  maximum possible sum of effective DoFs for normalization
        bins:   number of bins for histogram along (space,time)

    Returns:
        float: roughness index
    """
    total_d = 0.0
    for (m, n), arr in offsets.items():
        # Handle trivial cases: fewer than 2 or identical offsets
        if arr is None or arr.size < 2 or np.allclose(arr, arr[0], atol=1e-8):
            d_mn = 0.0
        else:
            # 2D histogram binning
            dx = arr[:,0]
            dt = arr[:,1]
            H, xedges, yedges = np.histogram2d(dx, dt, bins=bins)
            # Center
            Hc = H - H.mean()
            # SVD
            s = np.linalg.svd(Hc, compute_uv=False)
            # Eigenvalues = singular values squared
            lambdas = s**2
            # Effective DoF
            d_mn = np.sum(lambdas / (lambdas + sigma2))
        total_d += d_mn
    # Normalize
    return float(total_d / d_max)
