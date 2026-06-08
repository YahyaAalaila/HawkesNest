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

def alpha_ker(offsets: dict[tuple[int,int], np.ndarray],
              bins=(40,40)) -> float:
    nx, ny = bins
    d_max  = (nx-1)*(ny-1)*len(offsets)

    total = 0.0
    for arr in offsets.values():
        if arr is None or arr.shape[0] < 5:
            continue
        dx, dy = arr[:,0], arr[:,1]
        H, *_  = np.histogram2d(dx, dy, bins=bins)
        H      = H / H.sum()
        Hc     = H - H.mean()
        s      = np.linalg.svd(Hc, compute_uv=False)
        lamb   = s**2
        sigma2 = 1e-8 * lamb[0]**2
        total += (lamb / (lamb + sigma2)).sum()

    return float(total / d_max)
