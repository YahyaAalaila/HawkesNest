
from hawkesnest.background.base import BackgroundBase 
import numpy as np
from typing import Sequence, Tuple, Union

# class SpatialBackground(BackgroundBase):
#     """Static spatial field μ(s) represented as a Gaussian mixture."""

#     def __init__(self, centers: np.ndarray, weights: np.ndarray, sigma: float) -> None:
#         self.centers = centers  # (K, 2)
#         w = np.asarray(weights, dtype=float)
#         self.weights = w / w.sum()
#         self.sigma2 = sigma ** 2
#         self.norm = (2 * np.pi * self.sigma2) ** 1

#     def __call__(self, s: np.ndarray, t: float, m: int | None = None) -> float:  # noqa: D401,E501
#         diffs = s - self.centers  # (K, 2)
#         exps = np.exp(-0.5 * np.sum(diffs**2, axis=1) / self.sigma2)
#         return float((self.weights * exps).sum() / self.norm)
    

"""
Static spatial field μ(s) represented as a Gaussian mixture with per-component variances.
"""


class SpatialBackground(BackgroundBase):
    """Gaussian-mixture background with individual variances for each component."""

    def __init__(
        self,
        centers: Sequence[Tuple[float, float]],
        variances: Sequence[float],
        weights: Sequence[float]
    ) -> None:
        """
        Parameters:
            centers : Sequence of (x, y) tuples for K mixture centers.
            variances : Sequence of variances (σ²) for each Gaussian component.
            weights : Sequence of mixture weights for each component.
        """
        self.centers = np.asarray(centers, dtype=float)    # shape (K,2)
        self.variances = np.asarray(variances, dtype=float)  # shape (K,)
        w = np.asarray(weights, dtype=float)                # shape (K,)
        self.weights = w / w.sum()
        # Normalization constant for each Gaussian: 2πσ²
        self.norms = 2 * np.pi * self.variances           # shape (K,)

    def __call__(
        self,
        s: Union[Sequence[float], np.ndarray],
        t: float = None,
        m: int | None = None
    ) -> Union[float, np.ndarray]:
        """
        Evaluate μ(s) at spatial location(s) s.  Time and mark arguments ignored.

        Args:
            s: A single point (x, y) or array of points shape (N,2).
            t: Ignored (temporal placeholder).
            m: Ignored (mark placeholder).

        Returns:
            A scalar or array of intensities in [0, ∞).
        """
        s_arr = np.asarray(s, dtype=float)
        # Single point case
        if s_arr.ndim == 1 and s_arr.shape[0] == 2:
            diffs = self.centers - s_arr          # shape (K,2)
            d2    = np.sum(diffs**2, axis=1)      # shape (K,)
            exps  = np.exp(-0.5 * d2 / self.variances) / self.norms
            return float(np.dot(self.weights, exps))

        # Batch of points
        if s_arr.ndim == 2 and s_arr.shape[1] == 2:
            N = s_arr.shape[0]
            # Compute squared distances: shape (N,K)
            diffs = s_arr[:, None, :] - self.centers[None, :, :]
            d2    = np.sum(diffs**2, axis=2)
            exps  = np.exp(-0.5 * d2 / self.variances[None, :]) / self.norms[None, :]
            # Weighted sum over components: result shape (N,)
            return exps.dot(self.weights)

        raise ValueError(f"Invalid spatial input shape: {s_arr.shape}")
