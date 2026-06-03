"""
BarrierDomain — rectangle with rectangular exclusion zones.

Events may not originate inside a barrier, and the thinning algorithm rejects
proposed candidate locations that fall in a barrier.  Background and kernel
integrals ignore barrier area (rejection sampling preserves the correct
spatial marginal).
"""
from __future__ import annotations

import math
from typing import List, Tuple

from hawkesnest.domain.standard import RectangleDomain


class BarrierDomain(RectangleDomain):
    """
    Axis-aligned rectangular domain with rectangular exclusion zones.

    Parameters
    ----------
    x_min, x_max, y_min, y_max : float
        Bounding box of the domain.
    barriers : list of (bx0, bx1, by0, by1)
        Each tuple defines an excluded rectangle.  Any proposed point
        inside at least one barrier is rejected.
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        barriers: List[Tuple[float, float, float, float]],
    ) -> None:
        super().__init__(x_min, x_max, y_min, y_max)
        # normalise: ensure bx0 < bx1, by0 < by1
        self.barriers: List[Tuple[float, float, float, float]] = [
            (min(bx0, bx1), max(bx0, bx1), min(by0, by1), max(by0, by1))
            for bx0, bx1, by0, by1 in barriers
        ]
        blocked = sum(
            (bx1 - bx0) * (by1 - by0) for bx0, bx1, by0, by1 in self.barriers
        )
        self._area = (x_max - x_min) * (y_max - y_min) - blocked

    # ------------------------------------------------------------------
    def _in_barrier(self, x: float, y: float) -> bool:
        return any(
            bx0 <= x <= bx1 and by0 <= y <= by1
            for bx0, bx1, by0, by1 in self.barriers
        )

    def contains(self, point: Tuple[float, float]) -> bool:
        x, y = point
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and not self._in_barrier(x, y)
        )

    def sample_point(self, rng) -> Tuple[float, float]:
        """Rejection-sample a uniform point from the unblocked region."""
        while True:
            x = rng.uniform(self.x_min, self.x_max)
            y = rng.uniform(self.y_min, self.y_max)
            if not self._in_barrier(x, y):
                return (x, y)

    def distance(self, u: Tuple[float, float], v: Tuple[float, float]) -> float:
        return math.sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2)

    def domain_mask(self, nx: int = 100, ny: int = 100):
        """
        Return a binary (ny, nx) NumPy array: 1 = valid, 0 = barrier.
        Useful for storing the topology mask alongside NPZ exports.
        """
        import numpy as np

        xs = np.linspace(self.x_min, self.x_max, nx)
        ys = np.linspace(self.y_min, self.y_max, ny)
        mask = np.ones((ny, nx), dtype=np.float32)
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                if self._in_barrier(x, y):
                    mask[j, i] = 0.0
        return mask
