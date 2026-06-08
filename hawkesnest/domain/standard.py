# Module implementing simple rectangular spatial domains.
from typing import Tuple
import numpy as np

from hawkesnest.domain.base import SpatialDomain

class RectangleDomain(SpatialDomain):
    def __init__(self, x_min, x_max, y_min, y_max):
        super().__init__(x_min, x_max, y_min, y_max)
        self._area = (x_max - x_min) * (y_max - y_min)

    def area(self):
        return self._area
    
    def distance(self, u: Tuple[float, float], v: Tuple[float, float]) -> float:
        """Euclidean distance between two points in the domain."""
        dx = u[0] - v[0]
        dy = u[1] - v[1]
        return np.hypot(dx, dy)
