import math
from typing import Tuple
from hawkesnest.domain.base import SpatialDomain

class GridDomain(SpatialDomain):
    def __init__(self, x_min, x_max, y_min, y_max):
        self.bounds = (x_min, x_max, y_min, y_max)
        self._area = (x_max - x_min) * (y_max - y_min)
    def sample_point(self, rng) -> Tuple[float, float]:
        return rng.random(), rng.random()
    def contains(self, point):
        x, y = point
        x_min, x_max, y_min, y_max = self.bounds
        return x_min <= x <= x_max and y_min <= y <= y_max
    def area(self):
        return self._area
    
    def distance(self, u: Tuple[float, float],
                       v: Tuple[float, float]) -> float:
        return abs(u[0] - v[0]) + abs(u[1] - v[1])
