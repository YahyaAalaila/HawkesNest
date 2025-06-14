# Module implementing simple rectangular spatial domains.

from hawkesnest.domain.base import SpatialDomain


class RectangleDomain(SpatialDomain):
    def __init__(self, x_min, x_max, y_min, y_max):
        self.bounds = (x_min, x_max, y_min, y_max)
        self._area = (x_max - x_min) * (y_max - y_min)

    def sample_point(self, rng):
        x_min, x_max, y_min, y_max = self.bounds
        return (rng.uniform(x_min, x_max), rng.uniform(y_min, y_max))

    def contains(self, point):
        x, y = point
        x_min, x_max, y_min, y_max = self.bounds
        return x_min <= x <= x_max and y_min <= y <= y_max

    def area(self):
        return self._area
