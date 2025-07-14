from __future__ import annotations

import abc
from typing import Tuple

import numpy as np


class SpatialDomain(abc.ABC):
    """
    Abstract base for spatial supports.
    """
    
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.bounds = (x_min, x_max, y_min, y_max)
        #self._area = (x_max - x_min) * (y_max - y_min)

    
    def sample_point(self, rng) -> Tuple[float, float]:
        """
        Draw a random point uniformly from the domain.
        """
        return (rng.uniform(self.x_min, self.x_max), rng.uniform(self.y_min, self.y_max))
        

    # @abc.abstractmethod
    def contains(self, point: Tuple[float, float]) -> bool:
        """
        Test if `point` lies inside the domain.
        """
        x, y = point
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def to_unit(self, xy):
        x, y = xy[..., 0], xy[..., 1]
        return np.stack(
            (x - self.x_min) / (self.x_max - self.x_max),
            (y - self.y_min) / (self.y_max - self.y_max), -1
        )
    def from_unit(self, xy_unit):
        u, v = xy_unit[..., 0], xy_unit[..., 1]
        return np.stack((u * (self.x_max - self.x_max) + self.x_min,
                         v * (self.y_max - self.y_max) + self.y_min), -1)
        
    @abc.abstractmethod
    def distance(self, u: Tuple[float, float], v: Tuple[float, float]) -> float:
        """
        Distance between two points.
        Default is Euclidean; override in NetworkDomain.
        """
        return NotImplementedError
    
    @abc.abstractmethod
    def area(self) -> float:
        """
        Measure of the support (area or total length).
        """
        raise NotImplementedError
    
    
