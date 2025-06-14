# hawkesnest/domain/base.py
from __future__ import annotations
import abc
from typing import Tuple

class SpatialDomain(abc.ABC):
    """
    Abstract base for spatial supports.
    """

    @abc.abstractmethod
    def sample_point(self) -> Tuple[float, float]:
        """
        Draw a random point uniformly from the domain.
        """
        ...

    #@abc.abstractmethod
    def contains(self, point: Tuple[float, float]) -> bool:
        """
        Test if `point` lies inside the domain.
        """
        ...

    def geodesic(self, u: Tuple[float, float], v: Tuple[float, float]) -> float:
        """
        Distance between two points.  
        Default is Euclidean; override in NetworkDomain.
        """
        import math
        return math.hypot(u[0] - v[0], u[1] - v[1])

    def area(self) -> float:
        """
        Measure of the support (area or total length).
        """
        raise NotImplementedError
