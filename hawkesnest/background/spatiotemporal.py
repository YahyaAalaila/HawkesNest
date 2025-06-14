from typing import Callable
from hawkesnest.background.base import BackgroundBase 
from hawkesnest.background.spatial import SpatialBackground
from hawkesnest.background.temporal import TemporalProfile
import numpy as np

class SeparableBackground(BackgroundBase):
    """Product‑separable background μ(s, t) = f(s) g(t)."""

    def __init__(self, spatial: SpatialBackground, temporal: TemporalProfile) -> None:  # noqa: E501
        self.spatial = spatial
        self.temporal = temporal

    def __call__(self, s: np.ndarray, t: float, m: int | None = None) -> float:  # noqa: D401,E501
        return self.spatial(s, t) * self.temporal(t)
    
    
class EntangledBackground(BackgroundBase):
    """Fully entangled field μ(s, t) given as a thin‑plate spline surface."""

    def __init__(self, spline_surface: "Callable[[np.ndarray, float], float]") -> None:  # noqa: E501
        self.surface = spline_surface

    def __call__(self, s: np.ndarray, t: float, m: int | None = None) -> float:  # noqa: D401,E501
        return float(self.surface(s, t))