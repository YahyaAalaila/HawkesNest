from typing import Callable
import numpy as np

from hawkesnest.background.base import BackgroundBase
from hawkesnest.background.hetero_ladder import HeteroLadderBackground
from hawkesnest.background.spatial import SpatialBackground
from hawkesnest.background.temporal import TemporalProfile
from hawkesnest.domain.standard import RectangleDomain


class SeparableBackground(BackgroundBase):
    """Product‑separable background μ(s, t) = f(s) g(t)."""

    def __init__(
        self, spatial: SpatialBackground, temporal: TemporalProfile
    ) -> None:  # noqa: E501
        self.spatial = spatial
        self.temporal = temporal

    def __call__(
        self, s: np.ndarray, t: float, m: int | None = None
    ) -> float:  # noqa: D401,E501
        return self.spatial(s, t) * self.temporal(t)


class EntangledBackground(BackgroundBase):
    """Fully entangled field μ(s, t) given as a thin‑plate spline surface."""

    def __init__(
        self, spline_surface: "Callable[[np.ndarray, float], float]"
    ) -> None:  # noqa: E501
        self.surface = spline_surface

    def __call__(self, s: np.ndarray, t: float, m: int | None = None) -> float:
        val = float(self.surface(s, t))
        if val < 0.0:
            raise ValueError(
                f"Background intensity must be non-negative, got {val} at s={s}, t={t}"
            )
        return val
    


__all__ = [
    "SeparableBackground",
    "EntangledBackground",
    "HeteroLadderBackground",
]
