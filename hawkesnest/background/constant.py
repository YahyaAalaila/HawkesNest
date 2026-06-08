import numpy as np
from hawkesnest.background.base import BackgroundBase
class ConstantBackground(BackgroundBase):
    """Spatially and temporally homogeneous Poisson baseline."""

    def __init__(self, rate: float) -> None:
        self.rate = float(rate)

    def __call__(self, s: np.ndarray, t: float, m: int | None = None) -> float:  # noqa: D401,E501
        return self.rate

