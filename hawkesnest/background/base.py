from __future__ import annotations

import abc
from typing import Protocol, Tuple

import numpy as np

# Type alias for a single space–time coordinate
SpaceTime = Tuple[np.ndarray, float]  # (xy array of shape (2,), t)


class BackgroundBase(abc.ABC):
    """Abstract interface for background intensity functions.

    Subclasses must be *callable* and return a non‑negative float when
    evaluated at a single space–time coordinate.  They must also implement a
    vectorised variant: :py:meth:`batch` for performance‑critical looping.
    """

    @abc.abstractmethod
    def __call__(
        self, s: np.ndarray, t: float, m: int | None = None
    ) -> float:  # noqa: D401,E501
        """Evaluate μ(s, t)."""

    # ---------------------------------------------------------------------
    # Optional convenience helpers
    # ---------------------------------------------------------------------

    def batch(
        self, S: np.ndarray, T: np.ndarray, M: np.ndarray | None = None
    ) -> np.ndarray:
        """Vectorised evaluation on many points (used by the simulator).

        Parameters
        ----------
        S: (n, 2) array
            Spatial coordinates.
        T: (n,) array
            Time stamps.
        M: (n,) array, optional
            Marks; not all background models care about marks.
        """
        n = S.shape[0]
        out = np.empty(n)
        for i in range(n):
            out[i] = self(S[i], float(T[i]), int(M[i]) if M is not None else None)
        return out
