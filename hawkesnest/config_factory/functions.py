import numpy as np
from typing import Callable, Sequence

def _cosine_fn(space: np.ndarray, t: float, *, amp: float = 1.0, freq: float = 1.0) -> float:
    """cos(2π·freq·(t + r)) with **radial** space‑time coupling."""
    r = float(np.linalg.norm(space))
    return amp * np.cos(2 * np.pi * freq * (t + r))


def _sine_fn(space: np.ndarray, t: float, *, amp: float = 1.0, freq: float = 1.0) -> float:
    r = float(np.linalg.norm(space))
    return np.exp(amp * np.sin(2 * np.pi * freq * (t + r)))

def _sin_cos_fn(space: np.ndarray, t: float,
                *, amp: float = 1.0, freq: float = 1.0) -> float:
    """Purely spatial heterogeneity: sin(2πfx) + cos(2πfy)."""
    x, y = map(float, space[:2])
    return amp * (np.sin(2 * np.pi * freq * x) + np.cos(2 * np.pi * freq * y))
def _exp_sin_fn(space: np.ndarray, t: float,
                *, a0: float = 0.0,
                a1: float = 1.0,
                omega: float = 1.0) -> float:
    """Time‐varying rate: exp(a0 + a1·sin(ω·t))."""
    return np.exp(a0 + a1 * np.sin(omega * t))


def _cluster_mix_fn(space: np.ndarray, t: float,
                    *, centers: Sequence[tuple[float,float]],
                    sigma: float = 50.0, base: float = 1.0) -> float:
    """Sum of spatial Gaussians, time-invariant."""
    x, y = map(float, space[:2])
    z = 0.0
    for cx, cy in centers:
        z += np.exp(-((x-cx)**2 + (y-cy)**2)/(2*sigma**2))
    return base * (1 + z)                     # keep ≥ 0

def _moving_gauss_fn(space: np.ndarray, t: float,
                     *, start: tuple[float,float] = (300,300),
                     v: tuple[float,float] = (2,1),
                     sigma: float = 50.0, base: float = 1.0) -> float:
    """Gaussian whose centre drifts linearly with time t."""
    cx = start[0] + v[0]*t
    cy = start[1] + v[1]*t
    x, y = map(float, space[:2])
    z = np.exp(-((x-cx)**2 + (y-cy)**2)/(2*sigma**2))
    return base * (1 + z)


def _poly2_fn(space: np.ndarray, t: float, *,
              base: float = 1.0,
              a0: float,    # constant
              a_t: float,   # linear in t
              a_x: float,   # linear in x (or r if you want)
              a_tx: float,  # CROSS TERM t·x  ← this drives entanglement
              a_tt: float,  # t²
              a_xx: float,  # x²
              ) -> float:
    """
    2D (space) + time polynomial up to degree 2 with explicit cross-term.
    Entanglement comes from the t·x term.
    """
    dd = a0 + a_t * t + a_x * float(space[0]) + a_tt * (t**2) + a_xx * (float(space[0])**2) + a_tx * (t * float(space[0]))
    return dd  # return exp to ensure positivity
import numpy as np

def _poly_entangled(
        space: np.ndarray,
        t: float,
        *,
        aa0:  float = .0,        # constant offset
        ent:  float = 1.0,        # 1st-order cross (t·x, t·y)
        quad: float = 0.0,        # 2nd-order cross (t²x², t²y²)
        lin:  float = 0.0,        # linear terms (x+y+t)
        scale: float = 1.0,       # global multiplier you can tune in YAML
) -> float:
    """
    λ(s,t) = scale · exp(
                aa0
              + lin · (x + y + t)
              + ent · (t x + t y)
              + quad· (t² x² + t² y²) )

    • `scale` lets you dial the intensity up or down without touching the
      shape parameters.
    • If `target_mean` is given the first call will estimate the average
      of exp(·) over the unit cube × [0,1] and set `scale` automatically
      to hit that mean.
    """
    # --- 1. unpack space / compute the raw exponent --------------------
    x, y = map(float, space[:2])
    z_raw = (
        aa0
        + lin  * (x + y + t)
        + ent  * (t * x + t * y)
        + quad * (t*t * x*x + t*t * y*y)
    )

    lam = scale * np.exp(z_raw)
    return z_raw



def _poly_fn(space: np.ndarray, t: float, *, coeffs: Sequence[float]) -> float:
    r = float(np.linalg.norm(space))
    return np.polyval(list(reversed(coeffs)), t + r)


def gabor_travel(space, t,
                 *, a0=0.0, amp=3.0, fx=3.0, fy=3.0, ft=4.0,
                 sigma=0.2, cx=0.5, cy=0.5, phase=0.0):
    x, y = map(float, space[:2])
    gauss  = np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2))
    cosine = np.cos(2*np.pi*(fx*x + fy*y + ft*t) + phase)
    return np.exp(a0 + amp * gauss * cosine)
def gauss_shear(space, t, *, a0=0.0, sx=0.2, sy=0.2, kappa=1.0):
    x, y = map(float, space[:2])
    scale = 1.0 + kappa * t
    z = a0 - (x**2)/(2*(sx*scale)**2) - (y**2)/(2*(sy*scale)**2)
    return np.exp(z)
def vortex(space, t, *, a0=0.0, sigma=0.25,
           gamma=3.0, omega=6.283, cx=0.5, cy=0.5):
    x, y = map(float, space[:2])
    dx, dy = x-cx, y-cy
    r2 = dx*dx + dy*dy
    theta = np.arctan2(dy, dx)
    z = a0 - r2/(2*sigma**2) + gamma*np.sin(theta - omega*t)
    return np.exp(z)

def osc_cluster(space, t,
                *, centers, sigma=0.08,
                   amps, freqs, phases):
    x, y = map(float, space[:2])
    z = 0.0
    for (cx,cy), A, f, ph in zip(centers, amps, freqs, phases):
        r2 = (x-cx)**2 + (y-cy)**2
        z += A*(1+np.sin(2*np.pi*f*t+ph)) * np.exp(-r2/(2*sigma**2))
    return np.exp(z)


_FUNCTION_REGISTRY: dict[str, Callable[..., float]] = {
    "cos": _cosine_fn,
    "sine": _sine_fn,
    "sin_cos": _sin_cos_fn,
    "exp_sin": _exp_sin_fn,
    "cluster_mix": _cluster_mix_fn,
    "moving_gauss": _moving_gauss_fn,
    "poly2": _poly2_fn,  # 2D polynomial with cross-term
    "poly_entangled": _poly_entangled,  # 2D polynomial with cross-term
    "polynomial": _poly_fn,
    "gabor_travel": gabor_travel,
    "gauss_shear": gauss_shear,
    "vortex": vortex,
    "osc_cluster": osc_cluster,
    
}


poly_entangled_kernel = lambda r, dt, **kw: _poly_entangled(
        np.array([r, 0.0]), dt, **kw)        # r put on x-axis, y=0
gabor_travel_kernel = lambda r, dt, **kw: gabor_travel(
        np.array([r, 0.0]), dt, **kw)        # r

# register the wrapped version for **kernels**
_FUNCTION_REGISTRY["poly_entangled_kernel"] = poly_entangled_kernel
_FUNCTION_REGISTRY["gabor_travel_kernel"] = gabor_travel_kernel