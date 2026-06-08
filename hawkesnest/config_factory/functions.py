
from __future__ import annotations

import numpy as np
from typing import Callable, Sequence


# ----------------------------
# helpers
# ----------------------------

def _as_xy(space: np.ndarray) -> tuple[float, float]:
    x, y = map(float, space[:2])
    return x, y


def _r(space: np.ndarray) -> float:
    return float(np.linalg.norm(space[:2]))


def _safe_exp(z: float, *, z_min: float = -50.0, z_max: float = 50.0) -> float:
    # Prevent under/overflow; keep positivity.
    return float(np.exp(np.clip(z, z_min, z_max)))


# ----------------------------
# background / surface functions
# All must return > 0
# ----------------------------

def _cosine_fn(space: np.ndarray, t: float, *, a0: float = 0.0, amp: float = 1.0, freq: float = 1.0) -> float:
    """
    μ(s,t) = exp(a0 + amp * cos(2π freq (t + ||s||))).
    """
    r = _r(space)
    z = a0 + amp * np.cos(2 * np.pi * freq * (t + r))
    return _safe_exp(float(z))


def _sine_fn(space: np.ndarray, t: float, *, a0: float = 0.0, amp: float = 1.0, freq: float = 1.0) -> float:
    """
    μ(s,t) = exp(a0 + amp * sin(2π freq (t + ||s||))).
    """
    r = _r(space)
    z = a0 + amp * np.sin(2 * np.pi * freq * (t + r))
    return _safe_exp(float(z))


def _sin_cos_fn(space: np.ndarray, t: float, *, a0: float = 0.0, amp: float = 1.0, freq: float = 1.0) -> float:
    """
    Spatial heterogeneity (positive):
      μ(s,t) = exp(a0 + amp * ( sin(2π f x) + cos(2π f y) )).
    Note: independent of t by design (t unused but kept for uniform signature).
    """
    x, y = _as_xy(space)
    z = a0 + amp * (np.sin(2 * np.pi * freq * x) + np.cos(2 * np.pi * freq * y))
    return _safe_exp(float(z))


def _exp_sin_fn(space: np.ndarray, t: float, *, a0: float = 0.0, a1: float = 1.0, omega: float = 1.0) -> float:
    """
    Time-varying (positive):
      μ(s,t) = exp(a0 + a1 sin(ω t)).
    """
    z = a0 + a1 * np.sin(omega * t)
    return _safe_exp(float(z))


def _cluster_mix_fn(
    space: np.ndarray,
    t: float,
    *,
    centers: Sequence[tuple[float, float]],
    sigma: float = 0.1,
    a0: float = 0.0,
    amp: float = 1.0,
) -> float:
    """
    Spatial Gaussian mixture, time-invariant (positive):
      μ(s,t) = exp(a0 + amp * sum_k exp(-||s-c_k||^2/(2 sigma^2))).
    """
    x, y = _as_xy(space)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    zsum = 0.0
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for cx, cy in centers:
        dx = x - float(cx)
        dy = y - float(cy)
        zsum += np.exp(-(dx * dx + dy * dy) * inv2s2)

    z = a0 + amp * float(zsum)
    return _safe_exp(float(z))
import numpy as np


def _moving_hotspots_fn(
    space: np.ndarray,
    t: float,
    *,
    start: tuple[float, float] = (0.5, 0.5),
    v: tuple[float, float] = (0.2, 0.0),
    sigma: float = 0.05,
    a0: float = 0.05,
    amp: float = 1.0,
) -> float:
    """
    Moving hotspot mixture: a few sharp peaks whose centers move with time.
    - Uses (start, v) as a *global drift* applied to all hotspots (so signature params matter).
    - sigma controls hotspot sharpness.
    - a0 is baseline.
    - amp scales the overall field.
    """
    X, Y = _as_xy(space)

    # Base hotspot layout (relative to unit square)
    base_centers = np.array([[0.2, 0.2], [0.8, 0.3], [0.3, 0.8]], dtype=float)
    base_sigmas  = np.array([0.03, 0.04, 0.03], dtype=float) * (sigma / 0.05)
    base_amps    = np.array([10.0, 7.0, 5.0], dtype=float)

    # Add a global drift to all centers. Keep it bounded with wrap-around.
    cx0, cy0 = start
    vx, vy = v
    drift_x = (cx0 - 0.5) + vx * float(t)
    drift_y = (cy0 - 0.5) + vy * float(t)

    centers = base_centers.copy()
    centers[:, 0] = (centers[:, 0] + drift_x) % 1.0
    centers[:, 1] = (centers[:, 1] + drift_y) % 1.0

    field = np.zeros_like(X, dtype=float)
    for (cx, cy), sgm, a in zip(centers, base_sigmas, base_amps):
        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        field += a * np.exp(-r2 / (2.0 * sgm * sgm))

    field = a0 + amp * field
    return float(field) if np.isscalar(field) else field

def _moving_gauss_fn(
    space: np.ndarray,
    t: float,
    *,
    start: tuple[float, float] = (0.5, 0.5),   # unused; kept for compatibility
    v: tuple[float, float] = (0.2, 0.0),       # unused; kept for compatibility
    sigma: float = 0.03,                       # default narrow
    a0: float = 0.0,                           # phase offset
    amp: float = 1.0,                          # global multiplier
) -> float:

    X, Y = _as_xy(space)

    # --- base (mean) centers for three blobs
    base = np.array([
        [0.20, 0.20],
        [0.80, 0.30],
        [0.30, 0.80],
    ], dtype=float)

    # --- per-blob arc radii (small so they stay narrow and "hotspotty")
    # Elliptic arcs: (rx, ry)
    radii = np.array([
        [0.18, 0.10],
        [0.14, 0.16],
        [0.20, 0.12],
    ], dtype=float)

    # --- aggressive angular speeds (radians per unit time)
    omegas = np.array([10.0, 13.0, 11.0], dtype=float)

    # --- per-blob phases (separate trajectories)
    phases = a0 + np.array([0.0, 2.1, 4.2], dtype=float)

    # --- narrow sigmas per blob (override sigma if you want heterogeneity)
    sigmas = np.array([sigma, 1.2 * sigma, 0.9 * sigma], dtype=float)

    # --- amplitudes per blob (sharp peaks)
    amps = amp * np.array([10.0, 7.0, 5.0], dtype=float)

    # optional: add a slow drift so arcs sweep the domain more
    drift = 0.10 * np.array([np.cos(0.7 * t), np.sin(0.6 * t)], dtype=float)

    # compute moving centers with wrap-around to keep within [0,1)
    theta = omegas * t + phases
    centers = base + drift + np.column_stack([radii[:, 0] * np.cos(theta),
                                              radii[:, 1] * np.sin(theta)])
    centers = centers % 1.0

    baseline = 0.05
    field = np.zeros_like(X, dtype=float)
    for (cx, cy), sgm, a in zip(centers, sigmas, amps):
        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        field += a * np.exp(-r2 / (2.0 * sgm * sgm))

    field = baseline + field
    return float(field) if np.isscalar(field) else field
def _poly2_fn(
    space: np.ndarray,
    t: float,
    *,
    a0: float = 0.0,
    a_t: float = 0.0,
    a_x: float = 0.0,
    a_tx: float = 0.0,
    a_tt: float = 0.0,
    a_xx: float = 0.0,
    base: float = 1.0,
) -> float:
    """
    Polynomial log-intensity (positive):
      log μ = a0 + a_t t + a_x x + a_tt t^2 + a_xx x^2 + a_tx t x
      μ = base * exp(log μ)
    """
    if base <= 0:
        raise ValueError("base must be > 0")

    x = float(space[0])
    z = a0 + a_t * t + a_x * x + a_tt * (t * t) + a_xx * (x * x) + a_tx * (t * x)
    return float(base) * _safe_exp(float(z))


def _poly_entangled(
    space: np.ndarray,
    t: float,
    *,
    aa0: float = 0.0,
    ent: float = 1.0,
    quad: float = 0.0,
    lin: float = 0.0,
    scale: float = 1.0,
) -> float:
    """
    Positive entangled polynomial:
      μ(s,t) = scale * exp(
          aa0
        + lin  * (x + y + t)
        + ent  * (t x + t y)
        + quad * (t^2 x^2 + t^2 y^2)
      )
    """
    if scale <= 0:
        raise ValueError("scale must be > 0")

    x, y = _as_xy(space)
    z = (
        aa0
        + lin * (x + y + t)
        + ent * (t * x + t * y)
        + quad * (t * t * x * x + t * t * y * y)
    )
    return float(scale) * _safe_exp(float(z))


def _poly_fn(
    space: np.ndarray,
    t: float,
    *,
    coeffs: Sequence[float],
    a0: float = 0.0,
    use_radius: bool = True,
    center: float = 0.0,
    scale: float = 1.0,
) -> float:
    if scale == 0:
        return _safe_exp(float(a0))

    u = float(t + _r(space)) if use_radius else float(t)

    # np.polyval expects descending order; your original reversed(coeffs) was correct for ascending input.
    zpoly = float(np.polyval(list(reversed(list(coeffs))), u - float(center)))
    z = float(a0) + float(scale) * zpoly
    return _safe_exp(z)


def gabor_travel(
    space: np.ndarray,
    t: float,
    *,
    a0: float = 0.0,
    amp: float = 3.0,
    fx: float = 3.0,
    fy: float = 3.0,
    ft: float = 4.0,
    sigma: float = 0.2,
    cx: float = 0.5,
    cy: float = 0.5,
    phase: float = 0.0,
) -> float:
    x, y = _as_xy(space)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    gauss = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    cosine = np.cos(2 * np.pi * (fx * x + fy * y + ft * t) + phase)
    z = float(a0) + float(amp) * float(gauss) * float(cosine)
    return _safe_exp(z)


def gauss_shear(
    space: np.ndarray,
    t: float,
    *,
    a0: float = 0.0,
    sx: float = 0.2,
    sy: float = 0.2,
    kappa: float = 1.0,
) -> float:
    x, y = _as_xy(space)
    if sx <= 0 or sy <= 0:
        raise ValueError("sx, sy must be > 0")
    scale = 1.0 + kappa * t
    # avoid division by ~0 if scale crosses 0 (user error); enforce minimum
    scale = float(max(scale, 1e-6))
    z = float(a0) - (x * x) / (2 * (sx * scale) ** 2) - (y * y) / (2 * (sy * scale) ** 2)
    return _safe_exp(z)

def _stripes_fn(
    space: np.ndarray,
    t: float,
    *,
    a0: float = 0.0,
    amp: float = 1.0,
    freq: float = 6.0,
    mode: str = "sum",  # "sum" or "prod" or "checker"
) -> float:

    x, y = _as_xy(space)
    sx = np.sin(2 * np.pi * float(freq) * x)
    sy = np.sin(2 * np.pi * float(freq) * y)

    if mode == "sum":
        z = float(a0) + float(amp) * float(sx + sy)
    elif mode == "prod":
        z = float(a0) + float(amp) * float(sx * sy)
    elif mode == "checker":
        z = float(a0) + float(amp) * float(np.sign(sx) * np.sign(sy))
    else:
        raise ValueError(f"Unknown stripes mode={mode!r}")

    return _safe_exp(z)

def vortex(
    space: np.ndarray,
    t: float,
    *,
    a0: float = 0.0,
    sigma: float = 0.25,
    gamma: float = 3.0,
    omega: float = 6.283,
    cx: float = 0.5,
    cy: float = 0.5,
) -> float:
    x, y = _as_xy(space)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    dx, dy = x - cx, y - cy
    r2 = dx * dx + dy * dy
    theta = np.arctan2(dy, dx)
    z = float(a0) - r2 / (2 * sigma ** 2) + float(gamma) * float(np.sin(theta - omega * t))
    return _safe_exp(z)


def osc_cluster(
    space: np.ndarray,
    t: float,
    *,
    centers: Sequence[tuple[float, float]],
    sigma: float = 0.08,
    amps: Sequence[float],
    freqs: Sequence[float],
    phases: Sequence[float],
    a0: float = 0.0,
) -> float:
    x, y = _as_xy(space)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    z = 0.0
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for (cx, cy), A, f, ph in zip(centers, amps, freqs, phases):
        dx = x - float(cx)
        dy = y - float(cy)
        r2 = dx * dx + dy * dy
        # nonnegative modulation: 1 + sin in [0,2]
        mod = (1.0 + np.sin(2 * np.pi * float(f) * t + float(ph)))
        z += float(A) * float(mod) * float(np.exp(-r2 * inv2s2))

    # keep the same pattern: exp(a0 + z)
    return _safe_exp(float(a0) + float(z))

def _switching_regimes_fn(
    space: np.ndarray,
    t: float,
    *,
    start: tuple[float, float] = (0.5, 0.5),
    v: tuple[float, float] = (0.2, 0.0),
    sigma: float = 0.05,
    a0: float = 0.05,
    amp: float = 1.0,
) -> float:
    """
    Switching-regime hotspots: hotspot centers jump between K regimes over time.
    """
    X, Y = _as_xy(space)

    speed = float(np.hypot(v[0], v[1]))
    K = int(np.clip(np.round(2 + 12 * speed), 2, 20))  # 2..20 regimes

    u = float(t) % 1.0
    k = int(np.floor(K * u))  # 0..K-1


    seed = (12345 + 1000 * k + int(1e3 * start[0]) + 37 * int(1e3 * start[1])) % (2**32 - 1)
    rng = np.random.default_rng(seed)

    centers = rng.uniform(0.15, 0.85, size=(3, 2))
    centers[:, 0] = (centers[:, 0] + (start[0] - 0.5)) % 1.0
    centers[:, 1] = (centers[:, 1] + (start[1] - 0.5)) % 1.0

    base_sigmas = np.array([0.03, 0.04, 0.03], dtype=float) * (sigma / 0.05)
    base_amps   = np.array([10.0, 7.0, 5.0], dtype=float)

    field = np.zeros_like(X, dtype=float)
    for (cx, cy), sgm, a in zip(centers, base_sigmas, base_amps):
        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        field += a * np.exp(-r2 / (2.0 * sgm * sgm))

    field = a0 + amp * field
    return float(field) if np.isscalar(field) else field

def _moving_gauss_fn_slow(
    space: np.ndarray,
    t: float,
    *,
    start: tuple[float, float] = (0.5, 0.5),  # kept for compatibility
    v: tuple[float, float] = (0.2, 0.0),      # kept for compatibility
    sigma: float = 0.03,                      # narrow blobs
    a0: float = 0.0,
    amp: float = 1.0,
) -> float:
    """
    3 moving narrow hotspots with slow arc motion (periodic wrap in [0,1)^2).
    """
    X, Y = _as_xy(space)

    base = np.array([[0.20, 0.20],
                     [0.80, 0.30],
                     [0.30, 0.80]], dtype=float)

    radii = np.array([[0.18, 0.10],
                      [0.14, 0.16],
                      [0.20, 0.12]], dtype=float)

    
    omegas = np.array([0.5, 0.7, 0.4], dtype=float)
    phases = a0 + np.array([0.0, 2.1, 4.2], dtype=float)

    sigmas = np.array([sigma, 1.2 * sigma, 0.9 * sigma], dtype=float)
    amps = amp * np.array([10.0, 7.0, 5.0], dtype=float)

  
    drift = 0.06 * np.array([np.cos(0.08 * t), np.sin(0.06 * t)], dtype=float)

    theta = omegas * t + phases
    centers = base + drift + np.column_stack([
        radii[:, 0] * np.cos(theta),
        radii[:, 1] * np.sin(theta),
    ])

  
    centers = centers % 1.0

    baseline = 0.05
    field = np.zeros_like(X, dtype=float)
    for (cx, cy), sgm, a in zip(centers, sigmas, amps):
        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        field += a * np.exp(-r2 / (2.0 * sgm * sgm))

    field = baseline + field
    return float(field) if np.isscalar(field) else field


_FUNCTION_REGISTRY: dict[str, Callable[..., float]] = {
    "cos": _cosine_fn,
    "sine": _sine_fn,
    "sin_cos": _sin_cos_fn,
    "exp_sin": _exp_sin_fn,
    "cluster_mix": _cluster_mix_fn,
    "moving_gauss": _moving_gauss_fn,
    "moving_gauss_slow": _moving_gauss_fn_slow,
    "moving_hotspots": _moving_hotspots_fn,
    "poly2": _poly2_fn,
    "poly_entangled": _poly_entangled,
    "polynomial": _poly_fn,
    "gabor_travel": gabor_travel,
    "gauss_shear": gauss_shear,
    "vortex": vortex,
    "osc_cluster": osc_cluster,
    "switching_regimes": _switching_regimes_fn,
    
}


# ----------------------------
# kernel wrappers (r, dt) -> treat r as x and y=0
# ----------------------------

def poly_entangled_kernel(r: float, dt: float, **kw) -> float:
    return _poly_entangled(np.array([float(r), 0.0], dtype=float), float(dt), **kw)

def gabor_travel_kernel(r: float, dt: float, **kw) -> float:
    return gabor_travel(np.array([float(r), 0.0], dtype=float), float(dt), **kw)

_FUNCTION_REGISTRY["poly_entangled_kernel"] = poly_entangled_kernel
_FUNCTION_REGISTRY["gabor_travel_kernel"] = gabor_travel_kernel
_FUNCTION_REGISTRY["stripes"] = _stripes_fn
