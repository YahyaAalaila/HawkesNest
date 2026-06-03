from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def _as_xy(space: np.ndarray) -> tuple[float, float]:
    x, y = map(float, space[:2])
    return x, y


def _safe_exp(z: float, *, z_min: float = -50.0, z_max: float = 50.0) -> float:
    return float(np.exp(np.clip(z, z_min, z_max)))


def _cosine_fn(space: np.ndarray, t: float, *, amp: float = 1.0, freq: float = 1.0) -> float:
    r = float(np.linalg.norm(space))
    return amp * np.cos(2 * np.pi * freq * (t + r))


def _sine_fn(space: np.ndarray, t: float, *, amp: float = 1.0, freq: float = 1.0) -> float:
    r = float(np.linalg.norm(space))
    return np.exp(amp * np.sin(2 * np.pi * freq * (t + r)))


def _sin_cos_fn(space: np.ndarray, t: float, *, amp: float = 1.0, freq: float = 1.0) -> float:
    x, y = map(float, space[:2])
    return amp * (np.sin(2 * np.pi * freq * x) + np.cos(2 * np.pi * freq * y))


def _exp_sin_fn(space: np.ndarray, t: float, *, a0: float = 0.0, a1: float = 1.0, omega: float = 1.0) -> float:
    return np.exp(a0 + a1 * np.sin(omega * t))


def _cluster_mix_fn(space: np.ndarray, t: float, *, centers: Sequence[tuple[float, float]], sigma: float = 0.1, a0: float = 0.0, amp: float = 1.0) -> float:
    x, y = _as_xy(space)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    zsum = 0.0
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for cx, cy in centers:
        dx = x - float(cx)
        dy = y - float(cy)
        zsum += np.exp(-(dx * dx + dy * dy) * inv2s2)
    return _safe_exp(float(a0) + float(amp) * float(zsum))


def _moving_gauss_fn(space: np.ndarray, t: float, *, start: tuple[float, float] = (0.5, 0.5), v: tuple[float, float] = (0.2, 0.0), sigma: float = 0.03, a0: float = 0.0, amp: float = 1.0) -> float:
    x, y = _as_xy(space)
    center_x = (start[0] + v[0] * float(t)) % 1.0
    center_y = (start[1] + v[1] * float(t)) % 1.0
    r2 = (x - center_x) ** 2 + (y - center_y) ** 2
    return float(a0) + float(amp) * float(np.exp(-r2 / (2.0 * sigma * sigma)))


def _moving_hotspots_fn(space: np.ndarray, t: float, *, start: tuple[float, float] = (0.5, 0.5), v: tuple[float, float] = (0.2, 0.0), sigma: float = 0.05, a0: float = 0.05, amp: float = 1.0) -> float:
    x, y = _as_xy(space)
    base_centers = np.array([[0.2, 0.2], [0.8, 0.3], [0.3, 0.8]], dtype=float)
    base_sigmas = np.array([0.03, 0.04, 0.03], dtype=float) * (sigma / 0.05)
    base_amps = np.array([10.0, 7.0, 5.0], dtype=float)
    drift_x = (start[0] - 0.5) + v[0] * float(t)
    drift_y = (start[1] - 0.5) + v[1] * float(t)
    centers = base_centers.copy()
    centers[:, 0] = (centers[:, 0] + drift_x) % 1.0
    centers[:, 1] = (centers[:, 1] + drift_y) % 1.0
    field = 0.0
    for (cx, cy), sgm, scale in zip(centers, base_sigmas, base_amps):
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        field += scale * np.exp(-r2 / (2.0 * sgm * sgm))
    return float(a0) + float(amp) * float(field)


def _poly2_fn(space: np.ndarray, t: float, *, base: float = 1.0, a0: float, a_t: float, a_x: float, a_tx: float, a_tt: float, a_xx: float) -> float:
    return a0 + a_t * t + a_x * float(space[0]) + a_tt * (t**2) + a_xx * (float(space[0])**2) + a_tx * (t * float(space[0]))


def _poly_entangled(space: np.ndarray, t: float, *, aa0: float = 0.0, ent: float = 1.0, quad: float = 0.0, lin: float = 0.0, scale: float = 1.0) -> float:
    x, y = map(float, space[:2])
    return aa0 + lin * (x + y + t) + ent * (t * x + t * y) + quad * (t * t * x * x + t * t * y * y)


def _poly_fn(space: np.ndarray, t: float, *, coeffs: Sequence[float]) -> float:
    r = float(np.linalg.norm(space))
    return np.polyval(list(reversed(coeffs)), t + r)


def gabor_travel(space: np.ndarray, t: float, *, a0: float = 0.0, amp: float = 3.0, fx: float = 3.0, fy: float = 3.0, ft: float = 4.0, sigma: float = 0.2, cx: float = 0.5, cy: float = 0.5, phase: float = 0.0) -> float:
    x, y = _as_xy(space)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    gauss = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    cosine = np.cos(2 * np.pi * (fx * x + fy * y + ft * t) + phase)
    return _safe_exp(float(a0) + float(amp) * float(gauss) * float(cosine))


def gauss_shear(space, t, *, a0=0.0, sx=0.2, sy=0.2, kappa=1.0):
    x, y = map(float, space[:2])
    scale = 1.0 + kappa * t
    return np.exp(a0 - (x**2) / (2 * (sx * scale) ** 2) - (y**2) / (2 * (sy * scale) ** 2))


def vortex(space, t, *, a0=0.0, sigma=0.25, gamma=3.0, omega=6.283, cx=0.5, cy=0.5):
    x, y = map(float, space[:2])
    dx, dy = x - cx, y - cy
    r2 = dx * dx + dy * dy
    theta = np.arctan2(dy, dx)
    return np.exp(a0 - r2 / (2 * sigma**2) + gamma * np.sin(theta - omega * t))


def osc_cluster(space, t, *, centers, sigma=0.08, amps, freqs, phases):
    x, y = map(float, space[:2])
    z = 0.0
    for (cx, cy), A, f, ph in zip(centers, amps, freqs, phases):
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        z += A * (1 + np.sin(2 * np.pi * f * t + ph)) * np.exp(-r2 / (2 * sigma**2))
    return np.exp(z)


_FUNCTION_REGISTRY: dict[str, Callable[..., float]] = {"cos": _cosine_fn, "sine": _sine_fn, "sin_cos": _sin_cos_fn, "exp_sin": _exp_sin_fn, "cluster_mix": _cluster_mix_fn, "moving_gauss": _moving_gauss_fn, "moving_hotspots": _moving_hotspots_fn, "poly2": _poly2_fn, "poly_entangled": _poly_entangled, "polynomial": _poly_fn, "gabor_travel": gabor_travel, "gauss_shear": gauss_shear, "vortex": vortex, "osc_cluster": osc_cluster}
poly_entangled_kernel = lambda r, dt, **kw: _poly_entangled(np.array([r, 0.0]), dt, **kw)
gabor_travel_kernel = lambda r, dt, **kw: gabor_travel(np.array([r, 0.0]), dt, **kw)
_FUNCTION_REGISTRY["poly_entangled_kernel"] = poly_entangled_kernel
_FUNCTION_REGISTRY["gabor_travel_kernel"] = gabor_travel_kernel
