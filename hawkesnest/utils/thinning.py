import numpy as np

from hawkesnest.kernel.separable import ExponentialGaussianKernel


def thinning(background, kernel, time_horizon: float, domain, adjacency: np.ndarray, Lambda: float, n_acc: int = None, rng=None, *, debug: bool = False, tau_max: float | None = None):
    """Thinning algorithm to generate spatiotemporal events."""
    m = adjacency.shape[0]
    rng = rng or np.random.default_rng()
    Lambda = float(Lambda)
    if not np.isfinite(Lambda) or Lambda <= 0.0:
        raise ValueError(f"Lambda must be finite and positive, got {Lambda}")
    if not np.isfinite(time_horizon) or time_horizon <= 0.0:
        raise ValueError(f"time_horizon must be finite and positive, got {time_horizon}")
    if tau_max is not None and (not np.isfinite(tau_max) or tau_max <= 0.0):
        raise ValueError(f"tau_max must be finite and positive, got {tau_max}")
    if kernel is None:
        kernel = {(i, j): ExponentialGaussianKernel(branching_ratio=adjacency[i - 1, j - 1], spatial_scale=0.1, temporal_scale=1.0) for i in range(1, m + 1) for j in range(1, m + 1)}
    events = []
    t_current = 0.0

    def _propose_and_test():
        nonlocal t_current
        dt = -np.log(rng.uniform()) / Lambda
        t_current += dt
        if t_current > time_horizon:
            return "__STOP__"
        x_cand, y_cand = domain.sample_point(rng)
        mark_cand = int(rng.integers(1, m + 1))
        lam_bg = float(background((x_cand, y_cand), t_current, mark_cand))
        trig = 0.0
        for e in events:
            if e["t"] >= t_current:
                continue
            dt_e = t_current - e["t"]
            if tau_max is not None and dt_e > tau_max:
                continue
            dx = x_cand - e["x"]
            dy = y_cand - e["y"]
            ds = domain.distance((x_cand, y_cand), (e["x"], e["y"]))
            kobj = kernel[(e["type"], mark_cand)]
            s_arg = np.array([dx, dy], dtype=float) if getattr(kobj, "uses_vector_s", False) else ds
            eta = adjacency[mark_cand - 1, e["type"] - 1]
            trig += float(eta) * float(kobj(s_arg, dt_e))
        lam_tot = lam_bg + trig
        if lam_bg < 0.0 or trig < 0.0 or lam_tot < 0.0:
            raise ValueError(f"Negative intensity at t={t_current}: bg={lam_bg}, trig={trig}")
        if lam_tot > Lambda * 1.01:
            raise RuntimeError(f"Intensity {lam_tot:.6f} exceeds envelope Lambda={Lambda:.6f} at t={t_current:.4f}")
        if rng.uniform() < lam_tot / Lambda:
            return {"t": t_current, "x": float(x_cand), "y": float(y_cand), "type": int(mark_cand), "lambda": float(lam_tot), "is_triggered": bool(trig > 0.0)}
        return None

    while True:
        if n_acc is not None and len(events) >= n_acc:
            break
        evt = _propose_and_test()
        if evt == "__STOP__":
            break
        if evt is not None:
            events.append(evt)
        if n_acc is None and t_current >= time_horizon:
            break
    events.sort(key=lambda e: e["t"])
    return events


def auto_lambda(func, n_grid=200):
    """Crude upper bound by scanning the unit cube grid."""
    xs = np.linspace(0, 1, n_grid)
    ys = np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack((X.ravel(), Y.ravel()), axis=1)
    vals = np.empty(len(pts), dtype=float)
    for i, p in enumerate(pts):
        vals[i] = func(p, 300)
    lam = vals.reshape(X.shape)
    return float(np.nanmax(lam))
