import numpy as np
from hawkesnest.kernel.separable import ExponentialGaussianKernel

def thinning(
    background,
    kernel,
    time_horizon: float,
    domain,
    adjacency: np.ndarray,
    Lambda: float,
    n_acc: int = None,
    rng=None,
):
    """
    Thinning algorithm to generate spatiotemporal events.

    Stops when either time_horizon is exceeded or n_acc events have been accepted.
    """
    m = adjacency.shape[0]
    rng = rng or np.random.default_rng()

    # default separable kernel if none provided
    if kernel is None:
        kernel = {
            (i, j): ExponentialGaussianKernel(
                branching_ratio=adjacency[i-1, j-1],
                spatial_scale=0.1,
                temporal_scale=1.0,
            )
            for i in range(1, m+1)
            for j in range(1, m+1)
        }

    events = []
    t_current = 0.0

    def _propose_and_test():
        nonlocal t_current

        # 1) draw Δt and advance time
        dt = -np.log(rng.uniform()) / Lambda
        t_current += dt

        # 2) sample spatial location & mark
        x_cand, y_cand = domain.sample_point(rng)
        mark_cand = rng.integers(1, m+1)

        # 3) background intensity
        lam_bg = background((x_cand, y_cand), t_current, mark_cand)

        # 4) triggering contribution
        trig = 0.0
        for e in events:
            if e["t"] >= t_current:
                break  # events sorted by time
            ds = domain.distance((x_cand, y_cand), (e["x"], e["y"]))
            dt_e = t_current - e["t"]
            η = adjacency[mark_cand-1, e["type"]-1]
            trig += η * kernel[(e["type"], mark_cand)](ds, dt_e)

        # 5) accept/reject
        lam_tot = lam_bg + trig
        if rng.uniform() < lam_tot / Lambda:
            return {
                "t": t_current,
                "x": x_cand,
                "y": y_cand,
                "type": mark_cand,
                "lambda": lam_tot,
                "is_triggered": (trig > 0.0),
            }
        return None

    # main loop
    while True:
        # stop if we've reached desired count
        if n_acc is not None and len(events) >= n_acc:
            break

        evt = _propose_and_test()
        if evt is not None:
            events.append(evt)

        # stop if time horizon exceeded (and not counting by n_acc)
        if n_acc is None and t_current >= time_horizon:
            break

    # ensure sorted by time
    events.sort(key=lambda e: e["t"])
    return events


def auto_lambda(func, n_grid = 200):
    """Crude upper bound by scanning the unit cube grid."""
    # TODO: I am sure there is a better way to do this
    xs = np.linspace(0, 1, n_grid)
    ys = np.linspace(0, 1, n_grid)
    ts = np.linspace(0, 1, n_grid)
    # build grid of spatial points
    X, Y = np.meshgrid(xs, ys)                  # both shape (n_grid,n_grid)
    pts = np.stack((X.ravel(), Y.ravel()), axis=1)   # shape (n_grid*n_grid, 2)
    # evaluate background.surface (or kernel) at each point individually
    vals = np.empty(len(pts), dtype=float)
    for i, p in enumerate(pts):
        vals[i] = func(p, 300)                    # now p is a 1D array of length 2

    lam = vals.reshape(X.shape)                 # back to (n_grid,n_grid)
    return float(np.nanmax(lam))