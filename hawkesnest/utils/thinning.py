import numpy as np

from hawkesnest.kernel.separable import ExponentialGaussianKernel


def thinning(
    background,
    kernel,
    time_horizon: float,
    domain: list,
    adjacency: np.array,
    Lambda: float,
    n_acc: int,
    rng=None,
):
    """
    Thinning algorithm to generate spatiotemporal events.

    Parameters:
        composite_intensity: instance of BaseIntensity (e.g. CompositeIntensity).
        time_horizon: float, maximum simulation time.
        domain: list, e.g. [x_min, x_max, y_min, y_max].
        m: int, number of marks.
        Lambda: float, global upper bound on intensity.
        rng: np.random.Generator, defaults to np.random.default_rng().

    Returns:
        events: list of dicts with keys: 't', 'x', 'y', 'type', 'lambda'
    """
    m = adjacency.shape[0]  # Number of marks/types
    if kernel is None:
        kernel = {
            (i, j): ExponentialGaussianKernel(
                branching_ratio=adjacency[i - 1, j - 1],
                spatial_scale=0.1,
                temporal_scale=1.0,
            )
            for i in range(1, m + 1)
            for j in range(1, m + 1)
        }
    if len(domain.bounds) != 4:
        raise ValueError(
            "Domain must be a list of four elements: [x_min, x_max, y_min, y_max]."
        )
    if rng is None:
        rng = np.random.default_rng()

    events = []
    t_current = 0.0

    # Just added this, incase we want to control the number of events
    if n_acc is not None:
        while len(events) < n_acc:
            dt = -np.log(rng.uniform()) / Lambda
            t_candidate = t_current + dt
            x_candidate, y_candidate = domain.sample_point(rng)
            child_mark = rng.integers(1, m + 1)
            lam_val = background((x_candidate, y_candidate), t_candidate, child_mark)
            trig = 0.0
            for e in events:
                if e["t"] < t_candidate:
                    dt_candidate = t_candidate - e["t"]
                    dx = x_candidate - e["x"]
                    dy = y_candidate - e["y"]
                    parent_mark = e["type"]
                    ker = kernel[(parent_mark, child_mark)]
                    ds = np.hypot(dx, dy)
                    trig += ker(ds, dt_candidate)
            lambda_cand = lam_val + trig
            if rng.uniform() < lambda_cand / Lambda:
                events.append(
                    {
                        "t": t_candidate,
                        "x": x_candidate,
                        "y": y_candidate,
                        "type": child_mark,
                        "lambda": lambda_cand,
                    }
                )
            t_current = t_candidate
        events.sort(key=lambda e: e["t"])
        return events
    if time_horizon is None:
        raise ValueError(
            "time_horizon or n_acc must be specified to control simulation duration."
        )
    while t_current < time_horizon:
        dt = -np.log(rng.uniform()) / Lambda
        t_candidate = t_current + dt
        if t_candidate > time_horizon:
            break
        x_candidate, y_candidate = domain.sample_point(rng)
        child_mark = rng.integers(1, m + 1)
        lam_val = background((x_candidate, y_candidate), t_candidate, child_mark)
        trig = 0.0
        for e in events:
            if e["t"] < t_candidate:
                dt_candidate = t_candidate - e["t"]
                dx = x_candidate - e["x"]
                dy = y_candidate - e["y"]
                parent_mark = e["type"]
                ker = kernel[(parent_mark, child_mark)]
                ds = np.hypot(dx, dy)
                trig += kernel(ds, dt_candidate)

        lambda_cand = lam_val + trig
        if rng.uniform() < lambda_cand / Lambda:
            events.append(
                {
                    "t": t_candidate,
                    "x": x_candidate,
                    "y": y_candidate,
                    "type": child_mark,
                    "lambda": lambda_cand,
                }
            )
        t_current = t_candidate
    events.sort(key=lambda e: e["t"])
    return events
