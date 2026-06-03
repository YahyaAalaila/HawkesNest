
import os
import time
from collections import deque
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
    *,
    debug: bool = False,
    # --- DEBUG/SAFETY CONTROLS ---
    heartbeat_s: float = 5.0,      # print status every X seconds
    slow_s: float = 0.5,           # warn if any phase takes > slow_s
    tau_max: float | None = None,
):
    """
    Thinning algorithm to generate spatiotemporal events.

    Debug instrumentation pins stalls to specific phases:
      - sample_point
      - background eval
      - trigger loop: distance / kernel eval
      - accept/reject

    Stops when either time_horizon is exceeded or n_acc events have been accepted.
    """
    m = adjacency.shape[0]
    rng = rng or np.random.default_rng()
    



    # default separable kernel if none provided
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

    events: list[dict] = []
    active_events: deque[dict] = deque()
    t_current = 0.0

    
    
        # -----------------------------
    # HARD INVARIANTS (units/safety)
    # -----------------------------
    if not (np.isfinite(time_horizon) and time_horizon > 0):
        raise ValueError(f"time_horizon must be finite >0, got {time_horizon}")

    if tau_max is not None:
        if not (np.isfinite(tau_max) and tau_max > 0):
            raise ValueError(f"tau_max must be finite >0, got {tau_max}")
        if tau_max > time_horizon + 1e-12:
            raise ValueError(
                f"tau_max ({tau_max}) > time_horizon ({time_horizon}). "
                "Unit mismatch or invalid config."
            )

    Lambda = float(Lambda)
    if not (np.isfinite(Lambda) and Lambda > 0):
        raise ValueError(f"Lambda must be finite >0, got {Lambda}")

    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square, got shape {adjacency.shape}")
    if np.any(~np.isfinite(adjacency)) or np.any(adjacency < 0):
        raise ValueError("adjacency must be finite and nonnegative")

    # Kernel must be nonnegative over its support (basic sanity spot-check)
    if debug and kernel is not None:
        # random spot-check: small ds and small dt should not give negative
        kk = next(iter(kernel.values()))
        v = float(kk(0.0, 1e-3))
        if not np.isfinite(v) or v < -1e-12:
            raise ValueError(f"Kernel returns invalid value at (0,1e-3): {v}")
    
    
    
    
    
    
    pid = os.getpid()
    wall0 = time.perf_counter()


    # Counters
    n_prop = 0
    n_accept = 0
    n_reject = 0
    n_env_violate = 0
    n_neg_bg = 0
    n_neg_trig = 0

    # Timing accumulators
    t_dt = 0.0
    t_sample = 0.0
    t_bg = 0.0
    t_trig = 0.0
    t_dist = 0.0
    t_kernel = 0.0
    t_ar = 0.0


    # Progress diagnostics
    last_accept_prop = 0  # proposals count at last accept



    def _warn(phase: str, dt: float, extra: str = ""):
        if debug and dt >= slow_s:
            print(
                f"[THIN SLOW] pid={pid} phase={phase} dt={dt:.3f}s "
                f"t={t_current:.4f} prop={n_prop} acc={n_accept} {extra}",
                flush=True,
            )

    def _propose_and_test(iter_idx: int):
        nonlocal t_current
        nonlocal n_prop, n_accept, n_reject, n_env_violate, n_neg_bg, n_neg_trig
        nonlocal t_dt, t_sample, t_bg, t_trig, t_dist, t_kernel, t_ar
        nonlocal last_accept_prop

        n_prop += 1

        # 1) draw Δt and advance time
        t0 = time.perf_counter()
        dt = -np.log(rng.uniform()) / Lambda
        t_current += dt
        # NEW: enforce horizon before doing any work
        if t_current > time_horizon:
            return "__STOP__"
        t1 = time.perf_counter()
        t_dt += (t1 - t0)
        _warn("dt", t1 - t0)

        # 2) sample spatial location & mark
        t0 = time.perf_counter()
        if hasattr(domain, "sample_edgepoint"):
            (x_cand, y_cand), meta_cand = domain.sample_edgepoint(rng)
        else:
            x_cand, y_cand = domain.sample_point(rng)
            meta_cand = None
        mark_cand = int(rng.integers(1, m + 1))
        t1 = time.perf_counter()
        t_sample += (t1 - t0)
        _warn("sample_point", t1 - t0, extra=f"mark={mark_cand}")

        # 3) background intensity
        t0 = time.perf_counter()
        lam_bg = background((x_cand, y_cand), t_current, mark_cand)
        t1 = time.perf_counter()
        t_bg += (t1 - t0)
        _warn("background", t1 - t0, extra=f"lam_bg={lam_bg:.4f}")

        # 4) triggering contribution
        t0_trig = time.perf_counter()
        trig = 0.0

        # Maintain active window if tau_max is provided
        if tau_max is not None:
            while active_events and (t_current - active_events[0]["t"]) > tau_max:
                active_events.popleft()

        for e in active_events:
            if e["t"] >= t_current:
                continue

            # distance can be the killer on topo domains
            t0 = time.perf_counter()
            s_vec = None  # 2-D offset vector (only set for Euclidean domains)
            if meta_cand is not None and e.get("meta") is not None and hasattr(domain, "distance_edgepoints"):
                ds = domain.distance_edgepoints(meta_cand, e["meta"])
                if hasattr(domain, "_apsp"):
                    assert ds >= 0.0
            else:
                dx = x_cand - e["x"]
                dy = y_cand - e["y"]
                ds = domain.distance((x_cand, y_cand), (e["x"], e["y"]))
                s_vec = np.array([dx, dy], dtype=float)
            t1 = time.perf_counter()
            t_dist += (t1 - t0)
            if debug and (t1 - t0) >= slow_s:
                print(
                    f"[THIN SLOW] pid={pid} phase=distance dt={t1-t0:.3f}s "
                    f"ds={ds} t={t_current:.4f} (topo?)",
                    flush=True,
                )

            dt_e = t_current - e["t"]
                        # -----------------------------
            # TIME-LAG INVARIANTS (core)
            # -----------------------------
            if debug:
                if not np.isfinite(dt_e) or dt_e < 0:
                    raise RuntimeError(f"Bad dt_e={dt_e} (t_current={t_current}, e_t={e['t']})")

                # If time was accidentally normalized to [0,1] while horizon is large,
                # dt_e will almost never exceed ~1.
                # This is the exact failure mode you described.
                if time_horizon > 10.0 and dt_e <= 1.0 + 1e-12:
                    # Don't kill immediately on a single small dt; collect evidence.
                    pass
                
                
            eta = adjacency[e["type"] - 1, mark_cand - 1]
            #print(f"DEBUG: eta={eta} for parent type {e['type']} -> child type {mark_cand}")
            t0 = time.perf_counter()
            kobj = kernel[(e["type"], mark_cand)]
            s_arg = s_vec if (s_vec is not None and getattr(kobj, "uses_vector_s", False)) else ds
            trig += float(eta) * float(kobj(s_arg, dt_e))
            t1 = time.perf_counter()
            t_kernel += (t1 - t0)
            if debug and (t1 - t0) >= slow_s:
                print(
                    f"[THIN SLOW] pid={pid} phase=kernel dt={t1-t0:.3f}s "
                    f"eta={eta:.4f} dt_e={dt_e:.4f}",
                    flush=True,
                )

        t1_trig = time.perf_counter()
        t_trig += (t1_trig - t0_trig)
        _warn("trigger_loop_total", t1_trig - t0_trig, extra=f"n_hist={len(active_events)} trig={trig:.4f}")

        # 5) accept/reject
        lam_tot = lam_bg + trig

        if lam_bg < 0.0:
            n_neg_bg += 1
            raise ValueError(f"Negative background intensity: {lam_bg} at t={t_current}")
        if trig < 0.0:
            n_neg_trig += 1
            raise ValueError(f"Negative triggering intensity: {trig} at t={t_current}")
        if lam_tot < 0.0:
            raise ValueError(f"Total intensity negative: {lam_tot} at t={t_current}")
        if Lambda <= 0.0:
            raise ValueError(f"Lambda (envelope) must be positive, got {Lambda}")
        if lam_tot > Lambda * 1.01:
            n_env_violate += 1
            raise RuntimeError(
                f"Intensity {lam_tot:.6f} exceeds envelope Lambda={Lambda:.6f} "
                f"at t={t_current:.4f}. Increase lambda_max or fix background/kernel."
            )

        t0 = time.perf_counter()
        threshold = lam_tot / Lambda
        u = rng.uniform()
        accept = u < threshold
        t1 = time.perf_counter()
        t_ar += (t1 - t0)
        _warn("accept_reject", t1 - t0, extra=f"u={u:.4f} thr={threshold:.4f}")

        if debug and (iter_idx % 10_000 == 0):  # don't spam every step
            print(
                f"[THIN STEP] pid={pid} step={iter_idx} t={t_current:.4f} dt={dt:.4f} "
                f"mark={mark_cand} bg={lam_bg:.4f} trig={trig:.4f} tot={lam_tot:.4f} "
                f"u={u:.4f} thr={threshold:.4f} -> {'ACC' if accept else 'rej'}",
                flush=True,
            )

        if accept:
            n_accept += 1
            last_accept_prop = n_prop
            return {
                "t": t_current,
                "x": float(x_cand),
                "y": float(y_cand),
                "type": int(mark_cand),
                "lambda": float(lam_tot),
                "is_triggered": bool(trig > 0.0),
                "meta": meta_cand,
            }
        else:
            n_reject += 1
            return None

    # main loop
    iter_idx = 0
    while True:
        # stop if we've reached desired count
        if n_acc is not None and len(events) >= n_acc:
            if debug:
                print(f"[THIN] reached target event count {n_acc}", flush=True)
            break

        evt = _propose_and_test(iter_idx)
        iter_idx += 1
        
        # Enforce horizon before doing any work
        if evt == "__STOP__":
            break
        if evt is not None:
            events.append(evt)
            active_events.append(evt)

        # stop if time horizon exceeded (and not counting by n_acc)
        if n_acc is None and t_current >= time_horizon:
            if debug:
                print(f"[THIN] time horizon reached at t={t_current:.4f} (limit={time_horizon})", flush=True)
            break

    # ensure sorted by time
    events.sort(key=lambda e: e["t"])
    if debug:
        print(f"[THIN] accepted={len(events)} prop={n_prop} env_viol={n_env_violate} neg_bg={n_neg_bg} neg_trig={n_neg_trig}", flush=True)

    return events
