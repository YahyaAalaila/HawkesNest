import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.neighbors import KernelDensity
from hawkesnest.domain import RectangleDomain
from hawkesnest.background.base import BackgroundBase

def kernel_density_lambda(events: pd.DataFrame,
                          bandwidth: float = 0.05) -> np.ndarray:
    """
    Gaussian‐KDE estimate λ̂(s) at each (x,y).
    """
    X   = events[['x', 'y']].to_numpy()
    kde = KernelDensity(bandwidth=bandwidth).fit(X)
    return np.exp(kde.score_samples(X))  # length-N

def pairwise_indicators(events_u: pd.DataFrame,
                        events_v: pd.DataFrame,
                        h_thresh: float,
                        t_thresh: float) -> np.ndarray:
    """
    Matrix I{d_spatial ≤ h_thresh AND |t_i−t_j| ≤ t_thresh}.
    """
    d_sp = distance_matrix(events_u[['x','y']], events_v[['x','y']])
    d_t  = np.abs(
        events_u['t'].to_numpy()[:, None]
      - events_v['t'].to_numpy()[None, :]
    )
    return (d_sp <= h_thresh) & (d_t <= t_thresh)


def global_K_inhom(events: pd.DataFrame,
                   lam: np.ndarray,
                   h: float,
                   t: float,
                   area: float,
                   T: float) -> float:
    """
    K(h,t) = (A·T)/(N²) Σ_{i≠j} I{…}/[λ_i λ_j].
    """
    N = len(events)
    I = pairwise_indicators(events, events, h, t)
    np.fill_diagonal(I, False)
    lam_prod = lam[:,None] * lam[None,:]
    return float((area * T / (N*N)) * (I / lam_prod).sum())

def global_L_inhom(K: float, h: float, t: float) -> float:
    """L = sqrt(K/(π h² t)) − 1."""
    return np.sqrt(K / (np.pi * h*h * t)) - 1.0


def L_grid(events: pd.DataFrame,
           h_vals: np.ndarray,
           t_vals: np.ndarray,
           area: float,
           T: float,
           kde_bw: float = 0.05) -> np.ndarray:
    """
    Univariate inhomogeneous L over grid: returns array shape (len(h_vals), len(t_vals)).
    """
    lam = kernel_density_lambda(events, kde_bw)
    L   = np.zeros((len(h_vals), len(t_vals)))
    for i, h in enumerate(h_vals):
        for j, tt in enumerate(t_vals):
            K = global_K_inhom(events, lam, h, tt, area, T)
            L[i,j] = global_L_inhom(K, h, tt)
    return L


def csr_envelope(area_rect: tuple[float,float,float,float],
                 total_events: int,
                 h_vals: np.ndarray,
                 t_vals: np.ndarray,
                 T: float,
                 n_sims: int = 199,
                 percentile: float = 97.5) -> np.ndarray:
    """
    Monte-Carlo upper envelope of L under CSR on the same (h,t) grid.
    """
    xmin, xmax, ymin, ymax = area_rect
    # compute positive area
    area = (xmax - xmin) * (ymax - ymin)
    sims = np.zeros((n_sims, len(h_vals), len(t_vals)))

    for m in range(n_sims):
        xs = np.random.uniform(xmin, xmax, total_events)
        ys = np.random.uniform(ymin, ymax, total_events)
        ts = np.random.uniform(0.0, T, total_events)
        df = pd.DataFrame({'x': xs, 'y': ys, 't': ts})
        sims[m] = L_grid(df, h_vals, t_vals, area, T)

    return np.percentile(sims, percentile, axis=0)

def alpha_het(
    events: pd.DataFrame,
    L_env: np.ndarray,
    h_vals: np.ndarray,
    t_vals: np.ndarray,
    area: float,
    T: float,
    kde_bw: float = 0.05,
    min_n: int = 200,
) -> tuple[float, np.ndarray]:
    """
    Spatio-temporal heterogeneity index α_het ∈ [0,1].

    Parameters
    ----------
    events : DataFrame with columns ['x','y','t'].
    L_env  : array (len(h_vals), len(t_vals)) upper CSR envelope.
    h_vals : spatial radii grid.
    t_vals : temporal lags grid.
    area   : spatial window area |W|.
    T      : temporal horizon.
    kde_bw : bandwidth for spatial KDE.
    min_n  : minimum sample size for stable estimation.

    Returns
    -------
    alpha_het : float in [0,1].
    L_obs     : observed L(h,t) array.
    """
    n = len(events)
    if n < min_n:
        raise ValueError(f"alpha_het requires at least {min_n} events, got n={n}")

    L_obs = L_grid(events, h_vals, t_vals, area, T, kde_bw)

    D = np.maximum(L_obs - L_env, 0.0)

    # grid spacing (assume uniform)
    if len(h_vals) < 2 or len(t_vals) < 2:
        raise ValueError("h_vals and t_vals must contain at least 2 grid points each.")
    dh = float(np.diff(h_vals).mean())
    dt = float(np.diff(t_vals).mean())

    num = float((D * dh * dt).sum())
    den = float((L_env * dh * dt).sum())

    if den <= 0.0:
        # envelope degenerate, should not happen if built correctly
        return 0.0, L_obs

    alpha = num / den
    alpha = max(0.0, min(1.0, alpha))
    return alpha, L_obs

def alpha_het_weighted(
    events: pd.DataFrame,
    L_env: np.ndarray,
    h_vals: np.ndarray,
    t_vals: np.ndarray,
    area: float,
    T: float,
    kde_bw: float = 0.05,
    min_n: int = 200,
) -> tuple[float, np.ndarray]:
    """
    Self-weighted heterogeneity index α_het_weighted ∈ [0,1).

    Uses a D^2-weighted integral so that regions with strong excess
    clustering dominate the score:

        D(h,t) = [L_obs(h,t) - L_env(h,t)]_+.

    Raw index:
        a_raw = ∬ D(h,t)^2 dh dt / ∬ D(h,t) L_env(h,t) dh dt,
    then squashed to [0,1) as a = a_raw / (1 + a_raw).

    This avoids diluting the signal over parts of the (h,t) grid where
    D ≈ 0.
    """
    n = len(events)
    if n < min_n:
        raise ValueError(f"alpha_het_weighted requires at least {min_n} events, got n={n}")

    L_obs = L_grid(events, h_vals, t_vals, area, T, kde_bw)
    D = np.maximum(L_obs - L_env, 0.0)

    if len(h_vals) < 2 or len(t_vals) < 2:
        raise ValueError("h_vals and t_vals must contain at least 2 grid points each.")
    dh = float(np.diff(h_vals).mean())
    dt = float(np.diff(t_vals).mean())

    # If there is no positive excess at all, heterogeneity is zero.
    if not np.any(D > 0):
        return 0.0, L_obs

    num = float(((D ** 2) * dh * dt).sum())
    den = float(((D * L_env) * dh * dt).sum())

    if den <= 0.0:
        return 0.0, L_obs

    a_raw = num / den
    alpha = a_raw / (1.0 + a_raw)  # map to [0,1)
    return alpha, L_obs



def background_heterogeneity_index(
    background: BackgroundBase,
    domain: RectangleDomain,
    T: float,
    n_grid_space: int = 64,
    n_grid_time: int = 1,
) -> tuple[float, float]:
    """
    Deterministic heterogeneity index of a background μ(s,t).

    H = Var[μ] / E[μ]^2  over (s,t) ∈ W × [0,T]
    alpha = H / (1+H)   ∈ [0,1].

    Returns (alpha, H).
    """
    xs = np.linspace(domain.x_min, domain.x_max, n_grid_space)
    ys = np.linspace(domain.y_min, domain.y_max, n_grid_space)
    ts = np.linspace(0.0, T, n_grid_time)

    vals = []
    for x in xs:
        for y in ys:
            for t in ts:
                vals.append(background(np.array([x, y], dtype=float), float(t)))
    mu_vals = np.asarray(vals, dtype=float)

    mu_mean = float(mu_vals.mean())
    mu_var  = float(mu_vals.var())

    if mu_mean <= 0.0 or not np.isfinite(mu_mean):
        # If this happens, your background is degenerate; fail fast.
        raise ValueError(f"Invalid background: mean μ = {mu_mean}")

    H = mu_var / (mu_mean * mu_mean)
    alpha = H / (1.0 + H)
    return float(alpha), float(H)

def alpha_het_quantile(
    events: pd.DataFrame,
    L_env: np.ndarray,
    h_vals: np.ndarray,
    t_vals: np.ndarray,
    area: float,
    T: float,
    kde_bw: float = 0.05,
    min_n: int = 200,
    gamma: float = 0.9,
) -> tuple[float, np.ndarray]:
    """
    Quantile-restricted heterogeneity index α_het_quantile ∈ [0,1).

    Focuses only on the "hot region" of the (h,t) plane where excess
    clustering is largest.

    Let
        D(h,t) = [L_obs(h,t) - L_env(h,t)]_+.

    Define q_γ as the γ-quantile of positive D values (γ in (0,1)).
    Let S_γ = {(h,t): D(h,t) ≥ q_γ}.

    Raw index:
        a_raw = ∬_{S_γ} D(h,t) dh dt / ∬_{S_γ} L_env(h,t) dh dt,

    then squashed to [0,1) as a = a_raw / (1 + a_raw).

    This is an adaptive weighting: the integration domain moves
    automatically to wherever the strongest clustering occurs.
    """
    n = len(events)
    if n < min_n:
        raise ValueError(f"alpha_het_quantile requires at least {min_n} events, got n={n}")

    if not (0.0 < gamma < 1.0):
        raise ValueError(f"gamma must be in (0,1), got {gamma}")

    L_obs = L_grid(events, h_vals, t_vals, area, T, kde_bw)
    D = np.maximum(L_obs - L_env, 0.0)

    if len(h_vals) < 2 or len(t_vals) < 2:
        raise ValueError("h_vals and t_vals must contain at least 2 grid points each.")
    dh = float(np.diff(h_vals).mean())
    dt = float(np.diff(t_vals).mean())

    D_pos = D[D > 0.0]
    if D_pos.size == 0:
        return 0.0, L_obs

    q_gamma = float(np.quantile(D_pos, gamma))
    mask = D >= q_gamma

    num = float(((D * mask) * dh * dt).sum())
    den = float(((L_env * mask) * dh * dt).sum())

    if den <= 0.0:
        return 0.0, L_obs

    a_raw = num / den
    alpha = a_raw / (1.0 + a_raw)  # map to [0,1)
    return alpha, L_obs

def alpha_corr(
    events: pd.DataFrame,
    L_env: np.ndarray,
    h_vals: np.ndarray,
    t_vals: np.ndarray,
    area: float,
    T: float,
    kde_bw: float = 0.05,
    min_n: int = 200,
) -> tuple[float, np.ndarray]:
    """
    CSR-based correlation heterogeneity index α_corr ∈ [0,1).

    Conceptual definition (continuous):

        C = ∬ w(h,t) [L_obs(h,t) − L_env(h,t)]_+^2 dh dt,
        α_corr = C / (1 + C),

    with w(h,t) = 1/(h t), L_env the inhomogeneous CSR envelope,
    and L_obs the inhomogeneous L-function of the data.

    Numerically we approximate the integral on the (h_vals, t_vals) grid.

    Parameters
    ----------
    events : DataFrame with ['x','y','t'].
    L_env  : (len(h_vals), len(t_vals)) CSR envelope.
    h_vals : 1D array of positive spatial radii.
    t_vals : 1D array of positive temporal lags.
    area   : spatial window area |W| (passed to L_grid).
    T      : temporal horizon (passed to L_grid).
    kde_bw : bandwidth for spatial KDE in L_grid.
    min_n  : minimum event count for stability.

    Returns
    -------
    alpha_corr : float in [0,1).
    L_obs      : observed L(h,t) array.
    """
    n = len(events)
    if n < min_n:
        raise ValueError(f"alpha_corr requires at least {min_n} events, got n={n}")

    if np.any(h_vals <= 0) or np.any(t_vals <= 0):
        raise ValueError("alpha_corr requires strictly positive h_vals and t_vals")

    # observed inhomogeneous L surface
    L_obs = L_grid(events, h_vals, t_vals, area, T, kde_bw)

    # positive departure from CSR envelope
    D = np.maximum(L_obs - L_env, 0.0)

    # grid spacing
    if len(h_vals) < 2 or len(t_vals) < 2:
        raise ValueError("h_vals and t_vals must each have at least 2 grid points.")
    dh = float(np.diff(h_vals).mean())
    dt = float(np.diff(t_vals).mean())

    # weight w(h,t) = 1/(h t) on the grid
    H, TT = np.meshgrid(h_vals, t_vals, indexing="ij")  # shape (len(h), len(t))
    W = 1.0 / (H * TT)

    # integral approximation
    C = float(((D ** 2) * W * dh * dt).sum())

    if C < 0.0 or not np.isfinite(C):
        return 0.0, L_obs

    alpha = C / (1.0 + C)
    alpha = max(0.0, min(1.0, alpha)) 
    return alpha, L_obs

