import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.neighbors import KernelDensity

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

def alpha_het(events: pd.DataFrame,
              L_env: np.ndarray,
              h_vals: np.ndarray,
              t_vals: np.ndarray,
              area: float,
              T: float,
              kde_bw: float = 0.05) -> float:
    """
    α_het = ∬ [L_obs − L_env]_+ / ∬ L_env.
    """
    L_obs = L_grid(events, h_vals, t_vals, area, T, kde_bw)
    D     = np.maximum(L_obs - L_env, 0.0)
    dh    = np.diff(h_vals).mean()
    dt    = np.diff(t_vals).mean()
    num   = (D * dh * dt).sum()
    den   = (L_env * dh * dt).sum()
    return float(num / den), L_obs
