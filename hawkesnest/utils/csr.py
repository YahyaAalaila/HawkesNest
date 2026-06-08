# -----------------------------------------------------------------
import hashlib
import json
from pathlib import Path
from hawkesnest.metrics import csr_envelope
import numpy as np


CSR_DIR = Path("csr_cache"); CSR_DIR.mkdir(exist_ok=True)

def _dom_key(dom):
    """Hashable domain key."""
    return dom.x_min, dom.x_max, dom.y_min, dom.y_max

def _env_filename(dom_key: tuple,
                  grid_h: np.ndarray,
                  grid_t: np.ndarray,
                  n_events: int,
                  n_sims: int,
                  percentile: float,
                  horizon: float) -> Path:
    """Stable filename derived from *all* parameters that affect the envelope."""
    payload = dict(
        domain   = dom_key,
        grid_h   = grid_h.tolist(),
        grid_t   = grid_t.tolist(),
        n_events = n_events,
        n_sims   = n_sims,
        pct      = percentile,
        horizon  = horizon,
    )
    h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return CSR_DIR / f"csr_{h}.npz"

def load_or_build_csr(dom_key: tuple,
                      grid_h: np.ndarray,
                      grid_t: np.ndarray,
                      n_events: int,
                      horizon: float,
                      n_sims: int = 99,
                      pct: float = 90):
    """
    Return (L_env, grid_h, grid_t), loading from cache if possible.
    """
    fn = _env_filename(dom_key, grid_h, grid_t, n_events, n_sims, pct, horizon)
    if fn.exists():
        arr = np.load(fn)
        return arr["L_env"], arr["grid_h"], arr["grid_t"]

    # -- build fresh --
    L_env = csr_envelope(dom_key, n_events, grid_h, grid_t,
                         horizon, n_sims=n_sims, percentile=pct)
    np.savez(fn, L_env=L_env, grid_h=grid_h, grid_t=grid_t)
    print(f"[CSR] cached new envelope → {fn.name}")
    return L_env, grid_h, grid_t
