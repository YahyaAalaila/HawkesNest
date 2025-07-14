# hawkesnest/utils/simulate.py
from __future__ import annotations
import yaml, numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from hawkesnest.config import SimulatorConfig

# ----------------------------------------------------------------------
# Load all templates at import time (string or .yml file—both work)
# ----------------------------------------------------------------------
# Absolute directory  …/hawkesnest/datasets/templates
_TEMPLATE_DIR = (
    Path(__file__).resolve()          # …/hawkesnest/utils/simulate.py
          .parent                     # …/hawkesnest/utils
          / "../datasets/templates"   # → …/hawkesnest/datasets/templates
).resolve()
def _load_yaml(name: str) -> str:
    path = _TEMPLATE_DIR / f"{name}.yml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown pillar template “{name}.yml”")
    return path.read_text()

TEMPLATES: dict[str, str] = {
    pillar: _load_yaml(pillar)
    for pillar in ("entanglement", "heterogeneity", "topology")
}

def _resolve_sim_args(
    *,
    n_events: Optional[int],
    horizon: Optional[float],
    batch_mode: bool,
) -> Tuple[Optional[int], Optional[float]]:
    """
    Decide what to pass into `sim.simulate(n=…, horizon=…)`.
    - batch_mode=True: require horizon
    - batch_mode=False: allow exactly one of n_events or horizon
    """
    if batch_mode:
        if horizon is None:
            raise ValueError("`horizon` must be provided when `n_realisations` > 0")
        return None, horizon

    # single-run mode
    if (n_events is None) and (horizon is None):
        raise ValueError("For a single realization, supply either `n_events` or `horizon`")
    # if both are provided, you could choose to error, warn+choose one, or error:
    if (n_events is not None) and (horizon is not None):
        # here we choose to prefer horizon if both given
        return None, horizon

    return (n_events, None) if n_events is not None else (None, horizon)


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """In-place recursive dict merge: src into dst."""
    for k, v in src.items():
        if isinstance(v, dict) and k in dst and isinstance(dst[k], dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def simulate_pillar(
    pillar: str,
    *,
    overrides: Dict[str, Any] | None = None,
    n_realisations: int = 0,
    n_events: Optional[int] = None,
    horizon: Optional[float] = None,
    seed: int = 0,
) -> List[np.ndarray]:
    """
    Simulate one or many realizations for a given 'pillar'.  

    - If `n_realisations > 0`, we run *batch mode* (for training):  
      you **must** supply `horizon` (time‐window) and get back that
      many independent arrays of shape (num_events_i, 4).  
    - If `n_realisations == 0`, we run *inspect mode*:  
      you supply **either** `n_events` **or** `horizon` and get back
      a single array, wrapped in a list.

    Always returns a list of NumPy arrays with columns `[t, x, y, m]`.
    """
    # 1) load & override template
    cfg_dict = yaml.safe_load(TEMPLATES[pillar])
    if overrides:
        _deep_merge(cfg_dict, overrides)

    # 2) build sim
    cfg = SimulatorConfig.model_validate(cfg_dict)
    rng = np.random.default_rng(seed)
    sim = cfg.build()

    # 3) pick mode & resolve (n, horizon)
    batch_mode = n_realisations > 0
    runs = n_realisations if batch_mode else 1
    sim_args = _resolve_sim_args(
        n_events=n_events,
        horizon=horizon,
        batch_mode=batch_mode,
    )

    # 4) simulate
    outputs: List[np.ndarray] = []
    for _ in range(runs):
        # each call uses a fresh random seed
        s = int(rng.integers(1_000_000_000))
        ev_df, _ = sim.simulate(n=sim_args[0], horizon=sim_args[1], seed=s)

        arr = ev_df[["t", "x", "y", "m"]].to_numpy(dtype=np.float32)
        outputs.append(arr)

    return outputs