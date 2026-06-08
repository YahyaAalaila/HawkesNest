
from functools import partial
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Union
from hawkesnest.background.spatiotemporal import EntangledBackground

from hawkesnest.background.constant import ConstantBackground
from pydantic import BaseModel, Field, field_validator

from hawkesnest.config_factory.functions import _FUNCTION_REGISTRY
from hawkesnest.domain import RectangleDomain
from hawkesnest.domain.base import SpatialDomain


Vec2List = Sequence[Sequence[float]] 

class ConstantBackgroundCfg(BaseModel):
    type: Literal["constant"] = "constant"
    rate: float = 1.0
    def build(self, idx: int = 0):
        return ConstantBackground(self.rate)


class HeteroLadderBackgroundCfg(BaseModel):
    type: Literal["hetero_ladder"] = "hetero_ladder"
    lambda0: float
    theta_het: float
    T: float
    n_grid_space: int = 64
    n_grid_time: int = 64
    # name of the structured field function in _FUNCTION_REGISTRY, default "moving_gauss"
    field_name: str = "moving_gauss"

    def build(self, domain: RectangleDomain):
        from hawkesnest.background.hetero_ladder import HeteroLadderBackground
        from hawkesnest.config_factory.functions import _FUNCTION_REGISTRY

        g_fn = _FUNCTION_REGISTRY[self.field_name]
        print(f"[BackGround] building with {self.field_name}")
        return HeteroLadderBackground(
            lambda0=self.lambda0,
            g_fn=g_fn,
            domain=domain,
            T=self.T,
            theta_het=self.theta_het,
            n_grid_space=self.n_grid_space,
            n_grid_time=self.n_grid_time,
        )
   
# TODO: Break this up into smaller classes for each function type 
class FunctionBackgroundCfg(BaseModel):
    type: Literal["function"] = "function"
    name: str
    amp: Optional[float] = None
    freq: Optional[float] = None
    coeffs: Optional[Sequence[float]] = None
#     # parameters for cluster_mix
    centers: Sequence[Sequence[float]] | None = None
    
    c0  : Optional[float] = None
    c_t : Optional[float] = None
    c_x : Optional[float] = None
    c_tx: Optional[float] = None   # cross term t·x
    c_tt: Optional[float] = None
    c_xx: Optional[float] = None
    
    # parameters for poly_entangled
    ent: float | None = None
    quad: float | None = None
    lin:  float | None = None 
    aa0: Optional[float] = None  # a0
    
    ent_scale:     float | None                  = None
    start: Union[Sequence[float],  Vec2List, None] = None
    v:     Union[Sequence[float],  Vec2List, None] = None
    sigma: Union[float,    Sequence[float], None] = None
    base:  Union[float,    Sequence[float], None] = None
    a0:     Optional[float] = None
    a1:     Optional[float] = None
    omega:  Optional[float] = None
    freq_t: Optional[float] = None   # separate temporal frequency for gabor_travel

    @field_validator("name")
    @classmethod
    def _check_known(cls, v):
        if v not in _FUNCTION_REGISTRY:
            raise ValueError(f"Unknown background function '{v}'")
        return v

    #@staticmethod
    def _pick(self, x, idx: int, *, allow_vector: bool = False):
        """
        - If x is a single float/int, return it.
        - If x is a 2-tuple (e.g. [300,300]) but not a list-of-lists, return it as-is.
        - If x is a list of per-mark values, return x[idx].
        """
        if x is None:
            return None
        # broadcast scalars
        if isinstance(x, (int, float)):
            return x

        if isinstance(x, (list, tuple)):
            # treat length-2 numeric tuples as vectors only when requested
            if (
                allow_vector
                and len(x) == 2
                and not any(isinstance(el, (list, tuple)) for el in x)
                and all(isinstance(el, (int, float)) for el in x)
            ):
                return tuple(x)

            try:
                return x[idx]
            except (TypeError, IndexError):
                return x
        return x

    def build(self, idx: int = 0, domain: SpatialDomain | None = None):
        fn = _FUNCTION_REGISTRY[self.name]
        kwargs = {}
        if self.name in {"cos", "sine"}:
            kwargs["amp"] = self.amp or 1.0
            kwargs["freq"] = self.freq or 1.0
        elif self.name == "polynomial":
            kwargs["coeffs"] = list(self.coeffs or [2.0, 0.0, 0.0])
        elif self.name == "poly2":
            base = self.base or 1.0
            c0  = self.a0  or  1.0
            c_t = self.c_t or  0.0
            c_x = self.c_x or  0.0
            c_tx = self.c_tx or 1.0   # ← try 1.0, 2.0, 4.0 below
            c_tt = self.c_tt or 0.0
            c_xx = self.c_xx or 0.0
            kwargs.update(dict(base = base, a0=c0, a_t=c_t, a_x=c_x, a_tx=c_tx, a_tt=c_tt, a_xx=c_xx))
        elif self.name == "poly_entangled":
            # poly_entangled: a0, ent, quad, lin
            aa0 = self.aa0 or .0
            ent = self.ent or 1.0
            quad = self.quad or 0.0
            lin = self.lin or 0.0
            kwargs.update(dict(aa0=aa0, ent=ent, quad=quad, lin=lin))
        elif self.name == "cluster_mix":
            kwargs["centers"] = [tuple(c) for c in (self.centers or [])]
            kwargs["sigma"] = float(self._pick(self.sigma, idx) or 1.0)
            kwargs["a0"]    = float(self.a0 or 0.0)
            kwargs["amp"]   = float(self._pick(self.amp, idx) or 1.0) if self.amp is not None else 1.0
        elif self.name in ("moving_gauss", "moving_hotspots"):
            kwargs["start"] = tuple(self._pick(self.start, idx, allow_vector=True) or (0.5, 0.5))
            kwargs["v"]     = tuple(self._pick(self.v, idx, allow_vector=True) or (0.0, 0.0))
            kwargs["sigma"] = float(self._pick(self.sigma, idx) or 0.05)
            kwargs["a0"]    = float(self.a0 or 0.05)
            kwargs["amp"]   = float(self._pick(self.amp, idx) or 1.0) if self.amp is not None else 1.0
        elif self.name == "exp_sin":
            kwargs["a0"]    = self.a0 or 0.0
            kwargs["a1"]    = self.a1 or 1.0
            kwargs["omega"] = self.omega or 1.0
        elif self.name == "gabor_travel":
            kwargs["a0"]    = self.a0 or 0.0
            kwargs["amp"]   = self.amp or 3.0
            kwargs["fx"]    = self.freq or 3.0
            kwargs["fy"]    = self.freq or 3.0
            kwargs["ft"]    = self.freq_t if self.freq_t is not None else (self.freq or 4.0)
            kwargs["sigma"] = self.sigma or 0.2
            kwargs["cx"]    = self.start[0] if isinstance(self.start, (list, tuple)) else 0.5
            kwargs["cy"]    = self.start[1] if isinstance(self.start, (list, tuple)) else 0.5
            kwargs["phase"] = 0.0
        elif self.name == "gauss_shear":    
            kwargs["a0"]    = self.a0 or 0.0
            kwargs["sx"]    = self.sigma or 0.2
            kwargs["sy"]    = self.sigma or 0.2
            kwargs["kappa"] = self.base or 1.0
        elif self.name == "vortex":
            kwargs["a0"]    = self.a0 or 0.0
            kwargs["sigma"] = self.sigma or 0.25
            kwargs["gamma"] = self.amp or 3.0
            kwargs["omega"] = self.freq or 6.283
            kwargs["cx"]    = self.start[0] if isinstance(self.start, (list, tuple)) else 0.5
            kwargs["cy"]    = self.start[1] if isinstance(self.start, (list, tuple)) else 0.5
        elif self.name == "osc_cluster":
            # centers: list of [x,y]
            kwargs["centers"] = [tuple(c) for c in (self.centers or [])]
            kwargs["sigma"]   = float(self._pick(self.sigma, idx) or 0.08)
            kwargs["amps"]    = [float(a) for a in (self.amp or [1.0])]
            kwargs["freqs"]   = [float(f) for f in (self.freq or [1.0])]
            kwargs["phases"]  = [0.0] * len(kwargs["amps"])
        return EntangledBackground(partial(fn, **kwargs))



class GaussianComponentSpec(BaseModel):
    cx: float
    cy: float
    sigma: float
    amplitude: float = 1.0


class RegimeSpec(BaseModel):
    t_start: float
    t_end: float
    components: List[GaussianComponentSpec]
    base_rate: float = 0.0


class RegimeSwitchingBackgroundCfg(BaseModel):
    """
    Piecewise-stationary Gaussian mixture background.

    Each ``regimes`` entry specifies a time window and a list of Gaussian
    hotspots active in that window.  Between changepoints the hotspot
    locations and weights jump, testing whether models can track spatial
    drift.

    Example YAML::

        type: regime_switching
        lambda0: 2.0
        regimes:
          - t_start: 0.0
            t_end: 50.0
            components:
              - {cx: 0.2, cy: 0.3, sigma: 0.1, amplitude: 10.0}
          - t_start: 50.0
            t_end: 100.0
            components:
              - {cx: 0.8, cy: 0.7, sigma: 0.1, amplitude: 10.0}
    """
    type: Literal["regime_switching"] = "regime_switching"
    lambda0: float = 1.0
    regimes: List[RegimeSpec]

    def build(self, idx: int = 0, domain=None):
        from hawkesnest.background.regime_switch import (
            RegimeSwitchingBackground,
            GaussianComponent,
            Regime,
        )
        regime_objs = []
        for spec in self.regimes:
            comps = [
                GaussianComponent(
                    cx=c.cx, cy=c.cy, sigma=c.sigma, amplitude=c.amplitude
                )
                for c in spec.components
            ]
            regime_objs.append(
                Regime(
                    t_start=spec.t_start,
                    t_end=spec.t_end,
                    components=comps,
                    base_rate=spec.base_rate,
                )
            )
        return RegimeSwitchingBackground(regimes=regime_objs, lambda0=self.lambda0)


BackgroundCfg = Annotated[
    Union[
        ConstantBackgroundCfg,
        HeteroLadderBackgroundCfg,
        FunctionBackgroundCfg,
        RegimeSwitchingBackgroundCfg,
    ],
    Field(discriminator="type"),
]
