from functools import partial
from typing import Annotated, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field, field_validator

from hawkesnest.background.constant import ConstantBackground
from hawkesnest.background.spatiotemporal import EntangledBackground
from hawkesnest.config_factory.functions import _FUNCTION_REGISTRY


Vec2List = Sequence[Sequence[float]]


class ConstantBackgroundCfg(BaseModel):
    type: Literal["constant"] = "constant"
    rate: float = 1.0

    def build(self, idx: int = 0):
        return ConstantBackground(self.rate)


class FunctionBackgroundCfg(BaseModel):
    type: Literal["function"] = "function"
    name: str
    amp: Optional[float] = None
    freq: Optional[float] = None
    coeffs: Optional[Sequence[float]] = None
    centers: Sequence[Sequence[float]] | None = None
    c0: Optional[float] = None
    c_t: Optional[float] = None
    c_x: Optional[float] = None
    c_tx: Optional[float] = None
    c_tt: Optional[float] = None
    c_xx: Optional[float] = None
    ent: float | None = None
    quad: float | None = None
    lin: float | None = None
    aa0: Optional[float] = None
    ent_scale: float | None = None
    start: Union[Sequence[float], Vec2List, None] = None
    v: Union[Sequence[float], Vec2List, None] = None
    sigma: Union[float, Sequence[float], None] = None
    base: Union[float, Sequence[float], None] = None
    a0: Optional[float] = None
    a1: Optional[float] = None
    omega: Optional[float] = None
    freq_t: Optional[float] = None

    @field_validator("name")
    @classmethod
    def _check_known(cls, v):
        if v not in _FUNCTION_REGISTRY:
            raise ValueError(f"Unknown background function '{v}'")
        return v

    def _pick(self, x, idx: int, *, allow_vector: bool = False):
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return x
        if isinstance(x, (list, tuple)):
            if allow_vector and len(x) == 2 and not any(isinstance(el, (list, tuple)) for el in x) and all(isinstance(el, (int, float)) for el in x):
                return tuple(x)
            try:
                return x[idx]
            except (TypeError, IndexError):
                return x
        return x

    def build(self, idx: int = 0):
        fn = _FUNCTION_REGISTRY[self.name]
        kwargs = {}
        if self.name in {"cos", "sine"}:
            kwargs["amp"] = self.amp or 1.0
            kwargs["freq"] = self.freq or 1.0
        elif self.name == "polynomial":
            kwargs["coeffs"] = list(self.coeffs or [2.0, 0.0, 0.0])
        elif self.name == "poly2":
            kwargs.update(dict(base=self.base or 1.0, a0=self.a0 or 1.0, a_t=self.c_t or 0.0, a_x=self.c_x or 0.0, a_tx=self.c_tx or 1.0, a_tt=self.c_tt or 0.0, a_xx=self.c_xx or 0.0))
        elif self.name == "poly_entangled":
            kwargs.update(dict(aa0=self.aa0 or 0.0, ent=self.ent or 1.0, quad=self.quad or 0.0, lin=self.lin or 0.0))
        elif self.name == "cluster_mix":
            kwargs["centers"] = [tuple(c) for c in (self.centers or [])]
            kwargs["sigma"] = float(self._pick(self.sigma, idx) or 1.0)
            kwargs["a0"] = float(self.a0 or 0.0)
            kwargs["amp"] = float(self._pick(self.amp, idx) or 1.0) if self.amp is not None else 1.0
        elif self.name in ("moving_gauss", "moving_hotspots"):
            kwargs["start"] = tuple(self._pick(self.start, idx, allow_vector=True) or (0.5, 0.5))
            kwargs["v"] = tuple(self._pick(self.v, idx, allow_vector=True) or (0.0, 0.0))
            kwargs["sigma"] = float(self._pick(self.sigma, idx) or 0.05)
            kwargs["a0"] = float(self.a0 or 0.05)
            kwargs["amp"] = float(self._pick(self.amp, idx) or 1.0) if self.amp is not None else 1.0
        elif self.name == "exp_sin":
            kwargs["a0"] = self.a0 or 0.0
            kwargs["a1"] = self.a1 or 1.0
            kwargs["omega"] = self.omega or 1.0
        elif self.name == "gabor_travel":
            kwargs["a0"] = self.a0 or 0.0
            kwargs["amp"] = self.amp or 3.0
            kwargs["fx"] = self.freq or 3.0
            kwargs["fy"] = self.freq or 3.0
            kwargs["ft"] = self.freq_t if self.freq_t is not None else (self.freq or 4.0)
            kwargs["sigma"] = self.sigma or 0.2
            kwargs["cx"] = self.start[0] if isinstance(self.start, (list, tuple)) else 0.5
            kwargs["cy"] = self.start[1] if isinstance(self.start, (list, tuple)) else 0.5
            kwargs["phase"] = 0.0
        elif self.name == "gauss_shear":
            kwargs["a0"] = self.a0 or 0.0
            kwargs["sx"] = self.sigma or 0.2
            kwargs["sy"] = self.sigma or 0.2
            kwargs["kappa"] = self.base or 1.0
        elif self.name == "vortex":
            kwargs["a0"] = self.a0 or 0.0
            kwargs["sigma"] = self.sigma or 0.25
            kwargs["gamma"] = self.amp or 3.0
            kwargs["omega"] = self.freq or 6.283
            kwargs["cx"] = self.start[0] if isinstance(self.start, (list, tuple)) else 0.5
            kwargs["cy"] = self.start[1] if isinstance(self.start, (list, tuple)) else 0.5
        elif self.name == "osc_cluster":
            kwargs["centers"] = [tuple(c) for c in (self.centers or [])]
            kwargs["sigma"] = float(self._pick(self.sigma, idx) or 0.08)
            kwargs["amps"] = [float(a) for a in (self.amp or [1.0])]
            kwargs["freqs"] = [float(f) for f in (self.freq or [1.0])]
            kwargs["phases"] = [0.0] * len(kwargs["amps"])
        return EntangledBackground(partial(fn, **kwargs))


BackgroundCfg = Annotated[Union[ConstantBackgroundCfg, FunctionBackgroundCfg], Field(discriminator="type")]
