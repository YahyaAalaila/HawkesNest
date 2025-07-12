
from functools import partial
from typing import Annotated, Literal, Optional, Sequence, Union
from pydantic import BaseModel, Field, field_validator

from hawkesnest.kernel import (
    ExponentialGaussianKernel,
    MixtureKernel,
    RoughKernel,
    SpaceTimeKernel,
    NetworkKernel,
    ExponentialGeodesicKernel
)
from hawkesnest.config_factory.functions import _FUNCTION_REGISTRY



class NetworkKernelCfg(BaseModel):
    type: Literal["network"] = "network"
    json_path: str

    def build(self):
        return NetworkKernel.from_json(self.json_path)
    
class ExponentialGeodesicKernelCfg(BaseModel):
    type: Literal["exp_geodesic"] = "exp_geodesic"
    temporal_scale:  float = 1.0
    geodesic_scale:  float = 0.05

    def build(self):
        return ExponentialGeodesicKernel(
            temporal_scale=self.temporal_scale,
            geodesic_scale=self.geodesic_scale,
        )


class SeparableKernelCfg(BaseModel):
    type: Literal["separable"] = "separable"
    temporal_decay: float = 1.0
    spatial_sigma: float = 0.1

    def build(self):
        return ExponentialGaussianKernel(
            temporal_scale=self.temporal_decay,
            spatial_scale=self.spatial_sigma,
        )

class MixtureKernelCfg(BaseModel):
    type: Literal["mixture"] = "mixture"
    temporal_decay: float = 1.0
    weights: Sequence[float] = Field(default_factory=lambda: [1.0])
    sigmas: Sequence[float] = Field(default_factory=lambda: [0.1])

    def build(self):
        return MixtureKernel(
            temporal_scale=self.temporal_decay,
            weights=list(self.weights),
            sigmas=list(self.sigmas),
        )


class RoughKernelCfg(BaseModel):
    type: Literal["rough"] = "rough"
    hurst: float = 0.5
    spatial_scale: float = 0.5
    length_scale: float = 1.0

    def build(self):
        return RoughKernel(
            hurst=self.hurst,
            spatial_scale=self.spatial_scale,
            length_scale=self.length_scale,
        )

# TODO: Break this up into smaller classes for each function type
class EntangledKernelCfg(BaseModel):
    type: Literal["entangled"] = "entangled"

    name: str = "cos"  # {cos|sine|polynomial}
    amp: float | None = None
    freq: float | None = None
    coeffs: Sequence[float] | None = None
    coeff2: Sequence[float] | None = None  # for polynomial
    
    aa0: float = 1.0  # constant term 
    ent: float = 1.0  # linear term in t·x
    quad: float = 0.0  # quadratic term in t²·x²
    lin: float = 0.0  # linear term in t + x

    a0: float | None = None  # constant term for exp_sin
    a1: float | None = None  # linear term for exp_sin
    omega: float | None = None  # frequency for exp_sin 
    sigma: float | None = None  # spatial scale for gabor_travel, gauss_shear, vortex
    start: Union[Sequence[float]   , tuple[float, float]] = (0.5, 0.5)  # start position for gabor_travel, vortex
    centers: Sequence[Sequence[float]] | None = None  # centers for osc_cluster
    amp: Optional[float] | None = None  # amplitudes for osc_cluster
    freq: Optional[float] | None = None  # frequencies for osc_cluster
    phases: Optional[float] | None = None  # phases for osc_cluster
    base: float | None = None  # base value for poly2
    c0: float | None = None  # constant term for poly2
    c_t: float | None = None  # linear term in t for poly2
    c_x: float | None = None  # linear term in x for poly2
    c_tx: float | None = None  # cross term t·x for poly2
    c_tt: float | None = None  # quadratic term in t² for poly2
    c_xx: float | None = None  # quadratic term in x² for poly2


    @field_validator("name")
    @classmethod
    def _check_known(cls, v):
        if v not in _FUNCTION_REGISTRY:
            raise ValueError(f"Unknown entangled kernel '{v}'.")
        return v

    def build(self):
        fn = _FUNCTION_REGISTRY[self.name]
        kwargs: dict[str, object] = {}
        if self.name in {"cos", "sine"}:
            kwargs["amp"] = self.amp or 1.0
            kwargs["freq"] = self.freq or 1.0
        elif self.name == "polynomial":
            kwargs["coeffs"] = list(self.coeffs or (1, 0, 0, 0, 0, 0))
        elif self.name == "poly_entangled_kernel":
            kwargs["aa0"] = self.aa0
            kwargs["ent"] = self.ent
            kwargs["quad"] = self.quad
            kwargs["lin"] = self.lin
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
            print(f"[DEBUG] poly2: base={base} c0={c0}, c_t={c_t}, c_x={c_x}, c_tx={c_tx}, c_tt={c_tt}, c_xx={c_xx}")
            kwargs.update(dict(base = base, a0=c0, a_t=c_t, a_x=c_x, a_tx=c_tx, a_tt=c_tt, a_xx=c_xx))
        elif self.name == "poly_entangled":
            # poly_entangled: a0, ent, quad, lin
            aa0 = self.aa0 or 1.0
            ent = self.ent or 1.0
            quad = self.quad or 0.0
            lin = self.lin or 0.0
            print(f"[DEBUG] poly_entangled: aa0={aa0}, ent={ent}, quad={quad}, lin={lin}")
            kwargs.update(dict(aa0=aa0, ent=ent, quad=quad, lin=lin))
        elif self.name == "exp_sin":
            kwargs["a0"]    = self.a0 or 0.0
            kwargs["a1"]    = self.a1 or 1.0
            kwargs["omega"] = self.omega or 1.0
        elif self.name == "gabor_travel":
            kwargs["a0"]    = self.a0 or 0.0
            kwargs["amp"]   = self.amp or 3.0
            kwargs["fx"]    = self.freq or 3.0
            kwargs["fy"]    = self.freq or 3.0
            kwargs["ft"]    = self.freq or 4.0
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
        return SpaceTimeKernel(partial(fn, **kwargs))



KernelCfg = Annotated[
    Union[
        SeparableKernelCfg,
        MixtureKernelCfg,
        RoughKernelCfg,
        EntangledKernelCfg,
        NetworkKernelCfg,
        ExponentialGeodesicKernelCfg,
    ],
    Field(discriminator="type"),
]
