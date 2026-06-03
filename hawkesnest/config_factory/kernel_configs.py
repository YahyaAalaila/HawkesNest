from functools import partial
from typing import Annotated, Literal, Sequence, Union

from pydantic import BaseModel, Field, field_validator

from hawkesnest.config_factory.functions import _FUNCTION_REGISTRY
from hawkesnest.kernel import ExponentialGaussianKernel, ExponentialGeodesicKernel, MixtureKernel, NetworkKernel, RoughKernel, SpaceTimeKernel, TravelingWaveKernel


class SeparableKernelCfg(BaseModel):
    type: Literal["separable"] = "separable"
    temporal_decay: float = 1.0
    spatial_sigma: float = 1.0

    def build(self):
        return ExponentialGaussianKernel(temporal_scale=self.temporal_decay, spatial_scale=self.spatial_sigma)


class MixtureKernelCfg(BaseModel):
    type: Literal["mixture"] = "mixture"
    weights: list[float]
    spatial_sigmas: list[float]
    temporal_decays: list[float]

    def build(self):
        comps = [ExponentialGaussianKernel(s, t) for s, t in zip(self.spatial_sigmas, self.temporal_decays)]
        return MixtureKernel(comps, self.weights)


class NetworkKernelCfg(BaseModel):
    type: Literal["network"] = "network"
    edge_weight: float = 1.0
    temporal_decay: float = 1.0

    def build(self):
        return NetworkKernel(edge_weight=self.edge_weight, temporal_decay=self.temporal_decay)


class ExponentialGeodesicKernelCfg(BaseModel):
    type: Literal["exp_geodesic"] = "exp_geodesic"
    spatial_scale: float = 1.0
    temporal_scale: float = 1.0

    def build(self):
        return ExponentialGeodesicKernel(spatial_scale=self.spatial_scale, temporal_scale=self.temporal_scale)


class RoughKernelCfg(BaseModel):
    type: Literal["rough"] = "rough"
    hurst: float = 0.7
    length_scale: float = 1.0

    def build(self):
        return RoughKernel(hurst=self.hurst, length_scale=self.length_scale)


class EntangledKernelCfg(BaseModel):
    type: Literal["entangled"] = "entangled"
    name: str = "cos"
    amp: float | None = None
    freq: float | None = None
    coeffs: Sequence[float] | None = None
    coeff2: Sequence[float] | None = None
    aa0: float = 1.0
    ent: float = 1.0
    quad: float = 0.0
    lin: float = 0.0
    a0: float | None = None
    a1: float | None = None
    omega: float | None = None
    sigma: float | None = None
    start: Union[Sequence[float], tuple[float, float]] = (0.5, 0.5)
    centers: Sequence[Sequence[float]] | None = None
    base: float | None = None
    c0: float | None = None
    c_t: float | None = None
    c_x: float | None = None
    c_tx: float | None = None
    c_tt: float | None = None
    c_xx: float | None = None

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
            kwargs["coeffs"] = list(self.coeffs or [2.0, 0.0, 0.0])
        elif self.name == "poly_entangled":
            kwargs.update(dict(aa0=self.aa0, ent=self.ent, quad=self.quad, lin=self.lin))
        return SpaceTimeKernel(partial(fn, **kwargs))


class TravelingWaveKernelCfg(BaseModel):
    type: Literal["traveling_wave"] = "traveling_wave"
    v: float = 0.3
    theta_wave: float = 0.0
    sigma: float = 0.1
    temporal_scale: float = 1.0

    def build(self):
        return TravelingWaveKernel(v=self.v, theta_wave=self.theta_wave, sigma=self.sigma, temporal_scale=self.temporal_scale)


KernelCfg = Annotated[Union[SeparableKernelCfg, MixtureKernelCfg, RoughKernelCfg, EntangledKernelCfg, NetworkKernelCfg, ExponentialGeodesicKernelCfg, TravelingWaveKernelCfg], Field(discriminator="type")]
