
from functools import partial
from typing import Annotated, Literal, Optional, Sequence, Union
import numpy as np
from pydantic import BaseModel, Field, field_validator

from hawkesnest.kernel import (
    ExponentialGaussianKernel,
    MixtureKernel,
    RoughKernel,
    NetworkKernel,
    ExponentialGeodesicKernel,
    TravelingWaveKernel,
    TwoScaleKernel,
)
from hawkesnest.config_factory.functions import _FUNCTION_REGISTRY
from hawkesnest.kernel.entangled import EntangledExponentialGaussianKernel

# config_factory/kernel_config.py



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




class EntangledExpGaussianKernelCfg(BaseModel):
    type: Literal["entangled_exp_gaussian"] = "entangled_exp_gaussian"
    temporal_scale: float = 1
    spatial_scale: float = 0.1
    theta_ent: float
    r_max: float = 1
    tau_max: float = 5.0
    n_r: int = 64
    n_tau: int = 64
    ent_option: str = "s"  # "rt", "A", "early_tau", "sin_softplus", "multi_origin"

    def build(self):
        print("[EntangledExpGaussianKernelCfg] building with ent_option =", self.ent_option)
        base = ExponentialGaussianKernel(
            temporal_scale=self.temporal_scale,
            spatial_scale=self.spatial_scale,
        )

        # def shape_fn(R, Tau):
        #     return 1.0 + R * Tau + 1.2 * (R * Tau) ** 2
        
        def shape_fn(R, Tau):
            opt = self.ent_option

            if opt == "rt":
                return 1.0 + R * Tau

            if opt == "A":  # your diagonal propagation idea (localised)
                r_max = float(self.r_max)
                tau_max = float(self.tau_max)
                r_diag = 0.2 * r_max + 0.6 * (Tau / tau_max) * r_max
                sigma_r = 0.12 * r_max
                S = np.exp(-0.5 * ((R - r_diag) / sigma_r) ** 2)
                sigma_tau = 0.22 * tau_max
                center_tau = 0.6 * tau_max
                bump = np.exp(-0.5 * ((Tau - center_tau) / sigma_tau) ** 2)
                return 1.0 + 3.0 * S * bump

            if opt == "early_tau":  # “push deformation to τ≈0”
                #print("[EntangledExpGaussianKernelCfg] using early_tau shape_fn")
                tau_max = float(self.tau_max)
                return 1.0 + 3.0 * np.exp(-(Tau / (0.08 * tau_max)) ** 2) * np.exp(-(R / (0.25 * self.r_max)) ** 2)

            if opt == "sin_softplus":  # your “harder” one
                amp, k = 0.7, 3.0
                z = (R * Tau) + amp * np.sin(k * R * Tau)
                return 1.0 + np.log1p(np.exp(z))
            if opt == "multi_origin":
                #print("[EntangledExpGaussianKernelCfg] using multi_origin shape_fn")
                r_max = float(self.r_max)
                tau_max = float(self.tau_max)

                # Mode 1: very early, very local
                r1 = 0.08 * r_max
                t1 = 0.08 * tau_max

                # Mode 2: slightly delayed but still near origin
                r2 = 0.18 * r_max
                t2 = 0.20 * tau_max

                sigma_r1 = 0.03 * r_max
                sigma_t1 = 0.03 * tau_max
                sigma_r2 = 0.05 * r_max
                sigma_t2 = 0.05 * tau_max

                bump1 = np.exp(
                    -0.5 * ((R - r1) / sigma_r1) ** 2
                    -0.5 * ((Tau - t1) / sigma_t1) ** 2
                )

                bump2 = np.exp(
                    -0.5 * ((R - r2) / sigma_r2) ** 2
                    -0.5 * ((Tau - t2) / sigma_t2) ** 2
                )

                shape = bump1 + 0.8 * bump2

                return 1.0 + 4.0 * shape

            raise ValueError(f"Unknown ent_option={opt}")
        return EntangledExponentialGaussianKernel(
            base_kernel=base,
            shape_fn=shape_fn,
            theta_ent=self.theta_ent,
            r_max=self.r_max,
            tau_max=self.tau_max,
            n_r=self.n_r,
            n_tau=self.n_tau,
            renormalize=True,
        )
        



class TravelingWaveKernelCfg(BaseModel):
    """
    Traveling-wave triggering kernel config.

    φ(s_vec, τ) = exp(-τ/β) × exp(-‖s_vec − v·ê·τ‖² / (2σ²))

    Parameters
    ----------
    v : wave speed (0 = separable baseline)
    theta_wave : propagation direction in radians (0 = +x axis)
    sigma : spatial bandwidth
    temporal_scale : temporal decay β
    """
    type: Literal["traveling_wave"] = "traveling_wave"
    v: float = 0.3
    theta_wave: float = 0.0
    sigma: float = 0.1
    temporal_scale: float = 1.0

    def build(self):
        return TravelingWaveKernel(
            v=self.v,
            theta_wave=self.theta_wave,
            sigma=self.sigma,
            temporal_scale=self.temporal_scale,
        )


class TwoScaleKernelCfg(BaseModel):
    """
    Two-component temporal triggering kernel config.

    φ(r, τ) = [α_fast·exp(-τ/β_fast) + (1-α_fast)·exp(-τ/β_slow)] × exp(-r²/(2σ²))

    Parameters
    ----------
    alpha_fast : weight of the fast component (0 < α_fast < 1)
    beta_fast  : temporal decay of fast component
    beta_slow  : temporal decay of slow component (β_slow > β_fast recommended)
    sigma      : spatial bandwidth
    """
    type: Literal["two_scale"] = "two_scale"
    alpha_fast: float = 0.5
    beta_fast: float = 0.2
    beta_slow: float = 2.0
    sigma: float = 0.1

    def build(self):
        return TwoScaleKernel(
            alpha_fast=self.alpha_fast,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
            sigma=self.sigma,
        )


KernelCfg = Annotated[
    Union[
        SeparableKernelCfg,
        MixtureKernelCfg,
        RoughKernelCfg,
        EntangledExpGaussianKernelCfg,
        NetworkKernelCfg,
        ExponentialGeodesicKernelCfg,
        TravelingWaveKernelCfg,
        TwoScaleKernelCfg,
    ],
    Field(discriminator="type"),
]
