

from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field

from hawkesnest.domain import GridDomain, RectangleDomain, NetworkDomain, BarrierDomain


class GridDomainConfig(BaseModel):
    """Axis-aligned rectangular spatial domain."""
    
    type: Literal["manh"] = "manh"
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0

    def build(self) -> GridDomain:
        return GridDomain(self.x_min, self.x_max, self.y_min, self.y_max)

    @property
    def area(self) -> float:  # convenience
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)
    
# in hawkesnest/config.py  (DomainConfig registry part)
class NetworkDomainCfg(BaseModel):
    type: Literal["network"] = "network"
    pickle_path: str         # path to the pickled NetworkX graph

    def build(self):
        import pickle, pathlib
        G = pickle.load(pathlib.Path(self.pickle_path).open("rb"))
        return NetworkDomain(G)
    
class RectangleDomainConfig(BaseModel):
    """Axis-aligned rectangular spatial domain."""
    
    type: Literal["rectangle"] = "rectangle"
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0

    def build(self) -> RectangleDomain:
        return RectangleDomain(self.x_min, self.x_max, self.y_min, self.y_max)


class TopologyRGGDomainCfg(BaseModel):
    type: Literal["topology_rgg"] = "topology_rgg"
    theta_topo: float
    seed: int
    n_nodes: int = 400
    r_min: float = 0.05
    r_max: float = 1.0

    def build(self):
        from hawkesnest.domain.topology_utils import build_topology_domain

        return build_topology_domain(
            theta_topo=self.theta_topo,
            seed=self.seed,
            n_nodes=self.n_nodes,
            r_min=self.r_min,
            r_max=self.r_max,
        )
    
class BarrierDomainCfg(BaseModel):
    """Rectangle with rectangular exclusion zones (barriers)."""
    type: Literal["barrier"] = "barrier"
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    # List of [bx0, bx1, by0, by1] barriers
    barriers: list[list[float]] = []

    def build(self) -> BarrierDomain:
        return BarrierDomain(
            self.x_min, self.x_max, self.y_min, self.y_max,
            [tuple(b) for b in self.barriers],
        )


DomainConfig = Annotated[
    Union[
        GridDomainConfig,
        NetworkDomainCfg,
        RectangleDomainConfig,
        TopologyRGGDomainCfg,
        BarrierDomainCfg,
    ],
    Field(discriminator="type"),
]
