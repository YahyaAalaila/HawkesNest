

from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field

from hawkesnest.domain import GridDomain, RectangleDomain, NetworkDomain


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
    
DomainConfig = Annotated[
    Union[
        GridDomainConfig,
        NetworkDomainCfg,
        RectangleDomainConfig,
    ],
    Field(discriminator="type"),
]
