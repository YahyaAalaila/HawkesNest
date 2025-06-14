from pydantic import BaseModel, Field
from typing import Literal, Tuple

class DomainConfig(BaseModel):
    type: Literal["CartoonCity", "OSMArea"]
    osm_place: str | None = None

class BackgroundConfig(BaseModel):
    type: Literal["constant","spatial","separable","entangled","stochastic"]
    params: dict = Field(default_factory=dict)

class KernelConfig(BaseModel):
    type: Literal["expgau","mixture","entangled","network"]
    params: dict = Field(default_factory=dict)

class GraphConfig(BaseModel):
    type: Literal["diagonal","dense","block_modularity"]
    params: dict = Field(default_factory=dict)

class TopologyConfig(BaseModel):
    enabled: bool = False
    params: dict = Field(default_factory=dict)

class GeneratorConfig(BaseModel):
    domain: DomainConfig
    background: BackgroundConfig
    kernel: KernelConfig
    graph: GraphConfig
    topology: TopologyConfig
    alpha: Tuple[float, float, float, float, float] = (0.5,)*5
    n_events: int = 10000
    seed: int | None = None
