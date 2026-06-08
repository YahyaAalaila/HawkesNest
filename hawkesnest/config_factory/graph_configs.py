from typing import Annotated, Literal, Union

import numpy as np
from pydantic import BaseModel, Field


class BranchingMatrixCfg(BaseModel):
    theta_graph: float
    M: int = 3

    def build(self):
        from hawkesnest.graph.branching import build_branching_matrix
        rng = np.random.default_rng(123)
        return build_branching_matrix(theta_graph=self.theta_graph, M=self.M, rng=rng, mode="dcsbm")


GraphCfg = Annotated[
    Union[
        BranchingMatrixCfg,
    ],
    Field(discriminator="type"),
]
