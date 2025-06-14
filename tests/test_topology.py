import pytest
import numpy as np
from hawkesnest.metrics.topology import alpha_topo
from hawkesnest.domain import SpatialDomain

class DummyDomain(SpatialDomain):
    def sample_point(self):
        return (0,0)
    def geodesic(self, u, v):
        return 1.0  # constant

def test_topology_constant():
    dom = DummyDomain()
    at = alpha_topo(dom, n_pairs=10, D_max=2.0)
    assert at == pytest.approx((1.0-1)/(2.0-1))