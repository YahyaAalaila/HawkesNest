import pytest
import numpy as np
from hawkesnest.background import ConstantBackground, SpatialBackground, SeparableBackground, EntangledBackground, TemporalProfile, SinusoidProfile

@pytest.fixture
def point():
    return (0.5, 0.5)

@pytest.fixture
def time():
    return 12.0

def test_constant_background(point, time):
    bg = ConstantBackground(rate=3.14)
    assert bg(*point, time) == pytest.approx(3.14)

def test_spatial_background(point, time):
    centers = [(0,0),(1,1)]
    weights = [1.0, 0.5]
    vars = [0.1, 0.2]
    sb = SpatialBackground(centers=centers, variances=vars, weights=weights)
    val = sb(point, time)
    assert val >= 0

def test_separable_background(point, time):
    sb = SpatialBackground(centers=[point], variances=[0.1], weights=[1.0])
    tp = SinusoidProfile(period=24, amplitude=1.0)
    eb = SeparableBackground(spatial=sb, temporal=tp)
    assert eb(point, time) == pytest.approx(sb(point, time) * tp(time))

def test_entangled_background(point, time):
    eb = EntangledBackground(spline_surface=lambda s,t: (s[0]+s[1])*t)
    assert eb(point, time) == pytest.approx((point[0]+point[1]) * time)
