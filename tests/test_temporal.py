import pytest
import math
from hawkesnest.background.temporal import SinusoidProfile, StepProfile, SplineProfile

def test_sinusoid_profile():
    sp = SinusoidProfile(period=24, amplitude=2.0)
    assert sp(0) == pytest.approx(1.0)
    assert sp(6) == pytest.approx(1.0 + 2.0 * math.sin(2*math.pi*6/24))

def test_step_profile():
    steps = [(0,10,1.0), (10,20,0.5)]
    step = StepProfile(steps)
    assert step(5) == 1.0
    assert step(15) == 0.5