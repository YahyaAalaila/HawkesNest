import numpy as np
import pytest

from hawkesnest.metrics.kernel import alpha_ker


def test_alpha_ker_zero():
    offsets = {(0, 0): np.zeros((1, 2))}
    val = alpha_ker(offsets, sigma2=1.0, d_max=1.0)
    assert val == pytest.approx(0)
