import pytest

from hawkesnest.simulator.hawkes import HawkesSimulator


def test_simulator_runs():
    sim = HawkesSimulator()
    events, labels = sim.simulate(n=10, seed=42)
    assert len(events) == 10
    assert len(labels) == 10
