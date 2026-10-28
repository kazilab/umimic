"""A stochastic run that hits its event limit must say so to its callers."""

from __future__ import annotations

import numpy as np
import pytest

from umimic.data.synthetic import SyntheticDataGenerator
from umimic.dynamics.gillespie import GillespieSimulator
from umimic.dynamics.rates import RateSet
from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.tau_leaping import TauLeapingSimulator


def _pure_death():
    topology = ModelTopology(
        active_states=[CellType.P], transitions=[],
        division_states=[], death_states=[CellType.P],
    )
    rates = RateSet(birth_base=0.0, death_base={CellType.P: 1.0}, transition_base={})
    return topology, rates


@pytest.fixture
def one_event_limit(monkeypatch):
    original = GillespieSimulator.simulate

    def capped(self, x0, t_max, t_record=None, max_events=10_000_000):
        return original(self, x0, t_max, t_record, max_events=1)

    monkeypatch.setattr(GillespieSimulator, "simulate", capped)


def test_tau_leaping_reports_a_truncated_ssa_tail(one_event_limit):
    topology, rates = _pure_death()
    simulator = TauLeapingSimulator(
        rates, topology, lambda t: 0.0, tau=0.05, rng=np.random.default_rng(4), ssa_threshold=100.0
    )
    result = simulator.simulate(np.array([500.0]), 10.0, np.linspace(0.0, 10.0, 11))
    assert result.metadata["method"] == "tau_leaping(ssa_tail)"
    assert result.metadata["truncated"] is True
    assert result.metadata["ssa_handoff_time"] < result.metadata["t_reached"] < 10.0


def test_tau_leaping_without_a_limit_hit_is_not_truncated():
    topology, rates = _pure_death()
    simulator = TauLeapingSimulator(
        rates, topology, lambda t: 0.0, tau=0.05, rng=np.random.default_rng(4), ssa_threshold=100.0
    )
    result = simulator.simulate(np.array([500.0]), 10.0, np.linspace(0.0, 10.0, 11))
    assert result.metadata["truncated"] is False
    assert result.metadata["t_reached"] is None


@pytest.mark.parametrize("method", ["gillespie", "tau_leaping"])
def test_synthetic_plate_refuses_a_truncated_trajectory(one_event_limit, method):
    topology, rates = _pure_death()
    generator = SyntheticDataGenerator(rates, topology, rng=np.random.default_rng(1))
    with pytest.raises(RuntimeError, match="event limit"):
        generator.generate_invitro_plate(
            initial_cells=np.array([50.0]), concentrations=[0.0],
            n_wells_per_dose=1, t_max=5.0, dt_obs=1.0, method=method,
        )
