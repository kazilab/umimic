"""Tests for the Gillespie stochastic simulator."""

import numpy as np
import pytest

from umimic.dynamics.gillespie import GillespieSimulator, build_reactions
from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.rates import RateSet


class TestGillespieSimulator:
    def test_single_trajectory(self, two_state_topology, simple_rates, rng):
        """Single trajectory runs without errors."""
        sim = GillespieSimulator(
            simple_rates, two_state_topology, lambda t: 0.0, rng
        )
        result = sim.simulate(
            np.array([100.0, 0.0]), t_max=24, t_record=np.linspace(0, 24, 25)
        )
        assert len(result.times) == 25
        assert "P" in result.populations
        assert "Q" in result.populations

    def test_populations_nonnegative(self, two_state_topology, simple_rates, rng):
        """Populations should never go negative."""
        sim = GillespieSimulator(
            simple_rates, two_state_topology, lambda t: 0.0, rng
        )
        result = sim.simulate(
            np.array([50.0, 10.0]), t_max=72, t_record=np.linspace(0, 72, 100)
        )
        for name, pop in result.populations.items():
            assert np.all(pop >= 0), f"Negative population in {name}"

    def test_ensemble_mean_matches_ode(self, two_state_topology, rng):
        """Ensemble mean should approximate ODE solution for large populations."""
        rate_set = RateSet(
            birth_base=0.03,
            death_base={CellType.P: 0.01, CellType.Q: 0.005},
            transition_base={
                (CellType.P, CellType.Q): 0.002,
                (CellType.Q, CellType.P): 0.001,
            },
        )
        sim = GillespieSimulator(
            rate_set, two_state_topology, lambda t: 0.0, rng
        )
        t_record = np.linspace(0, 48, 25)
        ensemble = sim.simulate_ensemble(
            np.array([500.0, 0.0]), t_max=48, t_record=t_record, n_trajectories=200
        )

        # Compare to ODE
        from umimic.dynamics.ode_system import CellDynamicsODE
        ode = CellDynamicsODE(rate_set, two_state_topology, lambda t: 0.0)
        ode_result = ode.solve(np.array([500.0, 0.0]), (0, 48), t_record)

        ens_mean = ensemble.mean()
        # Allow ~10% relative tolerance for stochastic vs deterministic
        np.testing.assert_allclose(
            ens_mean["P"], ode_result.populations["P"], rtol=0.15, atol=20
        )

    def test_extinction_possible(self, two_state_topology, rng):
        """Small populations under strong drug should go extinct sometimes."""
        rate_set = RateSet(
            birth_base=0.01,
            death_base={CellType.P: 0.1, CellType.Q: 0.05},
        )
        sim = GillespieSimulator(
            rate_set, two_state_topology, lambda t: 0.0, rng
        )
        t_record = np.linspace(0, 100, 50)
        ensemble = sim.simulate_ensemble(
            np.array([10.0, 0.0]), t_max=100, t_record=t_record, n_trajectories=50
        )

        # At least some trajectories should go to zero
        final_P = [traj["P"][-1] for traj in ensemble.trajectories]
        assert any(p == 0 for p in final_P), "Expected some extinction with d >> b"

    def test_ensemble_statistics(self, two_state_topology, simple_rates, rng):
        """Ensemble result provides valid mean, std, and variance."""
        sim = GillespieSimulator(
            simple_rates, two_state_topology, lambda t: 0.0, rng
        )
        t_record = np.linspace(0, 24, 13)
        ensemble = sim.simulate_ensemble(
            np.array([100.0, 0.0]), t_max=24, t_record=t_record, n_trajectories=50
        )

        assert ensemble.n_trajectories == 50
        mean = ensemble.mean()
        std = ensemble.std()
        var = ensemble.variance()

        assert "P" in mean
        assert np.all(std["P"] >= 0)
        np.testing.assert_allclose(var["P"], std["P"] ** 2)


class TestBuildReactions:
    def test_reaction_count_two_state(self, two_state_topology, simple_rates):
        """Two-state model should have: 1 birth + 2 death + 2 transition = 5 reactions."""
        reactions = build_reactions(simple_rates, two_state_topology)
        names = [r.name for r in reactions]
        assert "birth_P" in names
        assert "death_P" in names
        assert "death_Q" in names
        assert "trans_P_Q" in names
        assert "trans_Q_P" in names

    def test_reaction_count_three_state(self, three_state_topology, cytotoxic_rates):
        """Three-state model adds clearance reaction."""
        reactions = build_reactions(cytotoxic_rates, three_state_topology)
        names = [r.name for r in reactions]
        assert "clearance_A" in names
