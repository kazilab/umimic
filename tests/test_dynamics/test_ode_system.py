"""Tests for the deterministic ODE system."""

import numpy as np

from umimic.dynamics.ode_system import CellDynamicsODE, build_ode_system
from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.rates import EmaxHill, RateSet


class TestCellDynamicsODE:
    def test_pure_exponential_growth(self):
        """With no death or transitions, P should grow exponentially."""
        topology = ModelTopology(
            active_states=[CellType.P],
            transitions=[],
            division_states=[CellType.P],
            death_states=[],
        )
        rate_set = RateSet(
            birth_base=0.04,
            death_base={},
            transition_base={},
        )
        ode = CellDynamicsODE(rate_set, topology, lambda t: 0.0)
        y0 = np.array([100.0])
        t_eval = np.linspace(0, 72, 100)
        result = ode.solve(y0, (0, 72), t_eval)

        # Analytical: P(t) = P0 * exp(b*t)
        expected = 100.0 * np.exp(0.04 * t_eval)
        np.testing.assert_allclose(
            result.populations["P"], expected, rtol=1e-4
        )

    def test_pure_death(self):
        """With no birth, P should decay exponentially."""
        topology = ModelTopology(
            active_states=[CellType.P],
            transitions=[],
            division_states=[],
            death_states=[CellType.P],
        )
        rate_set = RateSet(
            birth_base=0.0,
            death_base={CellType.P: 0.05},
            transition_base={},
        )
        ode = CellDynamicsODE(rate_set, topology, lambda t: 0.0)
        y0 = np.array([1000.0])
        t_eval = np.linspace(0, 72, 100)
        result = ode.solve(y0, (0, 72), t_eval)

        expected = 1000.0 * np.exp(-0.05 * t_eval)
        np.testing.assert_allclose(
            result.populations["P"], expected, rtol=1e-4
        )

    def test_net_growth_rate(self, two_state_topology, simple_rates):
        """Population growth rate matches b - d for simple case."""
        ode = CellDynamicsODE(simple_rates, two_state_topology, lambda t: 0.0)
        y0 = np.array([100.0, 0.0])
        result = ode.solve(y0, (0, 1), np.array([0, 1]))

        # After 1 hour, check direction is positive (b > d)
        assert result.populations["P"][-1] > 100.0

    def test_populations_nonnegative(self, two_state_topology, cytotoxic_rates):
        """Populations should never go negative."""
        ode = CellDynamicsODE(cytotoxic_rates, two_state_topology, lambda t: 100.0)
        y0 = np.array([50.0, 10.0])
        t_eval = np.linspace(0, 200, 200)
        result = ode.solve(y0, (0, 200), t_eval)

        for name, pop in result.populations.items():
            assert np.all(pop >= 0), f"Negative population in {name}"

    def test_dose_response(self, two_state_topology, cytotoxic_rates):
        """Higher concentration should reduce viable cells."""
        ode = CellDynamicsODE(cytotoxic_rates, two_state_topology, lambda t: 0.0)
        y0 = np.array([100.0, 0.0])
        results = ode.solve_dose_response(
            y0, (0, 72), [0.0, 1.0, 10.0], np.linspace(0, 72, 50)
        )

        final_0 = results[0.0].viable[-1]
        final_1 = results[1.0].viable[-1]
        final_10 = results[10.0].viable[-1]
        assert final_0 > final_1 > final_10

    def test_build_ode_system_constant(self, two_state_topology, simple_rates):
        """build_ode_system convenience function works."""
        ode = build_ode_system(simple_rates, two_state_topology,
                               constant_concentration=0.0)
        result = ode.solve(np.array([100.0, 0.0]), (0, 24))
        assert len(result.times) > 0


class TestThreeStateODE:
    def test_apoptotic_accumulation(self, three_state_topology, cytotoxic_rates):
        """Apoptotic cells should accumulate under drug treatment."""
        ode = CellDynamicsODE(cytotoxic_rates, three_state_topology, lambda t: 10.0)
        y0 = np.array([100.0, 0.0, 0.0])
        t_eval = np.linspace(0, 72, 50)
        result = ode.solve(y0, (0, 72), t_eval)

        # A should increase (cells dying)
        assert result.populations["A"][-1] > 0


class TestLinearFastPath:
    """The matrix-exponential path must be exact where it applies, and must
    decline to apply anywhere the system is not linear and time-invariant."""

    @staticmethod
    def _rates():
        return RateSet(
            birth_base=0.05,
            death_base={CellType.P: 0.012, CellType.Q: 0.006},
            death_modulation={
                CellType.P: EmaxHill(emax=0.4, ec50=2.0, hill=1.5)
            },
            transition_base={
                (CellType.P, CellType.Q): 0.005,
                (CellType.Q, CellType.P): 0.003,
            },
            clearance_rate=0.1,
        )

    def test_matches_integrator_across_topologies_and_doses(self):
        """Same answer as numerical integration, to integrator tolerance."""
        topologies = [
            ModelTopology.two_state(),
            ModelTopology.three_state(),
            ModelTopology.four_state(),
        ]
        t_eval = np.linspace(0, 88, 23)
        for topology in topologies:
            for conc in (0.0, 1.0, 25.0):
                ode = CellDynamicsODE(self._rates(), topology, lambda t, c=conc: c)
                y0 = np.zeros(len(topology.active_states))
                y0[0] = 500.0
                fast = ode.solve(y0, (0, 88), t_eval)
                slow = ode.solve(y0, (0, 88), t_eval, linear_fast_path=False)
                assert fast.metadata["method"] == "expm"
                for state in topology.active_states:
                    np.testing.assert_allclose(
                        fast.populations[state.name],
                        slow.populations[state.name],
                        rtol=1e-5,
                        atol=1e-6,
                    )

    def test_exact_for_known_closed_form(self):
        """Exponential growth is reproduced to machine precision, not to
        integrator tolerance -- the point of the fast path."""
        topology = ModelTopology(
            active_states=[CellType.P],
            transitions=[],
            division_states=[CellType.P],
            death_states=[],
        )
        ode = CellDynamicsODE(
            RateSet(birth_base=0.04, death_base={}, transition_base={}),
            topology,
            lambda t: 0.0,
        )
        t_eval = np.linspace(0, 72, 25)
        result = ode.solve(np.array([100.0]), (0, 72), t_eval)
        assert result.metadata["method"] == "expm"
        np.testing.assert_allclose(
            result.populations["P"], 100.0 * np.exp(0.04 * t_eval), rtol=1e-12
        )

    def test_non_uniform_time_grid(self):
        """Recorded times need not be evenly spaced."""
        topology = ModelTopology.three_state()
        t_eval = np.array([0.0, 1.0, 3.0, 7.0, 19.0, 48.0, 72.0])
        ode = CellDynamicsODE(self._rates(), topology, lambda t: 2.0)
        y0 = np.array([500.0, 0.0, 0.0])
        fast = ode.solve(y0, (0, 72), t_eval)
        slow = ode.solve(y0, (0, 72), t_eval, linear_fast_path=False)
        assert fast.metadata["method"] == "expm"
        for state in topology.active_states:
            np.testing.assert_allclose(
                fast.populations[state.name],
                slow.populations[state.name],
                rtol=1e-5,
                atol=1e-6,
            )

    def test_declines_time_varying_exposure(self):
        """A decaying PK exposure makes the generator time-varying."""
        topology = ModelTopology.three_state()
        ode = CellDynamicsODE(
            self._rates(), topology, lambda t: 5.0 * np.exp(-0.05 * t)
        )
        result = ode.solve(np.array([500.0, 0.0, 0.0]), (0, 72),
                           np.linspace(0, 72, 13))
        assert result.metadata["method"] != "expm"

    def test_declines_density_dependence(self):
        """Density-dependent growth is non-linear in the state."""
        topology = ModelTopology.three_state()
        topology.density_dependent = True
        topology.carrying_capacity = 1e4
        ode = CellDynamicsODE(self._rates(), topology, lambda t: 1.0)
        result = ode.solve(np.array([500.0, 0.0, 0.0]), (0, 72),
                           np.linspace(0, 72, 13))
        assert result.metadata["method"] != "expm"

    def test_declines_time_varying_rate_multiplier(self):
        topology = ModelTopology.three_state()
        ode = CellDynamicsODE(
            self._rates(), topology, lambda t: 1.0,
            rate_multiplier_fn=lambda t, key: 1.0 + 0.5 * np.sin(t),
        )
        result = ode.solve(np.array([500.0, 0.0, 0.0]), (0, 72),
                           np.linspace(0, 72, 13))
        assert result.metadata["method"] != "expm"
