"""Tests for the linear noise approximation (moment ODEs)."""

import numpy as np
import pytest

from umimic.dynamics.moment_equations import MomentODE
from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.rates import RateSet


class TestMomentODE:
    def test_mean_matches_ode(self, two_state_topology, simple_rates):
        """Moment ODE mean should match deterministic ODE."""
        from umimic.dynamics.ode_system import CellDynamicsODE

        exposure_fn = lambda t: 0.0
        mu0 = np.array([200.0, 0.0])

        # Moment ODE
        moment = MomentODE(simple_rates, two_state_topology, exposure_fn)
        t_eval = np.linspace(0, 48, 25)
        times, means, covs = moment.solve(mu0, t_span=(0, 48), t_eval=t_eval)

        # Deterministic ODE
        ode = CellDynamicsODE(simple_rates, two_state_topology, exposure_fn)
        result = ode.solve(mu0, (0, 48), t_eval)

        # Means should match
        np.testing.assert_allclose(
            means[:, 0], result.populations["P"], rtol=0.05, atol=5
        )

    def test_covariance_positive_semidefinite(self, two_state_topology, simple_rates):
        """Covariance matrices should be PSD at all times."""
        moment = MomentODE(simple_rates, two_state_topology, lambda t: 0.0)
        mu0 = np.array([100.0, 0.0])
        t_eval = np.linspace(0, 48, 25)
        times, means, covs = moment.solve(mu0, t_span=(0, 48), t_eval=t_eval)

        for k in range(len(times)):
            eigvals = np.linalg.eigvalsh(covs[k])
            assert np.all(eigvals >= -1e-6), f"Non-PSD covariance at t={times[k]}"

    def test_variance_grows_with_population(self, two_state_topology, simple_rates):
        """Variance should increase as population grows."""
        moment = MomentODE(simple_rates, two_state_topology, lambda t: 0.0)
        mu0 = np.array([100.0, 0.0])
        t_eval = np.linspace(0, 48, 25)
        times, means, covs = moment.solve(mu0, t_span=(0, 48), t_eval=t_eval)

        var_P = [covs[k][0, 0] for k in range(len(times))]
        # Variance should generally increase over time for growing population
        assert var_P[-1] > var_P[0]

    def test_rate_matrix_shape(self, two_state_topology, simple_rates):
        """Rate matrix A should be n_states x n_states."""
        moment = MomentODE(simple_rates, two_state_topology, lambda t: 0.0)
        A = moment.rate_matrix(0.0, np.array([100.0, 10.0]))
        assert A.shape == (2, 2)

    def test_diffusion_matrix_shape(self, two_state_topology, simple_rates):
        """Diffusion matrix D should be n_states x n_states."""
        moment = MomentODE(simple_rates, two_state_topology, lambda t: 0.0)
        D = moment.diffusion_matrix(0.0, np.array([100.0, 10.0]))
        assert D.shape == (2, 2)
        # D should be PSD (sum of outer products)
        eigvals = np.linalg.eigvalsh(D)
        assert np.all(eigvals >= -1e-10)
