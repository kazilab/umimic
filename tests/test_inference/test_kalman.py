"""Tests for Extended Kalman Filter."""

import numpy as np
import pytest

from umimic.inference.kalman import ExtendedKalmanFilter
from umimic.dynamics.moment_equations import MomentODE
from umimic.dynamics.states import ModelTopology
from umimic.dynamics.rates import RateSet
from umimic.observations.cell_counts import CellCountObservation
from umimic.data.schemas import TimeSeriesData


class TestExtendedKalmanFilter:
    def test_filter_runs(self, two_state_topology, simple_rates, sample_data):
        """EKF should run without errors."""
        moment_ode = MomentODE(simple_rates, two_state_topology, lambda t: 0.0)
        obs_model = CellCountObservation(overdispersion=10.0)
        ekf = ExtendedKalmanFilter(moment_ode, obs_model)

        mu0 = np.array([100.0, 0.0])
        result = ekf.filter(sample_data, mu0)

        assert result.filtered_means.shape == (len(sample_data.times), 2)
        assert result.filtered_covs.shape == (len(sample_data.times), 2, 2)

    def test_marginal_ll_finite(self, two_state_topology, simple_rates, sample_data):
        """Marginal log-likelihood should be finite."""
        moment_ode = MomentODE(simple_rates, two_state_topology, lambda t: 0.0)
        obs_model = CellCountObservation(overdispersion=10.0)
        ekf = ExtendedKalmanFilter(moment_ode, obs_model)

        mu0 = np.array([100.0, 0.0])
        ll = ekf.marginal_log_likelihood(sample_data, mu0)
        assert np.isfinite(ll)

    def test_filtered_means_positive(self, two_state_topology, simple_rates, sample_data):
        """Filtered means should be non-negative."""
        moment_ode = MomentODE(simple_rates, two_state_topology, lambda t: 0.0)
        obs_model = CellCountObservation(overdispersion=10.0)
        ekf = ExtendedKalmanFilter(moment_ode, obs_model)

        mu0 = np.array([100.0, 0.0])
        result = ekf.filter(sample_data, mu0)
        assert np.all(result.filtered_means >= 0)
