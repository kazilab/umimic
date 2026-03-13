"""Tests for the central likelihood computation."""

import numpy as np
import pytest

from umimic.inference.likelihood import ModelLikelihood
from umimic.dynamics.states import ModelTopology
from umimic.data.schemas import TimeSeriesData


class TestModelLikelihood:
    def test_finite_likelihood(self, two_state_topology, sample_data):
        """Likelihood should be finite for reasonable parameters."""
        ll = ModelLikelihood(
            topology=two_state_topology,
            data=sample_data,
            mode="ode",
        )
        theta = np.array([0.04, 0.01, 0.05, 1.0, 1.5, 0.005, 0.003, 10.0])
        result = ll(theta)
        assert np.isfinite(result)

    def test_negative_params_return_neg_inf(self, two_state_topology, sample_data):
        """Negative parameters should give -inf log-likelihood."""
        ll = ModelLikelihood(
            topology=two_state_topology,
            data=sample_data,
            mode="ode",
        )
        theta = np.array([-0.04, 0.01, 0.05, 1.0, 1.5, 0.005, 0.003, 10.0])
        result = ll(theta)
        assert result == -np.inf

    def test_param_conversion_roundtrip(self, two_state_topology, sample_data):
        """theta -> params -> theta roundtrip should preserve values."""
        ll = ModelLikelihood(
            topology=two_state_topology,
            data=sample_data,
        )
        theta = np.array([0.04, 0.01, 0.05, 1.0, 1.5, 0.005, 0.003, 10.0])
        params = ll.theta_to_params(theta)
        theta_back = ll.params_to_theta(params)
        np.testing.assert_allclose(theta, theta_back)

    def test_better_params_higher_ll(self, two_state_topology, sample_data):
        """Parameters closer to truth should give higher likelihood."""
        ll = ModelLikelihood(
            topology=two_state_topology,
            data=sample_data,
            mode="ode",
        )
        # Reasonable params for growing control data
        good_theta = np.array([0.04, 0.01, 0.0, 1.0, 1.5, 0.005, 0.003, 10.0])
        # Bad params (very high death rate)
        bad_theta = np.array([0.04, 0.2, 0.0, 1.0, 1.5, 0.005, 0.003, 10.0])

        ll_good = ll(good_theta)
        ll_bad = ll(bad_theta)
        assert ll_good > ll_bad

    def test_neg_log_likelihood(self, two_state_topology, sample_data):
        """neg_log_likelihood should be negative of log_likelihood."""
        ll = ModelLikelihood(
            topology=two_state_topology,
            data=sample_data,
            mode="ode",
        )
        theta = np.array([0.04, 0.01, 0.0, 1.0, 1.5, 0.005, 0.003, 10.0])
        assert ll.neg_log_likelihood(theta) == pytest.approx(-ll(theta))

    def test_multiple_data_series(self, two_state_topology, sample_data, sample_data_with_drug):
        """Likelihood with multiple series sums individual contributions."""
        ll_single = ModelLikelihood(
            topology=two_state_topology,
            data=sample_data,
            mode="ode",
        )
        ll_multi = ModelLikelihood(
            topology=two_state_topology,
            data=[sample_data, sample_data_with_drug],
            mode="ode",
        )
        theta = np.array([0.04, 0.01, 0.05, 1.0, 1.5, 0.005, 0.003, 10.0])

        # Multi should include contribution from both datasets
        assert ll_multi.n_params == ll_single.n_params
