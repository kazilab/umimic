"""Tests for maximum likelihood estimation."""

import numpy as np
import pytest

from umimic.inference.mle import MLEstimator, _minimize_options
from umimic.inference.likelihood import ModelLikelihood
from umimic.inference.priors import PriorSpec
from umimic.dynamics.states import ModelTopology
from umimic.data.schemas import TimeSeriesData


class TestMLEstimator:
    def test_mle_converges(self, two_state_topology, sample_data):
        """MLE should converge for reasonable data."""
        ll = ModelLikelihood(
            topology=two_state_topology,
            data=sample_data,
            mode="ode",
        )
        estimator = MLEstimator(ll, method="Nelder-Mead")
        result = estimator.fit(n_restarts=2)

        assert result.converged or result.log_likelihood > -np.inf
        assert "b0" in result.parameters
        assert result.parameters["b0"] > 0

    def test_mle_returns_valid_stats(self, two_state_topology, sample_data):
        """MLE result should have valid AIC/BIC."""
        ll = ModelLikelihood(
            topology=two_state_topology,
            data=sample_data,
            mode="ode",
        )
        estimator = MLEstimator(ll, method="Nelder-Mead")
        result = estimator.fit(n_restarts=1)

        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)
        assert np.isfinite(result.log_likelihood)

    def test_mle_with_priors(self, two_state_topology, sample_data):
        """MAP estimation (MLE + priors) should also work."""
        ll = ModelLikelihood(
            topology=two_state_topology,
            data=sample_data,
            mode="ode",
        )
        priors = PriorSpec.default_invitro()
        estimator = MLEstimator(ll, priors=priors, method="Nelder-Mead")
        result = estimator.fit(n_restarts=1)

        assert "b0" in result.parameters

    def test_solver_options_are_method_specific(self):
        """Nelder-Mead should use fatol; other methods should use ftol."""
        nelder = _minimize_options("Nelder-Mead")
        lbfgsb = _minimize_options("L-BFGS-B")

        assert "fatol" in nelder
        assert "ftol" not in nelder
        assert "ftol" in lbfgsb
