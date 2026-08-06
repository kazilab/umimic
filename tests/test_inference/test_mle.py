"""Tests for maximum likelihood estimation."""

import numpy as np
import pytest

from umimic.data.schemas import TimeSeriesData
from umimic.inference.mle import MLEstimator, _minimize_options
from umimic.inference.likelihood import ModelLikelihood
from umimic.inference.priors import PriorSpec


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

    def test_bic_uses_the_scored_sample_size(self, two_state_topology, sample_data):
        """BIC must count scored observations, not time points.

        The anchor observation sets the initial condition and is not scored,
        and every modality contributes its own terms. Counting time points
        instead is not a constant offset: the error scales with k, so it
        survives into the BIC *differences* used for model comparison.
        """
        ll = ModelLikelihood(
            topology=two_state_topology,
            data=sample_data,
            mode="ode",
        )
        n_times = sum(len(d.times) for d in ll.data_list)
        assert ll.n_observations == n_times - 1, "anchor should be excluded"

        estimator = MLEstimator(ll, method="Nelder-Mead")
        result = estimator.fit(n_restarts=1)

        theta_hat = ll.params_to_theta(result.parameters)
        assert result.bic == pytest.approx(ll.bic(theta_hat), rel=1e-9)
        # AIC carries no sample size at all, so it is unaffected.
        assert result.aic == pytest.approx(ll.aic(theta_hat), rel=1e-9)

    def test_bic_is_infinite_when_nothing_is_scored(
        self, two_state_topology
    ):
        """A design with only an anchor point scores nothing; BIC is undefined."""
        data = TimeSeriesData.from_counts(
            np.array([0.0, 6.0]), np.array([100.0, np.nan]), concentration=0.0
        )
        ll = ModelLikelihood(
            topology=two_state_topology, data=data, mode="ode"
        )
        assert ll.n_observations == 0

        result = MLEstimator(ll, method="Nelder-Mead").fit(n_restarts=1)
        assert np.isinf(result.bic)

    def test_solver_options_are_method_specific(self):
        """Nelder-Mead should use fatol; other methods should use ftol."""
        nelder = _minimize_options("Nelder-Mead")
        lbfgsb = _minimize_options("L-BFGS-B")

        assert "fatol" in nelder
        assert "ftol" not in nelder
        assert "ftol" in lbfgsb
