"""Tests for cell count observation model."""

import numpy as np
import pytest

from umimic.observations.cell_counts import CellCountObservation


class TestCellCountObservation:
    def test_log_likelihood_finite(self):
        """Log-likelihood should be finite for valid inputs."""
        obs = CellCountObservation(overdispersion=10.0)
        state = np.array([100.0, 20.0])
        ll = obs.log_likelihood(110, state)
        assert np.isfinite(ll)

    def test_log_likelihood_higher_near_mean(self):
        """Observations near the mean should have higher likelihood."""
        obs = CellCountObservation(overdispersion=10.0)
        state = np.array([100.0, 20.0])  # viable = 120
        ll_near = obs.log_likelihood(120, state)
        ll_far = obs.log_likelihood(200, state)
        assert ll_near > ll_far

    def test_sample_positive(self, rng):
        """Samples should be non-negative."""
        obs = CellCountObservation(overdispersion=10.0)
        state = np.array([100.0, 20.0])
        for _ in range(100):
            s = obs.sample(state, rng)
            assert s >= 0

    def test_sample_mean_near_expected(self, rng):
        """Mean of many samples should approximate expected value."""
        obs = CellCountObservation(overdispersion=10.0)
        state = np.array([200.0, 50.0])
        samples = [obs.sample(state, rng) for _ in range(5000)]
        mean_sample = np.mean(samples)
        expected = obs.expected_value(state)
        assert abs(mean_sample - expected) / expected < 0.1

    def test_overdispersion_affects_variance(self, rng):
        """Lower overdispersion -> higher variance."""
        state = np.array([200.0, 50.0])
        obs_low = CellCountObservation(overdispersion=3.0)
        obs_high = CellCountObservation(overdispersion=50.0)

        samples_low = [obs_low.sample(state, rng) for _ in range(2000)]
        samples_high = [obs_high.sample(state, rng) for _ in range(2000)]

        assert np.var(samples_low) > np.var(samples_high)

    def test_count_type_viable(self):
        """Viable count type should exclude apoptotic (index 2)."""
        obs = CellCountObservation(count_type="viable")
        state = np.array([100.0, 20.0, 50.0, 10.0])  # P, Q, A, R
        expected = obs.expected_value(state)
        assert expected == pytest.approx(130.0)  # P + Q + R = 100 + 20 + 10

    def test_count_type_dead(self):
        """Dead count type should return A only."""
        obs = CellCountObservation(count_type="dead")
        state = np.array([100.0, 20.0, 50.0])
        expected = obs.expected_value(state)
        assert expected == pytest.approx(50.0)
