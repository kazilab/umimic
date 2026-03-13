"""Tests for tumor volume observation model."""

import numpy as np
import pytest

from umimic.observations.tumor_volume import TumorVolumeObservation


class TestTumorVolumeObservation:
    def test_log_likelihood_finite(self):
        obs = TumorVolumeObservation(beta=1e-3, sigma_v=0.2)
        state = np.array([1000.0, 200.0])
        ll = obs.log_likelihood(1.0, state)
        assert np.isfinite(ll)

    def test_volume_scales_with_cells(self):
        obs = TumorVolumeObservation(beta=1e-3)
        state_small = np.array([100.0, 0.0])
        state_large = np.array([10000.0, 0.0])
        v_small = obs.expected_value(state_small)
        v_large = obs.expected_value(state_large)
        assert v_large > v_small
        assert v_large / v_small == pytest.approx(100.0, rel=0.01)

    def test_sample_positive(self, rng):
        obs = TumorVolumeObservation(beta=1e-3)
        state = np.array([1000.0, 200.0])
        for _ in range(100):
            v = obs.sample(state, rng)
            assert v > 0
