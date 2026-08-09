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

    def test_param_names_include_sigma_v(self):
        assert TumorVolumeObservation().param_names() == ["beta", "sigma_v"]

    def test_sample_respects_sigma_v_override(self, rng):
        state = np.array([1000.0, 0.0])
        obs = TumorVolumeObservation(beta=1e-3, sigma_v=0.05)
        tight = [obs.sample(state, rng, {"sigma_v": 0.05}) for _ in range(2000)]
        loose = [obs.sample(state, rng, {"sigma_v": 0.8}) for _ in range(2000)]
        assert np.std(np.log(loose)) > np.std(np.log(tight)) * 2

    def test_process_variance_inflates_sample_spread(self, rng):
        state = np.array([1000.0, 0.0])
        obs = TumorVolumeObservation(beta=1e-3, sigma_v=0.1)
        plain = [obs.sample(state, rng) for _ in range(2000)]
        with_proc = [
            obs.sample(state, rng, process_variance=1e6) for _ in range(2000)
        ]
        assert np.std(np.log(with_proc)) > np.std(np.log(plain))
