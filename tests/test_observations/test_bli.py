"""Tests for BLI observation model."""

import numpy as np
import pytest

from umimic.observations.bli import BLIObservation
from umimic.pk.luciferin import LuciferinKinetics, TissueAttenuation


class TestBLIObservation:
    def test_log_likelihood_finite(self):
        obs = BLIObservation(alpha=1000.0, sigma_log=0.3)
        state = np.array([100.0, 20.0])
        ll = obs.log_likelihood(100_000, state)
        assert np.isfinite(ll)

    def test_proportional_to_viable(self, rng):
        """Signal should scale with viable cell count."""
        obs = BLIObservation(alpha=1000.0, sigma_log=0.1)
        state_small = np.array([100.0, 0.0])
        state_large = np.array([1000.0, 0.0])

        samples_small = [obs.sample(state_small, rng) for _ in range(500)]
        samples_large = [obs.sample(state_large, rng) for _ in range(500)]

        assert np.mean(samples_large) > np.mean(samples_small) * 5

    def test_attenuation_reduces_signal(self, rng):
        """Tissue attenuation should reduce observed signal."""
        att = TissueAttenuation(mu_eff=0.5, reference_depth=2.0)
        obs_no_att = BLIObservation(alpha=1000.0, sigma_log=0.1)
        obs_with_att = BLIObservation(
            alpha=1000.0, sigma_log=0.1, attenuation=att
        )

        state = np.array([500.0, 0.0])
        signal_no_att = obs_no_att.expected_value(state)
        signal_with_att = obs_with_att._expected_signal(state, {"tumor_depth": 5.0})

        assert signal_with_att < signal_no_att

    def test_luciferin_kinetics_at_peak(self, rng):
        """Signal at peak luciferin time should be maximal."""
        luc = LuciferinKinetics(dose=150, ka_luc=0.5, ke_luc=0.05, km=50)
        obs = BLIObservation(
            alpha=1000.0, sigma_log=0.1,
            luciferin=luc,
            imaging_time_post_injection=luc.peak_time,
        )
        state = np.array([100.0, 0.0])
        signal_peak = obs._expected_signal(state)

        # At non-peak time
        obs_off = BLIObservation(
            alpha=1000.0, sigma_log=0.1,
            luciferin=luc,
            imaging_time_post_injection=luc.peak_time * 3,
        )
        signal_off = obs_off._expected_signal(state)

        assert signal_peak >= signal_off


class TestTissueAttenuation:
    def test_no_depth(self):
        att = TissueAttenuation(mu_eff=0.5)
        factor = att.attenuation_factor(depth=0.0)
        assert factor == pytest.approx(1.0)

    def test_deeper_less_signal(self):
        att = TissueAttenuation(mu_eff=0.5)
        f_shallow = att.attenuation_factor(depth=1.0)
        f_deep = att.attenuation_factor(depth=5.0)
        assert f_shallow > f_deep

    def test_volume_based_depth(self):
        att = TissueAttenuation(mu_eff=0.5, reference_depth=2.0)
        f_small = att.attenuation_factor(volume=10.0)
        f_large = att.attenuation_factor(volume=1000.0)
        assert f_small > f_large
