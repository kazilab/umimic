"""Tests for luciferin kinetics and tissue attenuation."""

import numpy as np
import pytest

from umimic.pk.luciferin import LuciferinKinetics, TissueAttenuation


class TestLuciferinKinetics:
    def test_signal_fraction_at_peak_is_one(self):
        luc = LuciferinKinetics()
        assert luc.signal_fraction(luc.peak_time) == pytest.approx(1.0)

    def test_rejects_non_positive_rates(self):
        with pytest.raises(ValueError, match="ka_luc"):
            LuciferinKinetics(ka_luc=0.0)
        with pytest.raises(ValueError, match="ke_luc"):
            LuciferinKinetics(ke_luc=-0.1)
        with pytest.raises(ValueError, match="km"):
            LuciferinKinetics(km=0.0)


class TestTissueAttenuation:
    def test_volume_average_exceeds_centroid(self):
        att = TissueAttenuation(mu_eff=0.5, reference_depth=2.0)
        vol = 500.0
        r = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        centroid = np.exp(-0.5 * (2.0 + r))
        assert att.attenuation_factor(volume=vol) > centroid

    def test_rejects_negative_mu(self):
        with pytest.raises(ValueError, match="mu_eff"):
            TissueAttenuation(mu_eff=-0.1)
