"""Tests for the unified exposure profile."""

import numpy as np
import pytest

from umimic.pk.exposure import ExposureProfile
from umimic.pk.dosing import DosingSchedule


class TestExposureProfile:
    def test_constant_profile(self):
        """Constant profile should return the same value at all times."""
        profile = ExposureProfile.constant(5.0)
        assert profile(0.0) == 5.0
        assert profile(100.0) == 5.0
        assert profile.is_constant

    def test_constant_array(self):
        """Array input should return array of constants."""
        profile = ExposureProfile.constant(3.0)
        t = np.array([0, 10, 20, 30])
        result = profile.concentration(t)
        np.testing.assert_allclose(result, 3.0)

    def test_callable_interface(self):
        """Profile should be callable for use as exposure_fn."""
        profile = ExposureProfile.constant(7.5)
        result = profile(10.0)
        assert isinstance(result, float)
        assert result == 7.5

    def test_from_pk(self):
        """PK-driven profile should give time-varying concentrations."""
        from umimic.pk.compartment import OneCompartmentPK

        pk = OneCompartmentPK(vd=10.0, ke=0.1)
        dosing = DosingSchedule.single_bolus(100.0)
        profile = ExposureProfile.from_pk(pk, dosing)

        assert not profile.is_constant
        c0 = profile(0.0)
        c24 = profile(24.0)
        assert c0 > c24  # should decay

    def test_no_drug(self):
        """No drug scenario returns zero."""
        profile = ExposureProfile()
        assert profile(10.0) == 0.0
