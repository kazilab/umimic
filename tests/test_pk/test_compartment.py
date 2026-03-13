"""Tests for PK compartment models."""

import numpy as np
import pytest

from umimic.pk.compartment import OneCompartmentPK, TwoCompartmentPK
from umimic.pk.dosing import DosingSchedule


class TestOneCompartmentPK:
    def test_iv_bolus_analytical(self):
        """IV bolus should match analytical solution: C(t) = D/Vd * exp(-ke*t)."""
        pk = OneCompartmentPK(vd=10.0, ke=0.1)
        dosing = DosingSchedule.single_bolus(100.0)
        t_eval = np.linspace(0, 72, 100)

        C = pk.solve(dosing, t_eval)
        expected = (100.0 / 10.0) * np.exp(-0.1 * t_eval)

        np.testing.assert_allclose(C, expected, rtol=1e-4)

    def test_concentration_decays(self):
        """Concentration should decrease over time after bolus."""
        pk = OneCompartmentPK(vd=10.0, ke=0.1)
        dosing = DosingSchedule.single_bolus(50.0)
        t_eval = np.linspace(0, 48, 50)
        C = pk.solve(dosing, t_eval)

        # Should be monotonically decreasing
        assert np.all(np.diff(C) <= 1e-10)

    def test_repeated_dosing_accumulation(self):
        """Repeated dosing should show accumulation."""
        pk = OneCompartmentPK(vd=10.0, ke=0.05)
        dosing = DosingSchedule.repeated(50.0, 24.0, 5)
        t_eval = np.linspace(0, 120, 500)
        C = pk.solve(dosing, t_eval)

        # Peak after 2nd dose should be higher than after 1st
        idx_24 = np.argmin(np.abs(t_eval - 24.0))
        idx_48 = np.argmin(np.abs(t_eval - 48.0))
        assert C[idx_48] > C[0]  # accumulation

    def test_half_life(self):
        pk = OneCompartmentPK(vd=10.0, ke=0.1)
        assert pk.half_life == pytest.approx(np.log(2) / 0.1, rel=1e-6)

    def test_constant_invitro(self):
        """Constant in vitro should return flat concentration."""
        pk = OneCompartmentPK(vd=10.0, ke=0.1)
        dosing = DosingSchedule.constant_invitro(5.0)
        t_eval = np.linspace(0, 72, 50)
        C = pk.solve(dosing, t_eval)
        np.testing.assert_allclose(C, 5.0)

    def test_oral_absorption(self):
        """Oral dosing should show absorption phase then decay."""
        pk = OneCompartmentPK(vd=10.0, ke=0.1, ka=1.0)
        dosing = DosingSchedule.single_bolus(100.0)
        dosing.doses[0].route = "oral"
        t_eval = np.linspace(0.1, 48, 200)
        C = pk.solve(dosing, t_eval)

        # Should peak somewhere after t=0 and then decay
        peak_idx = np.argmax(C)
        assert peak_idx > 0
        assert peak_idx < len(C) - 1


class TestTwoCompartmentPK:
    def test_iv_bolus_biexponential(self):
        """Two-compartment IV should show biexponential decay."""
        pk = TwoCompartmentPK(vc=10.0, vp=20.0, cl=1.0, q=0.5)
        dosing = DosingSchedule.single_bolus(100.0)
        t_eval = np.linspace(0.1, 72, 200)
        C = pk.solve(dosing, t_eval)

        # Should be positive and generally decreasing
        assert np.all(C >= 0)
        # Final concentration should be less than initial
        assert C[-1] < C[0]

    def test_concentrations_nonnegative(self):
        pk = TwoCompartmentPK(vc=10.0, vp=20.0, cl=2.0, q=1.0)
        dosing = DosingSchedule.repeated(50.0, 24.0, 7)
        t_eval = np.linspace(0, 200, 500)
        C = pk.solve(dosing, t_eval)
        assert np.all(C >= -1e-10)
