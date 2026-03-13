"""Tests for dose-response functions and rate parameterization."""

import numpy as np
import pytest

from umimic.dynamics.rates import EmaxHill, FourParameterLogistic, ConstantRate, RateSet
from umimic.dynamics.states import CellType


class TestEmaxHill:
    def test_zero_concentration(self):
        f = EmaxHill(emax=1.0, ec50=1.0, hill=1.0)
        assert f(0.0) == 0.0

    def test_high_concentration(self):
        f = EmaxHill(emax=1.0, ec50=1.0, hill=1.0)
        result = f(1e6)
        assert result == pytest.approx(1.0, abs=1e-3)

    def test_ec50_gives_half_emax(self):
        f = EmaxHill(emax=1.0, ec50=5.0, hill=1.0)
        result = f(5.0)
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_monotonic(self):
        f = EmaxHill(emax=0.1, ec50=1.0, hill=2.0)
        concs = np.logspace(-2, 2, 100)
        values = f(concs)
        assert np.all(np.diff(values) >= -1e-10)  # non-decreasing

    def test_array_input(self):
        f = EmaxHill(emax=1.0, ec50=1.0, hill=1.0)
        result = f(np.array([0, 1, 10]))
        assert result.shape == (3,)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)

    def test_hill_affects_steepness(self):
        f1 = EmaxHill(emax=1.0, ec50=1.0, hill=1.0)
        f2 = EmaxHill(emax=1.0, ec50=1.0, hill=3.0)
        # At EC50, both give 0.5
        assert f1(1.0) == pytest.approx(0.5)
        assert f2(1.0) == pytest.approx(0.5)
        # Steeper Hill: lower below EC50, higher above
        assert f2(0.5) < f1(0.5)
        assert f2(2.0) > f1(2.0)

    def test_negative_concentration_clamped(self):
        f = EmaxHill(emax=1.0, ec50=1.0, hill=1.0)
        assert f(-1.0) == 0.0


class TestFourParameterLogistic:
    def test_at_zero(self):
        f = FourParameterLogistic(top=1.0, bottom=0.0, ec50=1.0, hill=1.0)
        result = f(1e-10)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_at_ec50(self):
        f = FourParameterLogistic(top=1.0, bottom=0.0, ec50=1.0, hill=1.0)
        result = f(1.0)
        assert result == pytest.approx(0.5, abs=0.01)


class TestRateSet:
    def test_birth_rate_no_drug(self, simple_rates):
        b = simple_rates.birth_rate(0.0)
        assert b == pytest.approx(0.04)

    def test_birth_rate_cytostatic(self, cytostatic_rates):
        b_ctrl = cytostatic_rates.birth_rate(0.0)
        b_drug = cytostatic_rates.birth_rate(100.0)
        assert b_ctrl > b_drug
        assert b_drug >= 0

    def test_death_rate_cytotoxic(self, cytotoxic_rates):
        d_ctrl = cytotoxic_rates.death_rate(CellType.P, 0.0)
        d_drug = cytotoxic_rates.death_rate(CellType.P, 100.0)
        assert d_drug > d_ctrl

    def test_net_growth_positive_control(self, simple_rates):
        ng = simple_rates.net_growth_rate(0.0)
        assert ng > 0  # b0 > d0

    def test_net_growth_decreases_with_dose(self, cytotoxic_rates):
        ng0 = cytotoxic_rates.net_growth_rate(0.0)
        ng10 = cytotoxic_rates.net_growth_rate(10.0)
        assert ng0 > ng10

    def test_density_dependence(self):
        rs = RateSet(birth_base=0.04)
        b_empty = rs.birth_rate(0.0, total_cells=0, carrying_capacity=1000)
        b_full = rs.birth_rate(0.0, total_cells=1000, carrying_capacity=1000)
        assert b_empty == pytest.approx(0.04)
        assert b_full == pytest.approx(0.0)

    def test_transition_rate(self, simple_rates):
        u = simple_rates.transition_rate(CellType.P, CellType.Q, 0.0)
        assert u == pytest.approx(0.005)

    def test_rates_always_nonnegative(self, cytotoxic_rates):
        for c in np.logspace(-3, 3, 50):
            assert cytotoxic_rates.birth_rate(c) >= 0
            assert cytotoxic_rates.death_rate(CellType.P, c) >= 0
