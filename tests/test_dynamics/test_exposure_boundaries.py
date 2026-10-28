"""Treatment pulses must survive solver selection and idle populations."""

import numpy as np
import pytest

from umimic.dynamics.gillespie import GillespieSimulator
from umimic.dynamics.moment_equations import MomentODE
from umimic.dynamics.ode_system import CellDynamicsODE, build_ode_system
from umimic.dynamics.rates import EmaxHill, RateSet
from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.tau_leaping import TauLeapingSimulator
from umimic.pk.compartment import OneCompartmentPK
from umimic.pk.dosing import Dose, DosingSchedule
from umimic.pk.exposure import (
    ExposureProfile, exposure_breakpoints, is_constant_exposure, shifted_exposure,
)


@pytest.fixture
def delayed_treatment():
    topology = ModelTopology(
        active_states=[CellType.P], transitions=[], division_states=[],
        death_states=[CellType.P],
    )
    rates = RateSet(
        birth_base=0, death_base={CellType.P: 0}, transition_base={},
        death_modulation={CellType.P: EmaxHill(emax=1, ec50=1, hill=1)},
    )
    exposure = ExposureProfile.from_pk(
        OneCompartmentPK(vd=1, ke=2), DosingSchedule.single_bolus(100, time=5),
    )
    return topology, rates, exposure


def test_delayed_bolus_ode_and_moments_match_analytical_survival(delayed_treatment):
    topology, rates, exposure = delayed_treatment
    initial = np.array([1000.0])
    times = np.array([0.0, 4.0, 5.0, 10.0, 100.0])
    expected_survival = 1 / np.sqrt(101)
    ode = CellDynamicsODE(rates, topology, exposure).solve(initial, (0, 100), times)
    _, means, covariances = MomentODE(rates, topology, exposure).solve(
        initial, Sigma0=np.zeros((1, 1)), t_span=(0, 100), t_eval=times,
    )
    assert ode.metadata["method"] != "expm"
    np.testing.assert_allclose(ode.populations["P"][:3], 1000)
    assert ode.populations["P"][-1] == pytest.approx(1000 * expected_survival, rel=1e-5)
    assert means[-1, 0] == pytest.approx(1000 * expected_survival, rel=1e-5)
    assert covariances[-1, 0, 0] == pytest.approx(
        1000 * expected_survival * (1 - expected_survival), rel=1e-5,
    )


def test_unknown_callable_is_not_assumed_constant_between_matching_probes(delayed_treatment):
    topology, rates, _ = delayed_treatment

    def pulse(t):
        return 100.0 if 5 <= t < 6 else 0.0

    pulse.breakpoints = lambda start, end: [5.0, 6.0]
    ode = CellDynamicsODE(rates, topology, pulse).solve(
        np.array([1000.0]), (0, 100), np.array([0.0, 50.0, 100.0]),
    )
    assert ode.metadata["method"] != "expm"
    assert ode.populations["P"][-1] == pytest.approx(1000 * np.exp(-100 / 101), rel=1e-5)
    ssa = GillespieSimulator(rates, topology, pulse)
    assert ssa._resolve_method(100) == ("thinning", True)


def test_known_constant_exposure_retains_fast_paths(delayed_treatment):
    topology, rates, _ = delayed_treatment
    exposure = ExposureProfile.constant(1)
    ode = build_ode_system(rates, topology, constant_concentration=1).solve(
        np.array([1000.0]), (0, 1), np.array([0.0, 1.0]),
    )
    assert ode.metadata["method"] == "expm"
    assert ode.populations["P"][-1] == pytest.approx(1000 * np.exp(-0.5))
    assert GillespieSimulator(rates, topology, exposure)._resolve_method(1) == ("direct", True)
    # An unmarked callable is deliberately not a proof of constancy.
    assert GillespieSimulator(rates, topology, lambda t: 1)._resolve_method(1) == (
        "thinning", True,
    )


@pytest.mark.parametrize("method", ["ssa", "direct", "tau", "tau_ssa_fallback"])
def test_stochastic_solvers_wait_for_delayed_treatment(delayed_treatment, method):
    topology, rates, exposure = delayed_treatment
    initial = np.array([1000.0])
    times = np.array([0.0, 4.0, 5.0, 10.0, 20.0])
    rng = np.random.default_rng(7)
    if method in ("ssa", "direct"):
        solver = GillespieSimulator(
            rates, topology, exposure, rng,
            exposure_mode="direct" if method == "direct" else "auto",
        )
    else:
        solver = TauLeapingSimulator(
            rates, topology, exposure, rng=rng,
            ssa_threshold=2000 if method == "tau_ssa_fallback" else 0,
        )
    result = solver.simulate(initial, 20, times)
    np.testing.assert_array_equal(result.populations["P"][:3], [1000, 1000, 1000])
    # Analytically mean=99.5, SD=9.47. The approximate forced-direct method
    # freezes falling drug effects between events, but must still treat cells.
    assert 50 < result.populations["P"][-1] < 150
    if method == "ssa":
        assert result.metadata["n_events"] > 0
        assert result.metadata["t_reached"] == 20
        assert not result.metadata["extinct"]


def test_bound_method_and_clock_shift_preserve_exposure_metadata(delayed_treatment):
    _, _, exposure = delayed_treatment
    np.testing.assert_array_equal(exposure_breakpoints(exposure.concentration, 0, 10), [5])
    shifted = shifted_exposure(exposure, 2)
    np.testing.assert_array_equal(exposure_breakpoints(shifted, 0, 10), [3])
    assert shifted(3) == 100
    assert is_constant_exposure(shifted_exposure(ExposureProfile.constant(2), 8))


def test_exposure_boundaries_include_infusion_end_and_cache_knots():
    profile = ExposureProfile.from_pk(
        OneCompartmentPK(), DosingSchedule(doses=[Dose(2, 10, "iv_infusion", 3)]),
    )
    np.testing.assert_array_equal(profile.breakpoints(0, 10), [2, 5])
    profile.precompute(np.array([0, 2, 4, 5, 10]))
    np.testing.assert_array_equal(profile.breakpoints(0, 10), [2, 4, 5])


def test_constant_idle_state_is_not_reported_extinct(delayed_treatment):
    topology, rates, _ = delayed_treatment
    result = GillespieSimulator(rates, topology, ExposureProfile.constant(0)).simulate(
        np.array([1000.0]), 10, np.array([0.0, 10.0]),
    )
    assert not result.metadata["extinct"]
    assert result.metadata["t_reached"] == 10


def test_terminal_event_stops_segmented_integration(delayed_treatment):
    topology, rates, exposure = delayed_treatment

    def half_population(t, y):
        return y[0] - 500

    half_population.terminal = True
    result = CellDynamicsODE(rates, topology, exposure).solve(
        np.array([1000.0]), (0, 100), np.array([0.0, 4.0, 5.0, 10.0, 100.0]),
        events=half_population,
    )
    np.testing.assert_array_equal(result.times, [0, 4, 5])
    np.testing.assert_array_equal(result.populations["P"], [1000, 1000, 1000])


@pytest.mark.parametrize("kwargs", [{"lookahead": 0}, {"bound_samples": 1}, {"bound_safety": 0.5}])
def test_invalid_thinning_controls_cannot_stall_time(delayed_treatment, kwargs):
    topology, rates, exposure = delayed_treatment
    with pytest.raises(ValueError):
        GillespieSimulator(rates, topology, exposure, **kwargs)
