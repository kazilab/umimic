"""Regression coverage for exposure, conditioning and observation routing."""

import numpy as np
import pytest

from umimic.data.schemas import TimeSeriesData
from umimic.dynamics.ode_system import CellDynamicsODE
from umimic.dynamics.rates import EmaxHill, HillFoldChange, RateSet
from umimic.dynamics.states import CellType, ModelTopology
from umimic.inference.clonal import (
    clone_size_logpmf, clone_size_params, fit_clone_sizes,
    moments_to_rates, sample_clone_sizes,
)
from umimic.inference.likelihood import ModelLikelihood
from umimic.observations.cell_counts import CellCountObservation
from umimic.observations.multimodal import MultimodalObservation


@pytest.fixture
def single_state():
    return ModelTopology(
        active_states=[CellType.P], transitions=[],
        division_states=[CellType.P], death_states=[CellType.P],
    )


@pytest.mark.parametrize("mode", ["ode", "moment"])
@pytest.mark.parametrize("late_initial", [100., 150.])
def test_replicates_start_at_their_own_anchor(single_state, mode, late_initial):
    early = TimeSeriesData.from_counts([0, 10], [100, 200])
    late = TimeSeriesData.from_counts([10, 20], [late_initial, late_initial * 2])
    theta = np.array([np.log(2) / 10 + .01, .01])
    options = dict(topology=single_state, param_names=["b0", "d0_P"], mode=mode)
    joint = ModelLikelihood(data=[early, late], **options)
    separate = sum(ModelLikelihood(data=d, **options)(theta) for d in [early, late])
    assert joint(theta) == pytest.approx(separate, abs=1e-6)


@pytest.mark.parametrize("mode", ["ode", "moment"])
def test_missing_rows_before_anchor_do_not_add_growth(single_state, mode):
    complete = TimeSeriesData.from_counts([10, 20], [100, 200])
    missing = TimeSeriesData.from_counts([0, 10, 20], [np.nan, 100, 200])
    theta = np.array([np.log(2) / 10 + .01, .01])
    options = dict(topology=single_state, param_names=["b0", "d0_P"], mode=mode)
    assert ModelLikelihood(data=missing, **options)(theta) == pytest.approx(
        ModelLikelihood(data=complete, **options)(theta), abs=1e-6
    )


def test_observed_points_before_a_future_anchor_are_refused(single_state):
    data = TimeSeriesData(
        times=[0., 10.], observations={"cell_counts": [np.nan, 100.], "dead": [5., 10.]},
    )
    models = MultimodalObservation({
        "cell_counts": CellCountObservation(), "dead": CellCountObservation(),
    })
    with pytest.raises(ValueError, match="before the initial anchor"):
        ModelLikelihood(single_state, data, observation_model=models)


@pytest.mark.parametrize("mode", ["ode", "moment"])
def test_single_timepoint_is_scored_without_conditioning(single_state, mode):
    data = TimeSeriesData.from_counts([7.], [100.])
    likelihood = ModelLikelihood(
        single_state, data, param_names=["b0"], mode=mode,
        condition_on_first=False,
    )
    assert likelihood.n_observations == 1
    expected = likelihood.obs_model.log_likelihood(
        100., np.array([100.]), process_variance=100. if mode == "moment" else None,
    )
    assert likelihood(np.array([.04])) == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["ode", "moment"])
def test_explicit_time_varying_exposure_reaches_forward_model(single_state, mode):
    data = TimeSeriesData.from_counts([0., 2., 4.], [100., 95., 90.])
    exposure = lambda t: 10. * np.exp(-.1 * t)
    likelihood = ModelLikelihood(
        single_state, data, param_names=["b0", "d0_P", "emax_death"],
        mode=mode, exposure_fn=exposure,
    )
    theta = np.array([.04, .01, .1])
    rates = likelihood._build_rate_set(likelihood.theta_to_params(theta))
    times, means, _ = likelihood._forward_solutions(rates, 0., [data])[0]
    reference = CellDynamicsODE(rates, single_state, exposure).solve(
        np.array([100.]), (0., 4.), data.times,
    )
    np.testing.assert_allclose(means[:, 0], reference.populations["P"], rtol=1e-5)
    zero_effect = theta.copy()
    zero_effect[2] = 0.
    assert likelihood(theta) != pytest.approx(likelihood(zero_effect))


def test_sampled_rates_preserve_fixed_configuration(single_state):
    edge = (CellType.P, CellType.Q)
    base = RateSet(
        birth_base=.03, death_base={CellType.P: .02, CellType.Q: .007},
        birth_modulation=EmaxHill(.8, 2., 1.), clearance_rate=.6,
        transition_factor={edge: HillFoldChange(low=1., high=.2, ec50=3., hill=1.)},
        transition_induction={edge: EmaxHill(.02, 3., 1.)},
    )
    data = TimeSeriesData.from_counts([0., 1.], [100., 102.])
    likelihood = ModelLikelihood(
        single_state, data, param_names=["b0", "d0_P", "u_PQ"],
        mode="ode", base_rate_set=base,
    )
    rates = likelihood._build_rate_set({"b0": .05, "d0_P": .03, "u_PQ": .02})
    assert rates.birth_base == .05
    assert rates.death_base[CellType.P] == .03
    assert rates.death_base[CellType.Q] == .007
    assert rates.clearance_rate == .6
    assert rates.birth_modulation(2.) == pytest.approx(base.birth_modulation(2.))
    assert rates.transition_factor[edge](3.) == pytest.approx(.6)
    assert rates.transition_induction[edge](3.) == pytest.approx(.01)
    assert base.birth_base == .03
    assert base.transition_base[edge] == .005


@pytest.mark.parametrize("mode", ["ode", "moment"])
def test_custom_noise_parameter_changes_dead_channel_likelihood(mode):
    topology = ModelTopology.three_state()
    dead = CellCountObservation(count_type="dead", topology=topology, overdispersion_key="phi_dead")
    data = TimeSeriesData(
        times=[0., 10., 20.],
        observations={"cell_counts": [100., 110., 120.], "dead": [0., 40., 70.]},
    )
    likelihood = ModelLikelihood(
        topology, data, mode=mode,
        param_names=["b0", "d0_P", "overdispersion", "phi_dead"],
        observation_model=MultimodalObservation({"cell_counts": CellCountObservation(), "dead": dead}),
    )
    first = likelihood(np.array([.04, .02, 10., 1.]))
    second = likelihood(np.array([.04, .02, 10., 100.]))
    assert np.isfinite(first) and np.isfinite(second)
    assert first != pytest.approx(second)


@pytest.mark.parametrize("birth,death", [(.01, .05), (0., .05), (.05, 0.), (.05, .05)])
def test_clone_law_covers_declining_and_boundary_processes(birth, death):
    time = 24.
    alpha, beta = clone_size_params(birth, death, time)
    assert 0 <= alpha < 1 and 0 <= beta < 1
    probabilities = np.exp(clone_size_logpmf(np.arange(5000), birth, death, time))
    assert probabilities.sum() == pytest.approx(1.)
    recovered = moments_to_rates(alpha, beta, time)
    np.testing.assert_allclose(recovered, [birth, death], atol=1e-12)


def test_subcritical_clone_fit_can_recover_death_exceeding_birth():
    sizes = sample_clone_sizes(.01, .05, 24., 20000, np.random.default_rng(54))
    result = fit_clone_sizes(sizes, 24.)
    assert result.converged
    assert result.birth_rate == pytest.approx(.01, rel=.15)
    assert result.death_rate == pytest.approx(.05, rel=.08)
