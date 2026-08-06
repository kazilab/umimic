"""Tests for quantities whose *meaning* was wrong rather than their arithmetic.

A growth rate that is not the growth rate, a variance signature that reaches
only one modality, a likelihood that changes shape with the solver, and a
"pathway" whose drug term points the wrong way all produce plausible numbers.
These tests pin the semantics.
"""

from __future__ import annotations

import numpy as np
import pytest

from umimic.dynamics.moment_equations import MomentODE
from umimic.dynamics.rates import RateSet
from umimic.dynamics.states import CellType, ModelTopology
from umimic.observations.biomarkers import BiomarkerObservation
from umimic.observations.bli import BLIObservation
from umimic.observations.cell_counts import CellCountObservation
from umimic.observations.tumor_volume import TumorVolumeObservation
from umimic.pk.luciferin import TissueAttenuation
from umimic.signaling.models import ToyMapkAktNetwork


# ---------------------------------------------------------------------------
# 7. Growth rate must account for the whole state structure
# ---------------------------------------------------------------------------


def test_asymptotic_growth_rate_matches_simulated_growth():
    """The reported rate must be the one the population actually achieves."""
    topo = ModelTopology.two_state()
    rates = RateSet()

    growth = rates.asymptotic_growth_rate(0.0, topo)

    # Simulate long enough for the state mix to relax, then measure.
    moment = MomentODE(rates, topo, lambda t: 0.0)
    times = np.array([0.0, 400.0, 600.0])
    _, means, _ = moment.solve(
        np.array([1000.0, 1000.0]), t_span=(0, 600), t_eval=times
    )
    totals = means.sum(axis=1)
    realised = np.log(totals[2] / totals[1]) / (times[2] - times[1])

    assert growth == pytest.approx(realised, rel=1e-3)


def test_net_growth_rate_is_not_the_population_growth_rate():
    """Pin the discrepancy the naive formula produces."""
    topo = ModelTopology.two_state()
    rates = RateSet()
    naive = rates.net_growth_rate(0.0)
    true = rates.asymptotic_growth_rate(0.0, topo)

    assert naive == pytest.approx(0.030, abs=1e-6)
    assert true == pytest.approx(0.0254, abs=5e-4)
    assert naive > true  # transitions and Q death are ignored by the naive form


def test_doubling_time_uses_the_multi_state_rate():
    topo = ModelTopology.two_state()
    rates = RateSet()
    doubling = rates.doubling_time(0.0, topo)
    assert doubling == pytest.approx(np.log(2) / 0.0254, rel=0.02)
    # Not the birth-only 17 h, and not the naive 23 h.
    assert doubling > 25.0


def test_doubling_time_is_infinite_when_declining():
    topo = ModelTopology.two_state()
    rates = RateSet(birth_base=0.001, death_base={CellType.P: 0.1, CellType.Q: 0.1})
    assert rates.doubling_time(0.0, topo) == float("inf")


# ---------------------------------------------------------------------------
# 8. Quiescent drug sensitivity must be an explicit choice
# ---------------------------------------------------------------------------


def test_quiescent_cells_are_refractory_by_default_and_this_is_selectable():
    refractory = RateSet.cytotoxic_drug(emax_death=0.2)
    sensitive = RateSet.cytotoxic_drug(emax_death=0.2, quiescent_sensitivity=1.0)
    partial = RateSet.cytotoxic_drug(emax_death=0.2, quiescent_sensitivity=0.5)

    base_q = refractory.death_rate(CellType.Q, 0.0)
    assert refractory.death_rate(CellType.Q, 100.0) == pytest.approx(base_q)
    assert sensitive.death_rate(CellType.Q, 100.0) > partial.death_rate(
        CellType.Q, 100.0
    ) > refractory.death_rate(CellType.Q, 100.0)


def test_mixed_drug_also_exposes_quiescent_sensitivity():
    rates = RateSet.mixed_drug(emax_death=0.2, quiescent_sensitivity=0.5)
    assert CellType.Q in rates.death_modulation


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_quiescent_sensitivity_is_validated(bad):
    with pytest.raises(ValueError, match="quiescent_sensitivity"):
        RateSet.cytotoxic_drug(quiescent_sensitivity=bad)


# ---------------------------------------------------------------------------
# 9. Process variance must reach every modality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda topo: BLIObservation(alpha=1000.0, topology=topo),
        lambda topo: TumorVolumeObservation(beta=1e-5, topology=topo),
    ],
)
def test_lognormal_modalities_use_process_variance(model_factory):
    """BLI and volume previously accepted process_variance and ignored it."""
    topo = ModelTopology.two_state()
    model = model_factory(topo)
    state = np.array([1000.0, 100.0])
    observed = model.expected_value(state) * 3.0

    without = model.log_likelihood(observed, state, None, None)
    with_var = model.log_likelihood(observed, state, None, 1e5)

    assert without != pytest.approx(with_var), (
        "process variance had no effect on the likelihood"
    )
    # Extra variance widens the distribution, so an observation well out in
    # the tail becomes more probable.
    assert with_var > without


def test_process_variance_effect_scales_with_population():
    """Relative uncertainty from demographic noise falls as N grows."""
    topo = ModelTopology.two_state()
    model = BLIObservation(alpha=1000.0, topology=topo)

    small = model._with_process_variance(0.3, np.array([100.0, 0.0]), 1e4)
    large = model._with_process_variance(0.3, np.array([10000.0, 0.0]), 1e4)
    assert small > large > 0.3


# ---------------------------------------------------------------------------
# 10. The likelihood must not change model when the solver changes
# ---------------------------------------------------------------------------


def test_gaussian_branch_matches_negbin_moments_at_zero_process_variance():
    """The two forward modes must be the same statistical model."""
    topo = ModelTopology.two_state()
    model = CellCountObservation(overdispersion=8.0, topology=topo)
    mu, phi = 500.0, 8.0

    # Var(Y) = process + mu + mu^2/phi; at process = 0 this is the NegBin
    # variance, so switching solver no longer changes the assumed noise.
    assert model._total_variance(mu, phi, 0.0) == pytest.approx(mu + mu**2 / phi)


def test_process_variance_adds_to_rather_than_replaces_counting_noise():
    topo = ModelTopology.two_state()
    model = CellCountObservation(overdispersion=8.0, topology=topo)
    mu, phi = 500.0, 8.0
    assert model._total_variance(mu, phi, 1000.0) == pytest.approx(
        1000.0 + mu + mu**2 / phi
    )


def test_ode_and_moment_likelihoods_agree_in_the_low_noise_limit():
    """With negligible process variance the two modes must nearly coincide."""
    from umimic.data.schemas import TimeSeriesData
    from umimic.inference.likelihood import ModelLikelihood

    topo = ModelTopology.two_state()
    times = np.linspace(0, 48, 7)
    counts = np.array([500.0, 560.0, 640.0, 720.0, 810.0, 910.0, 1030.0])
    data = TimeSeriesData(
        times=times, observations={"cell_counts": counts}, concentration=0.0
    )
    theta = np.array([0.04, 0.01, 0.0, 1.0, 1.5, 0.005, 0.003, 10.0])

    ll_ode = ModelLikelihood(topo, data, mode="ode")(theta)
    ll_moment = ModelLikelihood(topo, data, mode="moment")(theta)

    # Not identical (Gaussian vs NegBin shape, plus real demographic noise),
    # but the same order of magnitude rather than a different model.
    assert np.isfinite(ll_ode) and np.isfinite(ll_moment)
    assert abs(ll_ode - ll_moment) < 0.5 * abs(ll_ode)


# ---------------------------------------------------------------------------
# 11. The initial state distribution must be explicit
# ---------------------------------------------------------------------------


def test_default_initial_state_is_all_proliferating():
    from umimic.data.schemas import TimeSeriesData
    from umimic.inference.likelihood import ModelLikelihood

    topo = ModelTopology.two_state()
    data = TimeSeriesData.from_counts(np.linspace(0, 24, 4), [100.0] * 4)
    lik = ModelLikelihood(topo, data)
    np.testing.assert_allclose(lik._initial_state(data), [100.0, 0.0])


def test_stable_initial_fractions_put_cells_in_both_states():
    from umimic.data.schemas import TimeSeriesData
    from umimic.inference.likelihood import ModelLikelihood

    topo = ModelTopology.two_state()
    rates = RateSet()
    data = TimeSeriesData.from_counts(np.linspace(0, 24, 4), [100.0] * 4)
    lik = ModelLikelihood(topo, data, initial_fractions="stable")

    state = lik._initial_state(data, rates)
    assert state.sum() == pytest.approx(100.0)
    assert state[0] > 0 and state[1] > 0, "a relaxed culture is not pure P"


def test_stable_fractions_are_a_fixed_point_of_the_dynamics():
    """Starting from the stable mix, the state proportions must not drift."""
    topo = ModelTopology.two_state()
    rates = RateSet()
    fractions = rates.stable_state_fractions(0.0, topo)

    moment = MomentODE(rates, topo, lambda t: 0.0)
    _, means, _ = moment.solve(
        1000.0 * fractions, t_span=(0, 200), t_eval=np.array([0.0, 200.0])
    )
    start = means[0] / means[0].sum()
    end = means[-1] / means[-1].sum()
    np.testing.assert_allclose(start, end, rtol=1e-4)


def test_explicit_initial_fractions_are_honoured_and_normalized():
    from umimic.data.schemas import TimeSeriesData
    from umimic.inference.likelihood import ModelLikelihood

    topo = ModelTopology.two_state()
    data = TimeSeriesData.from_counts(np.linspace(0, 24, 4), [100.0] * 4)
    lik = ModelLikelihood(topo, data, initial_fractions={"P": 3.0, "Q": 1.0})
    np.testing.assert_allclose(lik._initial_state(data), [75.0, 25.0])


def test_invalid_initial_fractions_are_rejected():
    from umimic.data.schemas import TimeSeriesData
    from umimic.inference.likelihood import ModelLikelihood

    topo = ModelTopology.two_state()
    data = TimeSeriesData.from_counts(np.linspace(0, 24, 4), [100.0] * 4)
    with pytest.raises(ValueError):
        ModelLikelihood(topo, data, initial_fractions="nonsense")._initial_state(data)
    with pytest.raises(ValueError):
        ModelLikelihood(topo, data, initial_fractions=[1.0, 2.0, 3.0])._initial_state(
            data
        )


# ---------------------------------------------------------------------------
# 12. Signaling direction and scope
# ---------------------------------------------------------------------------


def test_toy_network_treats_drug_as_inhibitory_by_default():
    """Most oncology agents suppress MAPK/AKT; the drug term must not raise it."""
    net = ToyMapkAktNetwork()
    y = net.initial_state()
    drug_free = net.rhs(0.0, y, 0.0)
    drugged = net.rhs(0.0, y, 1.0)
    assert drugged[0] < drug_free[0]
    assert drugged[1] < drug_free[1]


def test_stimulatory_direction_is_available_but_explicit():
    net = ToyMapkAktNetwork(direction="stimulatory")
    y = net.initial_state()
    assert net.rhs(0.0, y, 1.0)[0] > net.rhs(0.0, y, 0.0)[0]

    with pytest.raises(ValueError, match="direction"):
        ToyMapkAktNetwork(direction="sideways")


def test_signaling_activity_stays_bounded():
    net = ToyMapkAktNetwork(direction="stimulatory")
    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        lambda t, y: net.rhs(t, y, 1e3), (0, 200), net.initial_state(), max_step=1.0
    )
    assert np.all(sol.y <= 1.0 + 1e-6)
    assert np.all(sol.y >= -1e-6)


def test_signaling_coupling_is_rejected_for_methods_that_ignore_it():
    from umimic.pipeline.config import ExperimentConfig
    from umimic.pipeline.experiment import Experiment

    config = ExperimentConfig()
    config.signaling.enabled = True
    exp = Experiment(config)

    for method in ("gillespie", "tau_leaping"):
        with pytest.raises(NotImplementedError, match="deterministic ODE"):
            exp._reject_unsupported_signaling(method)


# ---------------------------------------------------------------------------
# 13. Tissue attenuation geometry
# ---------------------------------------------------------------------------


def test_volume_averaged_attenuation_exceeds_centroid_approximation():
    """Averaging over the emitting volume must not understate signal."""
    att = TissueAttenuation(mu_eff=0.5, reference_depth=2.0)
    volume = 500.0
    radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
    centroid_only = np.exp(-0.5 * (2.0 + radius))

    assert att.attenuation_factor(volume=volume) > centroid_only


def test_attenuation_bias_grows_with_tumour_size():
    """The old approximation looked like progressive cell loss as tumours grew."""
    att = TissueAttenuation(mu_eff=0.5, reference_depth=2.0)
    ratios = []
    for volume in (10.0, 100.0, 1000.0):
        radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
        centroid_only = np.exp(-0.5 * (2.0 + radius))
        ratios.append(att.attenuation_factor(volume=volume) / centroid_only)
    assert ratios[0] < ratios[1] < ratios[2]


def test_attenuation_is_bounded_and_monotone_in_volume():
    att = TissueAttenuation(mu_eff=0.5, reference_depth=2.0)
    values = [att.attenuation_factor(volume=v) for v in (1.0, 10.0, 100.0, 1000.0)]
    assert all(0.0 < v <= 1.0 for v in values)
    assert values == sorted(values, reverse=True)


def test_explicit_depth_still_uses_the_point_source_form():
    att = TissueAttenuation(mu_eff=0.5, reference_depth=2.0)
    assert att.attenuation_factor(depth=4.0) == pytest.approx(np.exp(-2.0))


def test_tiny_tumour_reduces_to_the_point_source_limit():
    att = TissueAttenuation(mu_eff=0.5, reference_depth=2.0)
    tiny = att.attenuation_factor(volume=1e-12)
    assert tiny == pytest.approx(np.exp(-0.5 * 2.0), rel=1e-4)


# ---------------------------------------------------------------------------
# 14. Biomarker edge cases
# ---------------------------------------------------------------------------


def test_ki67_counts_every_cycling_state_not_just_P():
    """With a proliferating R compartment, P/viable undercounts Ki-67."""
    topo = ModelTopology.four_state()  # P, Q, A, R -- both P and R divide
    model = BiomarkerObservation("ki67", topology=topo)
    state = np.array([100.0, 100.0, 50.0, 100.0])  # P, Q, A, R

    # Cycling = P + R = 200; viable = P + Q + R = 300.
    assert model._get_fraction(state) == pytest.approx(200.0 / 300.0)


def test_ki67_uses_P_alone_when_only_P_divides():
    topo = ModelTopology.two_state()
    model = BiomarkerObservation("ki67", topology=topo)
    assert model._get_fraction(np.array([75.0, 25.0])) == pytest.approx(0.75)


def test_caspase_fraction_is_apoptotic_over_total():
    topo = ModelTopology.three_state()
    model = BiomarkerObservation("caspase", topology=topo)
    assert model._get_fraction(np.array([70.0, 10.0, 20.0])) == pytest.approx(0.2)


def test_extinct_population_yields_no_biomarker_observation():
    """Zero cells must not produce a fabricated 50%-positive reading."""
    topo = ModelTopology.two_state()
    model = BiomarkerObservation("ki67", topology=topo)
    extinct = np.zeros(2)

    assert np.isnan(model._get_fraction(extinct))
    # The likelihood contributes nothing rather than scoring against 0.5.
    assert model.log_likelihood(0.5, extinct) == 0.0
    assert np.isnan(model.sample(extinct, np.random.default_rng(0)))


def test_unsupported_biomarker_type_is_rejected_not_silently_constant():
    with pytest.raises(ValueError, match="biomarker_type"):
        BiomarkerObservation("custom")


def test_biomarker_rejects_out_of_range_observations():
    topo = ModelTopology.two_state()
    model = BiomarkerObservation("ki67", topology=topo)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.log_likelihood(1.4, np.array([100.0, 50.0]))
