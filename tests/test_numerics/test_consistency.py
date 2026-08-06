"""Tests for single-source-of-truth and API-honesty properties.

Duplicate settings, inverted sign conventions, and configuration options the
orchestrator does not implement all fail quietly. These tests make each of
them fail loudly instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from umimic.dynamics.rates import (
    ConstantRate,
    EmaxHill,
    FourParameterLogistic,
    RateSet,
)
from umimic.dynamics.states import CellType, ModelTopology
from umimic.observations.bli import BLIObservation
from umimic.observations.cell_counts import CellCountObservation
from umimic.observations.multimodal import MultimodalObservation
from umimic.observations.tumor_volume import TumorVolumeObservation


# ---------------------------------------------------------------------------
# Clearance has one home
# ---------------------------------------------------------------------------


def test_topology_no_longer_carries_a_second_clearance_rate():
    topo = ModelTopology.two_state()
    assert not hasattr(topo, "apoptotic_clearance_rate")


def test_setting_the_removed_clearance_attribute_raises():
    """It used to accept the assignment and ignore it."""
    topo = ModelTopology.three_state()
    with pytest.raises(AttributeError, match="RateSet.clearance_rate"):
        topo.apoptotic_clearance_rate = 0.5


def test_clearance_configured_on_the_rate_set_reaches_the_simulation():
    from umimic.dynamics.moment_equations import MomentODE

    topo = ModelTopology.three_state()  # P, Q, A
    fast = RateSet(death_base={CellType.P: 0.05}, clearance_rate=1.0)
    slow = RateSet(death_base={CellType.P: 0.05}, clearance_rate=0.001)

    mu0 = np.array([500.0, 0.0, 100.0])
    times = np.array([0.0, 48.0])
    a_fast = MomentODE(fast, topo, lambda t: 0.0).solve(
        mu0, t_span=(0, 48), t_eval=times
    )[1][-1, 2]
    a_slow = MomentODE(slow, topo, lambda t: 0.0).solve(
        mu0, t_span=(0, 48), t_eval=times
    )[1][-1, 2]
    assert a_fast < a_slow


# ---------------------------------------------------------------------------
# Modulator sign convention
# ---------------------------------------------------------------------------


def test_emax_hill_is_a_valid_modulator_and_4pl_is_not():
    assert EmaxHill(emax=0.8, ec50=1.0, hill=1.0).is_increasing()
    assert not FourParameterLogistic(top=1.0, bottom=0.0).is_increasing()
    # A flat curve is admissible (no effect).
    assert ConstantRate(value=0.3).is_increasing()


def test_decreasing_modulator_is_rejected_at_construction():
    """A 4PL used directly as a modulator inverts the pharmacology."""
    viability = FourParameterLogistic(top=1.0, bottom=0.0, ec50=1.0, hill=1.0)

    with pytest.raises(ValueError, match="decreases with concentration"):
        RateSet(birth_base=0.05, birth_modulation=viability)
    with pytest.raises(ValueError, match="as_effect"):
        RateSet(death_base={CellType.P: 0.01}, death_modulation={CellType.P: viability})


def test_as_effect_reorients_a_viability_curve_into_a_modulator():
    viability = FourParameterLogistic(top=1.0, bottom=0.2, ec50=1.0, hill=1.5)
    effect = viability.as_effect()

    assert effect.is_increasing()
    assert float(effect(0.0)) == pytest.approx(0.0)
    assert float(effect(1e6)) == pytest.approx(0.8, rel=1e-3)

    rates = RateSet(birth_base=0.05, birth_modulation=effect)
    # Untreated growth is preserved and the drug suppresses it, which is the
    # opposite of what the unreoriented curve produced.
    assert rates.birth_rate(0.0) == pytest.approx(0.05)
    assert rates.birth_rate(1e6) < 0.05


def test_per_state_and_transition_modulators_are_validated_too():
    viability = FourParameterLogistic(top=1.0, bottom=0.0)
    with pytest.raises(ValueError, match="birth_modulation_by_state"):
        RateSet(birth_modulation_by_state={CellType.R: viability})
    with pytest.raises(ValueError, match="transition_modulation"):
        RateSet(transition_modulation={(CellType.P, CellType.Q): viability})


def test_builtin_factories_still_construct():
    """The new validation must not reject the package's own rate sets."""
    for factory in (
        RateSet.cytotoxic_drug,
        RateSet.cytostatic_drug,
        RateSet.mixed_drug,
        RateSet.resistant_clone,
    ):
        assert isinstance(factory(), RateSet)


# ---------------------------------------------------------------------------
# Synthetic generation must honour the configured modalities
# ---------------------------------------------------------------------------


def _multimodal_generator():
    from umimic.data.synthetic import SyntheticDataGenerator

    topo = ModelTopology.two_state()
    models = MultimodalObservation(
        {
            "cell_counts": CellCountObservation(10.0, topology=topo),
            "bli": BLIObservation(alpha=1000.0, topology=topo),
            "volume": TumorVolumeObservation(beta=1e-5, topology=topo),
        }
    )
    return SyntheticDataGenerator(
        RateSet(), topo, models, np.random.default_rng(0)
    )


def test_invitro_plate_emits_every_configured_modality():
    """It previously emitted cell counts only, whatever the model held."""
    gen = _multimodal_generator()
    dataset = gen.generate_invitro_plate(
        concentrations=[0.0], n_wells_per_dose=1, t_max=24.0, dt_obs=8.0,
        method="ode",
    )
    assert set(dataset.series[0].modalities) == {"cell_counts", "bli", "volume"}


def test_invivo_cohort_emits_every_configured_modality():
    gen = _multimodal_generator()
    dataset = gen.generate_invivo_cohort(
        n_animals=1, t_max=96.0, obs_times=np.array([0.0, 48.0, 96.0]),
        method="ode",
    )
    assert set(dataset.series[0].modalities) == {"cell_counts", "bli", "volume"}


def test_generated_modalities_are_mutually_consistent():
    """BLI and volume must come from the same observation models as the fit.

    The in vivo generator used to inline its own formulas, including a
    hardcoded 1e-3 mm^3/cell volume scale, so generated data did not match the
    likelihood later fitted to it.
    """
    gen = _multimodal_generator()
    dataset = gen.generate_invivo_cohort(
        n_animals=1, t_max=48.0, obs_times=np.array([0.0, 48.0]), method="ode"
    )
    series = dataset.series[0]
    # volume ~ beta * viable and bli ~ alpha * viable, so their ratio pins beta/alpha
    ratio = series.observations["volume"] / series.observations["bli"]
    assert np.all(ratio < 1e-6), "volume/BLI scales are inconsistent"


def test_requesting_an_unconfigured_modality_is_an_error():
    from umimic.data.synthetic import SyntheticDataGenerator

    topo = ModelTopology.two_state()
    gen = SyntheticDataGenerator(RateSet(), topo, rng=np.random.default_rng(0))
    assert gen.modalities == ["cell_counts"]
    with pytest.raises(ValueError, match="No observation model configured"):
        gen.generate_invitro_plate(modalities=["bli"], method="ode")


def test_single_modality_generator_still_works():
    from umimic.data.synthetic import SyntheticDataGenerator

    topo = ModelTopology.two_state()
    gen = SyntheticDataGenerator(RateSet(), topo, rng=np.random.default_rng(0))
    dataset = gen.generate_invitro_plate(
        concentrations=[0.0], n_wells_per_dose=1, t_max=24.0, dt_obs=8.0,
        method="ode",
    )
    assert dataset.series[0].modalities == ["cell_counts"]
    assert dataset.series[0].replicate_id is not None


# ---------------------------------------------------------------------------
# Configuration must not advertise unimplemented modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["smc", "hierarchical"])
def test_unimplemented_inference_modes_are_rejected_by_config(mode):
    """These parsed fine and then failed after the data had been loaded."""
    import pydantic

    from umimic.pipeline.config import ExperimentConfig

    with pytest.raises(pydantic.ValidationError):
        ExperimentConfig(inference={"mode": mode})


@pytest.mark.parametrize("mode", ["mle", "mcmc"])
def test_implemented_inference_modes_are_accepted(mode):
    from umimic.pipeline.config import ExperimentConfig

    assert ExperimentConfig(inference={"mode": mode}).inference.mode == mode


def test_fit_reports_unsupported_modes_with_a_pointer_to_the_direct_api():
    from umimic.data.schemas import TimeSeriesData
    from umimic.pipeline.config import ExperimentConfig
    from umimic.pipeline.experiment import Experiment

    config = ExperimentConfig()
    exp = Experiment(config)
    exp.config.inference.mode = "smc"  # bypass validation, as direct use would

    data = TimeSeriesData.from_counts(np.linspace(0, 24, 5), [100.0] * 5)
    with pytest.raises(ValueError, match="ParticleMCMC"):
        exp.fit(data)


# ---------------------------------------------------------------------------
# Luciferin units are documented as relative
# ---------------------------------------------------------------------------


def test_signal_fraction_is_normalized_to_the_peak():
    """The model is usable relatively even though its units are arbitrary."""
    from umimic.pk.luciferin import LuciferinKinetics

    kin = LuciferinKinetics()
    assert kin.signal_fraction(kin.peak_time) == pytest.approx(1.0)
    assert kin.signal_fraction(kin.peak_time * 0.1) < 1.0
    assert kin.signal_fraction(kin.peak_time * 5.0) < 1.0


def test_signal_fraction_is_scale_invariant_in_dose_and_km():
    """Only dose/km matters, which is why the units cancel."""
    from umimic.pk.luciferin import LuciferinKinetics

    a = LuciferinKinetics(dose=150.0, km=50.0)
    b = LuciferinKinetics(dose=1500.0, km=500.0)
    for t in (5.0, 15.0, 60.0):
        assert a.signal_fraction(t) == pytest.approx(b.signal_fraction(t), rel=1e-9)


# ---------------------------------------------------------------------------
# A backend the orchestrator does not dispatch must not be silently swapped
# ---------------------------------------------------------------------------


def test_particle_backend_is_rejected_rather_than_substituted():
    """Experiment.fit() ran emcee for backend='particle' and reported success.

    A config asking for particle inference and receiving an ensemble sampler
    is indistinguishable from one that worked, so this has to fail at
    construction, before any data is loaded.
    """
    from umimic.pipeline.config import ExperimentConfig

    with pytest.raises(ValueError, match="ParticleMCMC"):
        ExperimentConfig(inference={"mode": "mcmc", "backend": "particle"})


# ---------------------------------------------------------------------------
# "exact" in simulator metadata must mean exact
# ---------------------------------------------------------------------------


def _gillespie(exposure_fn, mode):
    from umimic.dynamics.gillespie import GillespieSimulator

    topo = ModelTopology.two_state()
    rates = RateSet(
        birth_base=0.05, death_base={CellType.P: 0.01, CellType.Q: 0.005}
    )
    return GillespieSimulator(
        rates, topo, exposure_fn, rng=np.random.default_rng(0),
        exposure_mode=mode,
    )


def test_direct_method_is_not_exact_under_a_time_varying_exposure():
    """Frozen propensities are an approximation; the flag must say so."""
    varying = _gillespie(lambda t: 1.0 + t, "direct").simulate(
        np.array([50.0, 0.0]), t_max=5.0, t_record=np.array([0.0, 5.0])
    )
    assert varying.metadata["exact"] is False

    constant = _gillespie(lambda _t: 1.0, "direct").simulate(
        np.array([50.0, 0.0]), t_max=5.0, t_record=np.array([0.0, 5.0])
    )
    assert constant.metadata["exact"] is True


def test_thinning_and_auto_remain_exact():
    thinning = _gillespie(lambda t: 1.0 + t, "thinning").simulate(
        np.array([50.0, 0.0]), t_max=5.0, t_record=np.array([0.0, 5.0])
    )
    assert thinning.metadata["exact"] is True
    assert thinning.metadata["method"] == "gillespie:thinning"

    # "auto" only picks the direct method after confirming constancy.
    auto = _gillespie(lambda t: 1.0 + t, "auto").simulate(
        np.array([50.0, 0.0]), t_max=5.0, t_record=np.array([0.0, 5.0])
    )
    assert auto.metadata["exact"] is True
    assert auto.metadata["method"] == "gillespie:thinning"


# ---------------------------------------------------------------------------
# The composite observation model must carry the LNA process variance
# ---------------------------------------------------------------------------


def test_multimodal_log_likelihood_uses_the_process_variance():
    """Variance fusion was a property of the orchestrated path only.

    Called directly, the composite dropped the LNA variance, so every
    modality saw measurement noise alone and the mechanistic
    birth-versus-death signature never reached the likelihood.
    """
    topo = ModelTopology.two_state()
    model = MultimodalObservation(
        {
            "cell_counts": CellCountObservation(10.0, topology=topo),
            "volume": TumorVolumeObservation(
                beta=1e-3, sigma_v=0.2, topology=topo
            ),
        }
    )
    state = np.array([400.0, 100.0])
    observations = {"cell_counts": 500.0, "volume": 0.5}

    without = model.log_likelihood(observations, state)
    with_var = model.log_likelihood(observations, state, None, 2500.0)

    assert without != pytest.approx(with_var)

    # A per-modality dict must reach each model independently.
    per_modality = model.log_likelihood(
        observations, state, None, {"cell_counts": 2500.0}
    )
    assert per_modality != pytest.approx(with_var)
    assert per_modality != pytest.approx(without)


# ---------------------------------------------------------------------------
# Induced plasticity potency is a separate parameter, not the cytotoxic one
# ---------------------------------------------------------------------------


def test_induced_transition_potency_defaults_to_death_but_can_be_freed():
    """Tying plasticity potency to killing potency is a parsimony choice.

    It must stay the default for backwards compatibility, and it must be
    escapable, because reading a fitted induced_* term as independent
    pharmacology would otherwise be wrong.
    """
    from umimic.inference.likelihood import build_rate_set

    base = {
        "b0": 0.05, "d0_P": 0.01, "emax_death": 0.03,
        "ec50_death": 2.0, "hill_death": 1.5, "induced_PQ": 0.01,
    }
    edge = (CellType.P, CellType.Q)

    tied = build_rate_set(base).transition_induction[edge]
    assert tied.ec50 == pytest.approx(2.0)
    assert tied.hill == pytest.approx(1.5)

    freed = build_rate_set(
        {**base, "ec50_induction": 20.0, "hill_induction": 3.0}
    ).transition_induction[edge]
    assert freed.ec50 == pytest.approx(20.0)
    assert freed.hill == pytest.approx(3.0)
    # The cytotoxic potency is untouched by the induction parameters.
    assert build_rate_set(
        {**base, "ec50_induction": 20.0}
    ).death_modulation[CellType.P].ec50 == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# A rejected parameter vector is a zero-likelihood point, not a crash
# ---------------------------------------------------------------------------


def test_degenerate_rate_parameters_give_minus_inf_not_an_exception():
    """EmaxHill now validates ec50 and hill, and samplers step anywhere.

    Raising out of the likelihood would abort an optimiser that merely tried a
    point outside the support; the point simply has zero density.
    """
    from umimic.data.schemas import TimeSeriesData
    from umimic.inference.likelihood import ModelLikelihood

    topo = ModelTopology.two_state()
    data = TimeSeriesData.from_counts(
        np.linspace(0, 24, 5), [100.0, 110.0, 125.0, 140.0, 160.0],
        concentration=1.0,
    )
    names = ["b0", "d0_P", "emax_death", "ec50_death", "hill_death"]
    lik = ModelLikelihood(topo, data, param_names=names, mode="ode")

    good = np.array([0.05, 0.01, 0.02, 1.0, 1.5])
    assert np.isfinite(lik(good))

    for bad in (
        np.array([0.05, 0.01, 0.02, 0.0, 1.5]),   # ec50 = 0
        np.array([0.05, 0.01, 0.02, 1.0, 0.0]),   # hill = 0
    ):
        assert lik(bad) == -np.inf


def test_the_quiescent_death_fallback_announces_itself(caplog):
    """d0_Q = 0.5 * d0_P is a modelling choice, not a neutral default.

    Nothing downstream distinguishes an estimated d0_Q from an imputed one,
    so the substitution has to be visible at construction.
    """
    import logging

    from umimic.data.schemas import TimeSeriesData
    from umimic.inference.likelihood import ModelLikelihood

    topo = ModelTopology.two_state()
    data = TimeSeriesData.from_counts(np.linspace(0, 24, 4), [100.0] * 4)

    with caplog.at_level(logging.WARNING, logger="umimic.inference.likelihood"):
        ModelLikelihood(topo, data, param_names=["b0", "d0_P"], mode="ode")
    assert "fixed at 0.5 * d0_P" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="umimic.inference.likelihood"):
        ModelLikelihood(topo, data, param_names=["b0", "d0_P", "d0_Q"], mode="ode")
    assert "fixed at 0.5 * d0_P" not in caplog.text


def test_exposure_profile_advertises_no_unused_cache():
    """precompute() filled a cache concentration() never read.

    It promised a speed-up it did not deliver, so it was removed rather than
    wired to an interpolation whose error the caller could not see.
    """
    from umimic.pk.exposure import ExposureProfile

    assert not hasattr(ExposureProfile, "precompute")
