"""Modelling-assumption regression tests.

These pin biological behaviour rather than numerics: a resistant clone must be
able to outgrow the sensitive population, corpses must not crowd out division,
scales must be physically plausible, and every capability the configuration
advertises must actually reach the solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from umimic.data.schemas import ExperimentalDataset, TimeSeriesData
from umimic.dynamics.ode_system import CellDynamicsODE
from umimic.dynamics.rates import RateSet
from umimic.dynamics.states import CellType, ModelTopology
from umimic.inference.likelihood import (
    DEFAULT_PARAM_NAMES,
    MECHANISM_PARAM_NAMES,
    PARAMETER_SETS,
)
from umimic.pipeline.config import DynamicsConfig, ExperimentConfig
from umimic.pipeline.experiment import Experiment


# ---------------------------------------------------------------------------
# 1. The resistant state must be a clone, not an inert sink
# ---------------------------------------------------------------------------


def test_resistant_state_can_proliferate():
    topo = ModelTopology.four_state()
    assert CellType.R in topo.division_states, (
        "R must divide; a non-proliferating resistant compartment is an "
        "absorbing sink that can never form a self-sustaining clone."
    )


def test_resistant_state_is_not_immortal_by_default():
    """R in death_states with no death_base entry gives death_rate(R) = 0."""
    rates = RateSet.resistant_clone()
    assert rates.death_rate(CellType.R, 0.0) > 0

    config = ExperimentConfig()
    config.dynamics.states = ["P", "Q", "A", "R"]
    exp = Experiment(config)
    assert exp.rate_set.death_rate(CellType.R, 0.0) > 0


def test_resistant_clone_escapes_drug_and_outgrows_sensitive_cells():
    """Under sustained drug, R must expand while P collapses."""
    topo = ModelTopology.four_state()
    rates = RateSet.resistant_clone(
        b0=0.05, d0=0.01, emax_death=0.30, ec50_death=1.0, resistance=1.0
    )
    ode = CellDynamicsODE(rates, topo, lambda t: 10.0)
    result = ode.solve(
        np.array([1000.0, 100.0, 0.0, 10.0]), (0, 300), np.array([0.0, 300.0])
    )
    assert result.populations["P"][-1] < 1.0
    assert result.populations["R"][-1] > 1000.0


def test_resistance_parameter_controls_drug_sensitivity():
    partial = RateSet.resistant_clone(emax_death=0.3, resistance=0.5)
    full = RateSet.resistant_clone(emax_death=0.3, resistance=1.0)
    d_p = partial.death_rate(CellType.P, 100.0)
    assert partial.death_rate(CellType.R, 100.0) < d_p
    assert full.death_rate(CellType.R, 100.0) < partial.death_rate(CellType.R, 100.0)


def test_fitness_cost_slows_the_resistant_clone():
    free = RateSet.resistant_clone(b0=0.05, fitness_cost=0.0)
    costly = RateSet.resistant_clone(b0=0.05, fitness_cost=0.4)
    assert free.birth_rate(0.0, cell_type=CellType.R) == pytest.approx(0.05)
    assert costly.birth_rate(0.0, cell_type=CellType.R) == pytest.approx(0.03)


def test_resistant_clone_ignores_cytostatic_birth_suppression():
    """Birth modulation applied to P must not silently suppress R."""
    from umimic.dynamics.rates import EmaxHill

    rates = RateSet.resistant_clone(b0=0.05)
    rates.birth_modulation = EmaxHill(emax=0.9, ec50=1.0, hill=1.0)
    assert rates.birth_rate(100.0, cell_type=CellType.P) < 0.01
    assert rates.birth_rate(100.0, cell_type=CellType.R) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 2. Density dependence must not count corpses
# ---------------------------------------------------------------------------


def test_density_mask_excludes_apoptotic_cells_by_default():
    topo = ModelTopology.three_state()  # [P, Q, A]
    np.testing.assert_allclose(topo.density_mask, [1.0, 1.0, 0.0])
    assert topo.density_total(np.array([100.0, 50.0, 400.0])) == 150.0


def test_corpses_do_not_suppress_regrowth_after_a_cytotoxic_pulse():
    rates = RateSet(
        birth_base=0.05,
        death_base={CellType.P: 0.01, CellType.Q: 0.005},
        clearance_rate=0.005,
    )
    populations = {}
    for counts_apoptotic in (True, False):
        topo = ModelTopology.three_state()
        topo.density_dependent = True
        topo.carrying_capacity = 1000.0
        topo.density_counts_apoptotic = counts_apoptotic
        ode = CellDynamicsODE(rates, topo, lambda t: 0.0)
        result = ode.solve(
            np.array([300.0, 0.0, 400.0]), (0, 120), np.array([0.0, 120.0])
        )
        populations[counts_apoptotic] = result.populations["P"][-1]

    # Well below K, viable cells must grow. Counting corpses makes them shrink.
    assert populations[False] > 300.0
    assert populations[False] > populations[True]


def test_density_counts_apoptotic_is_configurable():
    topo = ModelTopology.three_state()
    topo.density_counts_apoptotic = True
    assert topo.density_total(np.array([100.0, 50.0, 400.0])) == 550.0


def test_moment_jacobian_density_gradient_is_zero_in_apoptotic_column():
    """d(crowding)/d(A) = 0 when corpses do not occupy space."""
    from umimic.dynamics.moment_equations import MomentODE

    topo = ModelTopology.three_state()
    topo.density_dependent = True
    topo.carrying_capacity = 1000.0
    rates = RateSet(birth_base=0.05, death_base={CellType.P: 0.01})
    moment = MomentODE(rates, topo, lambda t: 0.0)

    J = moment.jacobian(0.0, np.array([200.0, 50.0, 100.0]))
    p_idx = topo.state_index(CellType.P)
    a_idx = topo.state_index(CellType.A)
    # The birth row's dependence on A comes only through the density term.
    assert J[p_idx, a_idx] == pytest.approx(0.0)


def test_moment_and_ode_means_still_agree_with_density_dependence():
    from umimic.dynamics.moment_equations import MomentODE

    topo = ModelTopology.three_state()
    topo.density_dependent = True
    topo.carrying_capacity = 2000.0
    rates = RateSet(
        birth_base=0.05,
        death_base={CellType.P: 0.01, CellType.Q: 0.005},
        clearance_rate=0.05,
    )
    mu0 = np.array([300.0, 30.0, 20.0])
    times = np.linspace(0, 96, 17)

    _, means, _ = MomentODE(rates, topo, lambda t: 0.0).solve(
        mu0, t_span=(0, 96), t_eval=times
    )
    ode = CellDynamicsODE(rates, topo, lambda t: 0.0).solve(mu0, (0, 96), times)
    np.testing.assert_allclose(means[:, 0], ode.populations["P"], rtol=1e-4)


# ---------------------------------------------------------------------------
# 3. Volume scale must be physically plausible
# ---------------------------------------------------------------------------


def test_default_volume_per_cell_is_physically_plausible():
    """beta must imply ~1e5-1e6 cells/mm^3, not ~1e3."""
    from umimic.observations.tumor_volume import TumorVolumeObservation

    for beta in (
        ExperimentConfig().observations.volume_beta,
        TumorVolumeObservation().beta,
    ):
        cells_per_mm3 = 1.0 / beta
        assert 0.99e5 <= cells_per_mm3 <= 1.01e6, (
            f"beta={beta} implies {cells_per_mm3:.3g} cells/mm^3, outside the "
            "plausible range for tumour tissue."
        )


def test_hundred_cubic_mm_tumour_is_about_ten_million_cells():
    from umimic.observations.tumor_volume import TumorVolumeObservation

    model = TumorVolumeObservation(topology=ModelTopology.two_state())
    n_cells = 1e7
    volume = model.expected_value(np.array([n_cells, 0.0]))
    assert 30.0 < volume < 300.0


# ---------------------------------------------------------------------------
# 4. Birth/death separation must be reachable on the default path
# ---------------------------------------------------------------------------


def test_default_forward_mode_supplies_process_variance():
    """LNA variance is what separates birth from death; it must be on by default."""
    assert ExperimentConfig().inference.forward_mode == "moment"


def test_forward_mode_is_independent_of_inference_mode():
    config = ExperimentConfig()
    config.inference.mode = "mle"
    assert config.inference.forward_mode == "moment"


def test_default_parameter_set_cannot_express_cytostatic_action():
    """Documented limitation: the default set has no birth-modulation terms."""
    assert not any("birth" in name for name in DEFAULT_PARAM_NAMES)
    assert any("birth" in name for name in MECHANISM_PARAM_NAMES)
    assert set(DEFAULT_PARAM_NAMES) < set(MECHANISM_PARAM_NAMES)


def test_mechanism_parameter_set_is_selectable_and_has_priors():
    from umimic.inference.priors import PriorSpec

    config = ExperimentConfig()
    config.inference.parameter_set = "mechanism"
    assert PARAMETER_SETS[config.inference.parameter_set] is MECHANISM_PARAM_NAMES

    priors = PriorSpec.default_mechanism()
    missing = [p for p in MECHANISM_PARAM_NAMES if p not in priors.distributions]
    assert not missing, f"mechanism priors missing {missing}"


def test_mechanism_set_distinguishes_cytostatic_from_cytotoxic():
    """Cytostatic and cytotoxic data must not give the same best fit."""
    from umimic.inference.likelihood import ModelLikelihood

    topo = ModelTopology.two_state()
    times = np.linspace(0, 72, 10)

    def series(rates):
        ode = CellDynamicsODE(rates, topo, lambda t: 5.0)
        out = ode.solve(np.array([500.0, 0.0]), (0, 72), times)
        counts = out.populations["P"] + out.populations["Q"]
        return TimeSeriesData(
            times=times, observations={"cell_counts": counts}, concentration=5.0
        )

    cytotoxic = series(RateSet.cytotoxic_drug(b0=0.05, emax_death=0.04))
    cytostatic = series(RateSet.cytostatic_drug(b0=0.05, emax_birth=0.7))

    theta = np.array([0.05, 0.01, 0.04, 1.0, 1.5, 0.0, 1.0, 1.5, 0.005, 0.003, 10.0])
    ll_toxic = ModelLikelihood(
        topo, cytotoxic, param_names=MECHANISM_PARAM_NAMES, mode="moment"
    )(theta)
    ll_static = ModelLikelihood(
        topo, cytostatic, param_names=MECHANISM_PARAM_NAMES, mode="moment"
    )(theta)
    # A cytotoxic parameter vector must fit cytotoxic data better than
    # cytostatic data; if the two were indistinguishable these would match.
    assert ll_toxic > ll_static


# ---------------------------------------------------------------------------
# 5. The orchestrator must use the observation models it builds
# ---------------------------------------------------------------------------


def _multimodal_dataset():
    t = np.linspace(0, 48, 7)
    counts = np.array([100.0, 120.0, 150.0, 185.0, 225.0, 270.0, 330.0])
    return ExperimentalDataset(
        series=[
            TimeSeriesData(
                times=t,
                observations={
                    "cell_counts": counts,
                    "bli": counts * 1000.0,
                    "volume": counts * 1e-5,
                },
                concentration=0.0,
            )
        ]
    )


def test_experiment_fit_uses_all_configured_modalities():
    config = ExperimentConfig()
    config.observations.modalities = ["cell_counts", "bli", "volume"]
    exp = Experiment(config)

    model = exp._likelihood_observation_model(_multimodal_dataset().series)
    assert set(model.models) == {"cell_counts", "bli", "volume"}


def test_experiment_fit_likelihood_is_sensitive_to_every_modality():
    from umimic.inference.likelihood import ModelLikelihood

    config = ExperimentConfig()
    config.observations.modalities = ["cell_counts", "bli", "volume"]
    exp = Experiment(config)
    dataset = _multimodal_dataset()
    model = exp._likelihood_observation_model(dataset.series)

    lik = ModelLikelihood(
        exp.topology, dataset.series, mode="moment", observation_model=model
    )
    assert set(lik._active_modalities) == {"cell_counts", "bli", "volume"}


def test_configured_modality_absent_from_data_is_reported():
    config = ExperimentConfig()
    config.observations.modalities = ["cell_counts"]
    exp = Experiment(config)
    t = np.linspace(0, 48, 5)
    only_bli = [
        TimeSeriesData(times=t, observations={"bli": np.full(5, 1e5)}, concentration=0.0)
    ]
    with pytest.raises(ValueError, match="configured modalities"):
        exp._likelihood_observation_model(only_bli)


# ---------------------------------------------------------------------------
# 6. Unimplemented options must fail, not no-op
# ---------------------------------------------------------------------------


def test_linear_chain_is_rejected_rather_than_silently_ignored():
    with pytest.raises(ValueError, match="not implemented"):
        DynamicsConfig(linear_chain={"P": 3})


def test_empty_linear_chain_is_accepted():
    assert DynamicsConfig(linear_chain=None).linear_chain is None
    assert DynamicsConfig(linear_chain={}).linear_chain == {}


def test_withdrawn_pymc_backend_is_not_a_configuration_choice():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ExperimentConfig(inference={"backend": "pymc"})
