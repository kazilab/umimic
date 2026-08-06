"""Tests that the inference side describes the same model as the simulation side.

The forward model gained resistance, persistence and plasticity before the
likelihood did. A state present in the topology but absent from the parameter
vector does not sit still -- it inherits the shared birth rate and gets no
death rate, becoming an immortal, fully fit clone that absorbs the whole
trajectory. The fit still converges, which is what makes it dangerous.
"""

from __future__ import annotations

import numpy as np
import pytest

from umimic.data.schemas import TimeSeriesData
from umimic.dynamics.rates import RateSet
from umimic.dynamics.states import CellType, ModelTopology
from umimic.inference.likelihood import (
    DEFAULT_PARAM_NAMES,
    PARAMETER_SETS,
    PERSISTER_PARAM_NAMES,
    RESISTANCE_PARAM_NAMES,
    ModelLikelihood,
)
from umimic.inference.priors import PriorSpec
from umimic.pipeline.config import DynamicsConfig, ExperimentConfig
from umimic.pipeline.experiment import Experiment


@pytest.fixture
def counts():
    return TimeSeriesData.from_counts(np.linspace(0, 48, 5), [100.0] * 5)


# ---------------------------------------------------------------------------
# Resistance is parameterised at fit time
# ---------------------------------------------------------------------------


def test_unparameterised_resistant_state_is_rejected(counts):
    """It used to fit R as an immortal, fully fit clone."""
    with pytest.raises(ValueError, match="immortal"):
        ModelLikelihood(ModelTopology.four_state(), counts, mode="ode")


def test_existing_topologies_are_unaffected(counts):
    """Q has a documented death fallback, so P/Q models must still work."""
    for topology in (ModelTopology.two_state(), ModelTopology.three_state()):
        assert ModelLikelihood(topology, counts, mode="ode") is not None


@pytest.mark.parametrize(
    "topology, param_set",
    [
        (ModelTopology.four_state(), "resistance"),
        (ModelTopology.persister_resistance(), "persister"),
    ],
)
def test_matching_parameter_sets_are_accepted(counts, topology, param_set):
    assert ModelLikelihood(
        topology, counts, param_names=PARAMETER_SETS[param_set], mode="ode"
    ) is not None


def test_resistant_state_gets_its_own_rates_at_fit_time(counts):
    topology = ModelTopology.four_state()
    likelihood = ModelLikelihood(
        topology, counts, param_names=RESISTANCE_PARAM_NAMES, mode="ode"
    )
    params = dict(
        zip(
            RESISTANCE_PARAM_NAMES,
            [0.04, 0.030, 0.01, 0.008, 0.06, 1.0, 1.5, 0.005, 0.005, 0.003, 1e-4, 10.0],
        )
    )
    rates = likelihood._build_rate_set(params)

    # Own division rate, not P's.
    assert rates.birth_rate(0.0, cell_type=CellType.R) == pytest.approx(0.030)
    assert rates.birth_rate(0.0, cell_type=CellType.P) == pytest.approx(0.04)
    # Mortal, and only partially drug-sensitive.
    assert rates.death_rate(CellType.R, 0.0) == pytest.approx(0.008)
    assert 0.008 < rates.death_rate(CellType.R, 100.0) < rates.death_rate(
        CellType.P, 100.0
    )


def test_resistant_state_does_not_inherit_cytostatic_suppression(counts):
    """R's drug response is its own; it must not pick up P's birth inhibition."""
    topology = ModelTopology.four_state()
    likelihood = ModelLikelihood(
        topology, counts, param_names=RESISTANCE_PARAM_NAMES, mode="ode"
    )
    params = dict.fromkeys(RESISTANCE_PARAM_NAMES, 0.0)
    params.update(
        b0=0.04, b0_R=0.030, d0_P=0.01, d0_R=0.008,
        emax_death=0.06, ec50_death=1.0, hill_death=1.5, overdispersion=10.0,
        emax_birth=0.9,
    )
    rates = likelihood._build_rate_set(params)
    assert rates.birth_rate(1e4, cell_type=CellType.R) == pytest.approx(0.030)


def test_persister_set_carries_the_induced_route_and_Q_sensitivity(counts):
    topology = ModelTopology.persister_resistance()
    likelihood = ModelLikelihood(
        topology, counts, param_names=PERSISTER_PARAM_NAMES, mode="ode"
    )
    params = dict.fromkeys(PERSISTER_PARAM_NAMES, 0.0)
    params.update(
        b0=0.04, b0_Q=0.002, b0_R=0.036, d0_P=0.01, d0_Q=0.003, d0_R=0.01,
        emax_death=0.06, ec50_death=1.0, hill_death=1.5,
        emax_death_Q=0.004, emax_death_R=0.03,
        u_PQ=1e-4, u_QP=3e-3, u_QR=0.0, induced_QR=2e-4, overdispersion=10.0,
    )
    rates = likelihood._build_rate_set(params)

    # Persisters divide slowly and are partly sensitive.
    assert rates.birth_rate(0.0, cell_type=CellType.Q) == pytest.approx(0.002)
    assert rates.death_rate(CellType.Q, 100.0) > rates.death_rate(CellType.Q, 0.0)
    # The persister route appears only under drug.
    assert rates.transition_rate(CellType.Q, CellType.R, 0.0) == 0.0
    assert rates.transition_rate(CellType.Q, CellType.R, 100.0) > 0.0


def test_every_parameter_set_has_matching_priors():
    for label, names in PARAMETER_SETS.items():
        spec = {
            "default": PriorSpec.default_invitro,
            "mechanism": PriorSpec.default_mechanism,
            "resistance": PriorSpec.default_resistance,
            "persister": PriorSpec.default_persister,
        }[label]()
        missing = [n for n in names if n not in spec.distributions]
        assert not missing, f"{label} priors missing {missing}"


def test_a_resistance_fit_actually_runs(counts):
    """End to end: the likelihood must be finite on the resistance set."""
    topology = ModelTopology.four_state()
    likelihood = ModelLikelihood(
        topology, counts, param_names=RESISTANCE_PARAM_NAMES, mode="ode"
    )
    theta = likelihood.params_to_theta(
        dict(
            zip(
                RESISTANCE_PARAM_NAMES,
                [0.04, 0.03, 0.01, 0.008, 0.06, 1.0, 1.5, 0.005, 0.005, 0.003,
                 1e-4, 10.0],
            )
        )
    )
    assert np.isfinite(likelihood(theta))


# ---------------------------------------------------------------------------
# Configuration coupling
# ---------------------------------------------------------------------------


def test_cytostatic_mechanism_requires_a_birth_parameter_set():
    """Otherwise the fit attributes reduced division to increased death."""
    with pytest.raises(ValueError, match="no birth-modulation"):
        ExperimentConfig(
            dynamics={"drug_mechanism": "cytostatic"},
            inference={"parameter_set": "default"},
        )
    # The matching combination is fine.
    ExperimentConfig(
        dynamics={"drug_mechanism": "cytostatic"},
        inference={"parameter_set": "mechanism", "forward_mode": "moment"},
    )


def test_mechanism_set_requires_the_moment_forward_model():
    """The mean confounds b and d; the LNA variance is what separates them."""
    with pytest.raises(ValueError, match="forward_mode='moment'"):
        ExperimentConfig(
            inference={"parameter_set": "mechanism", "forward_mode": "ode"}
        )


def test_simulation_only_config_with_R_is_still_allowed():
    """Forward-only use is legitimate; the fit-time check belongs at fit time."""
    config = ExperimentConfig()
    config.dynamics.states = ["P", "Q", "A", "R"]
    assert Experiment(config).topology.n_states == 4


def test_quiescent_sensitivity_reaches_the_rate_set():
    refractory = ExperimentConfig()
    refractory.dynamics.states = ["P", "Q"]
    refractory.dynamics.drug_mechanism = "cytotoxic"
    refractory.dynamics.emax_death = 0.2

    sensitive = refractory.model_copy(deep=True)
    sensitive.dynamics.quiescent_sensitivity = 0.5

    base = Experiment(refractory).rate_set.death_rate(CellType.Q, 0.0)
    assert Experiment(refractory).rate_set.death_rate(CellType.Q, 100.0) == pytest.approx(
        base
    )
    assert Experiment(sensitive).rate_set.death_rate(CellType.Q, 100.0) > base


# ---------------------------------------------------------------------------
# Initial conditions in the high-level API
# ---------------------------------------------------------------------------


def test_default_initial_state_is_all_proliferating():
    config = ExperimentConfig()
    exp = Experiment(config)
    y0 = exp._initial_state_vector(100.0)
    np.testing.assert_allclose(y0, [100.0, 0.0])


def test_stable_initial_fractions_reach_simulation():
    config = ExperimentConfig()
    config.dynamics.initial_fractions = "stable"
    exp = Experiment(config)
    y0 = exp._initial_state_vector(100.0)
    assert y0.sum() == pytest.approx(100.0)
    assert y0[0] > 0 and y0[1] > 0, "a relaxed culture is not pure P"


def test_explicit_initial_fractions_are_normalised():
    config = ExperimentConfig()
    config.dynamics.initial_fractions = {"P": 3.0, "Q": 1.0}
    y0 = Experiment(config)._initial_state_vector(100.0)
    np.testing.assert_allclose(y0, [75.0, 25.0])


def test_initial_fractions_reach_the_fit(counts):
    config = ExperimentConfig()
    config.dynamics.initial_fractions = "stable"
    exp = Experiment(config)
    model = exp._likelihood_observation_model([counts])
    likelihood = ModelLikelihood(
        exp.topology, counts, mode="ode", observation_model=model,
        initial_fractions=config.dynamics.initial_fractions,
    )
    state = likelihood._initial_state(counts, exp.rate_set)
    assert state[1] > 0, "initial_fractions did not reach the likelihood"


def test_invalid_initial_fractions_are_rejected():
    with pytest.raises(ValueError, match="initial_fractions"):
        DynamicsConfig(initial_fractions="nonsense")
    with pytest.raises(ValueError, match="not in states"):
        DynamicsConfig(states=["P", "Q"], initial_fractions={"R": 1.0})
    with pytest.raises(ValueError, match="positive"):
        DynamicsConfig(states=["P", "Q"], initial_fractions={"P": 0.0})


# ---------------------------------------------------------------------------
# Synthetic generation keeps every modality
# ---------------------------------------------------------------------------


def test_generate_synthetic_keeps_all_configured_modalities():
    """It previously took only the first observation model."""
    config = ExperimentConfig()
    config.observations.modalities = ["cell_counts", "bli", "volume"]
    config.simulation.t_max = 24.0
    config.simulation.dt_obs = 8.0
    config.simulation.n_replicates = 1
    config.simulation.method = "ode"
    config.dosing.concentrations = [0.0]

    dataset = Experiment(config).generate_synthetic()
    assert set(dataset.series[0].modalities) == {"cell_counts", "bli", "volume"}


def test_generated_multimodal_data_round_trips_into_a_fit():
    """Simulate-then-fit must exercise the experiment that was described."""
    config = ExperimentConfig()
    config.observations.modalities = ["cell_counts", "bli", "volume"]
    config.simulation.t_max = 24.0
    config.simulation.dt_obs = 8.0
    config.simulation.n_replicates = 1
    config.simulation.method = "ode"
    config.dosing.concentrations = [0.0]

    exp = Experiment(config)
    dataset = exp.generate_synthetic()
    model = exp._likelihood_observation_model(dataset.series)
    assert set(model.models) == {"cell_counts", "bli", "volume"}


# ---------------------------------------------------------------------------
# Diagnostic helper agrees with the solvers
# ---------------------------------------------------------------------------


def test_all_rates_at_uses_the_same_density_total_as_the_solvers():
    """Summing corpses reported birth = 0 with viable N far below K."""
    topology = ModelTopology.three_state()  # P, Q, A
    topology.density_dependent = True
    topology.carrying_capacity = 1000.0
    rates = RateSet(birth_base=0.05)

    # 100 viable cells, 900 corpses: well below K on any viable measure.
    state = np.array([100.0, 0.0, 900.0])
    reported = rates.all_rates_at(0.0, state, topology)
    assert reported["birth"] > 0.0

    # And it agrees with what the ODE actually uses.
    expected = rates.birth_rate(
        0.0, topology.density_total(state), topology.carrying_capacity,
        cell_type=CellType.P,
    )
    assert reported["birth"] == pytest.approx(expected)


def test_ode_docstring_describes_resistant_division():
    """The documented dR/dt omitted the birth term the code applies."""
    from umimic.dynamics.ode_system import CellDynamicsODE

    doc = CellDynamicsODE.__doc__
    line = next(line for line in doc.splitlines() if "dR/dt" in line)
    assert "bR" in line, f"dR/dt still omits R division: {line.strip()}"


def test_default_param_set_is_unchanged():
    """Guard against silently altering everyone's default model."""
    assert DEFAULT_PARAM_NAMES == [
        "b0", "d0_P", "emax_death", "ec50_death", "hill_death",
        "u_PQ", "u_QP", "overdispersion",
    ]


# ---------------------------------------------------------------------------
# Known limitations that needed more than documentation
# ---------------------------------------------------------------------------


def test_bli_attenuation_uses_per_time_point_volume():
    """A single mean volume erases the size-dependence being modelled."""
    from umimic.observations.bli import BLIObservation
    from umimic.observations.cell_counts import CellCountObservation
    from umimic.observations.multimodal import MultimodalObservation
    from umimic.observations.tumor_volume import TumorVolumeObservation
    from umimic.pk.luciferin import TissueAttenuation

    topology = ModelTopology.two_state()
    times = np.linspace(0, 96, 5)
    counts = np.array([5e4, 1.5e5, 4e5, 9e5, 1.6e6])
    data = TimeSeriesData(
        times=times,
        observations={
            "cell_counts": counts,
            "bli": counts * 1000.0,
            "volume": counts * 1e-5,
        },
        concentration=0.0,
    )
    models = MultimodalObservation(
        {
            "cell_counts": CellCountObservation(10.0, topology=topology),
            "bli": BLIObservation(
                alpha=1000.0,
                attenuation=TissueAttenuation(mu_eff=0.5, reference_depth=2.0),
                topology=topology,
            ),
            "volume": TumorVolumeObservation(beta=1e-5, topology=topology),
        }
    )
    likelihood = ModelLikelihood(
        topology, data, mode="ode", observation_model=models
    )
    theta = np.array([0.04, 0.01, 0.0, 1.0, 1.5, 0.005, 0.003, 10.0])
    params = likelihood.theta_to_params(theta)
    obs_params = likelihood._modality_params(
        "bli", params, data, np.arange(5), likelihood.scored_mask(data, "bli")
    )

    volumes = obs_params["tumor_volume"]
    assert isinstance(volumes, np.ndarray)
    assert volumes.shape == (5,)
    assert volumes[0] != volumes[-1], "volume collapsed to a single value"
    assert np.isfinite(likelihood(theta))


def test_bli_batch_indexes_array_valued_covariates():
    from umimic.observations.bli import BLIObservation
    from umimic.pk.luciferin import TissueAttenuation

    topology = ModelTopology.two_state()
    model = BLIObservation(
        alpha=1000.0,
        attenuation=TissueAttenuation(mu_eff=0.5, reference_depth=2.0),
        topology=topology,
    )
    states = np.array([[1e4, 0.0], [1e5, 0.0], [1e6, 0.0]])
    observed = np.array([1e6, 1e7, 1e8])

    growing = model.log_likelihood_batch(
        observed, states, {"tumor_volume": np.array([1.0, 10.0, 100.0])}
    )
    constant = model.log_likelihood_batch(
        observed, states, {"tumor_volume": 37.0}
    )
    assert np.isfinite(growing)
    assert growing != pytest.approx(constant)


def test_small_count_gaussian_branch_warns(caplog):
    """Matched moments do not make a Gaussian a negative binomial."""
    import logging

    from umimic.observations.cell_counts import CellCountObservation

    CellCountObservation._warned_small_counts = False
    model = CellCountObservation(10.0, topology=ModelTopology.two_state())
    with caplog.at_level(logging.WARNING):
        model.log_likelihood(5.0, np.array([5.0, 0.0]), None, 3.0)
    assert any("small counts" in r.message for r in caplog.records)


def test_large_counts_do_not_warn(caplog):
    import logging

    from umimic.observations.cell_counts import CellCountObservation

    CellCountObservation._warned_small_counts = False
    model = CellCountObservation(10.0, topology=ModelTopology.two_state())
    with caplog.at_level(logging.WARNING):
        model.log_likelihood(500.0, np.array([500.0, 0.0]), None, 100.0)
    assert not any("small counts" in r.message for r in caplog.records)


def test_biomarker_modality_is_reachable_from_configuration():
    """BiomarkerObservation existed but the builder had no branch for it."""
    config = ExperimentConfig()
    config.observations.modalities = ["cell_counts", "biomarker"]
    config.observations.biomarker_type = "ki67"

    exp = Experiment(config)
    assert "biomarker" in exp.observation_model.models
    fraction = exp.observation_model.models["biomarker"]._get_fraction(
        np.array([75.0, 25.0])
    )
    assert fraction == pytest.approx(0.75)


def test_unknown_modality_is_rejected_rather_than_dropped():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ExperimentConfig(observations={"modalities": ["bogus"]})
    with pytest.raises(pydantic.ValidationError):
        ExperimentConfig(observations={"modalities": []})
