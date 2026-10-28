"""Expected precision of a design, and the configured model's parameter values."""

from __future__ import annotations

import numpy as np
import pytest

from umimic.dynamics.states import CellType
from umimic.inference.design import evaluate_design
from umimic.pipeline.config import ExperimentConfig
from umimic.pipeline.experiment import Experiment


@pytest.mark.parametrize(
    ("states", "drug", "parameter_set", "forward_mode"),
    [
        (["P", "Q"], "cytotoxic", "default", "ode"),
        (["P", "Q"], "cytostatic", "mechanism", "moment"),
        (["P", "Q", "A", "R"], "cytotoxic", "resistance", "ode"),
    ],
)
def test_parameter_values_rebuild_the_configured_rates(states, drug, parameter_set, forward_mode):
    experiment = Experiment(
        ExperimentConfig(
            dynamics={"states": states, "drug_mechanism": drug},
            inference={"parameter_set": parameter_set, "forward_mode": forward_mode},
        )
    )
    values = experiment.parameter_values()
    dataset = experiment.generate_synthetic()
    likelihood = experiment._filter_likelihood(experiment._with_initial_cells(list(dataset.series)))
    rebuilt = likelihood._build_rate_set(values)
    original = experiment.rate_set
    assert rebuilt.birth_base == original.birth_base
    assert rebuilt.death_base == original.death_base
    # An explicit zero rate is the same as an absent one.
    assert {edge: rate for edge, rate in rebuilt.transition_base.items() if rate} == {
        edge: rate for edge, rate in original.transition_base.items() if rate
    }
    for concentration in (0.0, 1.0, 10.0):
        assert rebuilt.death_rate(CellType.P, concentration) == pytest.approx(original.death_rate(CellType.P, concentration))


def _design(concentrations, t_max, dt_obs, replicates):
    return Experiment(
        ExperimentConfig(
            dynamics={"states": ["P", "Q"], "drug_mechanism": "cytotoxic", "emax_death": 0.08},
            dosing={"type": "constant", "concentrations": concentrations},
            simulation={"method": "ode", "t_max": t_max, "dt_obs": dt_obs, "n_replicates": replicates, "initial_cells": 2000},
            inference={"forward_mode": "ode"},
        )
    )


def test_a_richer_design_constrains_the_drug_response_better():
    sparse = evaluate_design(_design([0.0, 10.0], 24.0, 12.0, 1))
    rich = evaluate_design(_design([0.0, 0.3, 1.0, 3.0, 10.0], 48.0, 6.0, 3))
    assert rich.n_observations > sparse.n_observations
    for name in ("emax_death", "ec50_death"):
        assert rich.relative_se[name] is not None
        assert sparse.relative_se[name] is None or rich.relative_se[name] < sparse.relative_se[name]
    # Two concentrations cannot separate EC50 from the Hill slope.
    assert sparse.scores["hill_death"] < rich.scores["hill_death"]


def test_more_replicates_shrink_the_errors_by_about_the_square_root():
    one = evaluate_design(_design([0.0, 0.3, 1.0, 3.0, 10.0], 48.0, 6.0, 1))
    four = evaluate_design(_design([0.0, 0.3, 1.0, 3.0, 10.0], 48.0, 6.0, 4))
    ratio = four.relative_se["emax_death"] / one.relative_se["emax_death"]
    assert ratio == pytest.approx(0.5, rel=0.05)


def test_design_is_evaluated_at_given_parameter_values():
    experiment = _design([0.0, 0.3, 1.0, 3.0, 10.0], 48.0, 6.0, 2)
    configured = evaluate_design(experiment)
    weaker = evaluate_design(experiment, {"emax_death": 0.01})
    assert weaker.values["emax_death"] == 0.01
    assert weaker.relative_se["emax_death"] > configured.relative_se["emax_death"]
    with pytest.raises(ValueError, match="not in the configured parameter set"):
        evaluate_design(experiment, {"b0_R": 0.01})
    # Noise is held fixed, at a given value when there is one.
    assert configured.fixed == {"overdispersion": 10.0}
    assert evaluate_design(experiment, {"overdispersion": 50.0}).fixed == {"overdispersion": 50.0}


def test_the_ode_likelihood_confounds_birth_and_death():
    # Mean counts depend on b0 - d0_P only; the split is not identifiable.
    report = evaluate_design(_design([0.0, 0.3, 1.0, 3.0, 10.0], 48.0, 6.0, 2))
    assert report.status["b0"] == report.status["d0_P"] == "confounded"
    assert report.relative_se["b0"] is None
    assert not np.isfinite(report.condition_number) or report.condition_number > 1e6
    assert report.status["emax_death"] == "constrained"
