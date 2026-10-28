"""Pathway history is independent of the requested observation schedule."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from umimic.pipeline.config import ExperimentConfig
from umimic.pipeline.experiment import Experiment


def signaling_config(schedule, *, delayed_pk=False):
    arguments = {
        "dynamics": {
            "states": ["P"], "transitions": [],
            "default_birth_base": 0.04, "default_death_base_p": 0,
        },
        "signaling": {"enabled": True, "model": "toy_mapk_akt"},
        "coupling": {
            "enabled": True, "targets": ["birth"],
            "parameters": {"max_effect": 5.0, "ec50": 0.2},
        },
        "simulation": {
            "method": "ode", "t_max": 30 if delayed_pk else 10,
            "observation_times": schedule,
        },
        "dosing": {"concentrations": [1]},
    }
    if delayed_pk:
        arguments["pk"] = {"model": "one_compartment", "vd": 1, "ke": 2}
        arguments["dosing"] = {
            "type": "single_bolus", "dose_amount": 100, "start_time": 5,
        }
    return ExperimentConfig(**arguments)


@pytest.mark.parametrize("schedule", [[5, 10], [5], [2, 7]])
def test_sparse_or_single_observation_preserves_pathway_history(schedule):
    reference = Experiment(signaling_config(list(range(11)))).simulate()
    sparse = Experiment(signaling_config(schedule)).simulate()
    np.testing.assert_allclose(
        sparse.populations["P"], reference.populations["P"][schedule], rtol=1e-8,
    )
    assert sparse.metadata["signaling"]["enabled"]


def test_delayed_pk_coupling_matches_joint_numerical_reference():
    times = np.array([0.0, 4.0, 5.0, 6.0, 10.0, 30.0])
    experiment = Experiment(signaling_config(times.tolist(), delayed_pk=True))
    result = experiment.simulate()

    # Independent augmented system: cells and pathway are integrated
    # together, with a small step bound across the delayed treatment.
    def rhs(t, state):
        population, mapk, akt = state
        concentration = experiment.exposure(t)
        target_mapk = np.clip(0.3 - 0.15 * concentration, 0, 1)
        target_akt = np.clip(0.25 - 0.10 * concentration, 0, 1)
        activity = 0.5 * (mapk + akt)
        multiplier = 1 + 5 * activity / (0.2 + activity)
        return [
            0.04 * multiplier * population,
            0.5 * (target_mapk - mapk),
            0.5 * (target_akt - akt),
        ]

    reference = solve_ivp(
        rhs, (0, 30), [100, 0.3, 0.25], t_eval=times,
        max_step=0.02, rtol=1e-10, atol=1e-12,
    )
    assert reference.success
    np.testing.assert_allclose(result.populations["P"], reference.y[0], rtol=1e-5)


def test_delayed_pk_signaling_is_independent_of_observation_density():
    full = Experiment(signaling_config(list(range(31)), delayed_pk=True)).simulate()
    sparse = Experiment(signaling_config([7], delayed_pk=True)).simulate()
    assert sparse.populations["P"][0] == pytest.approx(full.populations["P"][7], rel=1e-8)
