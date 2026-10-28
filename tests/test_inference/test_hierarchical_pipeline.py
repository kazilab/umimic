"""inference.mode hierarchical: groups are series, configured model, resumable."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("emcee")

from umimic.inference.checkpoint import MCMCCheckpoint  # noqa: E402
from umimic.inference.hierarchical import HierarchicalModel  # noqa: E402
from umimic.inference.priors import PriorSpec  # noqa: E402
from umimic.pipeline.config import ExperimentConfig  # noqa: E402
from umimic.pipeline.experiment import Experiment  # noqa: E402


def _config(**dynamics):
    return ExperimentConfig(
        seed=9,
        dynamics={"states": ["P", "Q"], "drug_mechanism": "cytotoxic", **dynamics},
        dosing={"type": "constant", "concentrations": [0.0]},
        simulation={"method": "ode", "t_max": 24.0, "dt_obs": 8.0, "n_replicates": 3, "initial_cells": 100},
        inference={
            "mode": "hierarchical",
            "random_effects": ["b0"],
            "forward_mode": "ode",
            "n_chains": 4,
            "n_warmup": 10,
            "n_samples": 15,
        },
    )


@pytest.fixture(scope="module")
def dataset():
    return Experiment(_config()).generate_synthetic()


class _Stop(BaseException):
    pass


def test_hierarchical_mode_runs_through_fit(dataset):
    seen = []
    result = Experiment(_config()).fit(dataset, progress=lambda *step: seen.append(step))
    assert result.method == "hierarchical"
    names = set(result.mcmc.samples)
    assert {"pop_mean_b0", "pop_sd_b0", "d0_P"} <= names
    assert len([name for name in names if name.startswith("b0_")]) == dataset.n_series
    assert result.mcmc.diagnostics["random_effect_params"] == ["b0"]
    assert result.metadata["inference_options"]["random_effects"] == ["b0"]
    assert seen[0] == ("warmup", 1, 10) and seen[-1] == ("production", 15, 15)


def test_hierarchical_fit_resumes_with_the_same_draws(dataset, tmp_path):
    path = tmp_path / "run.npz"

    def stop_at(phase, done, total):
        if (phase, done) == ("production", 7):
            raise _Stop

    with pytest.raises(_Stop):
        Experiment(_config()).fit(dataset, checkpoint=MCMCCheckpoint(path, every=5), progress=stop_at)
    resumed = Experiment(_config()).fit(dataset, checkpoint=MCMCCheckpoint(path, every=5))
    plain = Experiment(_config()).fit(dataset)
    assert resumed.mcmc.diagnostics["resumed_from"] == {"phase": "production", "steps_done": 5}
    for name, draws in plain.mcmc.samples.items():
        np.testing.assert_array_equal(draws, resumed.mcmc.samples[name])


def test_hierarchical_groups_use_the_configured_rates(dataset):
    crowded = Experiment(_config(density_dependent=True, carrying_capacity=120.0))
    priors = PriorSpec.default_invitro()
    args = (dataset, crowded.topology, ["d0_P"], ["b0"])
    configured = HierarchicalModel(*args, priors=priors, mode="ode", base_rate_set=crowded.rate_set)
    defaults = HierarchicalModel(*args, priors=priors, mode="ode")
    theta = np.concatenate([[0.01], [0.04], [np.log(0.1)], np.zeros(dataset.n_series)])
    assert configured.log_likelihood(theta) != defaults.log_likelihood(theta)


@pytest.mark.parametrize(
    ("random_effects", "message"),
    [([], "needs random_effects"), (["b0_R"], "not in parameter set"), (["b0", "b0"], "twice")],
)
def test_random_effects_are_checked(random_effects, message):
    with pytest.raises(ValueError, match=message):
        ExperimentConfig(inference={"mode": "hierarchical", "random_effects": random_effects})
