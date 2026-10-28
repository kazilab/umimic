"""inference.mode pmcmc: particle MCMC on the configured model, every series."""

from __future__ import annotations

import numpy as np
import pytest

from umimic.inference.smc import ParticleBudgetExceeded, ParticleMCMC
from umimic.pipeline.config import ExperimentConfig
from umimic.pipeline.experiment import Experiment


def _config(**dynamics):
    return ExperimentConfig(
        seed=2,
        dynamics={"states": ["P", "Q"], "drug_mechanism": "cytotoxic", **dynamics},
        dosing={"type": "constant", "concentrations": [0.0, 1.0]},
        simulation={"method": "ode", "t_max": 16.0, "dt_obs": 8.0, "n_replicates": 1, "initial_cells": 50},
        inference={"mode": "pmcmc", "n_particles": 10, "n_warmup": 5, "n_samples": 10, "forward_mode": "ode"},
    )


@pytest.fixture(scope="module")
def dataset():
    return Experiment(_config()).generate_synthetic()


def test_pmcmc_runs_through_experiment_fit(dataset):
    seen = []
    result = Experiment(_config()).fit(dataset, progress=lambda *step: seen.append(step))
    assert result.method == "pmcmc"
    assert all(draws.shape == (1, 10) for draws in result.mcmc.samples.values())
    assert result.mcmc.diagnostics["series"] == dataset.n_series
    assert result.mcmc.diagnostics["n_events"] > 0
    assert result.metadata["inference_options"]["n_particles"] == 10
    assert seen[0] == ("warmup", 1, 5) and seen[-1] == ("production", 10, 10) and len(seen) == 15


def _estimate(experiment, dataset, params):
    likelihood = experiment._filter_likelihood(experiment._with_initial_cells(list(dataset.series)))
    sampler = ParticleMCMC.from_likelihood(
        likelihood, priors=None, n_particles=20, rng=np.random.default_rng(4)
    )
    return sampler, sampler._particle_filter_ll(params)


def test_pmcmc_uses_the_configured_base_rates(dataset):
    params = {"b0": 0.04, "d0_P": 0.01}
    _, open_growth = _estimate(Experiment(_config()), dataset, params)
    _, crowded = _estimate(Experiment(_config(density_dependent=True, carrying_capacity=60.0)), dataset, params)
    # Carrying capacity is not a sampled parameter; the direct constructor
    # ignored it, the likelihood route keeps it.
    assert open_growth != crowded


def test_pmcmc_scores_every_series(dataset, monkeypatch):
    experiment = Experiment(_config())
    likelihood = experiment._filter_likelihood(experiment._with_initial_cells(list(dataset.series)))
    calls = []
    original = type(likelihood).particle_filter

    def counted(self, params, series, *args, **kwargs):
        calls.append(series)
        return original(self, params, series, *args, **kwargs)

    monkeypatch.setattr(type(likelihood), "particle_filter", counted)
    sampler = ParticleMCMC.from_likelihood(likelihood, priors=None, n_particles=5, rng=np.random.default_rng(1))
    assert np.isfinite(sampler._particle_filter_ll({"b0": 0.04}))
    assert len(calls) == dataset.n_series


def test_pmcmc_stops_at_its_event_budget(dataset):
    with pytest.raises(ParticleBudgetExceeded):
        Experiment(_config()).fit(dataset, max_total_events=50)


def test_particle_backend_is_accepted_for_pmcmc_only():
    assert ExperimentConfig(inference={"mode": "pmcmc", "backend": "particle"}).inference.mode == "pmcmc"
    with pytest.raises(ValueError, match="pmcmc"):
        ExperimentConfig(inference={"mode": "mle", "backend": "particle"})
