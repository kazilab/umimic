"""Experiment.filter runs the EKF on each series under the experiment's model."""

from __future__ import annotations

import numpy as np
import pytest

from umimic.pipeline.config import ExperimentConfig
from umimic.pipeline.experiment import Experiment


@pytest.fixture(scope="module")
def experiment_and_data():
    config = ExperimentConfig(
        seed=3,
        dynamics={"states": ["P", "Q"], "drug_mechanism": "cytotoxic"},
        dosing={"type": "constant", "concentrations": [0.0, 1.0]},
        simulation={"method": "ode", "t_max": 48.0, "dt_obs": 8.0, "n_replicates": 2, "initial_cells": 200},
        inference={"forward_mode": "ode"},
    )
    experiment = Experiment(config)
    return experiment, experiment.generate_synthetic()


def test_filters_every_series_from_its_anchor(experiment_and_data):
    experiment, dataset = experiment_and_data
    results = experiment.filter(dataset)
    assert len(results) == dataset.n_series
    for series, result in zip(dataset.series, results):
        assert not result.diverged
        assert np.isfinite(result.marginal_log_likelihood)
        np.testing.assert_array_equal(result.times, series.times)
        assert result.filtered_means.shape == (len(series.times), 2)
        # The anchor reading places the state and is not scored again.
        assert result.filtered_means[0].sum() == pytest.approx(series.observations["cell_counts"][0])
        assert np.all(np.diagonal(result.filtered_covs, axis1=1, axis2=2) >= 0)


def test_parameters_override_the_configured_rates(experiment_and_data):
    experiment, dataset = experiment_and_data
    configured = sum(result.marginal_log_likelihood for result in experiment.filter(dataset))
    slow = sum(result.marginal_log_likelihood for result in experiment.filter(dataset, {"b0": 0.001}))
    assert slow < configured


def test_unknown_parameters_are_refused(experiment_and_data):
    experiment, dataset = experiment_and_data
    with pytest.raises(ValueError, match="not in this likelihood's parameter set"):
        experiment.filter(dataset, {"b0_R": 0.01})


def test_particle_filter_is_seeded_and_reports_its_cost(experiment_and_data):
    experiment, dataset = experiment_and_data
    first = experiment.particle_filter(dataset, n_particles=30)
    again = experiment.particle_filter(dataset, n_particles=30)
    kalman = experiment.filter(dataset)
    assert len(first) == dataset.n_series
    for series, result, repeat, gaussian in zip(dataset.series, first, again, kalman):
        np.testing.assert_array_equal(result["times"], series.times)
        assert result["marginal_log_likelihood"] == repeat["marginal_log_likelihood"]
        assert np.isfinite(result["marginal_log_likelihood"])
        assert result["n_events"] > 0
        assert np.all((result["ess"] > 0) & (result["ess"] <= 30))
        assert np.all(result["viable_vars"] >= 0)
        # Same model and data: the exact and the Gaussian filter agree within
        # their combined spread (plus Monte Carlo error from 30 particles).
        ekf_mean = gaussian.filtered_means.sum(axis=1)
        ekf_sd = np.sqrt(gaussian.filtered_covs.sum(axis=(1, 2)))
        spread = 3 * (np.sqrt(result["viable_vars"]) + ekf_sd)
        assert np.all(np.abs(result["viable_means"] - ekf_mean) < spread)


def test_particle_filter_stops_at_its_event_budget(experiment_and_data):
    from umimic.inference.smc import ParticleBudgetExceeded

    experiment, dataset = experiment_and_data
    with pytest.raises(ParticleBudgetExceeded, match="Series 1 of"):
        experiment.particle_filter(dataset, n_particles=30, max_total_events=100)


def test_particle_filter_scores_marker_fractions_the_kalman_filter_cannot():
    config = ExperimentConfig(
        seed=5,
        dynamics={"states": ["P", "Q"], "drug_mechanism": "cytotoxic"},
        dosing={"type": "constant", "concentrations": [0.0]},
        observations={"modalities": ["cell_counts", "biomarker"]},
        simulation={"method": "ode", "t_max": 24.0, "dt_obs": 8.0, "n_replicates": 1, "initial_cells": 100},
        inference={"forward_mode": "ode"},
    )
    experiment = Experiment(config)
    dataset = experiment.generate_synthetic()
    assert dataset.series[0].has_modality("biomarker")
    with pytest.raises(ValueError, match="no Gaussian linearization"):
        experiment.filter(dataset)
    result = experiment.particle_filter(dataset, n_particles=40)[0]
    assert np.isfinite(result["marginal_log_likelihood"])
    # The marker readings move the estimate: without them it differs.
    counts_only = Experiment(config.model_copy(update={"observations": config.observations.model_copy(update={"modalities": ["cell_counts"]})}))
    assert counts_only.particle_filter(dataset, n_particles=40)[0]["marginal_log_likelihood"] != result["marginal_log_likelihood"]
