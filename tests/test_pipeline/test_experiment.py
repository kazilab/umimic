"""Tests for the Experiment orchestrator."""

import numpy as np
import pytest

from umimic.pipeline.config import ExperimentConfig
from umimic.pipeline.experiment import Experiment
from umimic.dynamics.rates import RateSet


class TestExperiment:
    def test_create_experiment(self):
        """Experiment should initialize from default config."""
        config = ExperimentConfig()
        exp = Experiment(config)
        assert exp.topology is not None
        assert exp.observation_model is not None

    def test_simulate_ode(self):
        """ODE simulation should produce valid results."""
        config = ExperimentConfig()
        exp = Experiment(config)
        result = exp.simulate(method="ode")

        assert len(result.times) > 0
        assert "P" in result.populations

    def test_simulate_dose_response(self):
        """Dose-response simulation should return results for each concentration."""
        config = ExperimentConfig()
        exp = Experiment(config)
        rs = RateSet.cytotoxic_drug()
        results = exp.simulate(
            rate_set=rs, method="ode",
            concentrations=[0, 1, 10],
        )
        assert 0 in results
        assert 1 in results
        assert 10 in results

    def test_generate_synthetic(self):
        """Synthetic data generation should produce valid dataset."""
        config = ExperimentConfig()
        config.dosing.concentrations = [0, 1, 10]
        config.simulation.n_replicates = 2
        config.simulation.method = "ode"

        exp = Experiment(config)
        dataset = exp.generate_synthetic()

        assert dataset.n_series > 0
        assert all(s.times is not None for s in dataset.series)
