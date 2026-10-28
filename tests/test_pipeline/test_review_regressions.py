"""Regression coverage for configuration, timing and reproducible artifacts."""

from argparse import Namespace
import json

import numpy as np
import pytest

from umimic.data.schemas import TimeSeriesData
from umimic.pipeline.config import ExperimentConfig, load_config, save_config
from umimic.pipeline.experiment import Experiment
from umimic.pipeline.results import dataset_fingerprint, load_result, save_result
from umimic.pipeline.runner import _run_simulate
from umimic.time_grid import observation_grid
from umimic.types import InferenceResult, MCMCResult, MLEResult


@pytest.mark.parametrize("context", ["in_vitro", "in_vivo"])
@pytest.mark.parametrize("method", ["ode", "gillespie", "tau_leaping"])
def test_simulation_and_generation_share_bounded_observations(context, method):
    cfg = ExperimentConfig(
        context=context,
        simulation={"method": method, "t_max": 10, "dt_obs": 4,
                    "initial_cells": 10, "n_replicates": 1},
    )
    result = Experiment(cfg).simulate()
    dataset = Experiment(cfg).generate_synthetic()
    np.testing.assert_array_equal(result.times, [0, 4, 8, 10])
    np.testing.assert_array_equal(dataset.series[0].times, result.times)


@pytest.mark.parametrize("dt", [10, 100])
def test_spacing_longer_than_study_includes_both_endpoints(dt):
    np.testing.assert_array_equal(observation_grid(3, dt), [0, 3])


def test_explicit_observation_schedule_reaches_simulation_and_generation():
    cfg = ExperimentConfig(context="in_vivo", simulation={
        "method": "ode", "t_max": 20, "observation_times": [0, 1, 6, 20],
        "n_replicates": 1,
    })
    np.testing.assert_array_equal(Experiment(cfg).simulate().times, [0, 1, 6, 20])
    np.testing.assert_array_equal(
        Experiment(cfg).generate_synthetic().series[0].times, [0, 1, 6, 20]
    )


@pytest.mark.parametrize("schedule", [[0, 5, 4], [-1, 0], [0, 11], [], [0, np.nan]])
def test_invalid_explicit_observation_schedule_rejected(schedule):
    with pytest.raises(ValueError, match="observation_times"):
        ExperimentConfig(simulation={"t_max": 10, "observation_times": schedule})


def test_invivo_tau_generation_calls_tau_solver(monkeypatch):
    from umimic.dynamics.tau_leaping import TauLeapingSimulator

    original = TauLeapingSimulator.simulate
    calls = []

    def capture(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TauLeapingSimulator, "simulate", capture)
    cfg = ExperimentConfig(context="in_vivo", simulation={
        "method": "tau_leaping", "t_max": 2, "dt_obs": 1, "n_replicates": 1,
    })
    Experiment(cfg).generate_synthetic()
    assert calls == [True]


@pytest.mark.parametrize("method", ["ode", "tau_leaping"])
def test_cli_preserves_configured_rates_and_method(tmp_path, method):
    cfg = ExperimentConfig(
        dynamics={"states": ["P"], "default_birth_base": .002,
                  "default_death_base_p": 0},
        simulation={"method": method, "t_max": 24, "dt_obs": 4},
    )
    path = tmp_path / "experiment.yaml"
    save_config(cfg, path)
    expected = Experiment(cfg).simulate()
    _run_simulate(Namespace(config=str(path), output=str(tmp_path), drug_type=None))
    artifact = json.loads((tmp_path / "simulation.json").read_text())
    simulated = artifact["simulations"][0]
    np.testing.assert_allclose(simulated["populations"]["P"], expected.populations["P"])
    assert simulated["metadata"]["method"] == expected.metadata["method"]
    assert load_config(tmp_path / "config.yaml").dynamics.default_birth_base == .002


def test_cli_preserves_time_varying_pk(tmp_path):
    cfg = ExperimentConfig(
        context="in_vivo",
        dynamics={"states": ["P"], "drug_mechanism": "cytotoxic"},
        pk={"model": "one_compartment"},
        dosing={"type": "single_bolus", "dose_amount": 100, "start_time": 2},
        simulation={"method": "ode", "t_max": 6, "dt_obs": 1},
    )
    path = tmp_path / "experiment.yaml"
    save_config(cfg, path)
    expected = Experiment(cfg).simulate()
    _run_simulate(Namespace(config=str(path), output=str(tmp_path), drug_type=None))
    artifact = json.loads((tmp_path / "simulation.json").read_text())
    assert "concentration" not in artifact["simulations"][0]
    np.testing.assert_allclose(
        artifact["simulations"][0]["populations"]["P"], expected.populations["P"]
    )


def test_explicit_cli_drug_override_retains_baseline_rates(tmp_path):
    cfg = ExperimentConfig(dynamics={"default_birth_base": .002},
                           simulation={"method": "ode", "t_max": 2, "dt_obs": 1})
    path = tmp_path / "experiment.yaml"
    save_config(cfg, path)
    _run_simulate(Namespace(config=str(path), output=str(tmp_path), drug_type="cytostatic"))
    effective = load_config(tmp_path / "config.yaml")
    assert effective.dynamics.default_birth_base == .002
    assert effective.dynamics.drug_mechanism == "cytostatic"


def test_pk_exposure_and_provenance_reach_fit(monkeypatch):
    captured = {}

    def capture(estimator, **kwargs):
        captured["likelihood"] = estimator.likelihood
        captured["options"] = kwargs
        return MLEResult({}, 0., 0., 0.)

    monkeypatch.setattr("umimic.pipeline.experiment.MLEstimator.fit", capture)
    cfg = ExperimentConfig(
        context="in_vivo", pk={"model": "one_compartment"},
        dosing={"type": "single_bolus", "dose_amount": 100},
        simulation={"method": "ode", "t_max": 4, "dt_obs": 1, "n_replicates": 1},
    )
    exp = Experiment(cfg)
    data = exp.generate_synthetic()
    result = exp.fit(data, n_restarts=2)
    likelihood = captured["likelihood"]
    assert likelihood.exposure_fn is exp.exposure
    assert captured["options"]["n_restarts"] == 2
    assert result.metadata["config"]["dosing"]["dose_amount"] == 100
    assert result.metadata["seed"] == cfg.seed
    assert result.metadata["data_sha256"] == dataset_fingerprint(data.series)


@pytest.mark.parametrize("method", ["gillespie", "tau_leaping"])
def test_fractional_initial_mixture_produces_whole_stochastic_cells(method):
    cfg = ExperimentConfig(
        dynamics={"initial_fractions": {"P": .5, "Q": .5}},
        simulation={"initial_cells": 1, "method": method, "t_max": 2, "dt_obs": 1,
                    "n_replicates": 1},
    )
    result = Experiment(cfg).simulate()
    for values in result.populations.values():
        np.testing.assert_array_equal(values, np.rint(values))
    assert result.total[0] == 1


def test_result_serialization_preserves_posterior_arrays(tmp_path):
    chain = MCMCResult(
        samples={"b0": np.array([[.1, .2], [.3, .4]])},
        log_likelihood_trace=np.array([[-1., -2.], [-3., -4.]]),
        log_posterior_trace=np.array([[-2., -3.], [-4., -5.]]),
        n_chains=2, n_samples=2,
    )
    path = tmp_path / "result.json"
    save_result(InferenceResult(method="mcmc", mcmc=chain,
                                metadata={"seed": 42}), path)
    loaded = load_result(path)
    np.testing.assert_array_equal(loaded["mcmc"]["samples"]["b0"], chain.samples["b0"])
    np.testing.assert_array_equal(
        loaded["mcmc"]["log_likelihood_trace"], chain.log_likelihood_trace
    )
    assert loaded["metadata"]["seed"] == 42


def test_data_fingerprint_changes_with_observations():
    first = TimeSeriesData.from_counts([0, 1], [100, 120])
    second = TimeSeriesData.from_counts([0, 1], [100, 121])
    assert dataset_fingerprint([first]) == dataset_fingerprint([first])
    assert dataset_fingerprint([first]) != dataset_fingerprint([second])


def test_data_fingerprint_includes_initial_condition_overrides():
    first = TimeSeriesData.from_counts([0, 1], [100, 120])
    second = TimeSeriesData.from_counts([0, 1], [100, 120])
    second.metadata["initial_fractions"] = {"P": .5, "Q": .5}
    assert dataset_fingerprint([first]) != dataset_fingerprint([second])


def test_noninteger_large_stochastic_population_rejected():
    from umimic.dynamics.initialization import integer_initial_state

    with pytest.raises(ValueError, match="integer total"):
        integer_initial_state(np.array([1_000_000.4, 0]), np.random.default_rng(0))


def test_pk_generation_uses_exposure_even_in_vitro():
    cfg = ExperimentConfig(
        dynamics={"states": ["P"], "default_birth_base": 0,
                  "default_death_base_p": 0, "drug_mechanism": "cytotoxic"},
        pk={"model": "one_compartment"},
        dosing={"type": "single_bolus", "dose_amount": 100, "start_time": 2},
        simulation={"method": "ode", "t_max": 6, "dt_obs": 1, "n_replicates": 1},
    )
    expected = Experiment(cfg).simulate()
    data = Experiment(cfg).generate_synthetic()
    assert data.context == "in_vitro"
    np.testing.assert_allclose(
        data.series[0].metadata["true_trajectory"]["P"], expected.populations["P"]
    )


def test_noncount_fit_uses_configured_initial_cells_without_mutating_data(monkeypatch):
    captured = {}

    def capture(estimator, **kwargs):
        likelihood = estimator.likelihood
        captured["initial"] = likelihood._initial_state(likelihood.data_list[0]).sum()
        return MLEResult({}, 0., 0., 0.)

    monkeypatch.setattr("umimic.pipeline.experiment.MLEstimator.fit", capture)
    cfg = ExperimentConfig(observations={"modalities": ["volume"]},
                           simulation={"initial_cells": 500})
    data = TimeSeriesData(times=[0, 1], observations={"volume": [1, 2]})
    Experiment(cfg).fit(data)
    assert captured["initial"] == 500
    assert data.metadata == {}


def test_posterior_predictive_check_preserves_replicate_anchor_times(monkeypatch):
    from umimic.dynamics.states import ModelTopology
    from umimic.inference.diagnostics import posterior_predictive_check
    from umimic.inference.likelihood import ModelLikelihood
    from umimic.observations.cell_counts import CellCountObservation

    model = CellCountObservation()
    monkeypatch.setattr(model, "sample", lambda state, *args, **kwargs: float(state.sum()))
    data = [TimeSeriesData.from_counts([0, 10], [100, 200]),
            TimeSeriesData.from_counts([10, 20], [100, 200])]
    likelihood = ModelLikelihood(
        ModelTopology.two_state(), data, param_names=["b0", "d0_P", "u_PQ", "u_QP"],
        mode="ode", observation_model=model,
    )
    posterior = MCMCResult(samples={
        "b0": np.array([[np.log(2) / 10]]), "d0_P": np.array([[0.]]),
        "u_PQ": np.array([[0.]]), "u_QP": np.array([[0.]]),
    }, n_samples=1, n_chains=1)
    check = posterior_predictive_check(posterior, likelihood, n_sim=1,
                                       rng=np.random.default_rng(0))
    np.testing.assert_allclose(check["modalities"]["cell_counts"]["replicates"], [[200, 200]])


def test_cli_yaml_stdout_is_machine_readable(tmp_path, monkeypatch, capsys):
    import yaml
    from umimic.pipeline.runner import main

    monkeypatch.setattr("sys.argv", [
        "umimic", "config", "suggest-coupling-ranges", "--format", "yaml",
        "--log-file", str(tmp_path / "utility.log"),
    ])
    main()
    captured = capsys.readouterr()
    assert "hill" in yaml.safe_load(captured.out)
    assert "Starting 'config'" in captured.err
