"""Tests for the Experiment orchestrator."""

import numpy as np
import pytest

from umimic.pipeline.config import ExperimentConfig
from umimic.pipeline.experiment import Experiment
from umimic.pipeline.transfer import TransferLearning
from umimic.dynamics.rates import RateSet
from umimic.pk.dosing import Dose, DosingSchedule
from umimic.types import InferenceResult, MLEResult


class TestExperiment:
    def test_create_experiment(self):
        """Experiment should initialize from default config."""
        config = ExperimentConfig()
        exp = Experiment(config)
        assert exp.topology is not None
        assert exp.observation_model is not None

    def test_pk_f_oral_reaches_one_compartment_model(self):
        """Incomplete oral absorption must scale C(t) through Experiment."""
        cfg = ExperimentConfig(
            pk={
                "model": "one_compartment",
                "vd": 10.0,
                "ke": 0.1,
                "ka": 0.5,
                "f_oral": 0.5,
            },
            dosing={
                "type": "oral",
                "dose_amount": 100.0,
                "interval": 24.0,
                "n_doses": 1,
                "start_time": 0.0,
            },
            simulation={"t_max": 24.0, "dt_obs": 6.0},
        )
        exp = Experiment(cfg)
        # Build equivalent full-F PK for comparison
        from umimic.pk.compartment import OneCompartmentPK

        full = OneCompartmentPK(vd=10.0, ke=0.1, ka=0.5, f_oral=1.0)
        half = OneCompartmentPK(vd=10.0, ke=0.1, ka=0.5, f_oral=0.5)
        dosing = DosingSchedule(doses=[Dose(0.0, 100.0, route="oral")])
        t = np.linspace(0.1, 24, 50)
        np.testing.assert_allclose(
            half.solve(dosing, t), 0.5 * full.solve(dosing, t), rtol=1e-8
        )
        # Experiment exposure uses the configured F
        c_exp = np.array([exp.exposure(float(tt)) for tt in t])
        np.testing.assert_allclose(c_exp, half.solve(dosing, t), rtol=1e-6)

    def test_transfer_shrinkage_must_be_positive(self):
        mle = MLEResult(
            parameters={"b0": 0.04},
            log_likelihood=0.0,
            aic=0.0,
            bic=0.0,
        )
        ir = InferenceResult(method="mle", mle=mle)
        with pytest.raises(ValueError, match="shrinkage"):
            TransferLearning(ir, transfer_params=["b0"], shrinkage=0.0)
        with pytest.raises(ValueError, match="shrinkage"):
            TransferLearning(ir, transfer_params=["b0"], shrinkage=1.5)
        # b0 is a rate, so it is no longer transferred without an explicit in
        # vitro -> in vivo scale factor: an unscaled rate produces a narrow
        # prior centred on the wrong value, which sparse in vivo data cannot
        # overcome. Supplying the factor makes it transferable again.
        ok = TransferLearning(
            ir,
            transfer_params=["b0"],
            shrinkage=0.5,
            scale_factors={"b0": 0.25},
        )
        assert "b0" in ok.build_priors().distributions

        # Without the factor it is skipped, with the reason recorded rather
        # than dropped silently.
        skipped = TransferLearning(ir, transfer_params=["b0"], shrinkage=0.5)
        assert "b0" not in skipped.build_priors().distributions
        assert "b0" in skipped.skipped

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

    def test_simulate_handles_extreme_concentrations(self, extreme_concentrations):
        """Dose-response remains finite across extreme concentration values."""
        config = ExperimentConfig(simulation={"method": "ode", "t_max": 24.0, "dt_obs": 6.0})
        exp = Experiment(config)
        results = exp.simulate(method="ode", concentrations=extreme_concentrations)

        assert set(results.keys()) == set(extreme_concentrations)
        for sim in results.values():
            for series in sim.populations.values():
                assert np.all(np.isfinite(series))

    def test_simulate_ode_with_signaling_enabled(self):
        """Signaling-enabled ODE simulation should run and attach signaling metadata."""
        config = ExperimentConfig(
            simulation={"method": "ode", "t_max": 12.0, "dt_obs": 3.0},
            signaling={
                "enabled": True,
                "model": "toy_mapk_akt",
                "parameters": {"decay": 0.4},
            },
            coupling={
                "enabled": True,
                "function": "hill",
                "targets": ["birth"],
                "parameters": {"max_effect": 0.2},
            },
        )
        exp = Experiment(config)
        result = exp.simulate(method="ode")
        assert len(result.times) > 0
        assert "P" in result.populations
        assert "signaling" in result.metadata
        assert result.metadata["signaling"]["enabled"] is True
        assert result.metadata["signaling"]["coupling_mode"] == "direct_rate_multiplier"
        assert result.metadata["signaling"]["direction"] == "inhibitory"
        assert "1 + max_effect" in result.metadata["signaling"]["coupling_formula"]

    def test_signaling_direction_from_config(self):
        """stimulatory direction must be selectable via SignalingConfig."""
        config = ExperimentConfig(
            simulation={"method": "ode", "t_max": 6.0, "dt_obs": 3.0},
            signaling={
                "enabled": True,
                "model": "toy_mapk_akt",
                "direction": "stimulatory",
            },
            coupling={
                "enabled": True,
                "targets": ["birth"],
                "parameters": {"max_effect": 0.1},
            },
        )
        result = Experiment(config).simulate(method="ode")
        assert result.metadata["signaling"]["direction"] == "stimulatory"

    def test_unknown_signaling_initial_state_is_rejected(self):
        config = ExperimentConfig(
            simulation={"method": "ode", "t_max": 6.0, "dt_obs": 3.0},
            signaling={
                "enabled": True,
                "model": "toy_mapk_akt",
                "initial_state": {"not_a_node": 0.5},
            },
            coupling={"enabled": True, "targets": ["birth"]},
        )
        with pytest.raises(ValueError, match="Unknown signaling initial_state"):
            Experiment(config).simulate(method="ode")

    def test_direct_rate_coupling_changes_trajectory(self):
        """Direct signaling rate multipliers should alter ODE trajectories."""
        base_cfg = ExperimentConfig(simulation={"method": "ode", "t_max": 12.0, "dt_obs": 3.0})
        coupled_cfg = ExperimentConfig(
            simulation={"method": "ode", "t_max": 12.0, "dt_obs": 3.0},
            signaling={
                "enabled": True,
                "model": "toy_mapk_akt",
                "parameters": {"decay": 0.3, "mapk_drive": 0.2, "akt_drive": 0.15},
            },
            coupling={
                "enabled": True,
                "function": "hill",
                "targets": ["birth"],
                "parameters": {"max_effect": 1.0, "ec50": 0.2, "hill": 1.0},
            },
        )

        base = Experiment(base_cfg).simulate(method="ode")
        coupled = Experiment(coupled_cfg).simulate(method="ode")
        assert not np.allclose(base.populations["P"], coupled.populations["P"])

    def test_state_specific_coupling_target_is_reported(self):
        """Specific targets like death:P should be accepted and tracked."""
        cfg = ExperimentConfig(
            simulation={"method": "ode", "t_max": 12.0, "dt_obs": 3.0},
            signaling={"enabled": True, "model": "toy_mapk_akt"},
            coupling={
                "enabled": True,
                "function": "hill",
                "targets": ["death:P"],
                "parameters": {"max_effect": 0.5, "ec50": 0.2, "hill": 1.0},
            },
        )
        res = Experiment(cfg).simulate(method="ode")
        assert "signaling" in res.metadata
        assert "death:P" in res.metadata["signaling"]["coupling_targets"]

    def test_max_effect_by_target_changes_specific_target_strength(self):
        """Per-target max_effect overrides should alter trajectories."""
        low_cfg = ExperimentConfig(
            simulation={"method": "ode", "t_max": 12.0, "dt_obs": 3.0},
            signaling={
                "enabled": True,
                "model": "toy_mapk_akt",
                "parameters": {"decay": 0.3, "mapk_drive": 0.2, "akt_drive": 0.15},
            },
            coupling={
                "enabled": True,
                "function": "hill",
                "targets": ["birth"],
                "parameters": {"max_effect": 0.2, "ec50": 0.2, "hill": 1.0},
                "max_effect_by_target": {"birth": 0.2},
            },
        )
        high_cfg = ExperimentConfig(
            simulation={"method": "ode", "t_max": 12.0, "dt_obs": 3.0},
            signaling={
                "enabled": True,
                "model": "toy_mapk_akt",
                "parameters": {"decay": 0.3, "mapk_drive": 0.2, "akt_drive": 0.15},
            },
            coupling={
                "enabled": True,
                "function": "hill",
                "targets": ["birth"],
                "parameters": {"max_effect": 0.2, "ec50": 0.2, "hill": 1.0},
                "max_effect_by_target": {"birth": 1.5},
            },
        )
        low = Experiment(low_cfg).simulate(method="ode")
        high = Experiment(high_cfg).simulate(method="ode")
        assert not np.allclose(low.populations["P"], high.populations["P"])
        assert high.metadata["signaling"]["max_effect_by_target"]["birth"] == pytest.approx(1.5)

    def test_ec50_by_target_changes_curve_shape(self):
        """Per-target ec50 overrides should change coupling sensitivity."""
        low_ec50_cfg = ExperimentConfig(
            simulation={"method": "ode", "t_max": 12.0, "dt_obs": 3.0},
            signaling={
                "enabled": True,
                "model": "toy_mapk_akt",
                "parameters": {"decay": 0.3, "mapk_drive": 0.2, "akt_drive": 0.15},
            },
            coupling={
                "enabled": True,
                "function": "hill",
                "targets": ["birth"],
                "parameters": {"max_effect": 1.0, "ec50": 1.0, "hill": 1.0},
                "ec50_by_target": {"birth": 0.1},
            },
        )
        high_ec50_cfg = ExperimentConfig(
            simulation={"method": "ode", "t_max": 12.0, "dt_obs": 3.0},
            signaling={
                "enabled": True,
                "model": "toy_mapk_akt",
                "parameters": {"decay": 0.3, "mapk_drive": 0.2, "akt_drive": 0.15},
            },
            coupling={
                "enabled": True,
                "function": "hill",
                "targets": ["birth"],
                "parameters": {"max_effect": 1.0, "ec50": 1.0, "hill": 1.0},
                "ec50_by_target": {"birth": 10.0},
            },
        )
        low = Experiment(low_ec50_cfg).simulate(method="ode")
        high = Experiment(high_ec50_cfg).simulate(method="ode")
        assert not np.allclose(low.populations["P"], high.populations["P"])
        assert low.metadata["signaling"]["ec50_by_target"]["birth"] == pytest.approx(0.1)

    def test_logistic_k_by_target_changes_curve_shape(self):
        """Per-target logistic k overrides should change trajectory sensitivity."""
        low_k_cfg = ExperimentConfig(
            simulation={"method": "ode", "t_max": 12.0, "dt_obs": 3.0},
            signaling={
                "enabled": True,
                "model": "toy_mapk_akt",
                "parameters": {"decay": 0.3, "mapk_drive": 0.2, "akt_drive": 0.15},
            },
            coupling={
                "enabled": True,
                "function": "logistic",
                "targets": ["birth"],
                "parameters": {"max_effect": 1.0, "k": 0.5, "center": 0.5},
                "k_by_target": {"birth": 0.5},
            },
        )
        high_k_cfg = ExperimentConfig(
            simulation={"method": "ode", "t_max": 12.0, "dt_obs": 3.0},
            signaling={
                "enabled": True,
                "model": "toy_mapk_akt",
                "parameters": {"decay": 0.3, "mapk_drive": 0.2, "akt_drive": 0.15},
            },
            coupling={
                "enabled": True,
                "function": "logistic",
                "targets": ["birth"],
                "parameters": {"max_effect": 1.0, "k": 0.5, "center": 0.5},
                "k_by_target": {"birth": 5.0},
                "center_by_target": {"birth": 0.2},
            },
        )
        low = Experiment(low_k_cfg).simulate(method="ode")
        high = Experiment(high_k_cfg).simulate(method="ode")
        assert not np.allclose(low.populations["P"], high.populations["P"])
        assert high.metadata["signaling"]["k_by_target"]["birth"] == pytest.approx(5.0)
