"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from umimic.pipeline.config import ExperimentConfig, load_config, save_config


class TestExperimentConfig:
    def test_default_config(self):
        """Default config should be valid."""
        config = ExperimentConfig()
        assert config.name == "experiment"
        assert config.context == "in_vitro"
        assert config.dynamics.states == ["P", "Q"]
        assert config.signaling.enabled is False
        assert config.coupling.enabled is False

    def test_yaml_roundtrip(self, tmp_path):
        """Config should survive YAML save/load roundtrip."""
        config = ExperimentConfig(name="test_roundtrip")
        path = tmp_path / "test_config.yaml"
        save_config(config, path)

        loaded = load_config(path)
        assert loaded.name == "test_roundtrip"
        assert loaded.context == "in_vitro"
        assert loaded.dynamics.states == ["P", "Q"]

    def test_load_invitro_config(self):
        """Load the example in vitro config."""
        path = Path(__file__).parent.parent.parent / "examples" / "configs" / "invitro_basic.yaml"
        if path.exists():
            config = load_config(path)
            assert config.context == "in_vitro"
            assert "cell_counts" in config.observations.modalities

    def test_custom_config(self):
        """Custom config with non-default values."""
        config = ExperimentConfig(
            name="custom",
            context="in_vivo",
            dynamics={"states": ["P", "Q", "A", "R"]},
            inference={"mode": "mcmc", "n_samples": 5000},
        )
        assert config.context == "in_vivo"
        assert len(config.dynamics.states) == 4
        assert config.inference.n_samples == 5000

    def test_reject_negative_concentrations(self):
        with pytest.raises(ValidationError, match="Concentrations must be non-negative"):
            ExperimentConfig(dosing={"concentrations": [0.1, -1.0]})

    def test_reject_negative_timing_values(self):
        with pytest.raises(ValidationError, match="start_time must be non-negative"):
            ExperimentConfig(dosing={"start_time": -0.5})

        with pytest.raises(ValidationError, match="t_max and dt_obs must be > 0"):
            ExperimentConfig(simulation={"t_max": -1.0})

    def test_signaling_config_accepts_finite_values(self):
        config = ExperimentConfig(
            signaling={
                "enabled": True,
                "model": "toy_mapk_akt",
                "initial_state": {"mapk": 0.2, "akt": 0.1},
                "parameters": {"decay": 0.4},
                "observed_nodes": ["mapk"],
            },
            coupling={
                "enabled": True,
                "function": "hill",
                "targets": ["birth", "death"],
                "parameters": {"ec50": 1.0, "hill": 2.0},
            },
        )
        assert config.signaling.enabled
        assert config.signaling.model == "toy_mapk_akt"
        assert config.signaling.direction == "inhibitory"
        assert set(config.coupling.targets) == {"birth", "death"}

    def test_signaling_direction_accepts_stimulatory(self):
        config = ExperimentConfig(
            signaling={"enabled": True, "model": "toy_mapk_akt", "direction": "stimulatory"}
        )
        assert config.signaling.direction == "stimulatory"

    def test_pk_f_oral_defaults_to_complete_absorption(self):
        assert ExperimentConfig().pk.f_oral == 1.0

    def test_pk_f_oral_rejects_zero_and_above_one(self):
        with pytest.raises(ValidationError, match="f_oral"):
            ExperimentConfig(pk={"f_oral": 0.0})
        with pytest.raises(ValidationError, match="f_oral"):
            ExperimentConfig(pk={"f_oral": 1.5})

    def test_signaling_rejects_non_finite_values(self):
        with pytest.raises(
            ValidationError, match="Signaling state/parameter values must be finite"
        ):
            ExperimentConfig(signaling={"parameters": {"decay": float("inf")}})

    def test_coupling_targets_allow_specific_state_and_transition(self):
        config = ExperimentConfig(
            coupling={"targets": ["death:P", "transition:P->R", "birth"]}
        )
        assert set(config.coupling.targets) == {"death:P", "transition:P->R", "birth"}

    def test_coupling_targets_reject_invalid_format(self):
        with pytest.raises(ValidationError, match="Invalid coupling target format"):
            ExperimentConfig(coupling={"targets": ["death:X"]})

    def test_max_effect_by_target_accepts_specific_keys(self):
        cfg = ExperimentConfig(
            coupling={
                "max_effect_by_target": {
                    "birth": 0.3,
                    "death:P": 1.2,
                    "transition:P->R": 0.4,
                }
            }
        )
        assert cfg.coupling.max_effect_by_target["death:P"] == pytest.approx(1.2)

    def test_max_effect_by_target_rejects_invalid_values(self):
        with pytest.raises(ValidationError, match="Invalid max_effect_by_target key"):
            ExperimentConfig(coupling={"max_effect_by_target": {"death:X": 0.2}})
        with pytest.raises(
            ValidationError,
            match="max_effect_by_target values must be finite and non-negative",
        ):
            ExperimentConfig(coupling={"max_effect_by_target": {"birth": -0.1}})

    def test_ec50_and_hill_by_target_validation(self):
        cfg = ExperimentConfig(
            coupling={
                "ec50_by_target": {"birth": 0.3, "death:P": 1.0},
                "hill_by_target": {"birth": 2.0, "death:P": 1.5},
            }
        )
        assert cfg.coupling.ec50_by_target["birth"] == pytest.approx(0.3)
        assert cfg.coupling.hill_by_target["death:P"] == pytest.approx(1.5)

        with pytest.raises(ValidationError, match="Invalid ec50_by_target key"):
            ExperimentConfig(coupling={"ec50_by_target": {"death:X": 0.1}})
        with pytest.raises(ValidationError, match="ec50_by_target values must be finite and > 0"):
            ExperimentConfig(coupling={"ec50_by_target": {"birth": 0.0}})
        with pytest.raises(ValidationError, match="Invalid hill_by_target key"):
            ExperimentConfig(coupling={"hill_by_target": {"transition:P->X": 1.0}})
        with pytest.raises(ValidationError, match="hill_by_target values must be finite and > 0"):
            ExperimentConfig(coupling={"hill_by_target": {"birth": -2.0}})

    def test_logistic_k_and_center_by_target_validation(self):
        cfg = ExperimentConfig(
            coupling={
                "k_by_target": {"birth": 2.0, "death:P": 1.2},
                "center_by_target": {"birth": 0.3, "transition:P->R": 0.7},
            }
        )
        assert cfg.coupling.k_by_target["birth"] == pytest.approx(2.0)
        assert cfg.coupling.center_by_target["transition:P->R"] == pytest.approx(0.7)

        with pytest.raises(ValidationError, match="Invalid k_by_target key"):
            ExperimentConfig(coupling={"k_by_target": {"death:X": 1.0}})
        with pytest.raises(ValidationError, match="k_by_target values must be finite and > 0"):
            ExperimentConfig(coupling={"k_by_target": {"birth": 0.0}})
        with pytest.raises(ValidationError, match="Invalid center_by_target key"):
            ExperimentConfig(coupling={"center_by_target": {"transition:P->X": 0.5}})

    def test_coupling_recommended_parameter_ranges_helper(self):
        ranges = ExperimentConfig().coupling.recommended_parameter_ranges()
        assert "hill" in ranges
        assert "logistic" in ranges
        assert "target_overrides" in ranges
        assert ranges["hill"]["ec50"][0] > 0
