"""Tests for configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from umimic.pipeline.config import ExperimentConfig, load_config, save_config


class TestExperimentConfig:
    def test_default_config(self):
        """Default config should be valid."""
        config = ExperimentConfig()
        assert config.name == "experiment"
        assert config.context == "in_vitro"
        assert config.dynamics.states == ["P", "Q"]

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
