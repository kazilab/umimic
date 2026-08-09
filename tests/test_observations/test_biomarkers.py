"""Tests for biomarker observation models."""

import numpy as np
import pytest

from umimic.dynamics.states import CellType, ModelTopology
from umimic.observations.biomarkers import BiomarkerObservation


class TestBiomarkerObservation:
    def test_param_names_match_inference_key(self):
        assert BiomarkerObservation().param_names() == ["biomarker_precision"]

    def test_ki67_includes_dividing_resistant_cells(self):
        topo = ModelTopology.four_state()
        obs = BiomarkerObservation("ki67", topology=topo)
        # P=50, Q=30, A=10, R=20 with R dividing
        state = np.array([50.0, 30.0, 10.0, 20.0])
        assert obs.expected_value(state) == pytest.approx((50 + 20) / (50 + 30 + 20))

    def test_caspase_is_dead_over_total(self):
        topo = ModelTopology.four_state()
        obs = BiomarkerObservation("caspase", topology=topo)
        state = np.array([50.0, 30.0, 10.0, 20.0])
        assert obs.expected_value(state) == pytest.approx(10.0 / 110.0)

    def test_extinct_population_contributes_zero_ll(self):
        obs = BiomarkerObservation("ki67")
        assert obs.log_likelihood(0.5, np.array([0.0, 0.0])) == 0.0

    def test_sample_respects_precision_override(self, rng):
        state = np.array([80.0, 20.0])
        obs = BiomarkerObservation("ki67", precision=10.0)
        tight = [
            obs.sample(state, rng, {"biomarker_precision": 200.0})
            for _ in range(2000)
        ]
        loose = [
            obs.sample(state, rng, {"biomarker_precision": 5.0})
            for _ in range(2000)
        ]
        assert np.nanvar(loose) > np.nanvar(tight)

    def test_legacy_precision_alias(self):
        obs = BiomarkerObservation("ki67", precision=50.0)
        state = np.array([80.0, 20.0])
        ll_new = obs.log_likelihood(0.8, state, {"biomarker_precision": 30.0})
        ll_old = obs.log_likelihood(0.8, state, {"precision": 30.0})
        assert ll_new == pytest.approx(ll_old)
