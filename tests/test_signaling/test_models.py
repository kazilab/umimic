"""Tests for signaling model interfaces and baseline implementations."""

import numpy as np
import pytest

from umimic.signaling.models import ToyMapkAktNetwork


class TestToyMapkAktNetwork:
    def test_initial_state_shape_matches_node_names(self):
        model = ToyMapkAktNetwork()
        y0 = model.initial_state()
        assert y0.shape == (len(model.node_names),)
        assert np.all(np.isfinite(y0))

    def test_rhs_is_finite_for_positive_concentration(self):
        model = ToyMapkAktNetwork()
        y0 = model.initial_state()
        dydt = model.rhs(0.0, y0, concentration=10.0)
        assert dydt.shape == y0.shape
        assert np.all(np.isfinite(dydt))

    def test_observable_map_exports_named_values(self):
        model = ToyMapkAktNetwork()
        values = model.observable_map(np.array([0.3, 0.4], dtype=float))
        assert set(values.keys()) == {"mapk", "akt"}
        assert values["mapk"] == 0.3

    def test_steady_state_matches_target(self):
        model = ToyMapkAktNetwork()
        c = 1.0
        y_star = np.array(
            [
                np.clip(model.mapk_baseline - model.mapk_drive * c, 0.0, 1.0),
                np.clip(model.akt_baseline - model.akt_drive * c, 0.0, 1.0),
            ]
        )
        np.testing.assert_allclose(model.rhs(0.0, y_star, c), 0.0, atol=1e-12)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"mapk_baseline": 1.5}, "mapk_baseline"),
            ({"akt_drive": -0.1}, "akt_drive"),
            ({"decay": 0.0}, "decay"),
            ({"direction": "sideways"}, "direction"),
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ToyMapkAktNetwork(**kwargs)



def test_simulation_reports_the_pathway_trajectory():
    import numpy as np

    from umimic.pipeline.config import ExperimentConfig
    from umimic.pipeline.experiment import Experiment

    config = ExperimentConfig(
        dosing={"type": "constant", "concentrations": [1.0]},
        simulation={"method": "ode", "t_max": 24.0, "dt_obs": 6.0},
        signaling={"enabled": True, "model": "toy_mapk_akt"},
        coupling={"enabled": True, "targets": ["birth"]},
    )
    result = Experiment(config).simulate()
    trajectory = result.metadata["signaling"]["trajectory"]
    assert trajectory["times"] == [0.0, 6.0, 12.0, 18.0, 24.0]
    for node in result.metadata["signaling"]["nodes"]:
        values = np.asarray(trajectory[node])
        assert values.shape == (5,) and np.all((values >= 0) & (values <= 1))
    # An inhibitor lowers pathway activity from its baseline.
    assert trajectory["mapk"][-1] < trajectory["mapk"][0]
