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
