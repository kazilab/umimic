"""Tests for signaling model interfaces and baseline implementations."""

import numpy as np

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
