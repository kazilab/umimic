"""Tests for dose-response visualization semantics."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from umimic.dynamics.rates import RateSet
from umimic.dynamics.states import ModelTopology
from umimic.visualization.dose_response import (
    _find_crossing,
    plot_net_growth_curve,
    plot_rate_dose_response,
)


def test_default_growth_curve_uses_asymptotic_not_naive_p():
    """Default metric must match multi-state g, not b - d_P."""
    rates = RateSet()
    topo = ModelTopology.two_state()
    naive = rates.net_growth_rate(0.0)
    asymptotic = rates.asymptotic_growth_rate(0.0, topo)
    assert asymptotic != pytest.approx(naive, rel=1e-3)

    fig = plot_net_growth_curve(rates, topology=topo, metric="asymptotic")
    ax = fig.axes[0]
    assert "Asymptotic" in ax.get_ylabel() or "Asymptotic" in ax.get_title()
    # Main series is the long line over the concentration grid; its value at
    # the leftmost (lowest) concentration should be ~ g(C_min) ≈ g(0).
    series = max(ax.lines, key=lambda ln: len(ln.get_xdata()))
    y0 = float(series.get_ydata()[0])
    assert y0 == pytest.approx(asymptotic, rel=0.05)
    assert y0 != pytest.approx(naive, rel=0.01)
    matplotlib.pyplot.close(fig)


def test_naive_metric_is_explicitly_available():
    rates = RateSet()
    fig = plot_net_growth_curve(rates, metric="naive_p")
    ax = fig.axes[0]
    assert "b" in ax.get_ylabel() or "d_P" in ax.get_ylabel() or "Naive" in ax.get_title()
    matplotlib.pyplot.close(fig)


def test_rate_dose_response_default_includes_asymptotic():
    rates = RateSet()
    fig = plot_rate_dose_response(rates, topology=ModelTopology.two_state())
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("Asymptotic" in lab or "multi-state" in lab for lab in labels)
    matplotlib.pyplot.close(fig)


def test_find_crossing_interpolates():
    c = np.array([1.0, 10.0])
    v = np.array([1.0, -1.0])
    x = _find_crossing(c, v, 0.0)
    assert x == pytest.approx(np.sqrt(10.0), rel=1e-6)
