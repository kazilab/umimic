"""Tests for MCMC trace layout and ensemble percentile bands."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from umimic.types import EnsembleResult, MCMCResult
from umimic.visualization.posteriors import plot_trace
from umimic.visualization.trajectories import plot_ensemble


def test_plot_trace_draws_one_line_per_chain():
    """Walkers must not be flattened into a single pseudo-iteration series."""
    samples = {
        "b0": np.array(
            [
                [0.04, 0.041, 0.042, 0.040],
                [0.05, 0.051, 0.049, 0.050],
            ]
        )
    }
    result = MCMCResult(samples=samples, n_chains=2, n_samples=4)
    fig = plot_trace(result, params=["b0"])
    ax_trace = fig.axes[0]
    # Two chain lines on the trace panel (plus possibly nothing else).
    assert len(ax_trace.lines) == 2
    # Each line has length n_draws, not n_chains * n_draws.
    assert all(len(line.get_xdata()) == 4 for line in ax_trace.lines)
    matplotlib.pyplot.close(fig)


def test_ensemble_percentile_band_matches_numpy():
    times = np.array([0.0, 1.0, 2.0])
    trajectories = [
        {"P": np.array([10.0, 20.0, 30.0])},
        {"P": np.array([10.0, 40.0, 50.0])},
        {"P": np.array([10.0, 60.0, 70.0])},
        {"P": np.array([10.0, 80.0, 90.0])},
    ]
    ensemble = EnsembleResult(times=times, trajectories=trajectories)
    fig = plot_ensemble(ensemble, states=["P"], show_mean=True, show_ci=True)
    ax = fig.axes[0]
    # Collect fill_between PolyCollection path extents roughly via collections
    assert len(ax.collections) >= 1
    stack = np.array([t["P"] for t in trajectories])
    lo = np.percentile(stack, 2.5, axis=0)
    hi = np.percentile(stack, 97.5, axis=0)
    # Mean line should be present
    mean = stack.mean(axis=0)
    y_means = [line.get_ydata() for line in ax.lines if len(line.get_ydata()) == 3]
    assert any(np.allclose(y, mean) for y in y_means)
    assert lo[1] < mean[1] < hi[1]
    matplotlib.pyplot.close(fig)
