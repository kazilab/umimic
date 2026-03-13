"""Population trajectory plotting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from umimic.visualization.style import (
    STATE_COLORS,
    MEAN_STYLE,
    CI_STYLE,
    DATA_STYLE,
    ENSEMBLE_STYLE,
    apply_umimic_style,
    get_concentration_colors,
)

if TYPE_CHECKING:
    from umimic.types import SimulationResult, EnsembleResult
    from umimic.data.schemas import TimeSeriesData


def plot_population_trajectories(
    result: SimulationResult,
    data: TimeSeriesData | None = None,
    states: list[str] | None = None,
    show_viable: bool = True,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot deterministic population trajectories.

    Args:
        result: SimulationResult from ODE solver.
        data: Optional observed data to overlay.
        states: Which states to plot (default: all).
        show_viable: Whether to show total viable line.
        ax: Matplotlib axes (created if None).
        title: Plot title.
    """
    apply_umimic_style()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    states_to_plot = states or list(result.populations.keys())

    for state_name in states_to_plot:
        if state_name in result.populations:
            color = STATE_COLORS.get(state_name, None)
            ax.plot(
                result.times,
                result.populations[state_name],
                label=state_name,
                color=color,
                **MEAN_STYLE,
            )

    if show_viable and len(states_to_plot) > 1:
        ax.plot(
            result.times,
            result.viable,
            label="Viable",
            color=STATE_COLORS["viable"],
            linestyle="--",
            **MEAN_STYLE,
        )

    if data is not None and "cell_counts" in data.observations:
        ax.plot(
            data.times,
            data.observations["cell_counts"],
            color="black",
            label="Observed",
            **DATA_STYLE,
        )

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cell count")
    ax.legend()
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


def plot_ensemble(
    ensemble: EnsembleResult,
    data: TimeSeriesData | None = None,
    states: list[str] | None = None,
    show_mean: bool = True,
    show_ci: bool = True,
    alpha_traj: float = 0.05,
    max_trajectories: int = 100,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot stochastic ensemble trajectories with mean and CI.

    Args:
        ensemble: EnsembleResult from Gillespie/tau-leaping.
        data: Optional observed data to overlay.
        states: Which states to plot.
        show_mean: Show ensemble mean.
        show_ci: Show 95% credible interval bands.
        alpha_traj: Opacity for individual trajectories.
        max_trajectories: Max individual trajectories to draw.
        ax: Matplotlib axes.
        title: Plot title.
    """
    apply_umimic_style()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    means = ensemble.mean()
    stds = ensemble.std()
    states_to_plot = states or list(means.keys())

    for state_name in states_to_plot:
        if state_name not in means:
            continue
        color = STATE_COLORS.get(state_name, None)

        # Individual trajectories
        n_show = min(max_trajectories, ensemble.n_trajectories)
        for i in range(n_show):
            traj = ensemble.trajectories[i]
            if state_name in traj:
                ax.plot(
                    ensemble.times,
                    traj[state_name],
                    color=color,
                    alpha=alpha_traj,
                    linewidth=0.5,
                )

        # Mean
        if show_mean:
            ax.plot(
                ensemble.times,
                means[state_name],
                color=color,
                label=f"{state_name} (mean)",
                **MEAN_STYLE,
            )

        # 95% CI
        if show_ci:
            lo = means[state_name] - 1.96 * stds[state_name]
            hi = means[state_name] + 1.96 * stds[state_name]
            ax.fill_between(
                ensemble.times,
                np.maximum(lo, 0),
                hi,
                color=color,
                **CI_STYLE,
            )

    if data is not None and "cell_counts" in data.observations:
        ax.plot(
            data.times,
            data.observations["cell_counts"],
            color="black",
            label="Observed",
            **DATA_STYLE,
        )

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cell count")
    ax.legend()
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


def plot_dose_response_trajectories(
    results: dict[float, SimulationResult],
    state: str = "P",
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot trajectories across multiple drug concentrations.

    Args:
        results: Dict mapping concentration -> SimulationResult.
        state: Which cell state to plot.
        ax: Matplotlib axes.
        title: Plot title.
    """
    apply_umimic_style()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    concentrations = sorted(results.keys())
    colors = get_concentration_colors(concentrations)

    for conc, color in zip(concentrations, colors):
        result = results[conc]
        if state in result.populations:
            label = f"C={conc}" if conc > 0 else "Control"
            ax.plot(
                result.times,
                result.populations[state],
                color=color,
                label=label,
                **MEAN_STYLE,
            )

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel(f"{state} cell count")
    ax.legend(title="Concentration", fontsize=8)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig
