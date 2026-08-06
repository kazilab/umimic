"""Visualization tools for trajectories, dose-response curves, posteriors, and diagnostics."""

from .diagnostics import plot_fit_quality, plot_residuals
from .dose_response import (
    plot_mechanism_comparison,
    plot_net_growth_curve,
    plot_rate_dose_response,
)
from .posteriors import plot_pair, plot_posterior_marginals, plot_trace
from .style import STATE_COLORS, apply_umimic_style, get_concentration_colors
from .trajectories import (
    plot_dose_response_trajectories,
    plot_ensemble,
    plot_population_trajectories,
)

__all__ = [
    # Style & Utilities
    "apply_umimic_style",
    "get_concentration_colors",
    "STATE_COLORS",
    # Trajectories
    "plot_population_trajectories",
    "plot_ensemble",
    "plot_dose_response_trajectories",
    # Dose-Response
    "plot_rate_dose_response",
    "plot_net_growth_curve",
    "plot_mechanism_comparison",
    # Posteriors
    "plot_posterior_marginals",
    "plot_trace",
    "plot_pair",
    # Diagnostics
    "plot_residuals",
    "plot_fit_quality",
]
