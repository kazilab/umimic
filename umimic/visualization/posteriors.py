"""Posterior distribution visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from umimic.visualization.style import apply_umimic_style

if TYPE_CHECKING:
    from umimic.types import MCMCResult


def plot_posterior_marginals(
    result: MCMCResult,
    params: list[str] | None = None,
    true_values: dict[str, float] | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot marginal posterior distributions for each parameter.

    Args:
        result: MCMC result with samples.
        params: Which parameters to plot (default: all).
        true_values: Optional ground-truth values to overlay.
        figsize: Figure size.
    """
    apply_umimic_style()
    params = params or list(result.samples.keys())
    n = len(params)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for i, name in enumerate(params):
        ax = axes[i]
        samples = result.samples[name].flatten()

        ax.hist(samples, bins=50, density=True, alpha=0.7, color="#2196F3",
                edgecolor="white", linewidth=0.5)

        # Mean and 95% CI
        mean = np.mean(samples)
        ci_lo, ci_hi = np.percentile(samples, [2.5, 97.5])
        ax.axvline(mean, color="#F44336", linestyle="-", linewidth=1.5,
                   label=f"Mean: {mean:.4f}")
        ax.axvline(ci_lo, color="#F44336", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(ci_hi, color="#F44336", linestyle="--", linewidth=0.8, alpha=0.7)

        if true_values and name in true_values:
            ax.axvline(true_values[name], color="#4CAF50", linestyle="-",
                       linewidth=2, label=f"True: {true_values[name]:.4f}")

        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7)

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Posterior Distributions", fontsize=13)
    fig.tight_layout()
    return fig


def plot_trace(
    result: MCMCResult,
    params: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot MCMC trace plots for convergence assessment.

    Args:
        result: MCMC result.
        params: Which parameters to plot.
        figsize: Figure size.
    """
    apply_umimic_style()
    params = params or list(result.samples.keys())
    n = len(params)

    if figsize is None:
        figsize = (12, 2.5 * n)

    fig, axes = plt.subplots(n, 2, figsize=figsize)
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, name in enumerate(params):
        samples = result.samples[name].flatten()

        # Trace plot
        axes[i, 0].plot(samples, linewidth=0.3, alpha=0.7, color="#333333")
        axes[i, 0].set_ylabel(name, fontsize=9)
        axes[i, 0].set_xlabel("Iteration")

        # Histogram
        axes[i, 1].hist(samples, bins=50, density=True, alpha=0.7,
                        color="#2196F3", edgecolor="white", linewidth=0.5)
        axes[i, 1].set_xlabel(name)

    axes[0, 0].set_title("Trace")
    axes[0, 1].set_title("Distribution")
    fig.tight_layout()
    return fig


def plot_pair(
    result: MCMCResult,
    params: list[str] | None = None,
    max_params: int = 6,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot pairwise posterior correlations (corner plot).

    Args:
        result: MCMC result.
        params: Which parameters (max ~6 for readability).
        max_params: Maximum parameters to include.
        figsize: Figure size.
    """
    apply_umimic_style()
    params = params or list(result.samples.keys())[:max_params]
    n = len(params)

    if figsize is None:
        figsize = (2.5 * n, 2.5 * n)

    fig, axes = plt.subplots(n, n, figsize=figsize)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j] if n > 1 else axes
            xi = result.samples[params[i]].flatten()
            xj = result.samples[params[j]].flatten()

            if i == j:
                # Diagonal: marginal histogram
                ax.hist(xi, bins=30, density=True, alpha=0.7,
                       color="#2196F3", edgecolor="white", linewidth=0.5)
            elif i > j:
                # Lower triangle: scatter
                ax.scatter(xj, xi, s=1, alpha=0.1, color="#333333")
            else:
                # Upper triangle: hide
                ax.set_visible(False)

            if j == 0:
                ax.set_ylabel(params[i], fontsize=8)
            if i == n - 1:
                ax.set_xlabel(params[j], fontsize=8)

            ax.tick_params(labelsize=6)

    fig.suptitle("Pairwise Posterior", fontsize=13)
    fig.tight_layout()
    return fig
