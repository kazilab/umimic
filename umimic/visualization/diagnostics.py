"""Diagnostic visualization for model checking."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from umimic.visualization.style import apply_umimic_style


def plot_residuals(
    observed: np.ndarray,
    predicted: np.ndarray,
    times: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    title: str = "Residual Analysis",
) -> plt.Figure:
    """Plot residuals (observed - predicted) for model checking."""
    apply_umimic_style()
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig = ax.figure
        axes = [ax, ax.twinx()]

    residuals = observed - predicted

    # Residuals vs time or index
    x = times if times is not None else np.arange(len(residuals))
    axes[0].scatter(x, residuals, s=20, alpha=0.6, color="#2196F3")
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Time" if times is not None else "Index")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Time")

    # QQ-like: residuals vs predicted
    axes[1].scatter(predicted, residuals, s=20, alpha=0.6, color="#FF9800")
    axes[1].axhline(y=0, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs Predicted")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_fit_quality(
    observed: np.ndarray,
    predicted: np.ndarray,
    ci_lower: np.ndarray | None = None,
    ci_upper: np.ndarray | None = None,
    times: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    title: str = "Model Fit",
) -> plt.Figure:
    """Plot observed data vs model predictions with credible intervals."""
    apply_umimic_style()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x = times if times is not None else np.arange(len(observed))

    ax.plot(x, predicted, color="#2196F3", linewidth=2, label="Predicted")
    ax.scatter(x, observed, color="black", s=30, zorder=5, label="Observed")

    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(x, ci_lower, ci_upper, color="#2196F3", alpha=0.2,
                        label="95% CI")

    ax.set_xlabel("Time" if times is not None else "Index")
    ax.set_ylabel("Cell count")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig
