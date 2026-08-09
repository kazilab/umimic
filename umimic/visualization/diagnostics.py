"""Diagnostic visualization for model checking."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from umimic.visualization.style import apply_umimic_style


def plot_residuals(
    observed: np.ndarray,
    predicted: np.ndarray,
    times: np.ndarray | None = None,
    axes: plt.Axes | Sequence[plt.Axes] | None = None,
    title: str = "Residual Analysis",
    ylabel: str = "Residual (obs − pred)",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot raw residuals (observed − predicted) for exploratory checking.

    These are additive residuals. For NegBin or lognormal observation models
    prefer Pearson/deviance residuals or residuals on the log scale; this plot
    is a diagnostic sketch, not a formal goodness-of-fit test.

    Args:
        axes: Two axes to draw into (residuals-vs-time, residuals-vs-predicted).
            A single axes is not accepted: the two panels plot different
            quantities on their x-axes.
        ax: Deprecated alias for `axes`. Passing one axes here used to add a
            `twinx`, which *shares the x-axis* -- so residuals-vs-time and
            residuals-vs-predicted were overplotted on one axis with time and
            fitted values silently conflated. A single axes now raises.
    """
    apply_umimic_style()

    if axes is None and ax is not None:
        axes = ax
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    else:
        if isinstance(axes, plt.Axes):
            raise ValueError(
                "plot_residuals draws two panels whose x-axes are different "
                "quantities (time and predicted value), so it needs two axes. "
                "Pass axes=(ax_time, ax_pred), e.g. from "
                "plt.subplots(1, 2), or omit `axes` to get its own figure. "
                "A single axes previously produced a twinx overlay that shared "
                "one x-axis between the two, which is not a readable figure."
            )
        axes = list(axes)
        if len(axes) != 2:
            raise ValueError(
                f"plot_residuals needs exactly two axes, got {len(axes)}."
            )
        fig = axes[0].figure

    residuals = np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)

    # Residuals vs time or index
    x = times if times is not None else np.arange(len(residuals))
    axes[0].scatter(x, residuals, s=20, alpha=0.6, color="#2196F3")
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Time" if times is not None else "Index")
    axes[0].set_ylabel(ylabel)
    axes[0].set_title("Residuals vs Time")

    # Residuals vs predicted (not a QQ plot)
    axes[1].scatter(predicted, residuals, s=20, alpha=0.6, color="#FF9800")
    axes[1].axhline(y=0, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel(ylabel)
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
    ylabel: str = "Observation",
    band_label: str = "Uncertainty band",
) -> plt.Figure:
    """Plot observed data vs model predictions with optional uncertainty band.

    The optional band is whatever the caller supplies (predictive interval,
    parameter CI, etc.), so it is labeled generically. It used to be hardcoded
    as "95% band" regardless of what was passed -- the figure asserted a
    coverage the function never computed, and the figure is what gets
    published. Pass `band_label` to state what the band actually is.
    """
    apply_umimic_style()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x = times if times is not None else np.arange(len(observed))

    ax.plot(x, predicted, color="#2196F3", linewidth=2, label="Predicted")
    ax.scatter(x, observed, color="black", s=30, zorder=5, label="Observed")

    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(
            x,
            ci_lower,
            ci_upper,
            color="#2196F3",
            alpha=0.2,
            label=band_label,
        )

    ax.set_xlabel("Time" if times is not None else "Index")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig
