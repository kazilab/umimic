"""Dose-response curve visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from umimic.visualization.style import apply_umimic_style, STATE_COLORS, MEAN_STYLE

if TYPE_CHECKING:
    from umimic.dynamics.rates import RateSet
    from umimic.types import InferenceResult


def plot_rate_dose_response(
    rate_set: RateSet,
    concentrations: np.ndarray | None = None,
    rates_to_plot: list[str] | None = None,
    ax: plt.Axes | None = None,
    title: str = "Mechanistic Dose-Response",
) -> plt.Figure:
    """Plot dose-response curves for individual mechanistic rates.

    Shows how birth rate, death rate, and transition rates change
    with drug concentration — the key mechanistic output of U-MIMIC.

    Args:
        rate_set: RateSet with dose-response parameterization.
        concentrations: Concentration range to plot.
        rates_to_plot: Which rates to include.
        ax: Matplotlib axes.
        title: Plot title.
    """
    apply_umimic_style()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if concentrations is None:
        concentrations = np.logspace(-2, 2, 200)

    from umimic.dynamics.states import CellType

    if rates_to_plot is None:
        rates_to_plot = ["birth", "death_P", "net_growth"]

    for rate_name in rates_to_plot:
        if rate_name == "birth":
            values = [rate_set.birth_rate(c) for c in concentrations]
            ax.semilogx(concentrations, values, label="Birth rate (b)",
                       color="#2196F3", **MEAN_STYLE)
        elif rate_name == "death_P":
            values = [rate_set.death_rate(CellType.P, c) for c in concentrations]
            ax.semilogx(concentrations, values, label="Death rate P (d_P)",
                       color="#F44336", **MEAN_STYLE)
        elif rate_name == "death_Q":
            values = [rate_set.death_rate(CellType.Q, c) for c in concentrations]
            ax.semilogx(concentrations, values, label="Death rate Q (d_Q)",
                       color="#FF5722", linestyle="--", **MEAN_STYLE)
        elif rate_name == "net_growth":
            values = [rate_set.net_growth_rate(c) for c in concentrations]
            ax.semilogx(concentrations, values, label="Net growth (b - d_P)",
                       color="#333333", linestyle="-.", **MEAN_STYLE)
            ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
        elif rate_name.startswith("trans_"):
            parts = rate_name.split("_")
            if len(parts) == 3:
                src = CellType[parts[1]]
                tgt = CellType[parts[2]]
                values = [rate_set.transition_rate(src, tgt, c) for c in concentrations]
                ax.semilogx(concentrations, values,
                           label=f"Transition {parts[1]}->{parts[2]}",
                           **MEAN_STYLE)

    ax.set_xlabel("Drug concentration")
    ax.set_ylabel("Rate (1/hour)")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_net_growth_curve(
    rate_set: RateSet,
    concentrations: np.ndarray | None = None,
    show_ng0: bool = True,
    show_ng50: bool = True,
    ax: plt.Axes | None = None,
    title: str = "Net Growth Rate vs. Concentration",
) -> plt.Figure:
    """Plot net growth rate curve with NG0 and NG50 annotations.

    NG0: concentration where net growth = 0 (growth arrest)
    NG50: concentration where net growth = 50% of control

    These are U-MIMIC's mechanistic alternatives to traditional IC50.
    """
    apply_umimic_style()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if concentrations is None:
        concentrations = np.logspace(-2, 2, 500)

    ng_values = np.array([rate_set.net_growth_rate(c) for c in concentrations])
    ng_control = rate_set.net_growth_rate(0.0)

    ax.semilogx(concentrations, ng_values, color="#333333", **MEAN_STYLE)
    ax.axhline(y=0, color="red", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(y=ng_control, color="blue", linestyle=":", linewidth=0.8, alpha=0.5)

    # Find and annotate NG0 (net growth = 0)
    if show_ng0 and ng_control > 0:
        crossings = np.where(np.diff(np.sign(ng_values)))[0]
        if len(crossings) > 0:
            idx = crossings[0]
            ng0 = concentrations[idx]
            ax.axvline(x=ng0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.annotate(
                f"NG0 = {ng0:.2f}",
                xy=(ng0, 0),
                xytext=(ng0 * 3, ng_control * 0.2),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red",
            )

    # Find and annotate NG50 (net growth = 50% of control)
    if show_ng50 and ng_control > 0:
        ng50_target = ng_control * 0.5
        crossings_50 = np.where(np.diff(np.sign(ng_values - ng50_target)))[0]
        if len(crossings_50) > 0:
            idx = crossings_50[0]
            ng50 = concentrations[idx]
            ax.axvline(x=ng50, color="blue", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.annotate(
                f"NG50 = {ng50:.2f}",
                xy=(ng50, ng50_target),
                xytext=(ng50 * 3, ng_control * 0.7),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="blue"),
                color="blue",
            )

    ax.set_xlabel("Drug concentration")
    ax.set_ylabel("Net growth rate (1/hour)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_mechanism_comparison(
    rate_set: RateSet,
    concentrations: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    title: str = "Cytostatic vs. Cytotoxic Decomposition",
) -> plt.Figure:
    """Plot birth and death rate changes to visualize drug mechanism.

    Shows at a glance whether a drug is cytostatic (reduces birth),
    cytotoxic (increases death), or mixed.
    """
    apply_umimic_style()
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig = ax.figure
        axes = [ax, ax.twinx()]

    if concentrations is None:
        concentrations = np.logspace(-2, 2, 200)

    from umimic.dynamics.states import CellType

    # Left panel: absolute rates
    b_vals = [rate_set.birth_rate(c) for c in concentrations]
    d_vals = [rate_set.death_rate(CellType.P, c) for c in concentrations]

    axes[0].semilogx(concentrations, b_vals, color="#2196F3",
                     label="Birth rate", **MEAN_STYLE)
    axes[0].semilogx(concentrations, d_vals, color="#F44336",
                     label="Death rate", **MEAN_STYLE)
    axes[0].set_xlabel("Drug concentration")
    axes[0].set_ylabel("Rate (1/hour)")
    axes[0].legend()
    axes[0].set_title("Rate Modulation")

    # Right panel: fold-change from control
    b_ctrl = rate_set.birth_rate(0.0)
    d_ctrl = rate_set.death_rate(CellType.P, 0.0)
    b_fc = [b / b_ctrl if b_ctrl > 0 else 1.0 for b in b_vals]
    d_fc = [d / d_ctrl if d_ctrl > 0 else 1.0 for d in d_vals]

    axes[1].semilogx(concentrations, b_fc, color="#2196F3",
                     label="Birth (fold-change)", **MEAN_STYLE)
    axes[1].semilogx(concentrations, d_fc, color="#F44336",
                     label="Death (fold-change)", **MEAN_STYLE)
    axes[1].axhline(y=1, color="gray", linestyle=":", linewidth=0.8)
    axes[1].set_xlabel("Drug concentration")
    axes[1].set_ylabel("Fold-change from control")
    axes[1].legend()
    axes[1].set_title("Mechanism Decomposition")

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    return fig
