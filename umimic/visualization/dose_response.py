"""Dose-response curve visualization."""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from umimic.visualization.style import apply_umimic_style, MEAN_STYLE

if TYPE_CHECKING:
    from umimic.dynamics.rates import RateSet
    from umimic.dynamics.states import ModelTopology


logger = logging.getLogger(__name__)

def _default_topology() -> ModelTopology:
    from umimic.dynamics.states import ModelTopology

    return ModelTopology.two_state()


def _asymptotic_growth(
    rate_set: RateSet, concentration: float, topology: ModelTopology
) -> float:
    """Dominant eigenvalue of the low-density multi-state rate matrix."""
    return float(rate_set.asymptotic_growth_rate(concentration, topology))


def _fix_log_axis_for_zero_dose(ax, concentrations) -> None:
    """Keep the untreated control visible on a logarithmic dose axis.

    ``semilogx`` cannot render C = 0, so the control point silently vanished:
    a ladder of [0, 0.1, 1, 10] produced x-limits of (0.079, 12.6) with no
    warning. The control is the reference every other point is judged against,
    and its absence is not visible in the finished figure.

    Switching to a symmetric-log scale, linear below the smallest positive
    dose, puts zero back on the axis without distorting the decades above it.
    """
    c = np.asarray(concentrations, dtype=float)
    if not np.any(c <= 0):
        return

    positive = c[c > 0]
    if positive.size == 0:
        ax.set_xscale("linear")
        return

    linthresh = float(positive.min())
    ax.set_xscale("symlog", linthresh=linthresh, linscale=0.5)
    ax.set_xlim(left=0.0)
    logger.warning(
        "Dose ladder contains a zero (untreated control). The x-axis uses a "
        "symlog scale with linthresh=%.3g so the control stays visible; a "
        "plain log axis would drop it silently.", linthresh,
    )


def _find_crossing(
    concentrations: np.ndarray, values: np.ndarray, level: float
) -> float | None:
    """Log-linear interpolate the first crossing of ``values`` through ``level``."""
    shifted = values - level
    crossings = np.where(np.diff(np.signbit(shifted)))[0]
    if len(crossings) == 0:
        return None
    idx = int(crossings[0])
    y0, y1 = float(shifted[idx]), float(shifted[idx + 1])
    if y0 == y1:
        return float(concentrations[idx])
    # Linear in log-concentration for log-spaced grids.
    c0, c1 = float(concentrations[idx]), float(concentrations[idx + 1])
    if c0 > 0 and c1 > 0:
        log_c = np.log(c0) + (np.log(c1) - np.log(c0)) * (-y0) / (y1 - y0)
        return float(np.exp(log_c))
    return float(c0 + (c1 - c0) * (-y0) / (y1 - y0))


def plot_rate_dose_response(
    rate_set: RateSet,
    concentrations: np.ndarray | None = None,
    rates_to_plot: list[str] | None = None,
    topology: ModelTopology | None = None,
    ax: plt.Axes | None = None,
    title: str = "Mechanistic Dose-Response",
) -> plt.Figure:
    """Plot dose-response curves for individual mechanistic rates.

    Shows how birth, death, transitions, and multi-state asymptotic growth
    change with drug concentration.

    Rate names
    ----------
    ``birth``, ``death_P``, ``death_Q``, ``trans_P_Q``, ... as before.

    ``asymptotic_growth`` (default): dominant eigenvalue of the low-density
    rate matrix (true multi-state exponential growth rate). Requires
    ``topology`` (defaults to two-state P/Q).

    ``net_growth`` / ``net_growth_P``: naive ``b - d_P`` only. Kept for
    comparison; do **not** treat as culture doubling rate when transitions
    or non-P death matter.

    Args:
        rate_set: RateSet with dose-response parameterization.
        concentrations: Concentration range to plot.
        rates_to_plot: Which rates to include.
        topology: Topology for asymptotic growth (default: two-state).
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

    if topology is None:
        topology = _default_topology()

    if rates_to_plot is None:
        rates_to_plot = ["birth", "death_P", "asymptotic_growth"]

    for rate_name in rates_to_plot:
        if rate_name == "birth":
            values = [rate_set.birth_rate(c) for c in concentrations]
            ax.semilogx(
                concentrations,
                values,
                label="Birth rate (b)",
                color="#2196F3",
                **MEAN_STYLE,
            )
        elif rate_name == "death_P":
            values = [rate_set.death_rate(CellType.P, c) for c in concentrations]
            ax.semilogx(
                concentrations,
                values,
                label="Death rate P (d_P)",
                color="#F44336",
                **MEAN_STYLE,
            )
        elif rate_name == "death_Q":
            values = [rate_set.death_rate(CellType.Q, c) for c in concentrations]
            ax.semilogx(
                concentrations,
                values,
                label="Death rate Q (d_Q)",
                color="#FF5722",
                linestyle="--",
                **MEAN_STYLE,
            )
        elif rate_name in ("asymptotic_growth", "growth"):
            values = [
                _asymptotic_growth(rate_set, c, topology) for c in concentrations
            ]
            ax.semilogx(
                concentrations,
                values,
                label="Asymptotic growth g (multi-state)",
                color="#333333",
                linestyle="-.",
                **MEAN_STYLE,
            )
            ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
        elif rate_name in ("net_growth", "net_growth_P"):
            values = [rate_set.net_growth_rate(c) for c in concentrations]
            ax.semilogx(
                concentrations,
                values,
                label="Naive net (b − d_P) only",
                color="#9E9E9E",
                linestyle=":",
                **MEAN_STYLE,
            )
            ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
        elif rate_name.startswith("trans_"):
            parts = rate_name.split("_")
            if len(parts) == 3:
                src = CellType[parts[1]]
                tgt = CellType[parts[2]]
                values = [
                    rate_set.transition_rate(src, tgt, c) for c in concentrations
                ]
                ax.semilogx(
                    concentrations,
                    values,
                    label=f"Transition {parts[1]}->{parts[2]}",
                    **MEAN_STYLE,
                )

    ax.set_xlabel("Drug concentration")
    ax.set_ylabel("Rate (1/hour)")
    ax.legend()
    ax.set_title(title)
    _fix_log_axis_for_zero_dose(ax, concentrations)
    fig.tight_layout()
    return fig


def plot_net_growth_curve(
    rate_set: RateSet,
    concentrations: np.ndarray | None = None,
    topology: ModelTopology | None = None,
    metric: str = "asymptotic",
    show_g0: bool = True,
    show_g50: bool = True,
    show_ng0: bool | None = None,
    show_ng50: bool | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot multi-state asymptotic growth rate vs concentration.

    By default this is the **dominant eigenvalue** of the low-density rate
    matrix (``RateSet.asymptotic_growth_rate``), not the naive
    ``b - d_P``. That matches doubling times and long-run culture growth.

    Annotations
    -----------
    - **g0**: concentration where asymptotic growth crosses zero (net arrest).
    - **g50**: concentration where growth is half the untreated control rate.

    Pass ``metric="naive_p"`` only to compare against the old ``b - d_P``
    curve (explicitly labeled).

    Args:
        rate_set: Rate set.
        concentrations: Concentration grid (log-spaced by default).
        topology: Model topology (default: two-state P/Q).
        metric: ``"asymptotic"`` (default) or ``"naive_p"`` for ``b - d_P``.
        show_g0, show_g50: Annotate zero and half-control crossings.
        show_ng0, show_ng50: Deprecated aliases for show_g0 / show_g50.
        ax: Matplotlib axes.
        title: Plot title.
    """
    apply_umimic_style()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if topology is None:
        topology = _default_topology()

    if show_ng0 is not None:
        show_g0 = show_ng0
    if show_ng50 is not None:
        show_g50 = show_ng50

    if concentrations is None:
        concentrations = np.logspace(-2, 2, 500)

    if metric == "asymptotic":
        g_values = np.array(
            [_asymptotic_growth(rate_set, c, topology) for c in concentrations]
        )
        g_control = _asymptotic_growth(rate_set, 0.0, topology)
        ylabel = "Asymptotic growth rate g (1/hour)"
        default_title = "Asymptotic Growth Rate vs. Concentration"
        series_label = "g(C) multi-state"
    elif metric == "naive_p":
        g_values = np.array(
            [rate_set.net_growth_rate(c) for c in concentrations]
        )
        g_control = rate_set.net_growth_rate(0.0)
        ylabel = "Naive net rate b − d_P (1/hour)"
        default_title = "Naive P Net Growth (b − d_P) vs. Concentration"
        series_label = "b − d_P (not multi-state g)"
    else:
        raise ValueError(
            f"Unknown metric {metric!r}; expected 'asymptotic' or 'naive_p'."
        )

    ax.semilogx(
        concentrations, g_values, color="#333333", label=series_label, **MEAN_STYLE
    )
    ax.axhline(y=0, color="red", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(y=g_control, color="blue", linestyle=":", linewidth=0.8, alpha=0.5)

    if show_g0 and g_control > 0:
        g0 = _find_crossing(concentrations, g_values, 0.0)
        if g0 is not None:
            ax.axvline(x=g0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.annotate(
                f"g0 = {g0:.2g}",
                xy=(g0, 0),
                xytext=(g0 * 3, max(g_control * 0.2, 1e-4)),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red",
            )

    if show_g50 and g_control > 0:
        g50_target = g_control * 0.5
        g50 = _find_crossing(concentrations, g_values, g50_target)
        if g50 is not None:
            ax.axvline(x=g50, color="blue", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.annotate(
                f"g50 = {g50:.2g}",
                xy=(g50, g50_target),
                xytext=(g50 * 3, g_control * 0.7),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="blue"),
                color="blue",
            )

    ax.set_xlabel("Drug concentration")
    ax.set_ylabel(ylabel)
    ax.set_title(title if title is not None else default_title)
    ax.legend(fontsize=8)
    _fix_log_axis_for_zero_dose(ax, concentrations)
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
    cytotoxic (increases death), or mixed — for the **P** rates only.
    Multi-state growth is not shown here; use :func:`plot_net_growth_curve`
    for asymptotic g(C).
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

    axes[0].semilogx(
        concentrations, b_vals, color="#2196F3", label="Birth rate", **MEAN_STYLE
    )
    axes[0].semilogx(
        concentrations, d_vals, color="#F44336", label="Death rate (P)", **MEAN_STYLE
    )
    axes[0].set_xlabel("Drug concentration")
    axes[0].set_ylabel("Rate (1/hour)")
    axes[0].legend()
    axes[0].set_title("Rate Modulation (P)")

    # Right panel: fold-change from control
    b_ctrl = rate_set.birth_rate(0.0)
    d_ctrl = rate_set.death_rate(CellType.P, 0.0)
    b_fc = [b / b_ctrl if b_ctrl > 0 else 1.0 for b in b_vals]
    d_fc = [d / d_ctrl if d_ctrl > 0 else 1.0 for d in d_vals]

    axes[1].semilogx(
        concentrations,
        b_fc,
        color="#2196F3",
        label="Birth (fold-change)",
        **MEAN_STYLE,
    )
    axes[1].semilogx(
        concentrations,
        d_fc,
        color="#F44336",
        label="Death (fold-change)",
        **MEAN_STYLE,
    )
    axes[1].axhline(y=1, color="gray", linestyle=":", linewidth=0.8)
    axes[1].set_xlabel("Drug concentration")
    axes[1].set_ylabel("Fold-change from control")
    axes[1].legend()
    axes[1].set_title("Mechanism Decomposition (P)")

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    _fix_log_axis_for_zero_dose(ax, concentrations)
    fig.tight_layout()
    return fig
