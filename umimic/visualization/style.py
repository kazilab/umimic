"""Consistent matplotlib style and theming for U-MIMIC plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl

# Color palette for cell states
STATE_COLORS = {
    "P": "#2196F3",  # blue - proliferating
    "Q": "#FF9800",  # orange - quiescent
    "A": "#F44336",  # red - apoptotic
    "R": "#4CAF50",  # green - resistant
    "viable": "#333333",  # dark gray - total viable
    "total": "#999999",  # light gray - total
}

# Color palette for dose-response
DOSE_CMAP = "viridis"

# Line styles
MEAN_STYLE = {"linewidth": 2.0, "solid_capstyle": "round"}
CI_STYLE = {"alpha": 0.2}
DATA_STYLE = {"marker": "o", "markersize": 5, "linestyle": "none", "alpha": 0.7}
ENSEMBLE_STYLE = {"linewidth": 0.5, "alpha": 0.1}


def apply_umimic_style():
    """Apply U-MIMIC publication-quality style to matplotlib."""
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 150,
            "font.size": 11,
            "font.family": "sans-serif",
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
        }
    )


def get_concentration_colors(concentrations: list[float]) -> list[str]:
    """Get a list of colors for a set of concentrations."""
    cmap = plt.get_cmap(DOSE_CMAP)
    n = len(concentrations)
    if n == 1:
        return [cmap(0.5)]
    return [cmap(i / (n - 1)) for i in range(n)]
