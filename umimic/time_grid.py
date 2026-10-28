"""Observation grids shared by the simulation and generation pipelines."""

from __future__ import annotations

import numpy as np


def observation_grid(t_max: float, dt_obs: float) -> np.ndarray:
    """Return a grid starting at zero and ending exactly at ``t_max``.

    The last interval may be shorter than ``dt_obs``. In particular, a spacing
    longer than the study gives two observations rather than an out-of-range
    observation beyond the study.
    """
    if not np.isfinite(t_max) or t_max <= 0:
        raise ValueError("t_max must be finite and positive")
    if not np.isfinite(dt_obs) or dt_obs <= 0:
        raise ValueError("dt_obs must be finite and positive")
    steps = int(np.floor(t_max / dt_obs))
    grid = np.arange(steps + 1, dtype=float) * dt_obs
    tolerance = 8 * np.finfo(float).eps * t_max
    grid = grid[grid < t_max - tolerance]
    return np.append(grid, float(t_max))


def validate_observation_times(times, t_max: float) -> np.ndarray:
    """Validate an explicit observation schedule without silently clipping it."""
    values = np.asarray(times, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("observation_times must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("observation_times must be finite")
    if np.any(values < 0) or np.any(values > t_max):
        raise ValueError("observation_times must lie between zero and t_max")
    if np.any(np.diff(values) <= 0):
        raise ValueError("observation_times must be strictly increasing")
    return values
