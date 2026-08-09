"""Data preprocessing and transformation utilities."""

from __future__ import annotations

import numpy as np

from umimic.data.schemas import TimeSeriesData


def log_transform(data: TimeSeriesData, modality: str = "cell_counts") -> TimeSeriesData:
    """Apply log-transformation to a modality."""
    new_obs = data.observations.copy()
    if modality in new_obs:
        new_obs[modality] = np.log(np.maximum(new_obs[modality], 1.0))
    return TimeSeriesData(
        times=data.times,
        observations=new_obs,
        concentrations=data.concentrations,
        concentration=data.concentration,
        group_id=data.group_id,
        metadata={**data.metadata, f"log_transform_{modality}": True},
    )


def normalize_to_control(
    data: TimeSeriesData,
    control_value: float,
    modality: str = "cell_counts",
) -> TimeSeriesData:
    """Normalize observations relative to control (fold-change)."""
    new_obs = data.observations.copy()
    if modality in new_obs and control_value > 0:
        new_obs[modality] = new_obs[modality] / control_value
    return TimeSeriesData(
        times=data.times,
        observations=new_obs,
        concentrations=data.concentrations,
        concentration=data.concentration,
        group_id=data.group_id,
        metadata={**data.metadata, f"normalized_{modality}": True},
    )


def interpolate_missing(
    data: TimeSeriesData,
    modality: str = "cell_counts",
) -> TimeSeriesData:
    """Interpolate NaN values in observation data."""
    new_obs = data.observations.copy()
    if modality in new_obs:
        values = new_obs[modality].copy()
        nans = np.isnan(values)
        if np.any(nans) and not np.all(nans):
            values[nans] = np.interp(
                data.times[nans], data.times[~nans], values[~nans]
            )
        new_obs[modality] = values
    return TimeSeriesData(
        times=data.times,
        observations=new_obs,
        concentrations=data.concentrations,
        concentration=data.concentration,
        group_id=data.group_id,
        metadata={**data.metadata, "interpolated": True},
    )
