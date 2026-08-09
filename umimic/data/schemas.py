"""Data format definitions and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TimeSeriesData:
    """Container for experimental time-series data.

    Supports multiple observation modalities (cell counts, BLI, tumor volume,
    biomarkers) at multiple time points, with optional concentration and
    replicate information.

    Missing observations are represented by NaN. Every modality array has the
    same length as `times`; use :meth:`observed_mask` to select the entries
    that are actually present.
    """

    times: np.ndarray
    observations: dict[str, np.ndarray]  # modality -> values at each time
    concentrations: np.ndarray | None = None  # per-observation concentrations
    concentration: float | None = None  # single concentration (in vitro)
    group_id: str | None = None  # experimental group identifier
    replicate_id: str | None = None  # replicate/animal identifier within a group
    units: dict[str, str] = field(default_factory=dict)  # modality -> unit label
    metadata: dict[str, Any] = field(default_factory=dict)

    # Modalities that cannot be negative.
    _NON_NEGATIVE = ("cell_counts", "bli", "volume", "tumor_volume")

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)

        if self.times.ndim != 1 or self.times.size == 0:
            raise ValueError("times must be a non-empty 1-D array.")
        if not np.all(np.isfinite(self.times)):
            raise ValueError("times must contain only finite values.")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError(
                "times must be strictly increasing; duplicate or out-of-order "
                "time points are ambiguous for a single series."
            )

        for key in list(self.observations.keys()):
            arr = np.asarray(self.observations[key], dtype=float)
            if arr.shape != self.times.shape:
                raise ValueError(
                    f"Observation array for modality {key!r} has length "
                    f"{arr.shape[0] if arr.ndim else 'scalar'} but there are "
                    f"{len(self.times)} time points."
                )
            present = ~np.isnan(arr)
            if np.any(np.isinf(arr[present])):
                raise ValueError(
                    f"Modality {key!r} contains infinite values; use NaN to "
                    "mark missing observations."
                )
            if key in self._NON_NEGATIVE and np.any(arr[present] < 0):
                raise ValueError(
                    f"Modality {key!r} must be non-negative, found negative values."
                )
            self.observations[key] = arr

        if self.concentrations is not None:
            self.concentrations = np.asarray(self.concentrations, dtype=float)
            if self.concentrations.shape != self.times.shape:
                raise ValueError(
                    "concentrations must have the same length as times."
                )
            if np.any(self.concentrations < 0):
                raise ValueError("concentrations must be non-negative.")

        if self.concentration is not None and self.concentration < 0:
            raise ValueError("concentration must be non-negative.")

    @property
    def n_timepoints(self) -> int:
        return len(self.times)

    @property
    def modalities(self) -> list[str]:
        return list(self.observations.keys())

    def observed_mask(self, modality: str) -> np.ndarray:
        """Boolean mask of time points where this modality was measured."""
        if modality not in self.observations:
            return np.zeros(len(self.times), dtype=bool)
        return ~np.isnan(self.observations[modality])

    def n_observations(self, modalities: list[str] | None = None) -> int:
        """Total count of non-missing observations across modalities.

        This is the sample size that should enter information criteria; a NaN
        placeholder is not an observation.
        """
        names = modalities if modalities is not None else self.modalities
        return int(
            sum(int(np.count_nonzero(self.observed_mask(m))) for m in names)
        )

    def get_observation(self, modality: str, idx: int) -> float:
        """Get a single observation value."""
        return float(self.observations[modality][idx])

    def has_modality(self, modality: str) -> bool:
        return modality in self.observations

    def subset_times(self, indices: np.ndarray | list[int]) -> TimeSeriesData:
        """Create a subset with selected time indices."""
        indices = np.asarray(indices)
        return TimeSeriesData(
            times=self.times[indices],
            observations={k: v[indices] for k, v in self.observations.items()},
            concentrations=(
                self.concentrations[indices]
                if self.concentrations is not None
                else None
            ),
            concentration=self.concentration,
            group_id=self.group_id,
            replicate_id=self.replicate_id,
            units=dict(self.units),
            metadata=self.metadata.copy(),
        )

    @classmethod
    def from_counts(
        cls,
        times: np.ndarray,
        counts: np.ndarray,
        concentration: float = 0.0,
        group_id: str | None = None,
        replicate_id: str | None = None,
        units: dict[str, str] | None = None,
    ) -> TimeSeriesData:
        """Create from simple viable cell count data."""
        return cls(
            times=np.asarray(times),
            observations={"cell_counts": np.asarray(counts, dtype=float)},
            concentration=concentration,
            group_id=group_id,
            replicate_id=replicate_id,
            units=units or {"cell_counts": "cells"},
        )


@dataclass
class ExperimentalDataset:
    """Collection of TimeSeriesData across multiple conditions/replicates."""

    series: list[TimeSeriesData] = field(default_factory=list)
    name: str = ""
    context: str = "in_vitro"  # or "in_vivo"

    @property
    def n_series(self) -> int:
        return len(self.series)

    @property
    def concentrations(self) -> list[float]:
        """Unique concentrations across all series."""
        concs = set()
        for s in self.series:
            if s.concentration is not None:
                concs.add(s.concentration)
        return sorted(concs)

    @property
    def group_ids(self) -> list[str]:
        return [s.group_id for s in self.series if s.group_id is not None]

    def by_concentration(self, concentration: float) -> list[TimeSeriesData]:
        """Get all series at a given concentration."""
        return [s for s in self.series if s.concentration == concentration]

    def by_group(self, group_id: str) -> TimeSeriesData | None:
        for s in self.series:
            if s.group_id == group_id:
                return s
        return None
