"""Data loading and saving utilities for CSV experimental data.

Dataset interchange format
--------------------------
U-MIMIC reads and writes a single tidy CSV layout, one row per (replicate,
time point):

    hours,viable_count,concentration,replicate_id[,bli][,volume]
    0,100,0.0,well_1
    4,150,0.0,well_1

`hours` is the observation time, `viable_count` the cell count, and
`concentration` the drug concentration for that replicate. Optional `bli` and
`volume` columns carry the other modalities. Empty cells denote missing
observations and are loaded as NaN.

:func:`save_dataset` writes this format and :func:`load_csv` reads it, so a
dataset produced by ``umimic generate`` can be fitted by ``umimic fit``
without manual conversion.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from umimic.data.schemas import TimeSeriesData, ExperimentalDataset

#: Column name written for each modality key.
MODALITY_COLUMNS = {
    "cell_counts": "viable_count",
    "bli": "bli",
    "volume": "volume",
}


def save_dataset(dataset: ExperimentalDataset, path: str | Path) -> Path:
    """Write an ExperimentalDataset to the interchange CSV format.

    Args:
        dataset: Dataset to write.
        path: Destination CSV path.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    present = []
    for key, column in MODALITY_COLUMNS.items():
        if any(s.has_modality(key) for s in dataset.series):
            present.append((key, column))
    if not present:
        raise ValueError("Dataset contains no recognized modalities to save.")

    header = ["hours"] + [c for _, c in present] + ["concentration", "replicate_id"]

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, series in enumerate(dataset.series):
            replicate = series.replicate_id or series.group_id or f"series_{i}"
            conc = "" if series.concentration is None else series.concentration
            for k, t in enumerate(series.times):
                row = [t]
                for key, _ in present:
                    if series.has_modality(key):
                        value = series.observations[key][k]
                        row.append("" if np.isnan(value) else value)
                    else:
                        row.append("")
                row.extend([conc, replicate])
                writer.writerow(row)

    return path


def load_csv(
    path: str | Path,
    data_config=None,
    time_column: str = "hours",
    count_column: str = "viable_count",
    concentration_column: str = "concentration",
    replicate_column: str = "replicate_id",
) -> ExperimentalDataset:
    """Load experimental data from a CSV file.

    Expected CSV format:
        hours, viable_count, concentration, replicate_id
        0, 100, 0.0, well_1
        4, 150, 0.0, well_1
        ...

    Args:
        path: Path to CSV file.
        data_config: Optional DataConfig for column names.
        time_column: Name of time column.
        count_column: Name of count column.
        concentration_column: Name of concentration column.
        replicate_column: Name of replicate/group column.

    Returns:
        ExperimentalDataset with one TimeSeriesData per replicate.
    """
    try:
        import pandas as pd
    except ImportError:
        return _load_csv_numpy(path, time_column, count_column,
                                concentration_column, replicate_column)

    if data_config is not None:
        time_column = data_config.time_column
        count_column = data_config.count_column
        concentration_column = data_config.concentration_column
        replicate_column = data_config.replicate_column

    df = pd.read_csv(path)

    series = []
    groups = df[replicate_column].unique() if replicate_column in df.columns else ["all"]

    for group_id in groups:
        if replicate_column in df.columns:
            sub = df[df[replicate_column] == group_id].sort_values(time_column)
        else:
            sub = df.sort_values(time_column)

        times = sub[time_column].values.astype(float)
        counts = sub[count_column].values.astype(float)

        conc = None
        if concentration_column in sub.columns:
            conc = float(sub[concentration_column].iloc[0])

        observations = {"cell_counts": counts}

        # Check for other modalities
        for col in ["bli", "bioluminescence", "photon_flux"]:
            if col in sub.columns:
                observations["bli"] = sub[col].values.astype(float)
        for col in ["volume", "tumor_volume"]:
            if col in sub.columns:
                observations["volume"] = sub[col].values.astype(float)

        ts = TimeSeriesData(
            times=times,
            observations=observations,
            concentration=conc,
            group_id=str(group_id),
            replicate_id=str(group_id),
        )
        series.append(ts)

    return ExperimentalDataset(series=series, name=Path(path).stem)


def _load_csv_numpy(
    path: str | Path,
    time_col: str,
    count_col: str,
    conc_col: str,
    rep_col: str,
) -> ExperimentalDataset:
    """Fallback CSV loader using only numpy (no pandas dependency)."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)

    # Simple: load as single series
    times = data[time_col].astype(float)
    counts = data[count_col].astype(float)

    ts = TimeSeriesData(
        times=times,
        observations={"cell_counts": counts},
    )
    return ExperimentalDataset(series=[ts], name=Path(path).stem)
