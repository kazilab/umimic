"""CSV interchange for experimental datasets, without a pandas dependency.

One row records one time point in a series. At least one observation column
is required: ``viable_count``, ``bli``, ``volume``, or ``biomarker``. Empty
observation cells represent missing values. ``save_dataset`` also writes
series/group identifiers so replicate labels can be reused between groups.
Legacy CSVs without these extra columns remain supported.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import lzma
from pathlib import Path

import numpy as np

from umimic.data.schemas import ExperimentalDataset, TimeSeriesData


MODALITY_COLUMNS = {
    "cell_counts": "viable_count",
    "bli": "bli",
    "volume": "volume",
    "biomarker": "biomarker",
}
_MODALITY_ALIASES = {
    "bli": ("bli", "bioluminescence", "photon_flux"),
    "volume": ("volume", "tumor_volume"),
    "biomarker": ("biomarker",),
}
_OBSERVATION_PREFIX = "observation:"


def _open_csv(path: Path, mode: str):
    """Keep compressed CSV support without relying on pandas compression inference."""
    opener = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}.get(path.suffix)
    encoding = "utf-8-sig" if mode == "r" else "utf-8"
    if opener is not None:
        return opener(path, mode + "t", newline="", encoding=encoding)
    return path.open(mode, newline="", encoding=encoding)


def _optional_number(value: str | None) -> float | None:
    """Treat blank/NaN scalar concentrations as unspecified, never as NaN C."""
    if value is None or not value.strip():
        return None
    number = float(value)
    if np.isnan(number):
        return None
    if not np.isfinite(number) or number < 0:
        raise ValueError("concentration must be finite and non-negative.")
    return number


def _observation_number(value: str | None) -> float:
    return float(value) if value is not None and value.strip() else float("nan")


def save_dataset(dataset: ExperimentalDataset, path: str | Path) -> Path:
    """Write all observation modalities and series identities to a tidy CSV.

    Unrecognized modality names use ``observation:<name>`` columns, avoiding
    ambiguity with annotation columns in user-supplied files. Per-observation
    concentrations, when supplied, use ``concentration_at_time`` separately
    from the scalar concentration. Dataset context is retained as well.
    """
    present: dict[str, str] = {}
    for series in dataset.series:
        for key in series.modalities:
            present.setdefault(key, MODALITY_COLUMNS.get(key, f"{_OBSERVATION_PREFIX}{key}"))
    if not present:
        raise ValueError("Dataset contains no observation modalities to save.")

    has_concentrations = any(s.concentrations is not None for s in dataset.series)
    header = [
        "hours",
        *present.values(),
        "concentration",
        "replicate_id",
        "group_id",
        "series_id",
        "context",
    ]
    if has_concentrations:
        header.append("concentration_at_time")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_csv(path, "w") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, series in enumerate(dataset.series):
            for k, t in enumerate(series.times):
                row = [t]
                for key in present:
                    value = series.observations[key][k] if series.has_modality(key) else np.nan
                    row.append("" if np.isnan(value) else value)
                row.extend(
                    [
                        "" if series.concentration is None else series.concentration,
                        series.replicate_id if series.replicate_id is not None else "",
                        series.group_id if series.group_id is not None else "",
                        str(i),
                        dataset.context,
                    ]
                )
                if has_concentrations:
                    row.append("" if series.concentrations is None else series.concentrations[k])
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
    """Load a tidy CSV with one series per group, replicate and concentration.

    A count column is optional when other observation modalities are present.
    Exported ``series_id`` takes precedence over inferred grouping. Blank
    concentrations remain ``None``; empty observations become NaN. Custom
    column names from ``data_config`` work without optional dependencies.
    """
    if data_config is not None:
        time_column = data_config.time_column
        count_column = data_config.count_column
        concentration_column = data_config.concentration_column
        replicate_column = data_config.replicate_column
    return _load_csv_rows(path, time_column, count_column, concentration_column, replicate_column)


def _load_csv_rows(
    path: str | Path,
    time_col: str,
    count_col: str,
    conc_col: str,
    rep_col: str,
) -> ExperimentalDataset:
    with _open_csv(Path(path), "r") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if time_col not in fields:
            raise ValueError(f"CSV is missing time column {time_col!r}.")
        modality_columns = {}
        if count_col in fields:
            modality_columns["cell_counts"] = count_col
        for key, aliases in _MODALITY_ALIASES.items():
            column = next((name for name in aliases if name in fields), None)
            if column is not None:
                modality_columns[key] = column
        for column in fields:
            if column.startswith(_OBSERVATION_PREFIX) and column[len(_OBSERVATION_PREFIX) :]:
                modality_columns[column[len(_OBSERVATION_PREFIX) :]] = column
        if not modality_columns:
            raise ValueError("CSV contains no recognized observation columns.")

        grouped = {}
        contexts = set()
        for row in reader:
            conc = _optional_number(row.get(conc_col))
            group = row.get("group_id") or None
            replicate = row.get(rep_col) or None
            series_id = row.get("series_id")
            # Old plate CSVs often reuse replicate numbers at each dose.
            key = ("series", series_id) if series_id else ("legacy", group, replicate, conc)
            grouped.setdefault(key, []).append((row, conc, group, replicate))
            if row.get("context"):
                contexts.add(row["context"])
    if len(contexts) > 1:
        raise ValueError("CSV contains conflicting dataset contexts.")

    series = []
    for rows in grouped.values():
        rows.sort(key=lambda item: float(item[0][time_col]))
        _, conc, group, replicate = rows[0]
        if any(item[1:] != (conc, group, replicate) for item in rows):
            raise ValueError(
                "Rows sharing a series_id have conflicting concentration or identifiers."
            )
        observations = {
            key: np.array([_observation_number(item[0].get(column)) for item in rows])
            for key, column in modality_columns.items()
        }
        concentrations = None
        if "concentration_at_time" in fields:
            values = [_optional_number(item[0].get("concentration_at_time")) for item in rows]
            if any(value is not None for value in values):
                if any(value is None for value in values):
                    raise ValueError(
                        "concentration_at_time must be specified at every series time point."
                    )
                concentrations = np.asarray(values, dtype=float)
        # Files written before group_id was added used the replicate as group.
        if "group_id" not in fields:
            group = replicate or "all"
        series.append(
            TimeSeriesData(
                times=np.array([float(item[0][time_col]) for item in rows]),
                observations=observations,
                concentration=conc,
                concentrations=concentrations,
                group_id=group,
                replicate_id=replicate,
            )
        )
    return ExperimentalDataset(
        series=series,
        name=Path(path).stem,
        context=next(iter(contexts), "in_vitro"),
    )


def _load_csv_numpy(
    path: str | Path,
    time_col: str,
    count_col: str,
    conc_col: str,
    rep_col: str,
) -> ExperimentalDataset:
    """Backward-compatible entry point for the dependency-free CSV parser."""
    return _load_csv_rows(path, time_col, count_col, conc_col, rep_col)
