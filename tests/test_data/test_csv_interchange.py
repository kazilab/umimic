"""Regression coverage for the generated-data CSV interchange contract."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from umimic.data import ExperimentalDataset, TimeSeriesData, load_csv, save_dataset


@pytest.mark.parametrize("modality", ["volume", "bli", "biomarker", "dead_count"])
def test_single_modality_without_counts_roundtrips(tmp_path, modality):
    original = TimeSeriesData(
        times=[0, 4],
        observations={modality: [0.25, np.nan]},
        group_id="treated",
        replicate_id="mouse_1",
    )
    path = save_dataset(
        ExperimentalDataset(series=[original], context="in_vivo"), tmp_path / "observations.csv"
    )
    restored = load_csv(path)
    assert restored.context == "in_vivo"
    assert restored.n_series == 1
    actual = restored.series[0]
    assert actual.modalities == [modality]
    np.testing.assert_equal(actual.observations[modality], original.observations[modality])
    assert actual.concentration is None
    assert actual.group_id == "treated"
    assert actual.replicate_id == "mouse_1"


def test_series_identity_concentrations_and_missing_values_roundtrip(tmp_path):
    original = ExperimentalDataset(
        series=[
            TimeSeriesData(
                times=[0, 4],
                observations={"cell_counts": [100, 150], "bli": [1, np.nan]},
                concentration=0,
                group_id="control",
                replicate_id="1",
            ),
            TimeSeriesData(
                times=[0, 4],
                observations={"cell_counts": [100, 80], "bli": [2, 3]},
                concentration=10,
                group_id="treated",
                replicate_id="1",
            ),
            # Identical public labels still identify separate series in an export.
            TimeSeriesData(
                times=[0, 4],
                observations={"cell_counts": [90, 75], "bli": [4, 5]},
                concentration=10,
                group_id="treated",
                replicate_id="1",
            ),
            TimeSeriesData(
                times=[0, 4],
                observations={"cell_counts": [100, 85], "bli": [6, 7]},
                concentrations=[2, 1],
            ),
        ]
    )
    restored = load_csv(save_dataset(original, tmp_path / "plate.csv"))
    assert restored.n_series == original.n_series
    for expected, actual in zip(original.series, restored.series):
        assert actual.group_id == expected.group_id
        assert actual.replicate_id == expected.replicate_id
        assert actual.concentration == expected.concentration
        if expected.concentrations is None:
            assert actual.concentrations is None
        else:
            np.testing.assert_equal(actual.concentrations, expected.concentrations)
        np.testing.assert_equal(actual.times, expected.times)
        for key in expected.modalities:
            np.testing.assert_equal(actual.observations[key], expected.observations[key])


def test_legacy_csv_groups_replicates_within_concentration_and_sorts(tmp_path):
    path = tmp_path / "plate.csv"
    path.write_text(
        "hours,viable_count,concentration,replicate_id\n"
        "4,80,10,1\n0,100,0,1\n0,100,10,1\n4,150,0,1\n",
        encoding="utf-8",
    )
    restored = load_csv(path)
    assert restored.n_series == 2
    assert restored.concentrations == [0, 10]
    for series in restored.series:
        np.testing.assert_equal(series.times, [0, 4])


def test_custom_columns_and_single_row_work_without_pandas(tmp_path, monkeypatch):
    path = tmp_path / "single.csv"
    path.write_text('time,n,dose,well,photon_flux\n4,80,10,"001, A",2\n', encoding="utf-8")
    config = SimpleNamespace(
        time_column="time", count_column="n", concentration_column="dose", replicate_column="well"
    )
    monkeypatch.setitem(sys.modules, "pandas", None)
    restored = load_csv(path, config)
    series = restored.series[0]
    assert series.replicate_id == "001, A"
    assert series.concentration == 10
    np.testing.assert_equal(series.observations["cell_counts"], [80])
    np.testing.assert_equal(series.observations["bli"], [2])


def test_duplicate_times_are_rejected_without_dropping_measurements(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text("hours,volume\n0,1\n0,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strictly increasing"):
        load_csv(path)


@pytest.mark.parametrize("concentration", ["", "NaN"])
def test_unspecified_concentration_remains_none(tmp_path, concentration):
    path = tmp_path / "missing_concentration.csv"
    path.write_text(f"hours,bli,concentration\n0,1,{concentration}\n", encoding="utf-8")
    assert load_csv(path).series[0].concentration is None


@pytest.mark.parametrize("concentration", ["inf", "-1"])
def test_invalid_concentration_is_rejected(tmp_path, concentration):
    path = tmp_path / "bad_concentration.csv"
    path.write_text(f"hours,bli,concentration\n0,1,{concentration}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="concentration"):
        load_csv(path)


def test_exported_series_cannot_change_identifiers_midway(tmp_path):
    path = tmp_path / "conflicting_series.csv"
    path.write_text(
        "hours,bli,concentration,replicate_id,series_id\n0,1,0,A,0\n4,2,10,A,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicting"):
        load_csv(path)


@pytest.mark.parametrize("suffix", [".csv.xz", ".csv.gz", ".csv.bz2"])
def test_compressed_csv_roundtrip(tmp_path, suffix):
    dataset = ExperimentalDataset(
        series=[
            TimeSeriesData(times=[0, 4], observations={"bli": [1, 2]}, concentration=10),
        ]
    )
    restored = load_csv(save_dataset(dataset, tmp_path / ("data" + suffix)))
    np.testing.assert_equal(restored.series[0].observations["bli"], [1, 2])
    assert restored.series[0].concentration == 10
