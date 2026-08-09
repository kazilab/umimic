"""NCI-60 honest loader: filters, reconstructions, safety caps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from umimic.data import (
    NCI60_CITATION,
    load_nci60,
    resolve_nci60_nsc,
)
from umimic.data.public_datasets import DATA_ROOT


@pytest.fixture(scope="module")
def nci_product() -> Path:
    path = DATA_ROOT / "nci60" / "nci60_umimic.csv.xz"
    if not path.exists():
        pytest.skip("NCI-60 product not present")
    return path


def test_resolve_nci60_nsc_aliases():
    assert resolve_nci60_nsc("cisplatin") == "119875"
    assert resolve_nci60_nsc(119875) == "119875"
    assert resolve_nci60_nsc("119875") == "119875"
    assert resolve_nci60_nsc(None) is None


def test_load_nci60_cisplatin_mcf7_reconstructed(nci_product: Path):
    ds = load_nci60(nsc="cisplatin", cell_line="MCF7")
    assert ds.context == "in_vitro"
    assert ds.name == "nci60_dtp"
    assert len(ds.series) >= 3
    assert all(s.n_timepoints == 2 for s in ds.series)
    assert all("cell_counts" in s.observations for s in ds.series)
    s0 = ds.series[0]
    assert s0.times[0] == 0.0
    assert s0.times[1] == 48.0
    assert s0.observations["cell_counts"][0] == 10_000.0
    assert s0.metadata["citation"] == NCI60_CITATION
    assert s0.metadata["NSC"] == "119875"
    assert "MCF7" in str(s0.metadata["cell_line"]).upper()
    assert "reconstruction" in s0.metadata
    # Endpoint non-negative after floor
    for s in ds.series:
        assert np.all(s.observations["cell_counts"] >= 1.0)


def test_load_nci60_endpoint_metrics(nci_product: Path):
    ds = load_nci60(
        nsc=119875,
        cell_line="MCF7",
        representation="endpoint_metrics",
    )
    assert len(ds.series) >= 1
    s0 = ds.series[0]
    assert s0.n_timepoints == 1
    assert "growth_inhibition_pct" in s0.observations
    assert "percent_of_control" in s0.observations
    assert s0.metadata["representation"] == "endpoint_metrics"


def test_load_nci60_panel_filter(nci_product: Path):
    ds = load_nci60(nsc="cisplatin", panel="Breast", max_series=50)
    assert len(ds.series) >= 1
    assert all("breast" in str(s.metadata.get("panel", "")).lower() for s in ds.series)


def test_load_nci60_no_match(nci_product: Path):
    with pytest.raises(FileNotFoundError, match="No NCI-60 rows"):
        load_nci60(nsc=119875, cell_line="DefinitelyNotARealLineXYZ")


def test_load_nci60_invalid_representation(nci_product: Path):
    with pytest.raises(ValueError, match="representation"):
        load_nci60(nsc=119875, cell_line="MCF7", representation="magic")


def test_load_nci60_aggregate_none_has_experiment_ids(nci_product: Path):
    ds_mean = load_nci60(nsc=119875, cell_line="MCF7", aggregate="mean")
    ds_none = load_nci60(nsc=119875, cell_line="MCF7", aggregate="none")
    # Keeping experiments should not shrink the set
    assert len(ds_none.series) >= len(ds_mean.series)
    assert any(s.metadata.get("experiment_id") for s in ds_none.series)
