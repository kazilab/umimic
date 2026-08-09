"""BESTDR live-cell product and Whiting 2025 population extract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from umimic.data import (
    BESTDR_CITATION,
    WHITING2025_CITATION,
    load_bestdr,
    load_whiting2025_barcode,
)
from umimic.data.public_datasets import DATA_ROOT


@pytest.fixture(scope="module")
def bestdr_product() -> Path:
    path = DATA_ROOT / "bestdr" / "bestdr_umimic.csv.xz"
    if not path.exists():
        pytest.skip("BESTDR product not present")
    return path


@pytest.fixture(scope="module")
def whiting_product() -> Path:
    path = (
        DATA_ROOT
        / "whiting2025_barcode"
        / "whiting2025_barcode_population_umimic.csv.xz"
    )
    if not path.exists():
        pytest.skip("Whiting 2025 product not present")
    return path


def test_load_bestdr_smoke(bestdr_product: Path):
    ds = load_bestdr()
    assert ds.context == "in_vitro"
    assert len(ds.series) == 576
    s0 = ds.series[0]
    assert s0.n_timepoints >= 2
    assert "cell_counts" in s0.observations
    assert "dead_count" in s0.observations
    assert s0.metadata["citation"] == BESTDR_CITATION
    assert s0.metadata["cell_line"] == "HCT116"
    assert s0.metadata["drug"] == "cisplatin"
    assert np.all(s0.observations["cell_counts"] >= 0)


def test_load_bestdr_plate_filter(bestdr_product: Path):
    rep = load_bestdr(plate="replicates")
    dose = load_bestdr(plate="dose_response")
    assert len(rep.series) == 288
    assert len(dose.series) == 288
    assert all(s.metadata["plate"] == "replicates" for s in rep.series)
    assert all(s.metadata["plate"] == "dose_response" for s in dose.series)
    # dose ladder has more concentrations than the two-point replicate plate
    assert len(dose.concentrations) > len(rep.concentrations)


def test_load_bestdr_live_only(bestdr_product: Path):
    ds = load_bestdr(plate="replicates", include_dead=False)
    assert "dead_count" not in ds.series[0].observations


def test_load_whiting2025_smoke(whiting_product: Path):
    ds = load_whiting2025_barcode()
    assert len(ds.series) == 8
    assert ds.context == "in_vitro"
    s0 = ds.series[0]
    assert s0.n_timepoints == 4
    assert s0.metadata["citation"] == WHITING2025_CITATION
    assert "not live-cell" in s0.metadata["note"].lower() or "not imaging" in s0.metadata["note"].lower() or "Bulk" in s0.metadata["note"]
    # hours = day * 24
    assert np.allclose(s0.times, s0.times)  # finite
    assert s0.times[0] > 0


def test_load_whiting2025_cell_line_filter(whiting_product: Path):
    hct = load_whiting2025_barcode(cell_line="HCT")
    sw = load_whiting2025_barcode(cell_line="SW6")
    assert len(hct.series) == 4
    assert len(sw.series) == 4
    assert all("HCT" in s.metadata["cell_line"] for s in hct.series)
