"""TSHS public product: citation, QC exclusions, loader smoke."""

from __future__ import annotations

import csv
import lzma
from pathlib import Path

import pytest

from umimic.data import TSHS_CITATION, load_catalog, load_tshs_tumor
from umimic.data.public_datasets import DATA_ROOT


@pytest.fixture(scope="module")
def tshs_dir() -> Path:
    d = DATA_ROOT / "tshs_tumor"
    product = d / "tshs_tumor_umimic.csv.xz"
    if not product.exists():
        pytest.skip("TSHS product not present in package data tree")
    return d


def test_catalog_has_tshs_citation_and_exclusions():
    catalog = load_catalog()
    entry = catalog["datasets"]["tshs_tumor"]
    assert "Daskalakis" in entry["citation"]
    excluded = entry["data_quality"]["excluded_values"]
    vols = {float(x["tumor_volume"]) for x in excluded}
    assert 623.7 in vols and 734.4 in vols


def test_product_excludes_orange_values(tshs_dir: Path):
    path = tshs_dir / "tshs_tumor_umimic.csv.xz"
    with lzma.open(path, "rt") as f:
        rows = list(csv.DictReader(f))
    # Mouse 101 day 10 legitimately has 623.7; the excluded point is mouse 202 day 3.
    bad_pairs = {
        (r["mouse_id"], float(r["day"]), float(r["tumor_volume"]))
        for r in rows
    }
    assert ("202", 3.0, 623.7) not in bad_pairs
    assert ("302", 26.0, 734.4) not in bad_pairs
    assert len(rows) == 573
    assert all(r["quality_flag"] != "exclude_clear_error" for r in rows)


def test_excluded_audit_file(tshs_dir: Path):
    path = tshs_dir / "EXCLUDED_VALUES.csv"
    assert path.exists()
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {float(r["tumor_volume"]) for r in rows} == {623.7, 734.4}


def test_load_tshs_tumor_smoke(tshs_dir: Path):
    ds = load_tshs_tumor()
    assert ds.context == "in_vivo"
    assert len(ds.series) == 37
    assert all("volume" in s.observations for s in ds.series)
    assert all(s.metadata.get("citation") == TSHS_CITATION for s in ds.series)
    # No series should contain the excluded orange cells
    for s in ds.series:
        mouse = str(s.metadata.get("mouse_id", ""))
        days = s.times / 24.0
        vols = s.observations["volume"]
        for day, vol in zip(days, vols):
            if mouse == "202":
                assert not (abs(day - 3.0) < 1e-9 and abs(vol - 623.7) < 1e-6)
            if mouse == "302":
                assert not (abs(day - 26.0) < 1e-9 and abs(vol - 734.4) < 1e-6)


def test_load_tshs_treatment_filter(tshs_dir: Path):
    control = load_tshs_tumor(treatment_group="control")
    assert len(control.series) == 8
    assert all(
        "CONTROL" in str(s.metadata.get("treatment_group", "")).upper()
        for s in control.series
    )
