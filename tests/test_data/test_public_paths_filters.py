"""External-cache defaults and documented public-data filters."""

import csv
import importlib
import lzma
from pathlib import Path

import pytest

from umimic.data import public_datasets


@pytest.mark.parametrize("alias", ["mixed", "mixture", "mix", "MIXED"])
def test_phenopop_mixture_aliases_match_canonical_product(tmp_path, alias):
    dest = tmp_path / "phenopop"
    dest.mkdir()
    with lzma.open(dest / "phenopop_umimic.csv.xz", "wt", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["hours", "viable_count", "concentration", "replicate_id", "population"])
        writer.writerows(
            [
                [0, 100, 0, "mix_1to1_rep0", "mix_1to1"],
                [4, 150, 0, "mix_1to1_rep0", "mix_1to1"],
                [0, 100, 0, "sensitive_rep0", "sensitive"],
                [4, 150, 0, "sensitive_rep0", "sensitive"],
            ]
        )
    restored = public_datasets.load_phenopop(data_root=tmp_path, population=alias)
    assert restored.n_series == 1
    assert restored.series[0].metadata["population"] == "mix_1to1"


def test_nonexistent_external_root_is_respected(tmp_path, monkeypatch):
    target = tmp_path / "new" / "cache"
    monkeypatch.setenv("UMIMIC_DATA_ROOT", str(target))
    assert public_datasets._resolve_default_data_root() == target.resolve()


def test_downloader_uses_and_creates_external_cache(tmp_path, monkeypatch):
    from umimic.data.public import download_datasets

    target = tmp_path / "new" / "cache"
    original = download_datasets.DATA_DIR
    monkeypatch.setenv("UMIMIC_DATA_ROOT", str(target))
    importlib.reload(download_datasets)
    paths = []
    monkeypatch.setattr(
        download_datasets, "download_file", lambda url, dest, desc="": paths.append(dest)
    )
    try:
        assert download_datasets.DATA_DIR == target.resolve()
        download_datasets.download_tshs_tumor()
        assert (target / "tshs_tumor").is_dir()
        assert paths and all(path.is_relative_to(target) for path in paths)
    finally:
        download_datasets.DATA_DIR = original


def test_tshs_converter_source_default_uses_external_cache(tmp_path, monkeypatch):
    from umimic.data.public import convert_tshs

    monkeypatch.setenv("UMIMIC_DATA_ROOT", str(tmp_path))
    calls = []
    monkeypatch.setattr(convert_tshs, "convert", lambda *args, **kwargs: calls.append(args))
    convert_tshs.main([])
    assert calls[0][0] == Path(tmp_path) / "tshs_tumor" / "tumorgrowth.xlsx"
