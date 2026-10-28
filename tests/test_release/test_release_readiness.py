"""Release-readiness smoke checks for packaging and data path defaults."""

from __future__ import annotations

import tomllib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from umimic.data import public_datasets


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_default_data_root_points_to_package_public_dir(monkeypatch):
    """Default dataset root should resolve relative to the installed module."""
    expected = Path(public_datasets.__file__).resolve().parent / "public"
    monkeypatch.delenv("UMIMIC_DATA_ROOT", raising=False)
    assert public_datasets._resolve_default_data_root() == expected


def test_pyproject_readme_exists():
    """project.readme should point to an existing file for wheel metadata."""
    pyproject = PROJECT_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    readme = data["project"]["readme"]
    assert (PROJECT_ROOT / readme).exists()


def test_setuptools_package_discovery_not_src_only():
    """Setuptools find config should match this repository's non-src layout."""
    pyproject = PROJECT_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    where = data["tool"]["setuptools"]["packages"]["find"]["where"]
    assert "." in where


def test_readthedocs_config_exists():
    """Read the Docs build config should be present at repository root."""
    assert (PROJECT_ROOT / ".readthedocs.yaml").exists()


def test_docs_entrypoints_exist():
    """Docs landing files needed by RTD/Sphinx should exist."""
    assert (PROJECT_ROOT / "docs" / "conf.py").exists()
    assert (PROJECT_ROOT / "docs" / "index.md").exists()


def test_pypi_publish_workflow_exists():
    """Trusted publishing workflow should be present for PyPI releases."""
    if not (PROJECT_ROOT / ".github").exists():
        pytest.skip("Source export does not include GitHub workflow metadata.")
    workflow = PROJECT_ROOT / ".github" / "workflows" / "publish-pypi.yml"
    assert workflow.exists()


def test_version_has_one_build_and_runtime_source():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" in pyproject["project"]["dynamic"]
    assert "version" not in pyproject["project"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "umimic.__version__"}


def test_installed_wheel_includes_public_data_and_matching_version(tmp_path):
    """Build and install outside the checkout so source files cannot mask omissions."""
    source = tmp_path / "source"
    source.mkdir()
    for filename in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(PROJECT_ROOT / filename, source / filename)
    shutil.copytree(
        PROJECT_ROOT / "umimic",
        source / "umimic",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    env = dict(os.environ)
    env.pop("UMIMIC_DATA_ROOT", None)
    env.pop("PYTHONPATH", None)
    wheels = tmp_path / "wheels"
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--disable-pip-version-check",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--wheel-dir",
            str(wheels),
            str(source),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    wheel = next(wheels.glob("umimic-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        for relative in (
            "catalog/datasets.json",
            "bestdr/bestdr_umimic.csv.xz",
            "phenopop/phenopop_umimic.csv.xz",
            "hafner_gr/hafner_gr_umimic.csv.xz",
            "nci60/nci60_umimic.csv.xz",
            "tshs_tumor/tshs_tumor_umimic.csv.xz",
            "tshs_tumor/EXCLUDED_VALUES.csv",
            "tshs_tumor/tumorgrowth.xlsx",
            "whiting2025_barcode/whiting2025_barcode_population_umimic.csv.xz",
            "whiting2025_barcode/whiting2025_barcode_lineage_umimic.csv.xz",
            "csc_dip/csc_dip_reference_aic_umimic.csv",
            "phenopop/phenopop_reference_fits_umimic.csv",
            "phenopop/phenopop_reference_intervals_umimic.csv",
        ):
            assert f"umimic/data/public/{relative}" in names
        assert "umimic/dynamics/SCIENTIFIC_ASSUMPTIONS.md" in names

    installed = tmp_path / "installed"
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--disable-pip-version-check",
            "install",
            "--no-deps",
            "--no-index",
            "--no-compile",
            "--target",
            str(installed),
            str(wheel),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, install.stdout + install.stderr
    env["PYTHONPATH"] = str(installed)
    smoke = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
from importlib.metadata import version
from pathlib import Path
import umimic
from umimic.data import load_catalog, load_tshs_tumor, load_phenopop, load_bestdr
assert Path(umimic.__file__).is_relative_to(Path.cwd() / 'installed')
assert umimic.__version__ == version('umimic')
assert load_catalog()['datasets']
assert load_tshs_tumor().n_series == 37
assert load_phenopop(population='mixed').n_series > 0
assert load_bestdr(cell_line='HCT116', drug='cisplatin').n_series > 0
print(json.dumps({'version': umimic.__version__}))
""",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert smoke.returncode == 0, smoke.stdout + smoke.stderr
    from umimic import __version__

    assert json.loads(smoke.stdout)["version"] == __version__
