"""Release-readiness smoke checks for packaging and data path defaults."""

from __future__ import annotations

import tomllib
from pathlib import Path

from umimic.data import public_datasets


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_default_data_root_points_to_package_public_dir():
    """Default dataset root should resolve relative to the installed module."""
    expected = Path(public_datasets.__file__).resolve().parent / "public"
    assert public_datasets.DATA_ROOT == expected


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
    workflow = PROJECT_ROOT / ".github" / "workflows" / "publish-pypi.yml"
    assert workflow.exists()
