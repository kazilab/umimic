"""Tests for CLI run logging utilities."""

from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path

from umimic.pipeline.runner import _configure_logging, _default_log_path, _run_config


def test_default_log_path_uses_output_dir(tmp_path):
    args = Namespace(command="simulate", output=str(tmp_path), log_file=None, log_level="INFO")
    log_path = _default_log_path(args)

    assert log_path.parent == tmp_path
    assert log_path.name.startswith("simulate_")
    assert log_path.suffix == ".log"


def test_configure_logging_writes_to_file(tmp_path):
    log_path = tmp_path / "run.log"
    args = Namespace(
        command="generate",
        output=str(tmp_path),
        log_file=str(log_path),
        log_level="INFO",
    )

    configured_path = _configure_logging(args)
    assert configured_path == log_path

    logger = logging.getLogger("umimic.cli")
    logger.info("smoke-log-entry")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert log_path.exists()
    assert "smoke-log-entry" in Path(log_path).read_text(encoding="utf-8")


def test_run_config_suggest_coupling_ranges_outputs_json(capsys):
    args = Namespace(
        command="config",
        config_command="suggest-coupling-ranges",
        output_format="json",
    )
    _run_config(args)
    out = capsys.readouterr().out
    assert '"hill"' in out
    assert '"logistic"' in out
    assert '"target_overrides"' in out


def test_run_config_suggest_coupling_ranges_outputs_yaml(capsys):
    args = Namespace(
        command="config",
        config_command="suggest-coupling-ranges",
        output_format="yaml",
    )
    _run_config(args)
    out = capsys.readouterr().out
    assert "hill:" in out
    assert "logistic:" in out
    assert "target_overrides:" in out
