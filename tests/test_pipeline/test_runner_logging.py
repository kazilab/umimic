"""Tests for CLI run logging utilities."""

from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path

from umimic.pipeline.runner import _configure_logging, _default_log_path


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
