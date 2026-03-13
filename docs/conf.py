"""Sphinx configuration for U-MIMIC documentation."""

from __future__ import annotations

from datetime import datetime

project = "U-MIMIC"
author = "Data Analysis Team @KaziLab.se"
copyright = f"{datetime.now().year}, @kazilab.se"

extensions = [
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST markdown support
myst_enable_extensions = [
    "colon_fence",
]
