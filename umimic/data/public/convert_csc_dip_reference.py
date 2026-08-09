#!/usr/bin/env python3
"""Extract the CSC / drug-induced-plasticity authors' published AIC results.

Wu, Gunnarsson, Foo & Leder (2025) test for *therapy-induced* resistance by
model selection: they fit competing models to the same drug-screen data and
compare information criteria. Their repository ships those criteria but **not
the measurements they were computed from** -- every data input is loaded from a
workbook or ``.mat`` that is absent from the repository (see the ``csc_dip``
entry in ``catalog/datasets.json`` for the full list).

That makes this extraction narrow but worth keeping: their *conclusions* can be
compared against U-MIMIC's on the same model-selection question, even though
their fit cannot be reproduced. Without it, deleting the source repository
would lose the only comparable output from the paper.

Product written to ``csc_dip/``:

``csc_dip_reference_aic_umimic.csv``
    One row per (experiment, model, criterion): the published AIC or AICc.

Model naming
------------
``DIP`` / ``nDIP``
    With and without drug-induced plasticity -- the hypothesis under test.
``Asy`` / ``nAsy``
    With and without asymmetric division.
``TD2``
    Time-delay variant of the AGS analysis.
``Model_I`` / ``Model_Ia`` / ``Model_II`` / ``Model_IIa``
    The manuscript's labels, retained where the authors used them; these are
    aliases of the DIP/nDIP pairs in the same file rather than extra models.

Only scalar criteria are extracted. Parameter estimates are deliberately not
exported: without the underlying data their scale cannot be checked, and an
uncheckable parameter vector invites exactly the kind of misuse this package
tries to avoid.

The raw source is the repository zip, expected at
``umimic/data/tmp/Cancer-Stem-Cells-Drug-induced-Plasticity-main.zip`` or
passed via ``--source``. It is **not shipped with the package**. Re-running
requires re-downloading from
https://github.com/chenyuwu233/Cancer-Stem-Cells-Drug-induced-Plasticity
(doi:10.1038/s41540-025-00560-8).

Usage::

    python -m umimic.data.public.convert_csc_dip_reference
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import zipfile
from pathlib import Path

import numpy as np

FIELDS = [
    "experiment",
    "context",
    "model",
    "criterion",
    "value",
    "source_file",
]

#: Which experiment each result file belongs to, and whether it is a real
#: measurement or a simulation study.
_EXPERIMENTS = {
    "In_vitro_AGS_AIC": ("AGS_gastric_CPX-O", "in_vitro"),
    "In_vitro_AGS_Time_Delay_AIC": ("AGS_gastric_CPX-O_time_delay", "in_vitro"),
    "In_vitro_COLO858_60h_AIC": ("COLO858_vemurafenib_60h", "in_vitro"),
}

_CRITERION = re.compile(r"(AICc|AIC)")


def _classify(key: str) -> tuple[str, str] | None:
    """Split a variable name into (model, criterion), or None if not a criterion."""
    match = _CRITERION.search(key)
    if match is None:
        return None
    criterion = match.group(1)
    model = (key[: match.start()] + key[match.end() :]).strip("_")
    return (model or "default"), criterion


def convert(source: Path, dest_dir: Path | None = None) -> Path:
    import scipy.io

    dest = dest_dir or Path(__file__).resolve().parent / "csc_dip"
    dest.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with zipfile.ZipFile(source) as z:
        names = [
            n
            for n in z.namelist()
            if "/In vitro experiments/Result/" in n and n.endswith(".mat")
        ]
        if not names:
            raise SystemExit(
                f"No 'In vitro experiments/Result/*.mat' entries in {source}. "
                "Is this the Cancer-Stem-Cells-Drug-induced-Plasticity zip?"
            )
        for name in sorted(names):
            stem = Path(name).stem
            experiment, context = _EXPERIMENTS.get(stem, (stem, "unknown"))
            mat = scipy.io.loadmat(
                io.BytesIO(z.read(name)), squeeze_me=True, struct_as_record=False
            )
            for key, value in sorted(mat.items()):
                if key.startswith("__"):
                    continue
                arr = np.asarray(value)
                if arr.shape != () or arr.dtype.kind not in "fiu":
                    continue
                classified = _classify(key)
                if classified is None:
                    continue
                model, criterion = classified
                rows.append(
                    {
                        "experiment": experiment,
                        "context": context,
                        "model": model,
                        "criterion": criterion,
                        "value": float(arr),
                        "source_file": f"{stem}.mat",
                    }
                )

    out = dest / "csc_dip_reference_aic_umimic.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSC/DIP reference AIC: {len(rows)} rows -> {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    default = (
        Path(__file__).resolve().parent.parent
        / "tmp"
        / "Cancer-Stem-Cells-Drug-induced-Plasticity-main.zip"
    )
    p.add_argument("--source", type=Path, default=default)
    p.add_argument("--dest", type=Path, default=None)
    args = p.parse_args(argv)
    if not args.source.exists():
        raise SystemExit(
            f"Source not found: {args.source}\n"
            "Download from https://github.com/chenyuwu233/"
            "Cancer-Stem-Cells-Drug-induced-Plasticity"
        )
    convert(args.source, args.dest)


if __name__ == "__main__":
    main()
