#!/usr/bin/env python3
"""Extract the PhenoPop-stochastic authors' published fits as a reference set.

Wu et al. (2024) ship the *results* of their MATLAB analysis alongside the raw
data: per-mixture optima, AIC values and bootstrap confidence intervals for
three competing models. Those numbers are what makes a head-to-head possible
without re-running MATLAB -- U-MIMIC can be fitted to the same mixtures and
compared against the published estimates directly.

Two products are written to ``phenopop/``:

``phenopop_reference_fits_umimic.csv``
    One row per (dataset, variant, model): AIC, objective value at the
    optimum, number of free parameters, and the full optimum parameter vector
    kept verbatim as a JSON list.

``phenopop_reference_intervals_umimic.csv``
    One row per (dataset, variant, model, quantity): bootstrap CI bounds for
    the mixture proportion ``p`` and for the two subpopulation growth rates
    ``GR1``/``GR2``.

Models
------
``sto``
    Stochastic birth-death model -- the paper's contribution.
``dyn``
    Deterministic ("dynamic") mean model.
``hl``
    The original PhenoPop Hill-based deconvolution, i.e. the prior method the
    paper improves on.

Interpreting the parameter vectors
----------------------------------
Only element 0 of the birth-death vectors (``sto``/``dyn``) is interpreted
here, as the mixture proportion ``p``; it is the one position confirmed by
cross-checking against the published ``ci_*_p`` intervals. The remaining
entries are per-subpopulation growth/response parameters whose ordering is
defined by the authors' MATLAB objective, and are preserved verbatim rather
than guessed at. The ``hl`` vector has a different length and layout again, so
no point estimate is extracted from it -- only its published CIs.

The raw source is the PhenoPop_stochastic repository zip, expected at
``umimic/data/tmp/PhenoPop_stochastic-main.zip`` or passed via ``--source``.
That zip is **not shipped with the package**; only the converted products are.
Re-running this script therefore requires re-downloading it from
https://github.com/chenyuwu233/PhenoPop_stochastic (doi:10.1371/journal.pcbi.1011888).

Usage::

    python -m umimic.data.public.convert_phenopop_reference
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np

FIT_FIELDS = [
    "dataset",
    "variant",
    "model",
    "aic",
    "objective",
    "n_params",
    "mixture_proportion",
    "params_json",
    "source_file",
]

INTERVAL_FIELDS = [
    "dataset",
    "variant",
    "model",
    "quantity",
    "ci_low",
    "ci_high",
    "ci_width",
    "source_file",
]

#: Nominal sensitive:resistant seeding ratio per mixture, from the paper's
#: naming (BF_<a><b> = a:b). These are the ground truth a deconvolution has to
#: recover, and the reason this dataset is worth benchmarking against.
NOMINAL_RATIO = {
    "BF_11": "1:1",
    "BF_12": "1:2",
    "BF_21": "2:1",
    "BF_41": "4:1",
}

_MODELS = ("sto", "dyn", "hl")


def _scalar(value) -> float | None:
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size != 1 or not np.isfinite(arr[0]):
        return None
    return float(arr[0])


def _parse_name(stem: str) -> tuple[str, str]:
    """Split a result filename into (dataset, variant).

    ``CI_BF21_500c`` -> ``("BF_21", "ci_500c")``; ``BF_41`` -> ``("BF_41", "point")``.
    """
    is_ci = stem.startswith("CI_")
    body = stem[3:] if is_ci else stem

    match = re.match(r"^BF_?(\d{2})(.*)$", body)
    if match:
        dataset = f"BF_{match.group(1)}"
        suffix = match.group(2).strip("_")
    else:
        dataset = body
        suffix = ""

    if is_ci:
        variant = f"ci_{suffix}" if suffix else "ci"
    else:
        variant = suffix or "point"
    return dataset, variant


def _extract(mat: dict, stem: str) -> tuple[list[dict], list[dict]]:
    dataset, variant = _parse_name(stem)
    fits: list[dict] = []
    intervals: list[dict] = []

    for model in _MODELS:
        params = mat.get(f"opt_xx_{model}")
        aic = _scalar(mat.get(f"aic_{model}")) if f"aic_{model}" in mat else None
        fval = (
            _scalar(mat.get(f"opt_fval_{model}"))
            if f"opt_fval_{model}" in mat
            else None
        )

        if params is not None:
            vec = np.asarray(params, dtype=float).ravel()
            # Only the birth-death vectors have p in position 0; the Hill
            # vector uses a different layout, so no point estimate is claimed.
            proportion = float(vec[0]) if model in ("sto", "dyn") else None
            fits.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "model": model,
                    "aic": aic,
                    "objective": fval,
                    "n_params": int(vec.size),
                    "mixture_proportion": proportion,
                    "params_json": json.dumps([round(v, 10) for v in vec.tolist()]),
                    "source_file": f"{stem}.mat",
                }
            )

        for quantity, key in (
            ("p", f"ci_{model}_p"),
            ("GR1", f"ci_{model}_GR1"),
            ("GR2", f"ci_{model}_GR2"),
        ):
            if key not in mat:
                continue
            bounds = np.asarray(mat[key], dtype=float).ravel()
            if bounds.size != 2 or not np.all(np.isfinite(bounds)):
                continue
            low, high = float(bounds[0]), float(bounds[1])
            intervals.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "model": model,
                    "quantity": quantity,
                    "ci_low": low,
                    "ci_high": high,
                    "ci_width": high - low,
                    "source_file": f"{stem}.mat",
                }
            )

    return fits, intervals


def convert(source: Path, dest_dir: Path | None = None) -> tuple[Path, Path]:
    import scipy.io

    dest = dest_dir or Path(__file__).resolve().parent / "phenopop"
    dest.mkdir(parents=True, exist_ok=True)

    fits: list[dict] = []
    intervals: list[dict] = []

    with zipfile.ZipFile(source) as z:
        names = [
            n
            for n in z.namelist()
            if "/In vitro experiment/Result/" in n and n.endswith(".mat")
        ]
        if not names:
            raise SystemExit(
                f"No 'In vitro experiment/Result/*.mat' entries in {source}. "
                "Is this the PhenoPop_stochastic repository zip?"
            )
        for name in sorted(names):
            stem = Path(name).stem
            try:
                mat = scipy.io.loadmat(
                    io.BytesIO(z.read(name)), squeeze_me=True, struct_as_record=False
                )
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  ! skipping {stem}.mat ({type(exc).__name__}: {exc})")
                continue
            f, i = _extract(mat, stem)
            fits.extend(f)
            intervals.extend(i)

    for row in fits:
        row["nominal_ratio"] = NOMINAL_RATIO.get(row["dataset"], "")
    for row in intervals:
        row["nominal_ratio"] = NOMINAL_RATIO.get(row["dataset"], "")

    fits_path = dest / "phenopop_reference_fits_umimic.csv"
    with fits_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIT_FIELDS + ["nominal_ratio"])
        w.writeheader()
        w.writerows(fits)

    intervals_path = dest / "phenopop_reference_intervals_umimic.csv"
    with intervals_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INTERVAL_FIELDS + ["nominal_ratio"])
        w.writeheader()
        w.writerows(intervals)

    print(f"  PhenoPop reference fits:      {len(fits)} rows -> {fits_path}")
    print(f"  PhenoPop reference intervals: {len(intervals)} rows -> {intervals_path}")
    return fits_path, intervals_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    default = (
        Path(__file__).resolve().parent.parent
        / "tmp"
        / "PhenoPop_stochastic-main.zip"
    )
    p.add_argument("--source", type=Path, default=default)
    p.add_argument("--dest", type=Path, default=None)
    args = p.parse_args(argv)
    if not args.source.exists():
        raise SystemExit(
            f"Source not found: {args.source}\n"
            "Download from https://github.com/chenyuwu233/PhenoPop_stochastic"
        )
    convert(args.source, args.dest)


if __name__ == "__main__":
    main()
