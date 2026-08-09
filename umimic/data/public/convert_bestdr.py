#!/usr/bin/env python3
"""Build ``bestdr/bestdr_umimic.csv.xz`` from the public BESTDR count CSVs.

Expects either:

* ``umimic/data/tmp/bestdr_data.zip`` (local cache; **not shipped with the
  package** -- only the converted product is, so re-running this script means
  re-downloading the source), or
* ``--replicates`` / ``--dose-response`` CSV paths.

Usage::

    python -m umimic.data.public.convert_bestdr
"""

from __future__ import annotations

import argparse
import csv
import io
import lzma
import zipfile
from pathlib import Path

FIELDS = [
    "hours",
    "viable_count",
    "dead_count",
    "total_count",
    "concentration",
    "well",
    "img",
    "fov_id",
    "seed",
    "cell_line",
    "drug",
    "plate",
    "replicate_id",
    "quality_flag",
]


def _rows_from_csv(text: str, plate: str) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(io.StringIO(text))
    rows: list[dict] = []
    for _, r in df.iterrows():
        live = float(r["intersect_live_cells"])
        dead = float(r["intersect_dead_cells"])
        tot = float(r["tot_cells"])
        flag = "ok"
        if live < 0:
            flag = "negative_live_clipped"
            live = 0.0
        if dead < 0:
            flag = "negative_dead_clipped" if flag == "ok" else flag + "+dead"
            dead = 0.0
        well = str(r["well"])
        img = int(r["img"])
        seed = str(r["seed"])
        conc = float(r["concentration"])
        rows.append(
            {
                "hours": float(r["Elapsed"]),
                "viable_count": live,
                "dead_count": dead,
                "total_count": tot if tot >= 0 else live + dead,
                "concentration": conc,
                "well": well,
                "img": img,
                "fov_id": f"{well}_img{img}",
                "seed": seed,
                "cell_line": "HCT116",
                "drug": "cisplatin",
                "plate": plate,
                "replicate_id": f"{plate}_{well}_img{img}_seed{seed}_c{conc:g}",
                "quality_flag": flag,
            }
        )
    return rows


def convert(
    zip_path: Path | None = None,
    replicates_csv: Path | None = None,
    dose_csv: Path | None = None,
    dest_dir: Path | None = None,
) -> Path:
    dest = dest_dir or Path(__file__).resolve().parent / "bestdr"
    dest.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    if zip_path and zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            rows.extend(
                _rows_from_csv(
                    z.read("HCT116_Cisplatin_Replicates.csv").decode(),
                    "replicates",
                )
            )
            rows.extend(
                _rows_from_csv(
                    z.read("HCT116_Cisplatin_DoseResponse.csv").decode(),
                    "dose_response",
                )
            )
    else:
        if replicates_csv is None or dose_csv is None:
            raise FileNotFoundError(
                "Provide bestdr_data.zip or both --replicates and --dose-response CSVs."
            )
        rows.extend(_rows_from_csv(replicates_csv.read_text(), "replicates"))
        rows.extend(_rows_from_csv(dose_csv.read_text(), "dose_response"))

    rows.sort(
        key=lambda x: (x["plate"], x["concentration"], x["well"], x["img"], x["hours"])
    )
    out = dest / "bestdr_umimic.csv.xz"
    with lzma.open(out, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  BESTDR: {len(rows)} rows -> {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    default_zip = Path(__file__).resolve().parent.parent / "tmp" / "bestdr_data.zip"
    p.add_argument("--zip", type=Path, default=default_zip)
    p.add_argument("--replicates", type=Path, default=None)
    p.add_argument("--dose-response", type=Path, default=None)
    p.add_argument("--dest", type=Path, default=None)
    args = p.parse_args(argv)
    convert(args.zip if args.zip.exists() else None, args.replicates, args.dose_response, args.dest)


if __name__ == "__main__":
    main()
