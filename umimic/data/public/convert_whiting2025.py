#!/usr/bin/env python3
"""Build the Whiting 2025 products from Supplementary Data 2.

Two products are written:

``whiting2025_barcode_population_umimic.csv.xz``
    Bulk population sizes per cell line, timepoint and replicate (32 rows).

``whiting2025_barcode_lineage_umimic.csv.xz``
    Long-form barcode abundances: one row per (cell line, barcode, sampling
    timepoint, plate). This is the lineage-tracing measurement the paper is
    about -- 702k source rows across HCT116 and SW620 -- and without it the
    population product alone cannot support any lineage-informed analysis.

The raw source is a Nature Communications ESM zip (MOESM4), expected at
``umimic/data/tmp/41467_2025_59479_MOESM4_ESM.zip`` or passed via ``--source``.
That zip is **not shipped with the package**; only the converted products are.
Re-running this script therefore requires downloading the supplement again
from doi:10.1038/s41467-025-59479-7.

Usage::

    python -m umimic.data.public.convert_whiting2025
"""

from __future__ import annotations

import argparse
import csv
import io
import lzma
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

FIELDS = [
    "hours",
    "day",
    "population_size",
    "concentration",
    "group_id",
    "cell_line",
    "replicate",
    "modality_note",
]

LINEAGE_FIELDS = [
    "cell_line",
    "barcode",
    "timepoint",
    "plate",
    "sample",
    "count",
]


def _sheet_rows(xlsx_bytes: bytes, sheet_index: int) -> list[list[str | None]]:
    z = zipfile.ZipFile(io.BytesIO(xlsx_bytes))
    sst: list[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall(
            "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"
        ):
            texts = [
                t.text or ""
                for t in si.iter(
                    "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
                )
            ]
            sst.append("".join(texts))
    sheets = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))
    sheet = ET.fromstring(z.read(sheets[sheet_index]))
    out: list[list[str | None]] = []
    for row in sheet.findall(
        "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheetData/"
        "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"
    ):
        vals: list[str | None] = []
        for c in row.findall(
            "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"
        ):
            t = c.attrib.get("t")
            v = c.find(
                "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v"
            )
            val = v.text if v is not None else None
            if t == "s" and val is not None:
                val = sst[int(val)]
            vals.append(val)
        out.append(vals)
    return out


def convert(source: Path, dest_dir: Path | None = None) -> Path:
    dest = dest_dir or Path(__file__).resolve().parent / "whiting2025_barcode"
    dest.mkdir(parents=True, exist_ok=True)

    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            xlsx_bytes = z.read("Supp_Data/Supp_Data_2.xlsx")
    else:
        xlsx_bytes = source.read_bytes()

    rows: list[dict] = []
    for sheet_i, line, n_tp in ((0, "HCT116_barcode", 4), (2, "SW620_barcode", 4)):
        body = _sheet_rows(xlsx_bytes, sheet_i)[1:]
        pts = [(float(r[0]), float(r[1])) for r in body if r and r[0] is not None]
        n_rep = len(pts) // n_tp
        for rep in range(n_rep):
            block = pts[rep * n_tp : (rep + 1) * n_tp]
            for day, pop in block:
                rows.append(
                    {
                        "hours": day * 24.0,
                        "day": day,
                        "population_size": pop,
                        "concentration": 0.0,
                        "group_id": f"{line}_rep{rep + 1}",
                        "cell_line": line,
                        "replicate": rep + 1,
                        "modality_note": "bulk_population_size_not_imaging",
                    }
                )

    out = dest / "whiting2025_barcode_population_umimic.csv.xz"
    with lzma.open(out, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  Whiting 2025 population: {len(rows)} rows -> {out}")

    convert_lineage(xlsx_bytes, dest)
    return out


def _parse_sample_column(name: str) -> tuple[int, int]:
    """Split a barcode column header like ``DT3_P2`` into (timepoint, plate).

    ``DT`` indexes the sequencing timepoint and ``P`` the replicate plate, so
    the wide matrix is really a (barcode x timepoint x plate) array.
    """
    dt_part, p_part = name.split("_")
    return int(dt_part[2:]), int(p_part[1:])


def convert_lineage(xlsx_bytes: bytes, dest: Path) -> Path:
    """Write the long-form barcode abundance product.

    The source sheets are wide (one column per timepoint x plate); long form is
    what every downstream consumer wants, and it keeps zero counts explicit --
    a barcode absent at a timepoint is an extinction event, not missing data,
    and dropping those rows would silently bias any lineage analysis.
    """
    rows: list[dict] = []
    for sheet_i, line in ((1, "HCT116_barcode"), (3, "SW620_barcode")):
        body = _sheet_rows(xlsx_bytes, sheet_i)
        if not body:
            continue
        header = [h for h in body[0] if h]
        sample_cols = header[1:]
        for record in body[1:]:
            if not record or record[0] is None:
                continue
            barcode = str(record[0])
            for col_i, col_name in enumerate(sample_cols, start=1):
                if col_i >= len(record):
                    continue
                raw = record[col_i]
                if raw is None:
                    continue
                timepoint, plate = _parse_sample_column(str(col_name))
                rows.append(
                    {
                        "cell_line": line,
                        "barcode": barcode,
                        "timepoint": timepoint,
                        "plate": plate,
                        "sample": str(col_name),
                        "count": int(float(raw)),
                    }
                )

    out = dest / "whiting2025_barcode_lineage_umimic.csv.xz"
    with lzma.open(out, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LINEAGE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  Whiting 2025 lineage: {len(rows)} rows -> {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    default = (
        Path(__file__).resolve().parent.parent
        / "tmp"
        / "41467_2025_59479_MOESM4_ESM.zip"
    )
    p.add_argument("--source", type=Path, default=default)
    p.add_argument("--dest", type=Path, default=None)
    args = p.parse_args(argv)
    if not args.source.exists():
        raise SystemExit(f"Source not found: {args.source}")
    convert(args.source, args.dest)


if __name__ == "__main__":
    main()
