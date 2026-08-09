#!/usr/bin/env python3
"""Convert the TSHS tumor-growth Excel workbook to U-MIMIC long format.

The source workbook (``tumorgrowth.xlsx``) encodes measurement quality in
cell formatting, as described by the data owner (Daskalakis, 2016 / personal
correspondence used with permission for distribution under citation):

* **Red font** — modest non-monotonic fluctuation; keep (normal caliper error).
* **Green fill** — larger / suspicious fluctuation; keep by default (may be
  real late shrinkage).
* **Orange fill** — clearly wrong values; **excluded** from the distributed
  product (two cells: 623.7 and 734.4).

Citation required when using the data::

    Daskalakis C. Tumor Growth Dataset. TSHS Resources Portal, 2016.
    Available at https://www.causeweb.org/tshs/tumor-growth/.

Usage::

    python -m umimic.data.public.convert_tshs
    python -m umimic.data.public.convert_tshs --xlsx path/to/tumorgrowth.xlsx
"""

from __future__ import annotations

import argparse
import csv
import lzma
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
FIELDS = [
    "hours",
    "day",
    "tumor_volume",
    "concentration",
    "group_id",
    "treatment",
    "mouse_id",
    "quality_flag",
]

# Excel theme mapping from the TSHS workbook styles.xml
# fillId 2 = green (suspicious), fillId 3 = orange (clear error)
# fontId 1 = red (non-monotonic)
ORANGE_FILL_ID = 3
GREEN_FILL_ID = 2
RED_FONT_ID = 1


def _col_to_index(col: str) -> int:
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - 64)
    return n


def _parse_xlsx(xlsx: Path) -> tuple[list[dict], list[dict]]:
    """Return (kept_records, excluded_records) from the colour-coded workbook."""
    with zipfile.ZipFile(xlsx) as z:
        styles = ET.fromstring(z.read("xl/styles.xml"))
        xfs = styles.findall("m:cellXfs/m:xf", NS)
        style_map = [
            (int(xf.attrib.get("fontId", 0)), int(xf.attrib.get("fillId", 0)))
            for xf in xfs
        ]

        sst: dict[int, str] = {}
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for i, si in enumerate(root.findall("m:si", NS)):
                texts = [
                    t.text or ""
                    for t in si.iter(
                        "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
                    )
                ]
                sst[i] = "".join(texts)

        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        grid: dict[tuple[str, int], tuple[str | None, str]] = {}
        for row in sheet.findall("m:sheetData/m:row", NS):
            for c in row.findall("m:c", NS):
                ref = c.attrib["r"]
                col = "".join(ch for ch in ref if ch.isalpha())
                r = int("".join(ch for ch in ref if ch.isdigit()))
                t = c.attrib.get("t")
                s = int(c.attrib.get("s", 0))
                v = c.find("m:v", NS)
                val = v.text if v is not None else None
                if t == "s" and val is not None:
                    val = sst.get(int(val), val)
                font_id, fill_id = style_map[s] if s < len(style_map) else (0, 0)
                if fill_id == ORANGE_FILL_ID:
                    flag = "exclude_clear_error"
                elif fill_id == GREEN_FILL_ID:
                    flag = "suspicious_green"
                elif font_id == RED_FONT_ID:
                    flag = "nonmonotonic_red"
                else:
                    flag = "ok"
                grid[(col, r)] = (val, flag)

    mice: dict[str, str] = {}
    for (col, r), (val, _flag) in grid.items():
        if r == 2 and col != "A" and val is not None and str(val).strip() != "":
            mice[col] = str(int(float(val)))

    headers = {
        col: val
        for (col, r), (val, _flag) in grid.items()
        if r == 1 and val is not None and str(val).strip() != ""
    }

    def treatment_for(col: str) -> str:
        ci = _col_to_index(col)
        best, best_i = "unknown", -1
        for hcol, hval in headers.items():
            hi = _col_to_index(hcol)
            if hi <= ci and hi > best_i:
                best_i, best = hi, str(hval)
        return best

    days: dict[int, float] = {}
    for (col, r), (val, _flag) in grid.items():
        if col == "A" and r >= 3 and val is not None and str(val).strip() != "":
            days[r] = float(val)

    kept: list[dict] = []
    excluded: list[dict] = []
    for col, mouse in mice.items():
        treat = treatment_for(col)
        for r, day in days.items():
            cell = grid.get((col, r))
            if not cell or cell[0] is None or str(cell[0]).strip() == "":
                continue
            try:
                vol = float(cell[0])
            except (TypeError, ValueError):
                continue
            rec = {
                "hours": day * 24.0,
                "day": day,
                "tumor_volume": float(f"{vol:.10g}"),
                "concentration": 0.0,
                "group_id": f"mouse_{mouse}",
                "treatment": treat,
                "mouse_id": mouse,
                "quality_flag": cell[1],
            }
            if cell[1] == "exclude_clear_error":
                excluded.append(rec)
            else:
                kept.append(rec)

    kept.sort(key=lambda row: (row["treatment"], int(row["mouse_id"]), row["day"]))
    excluded.sort(key=lambda row: (row["treatment"], int(row["mouse_id"]), row["day"]))
    return kept, excluded


def convert(
    xlsx: Path,
    dest_dir: Path | None = None,
    *,
    write_uncompressed: bool = False,
) -> Path:
    """Convert workbook → ``tshs_tumor_umimic.csv.xz`` (orange cells removed)."""
    xlsx = Path(xlsx)
    if not xlsx.exists():
        raise FileNotFoundError(f"TSHS Excel file not found: {xlsx}")

    dest = Path(dest_dir) if dest_dir is not None else xlsx.parent
    dest.mkdir(parents=True, exist_ok=True)

    kept, excluded = _parse_xlsx(xlsx)
    counts = Counter(r["quality_flag"] for r in kept)
    print(
        f"  TSHS: kept {len(kept)} observations "
        f"({dict(counts)}), excluded {len(excluded)} clear errors"
    )
    for row in excluded:
        print(
            f"    excluded mouse {row['mouse_id']} day {row['day']}: "
            f"volume={row['tumor_volume']} ({row['treatment']})"
        )

    xz_path = dest / "tshs_tumor_umimic.csv.xz"
    with lzma.open(xz_path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(kept)
    print(f"  -> {xz_path}")

    if write_uncompressed:
        csv_path = dest / "tshs_tumor_umimic.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(kept)
        print(f"  -> {csv_path}")

    excl_path = dest / "EXCLUDED_VALUES.csv"
    with excl_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS + ["reason"])
        writer.writeheader()
        for row in excluded:
            out = dict(row)
            out["reason"] = (
                "Orange highlight in source Excel: clearly wrong per data "
                "owner (Daskalakis); excluded from the distributed product."
            )
            writer.writerow(out)
    print(f"  -> {excl_path}")
    return xz_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=Path(__file__).resolve().parent / "tshs_tumor" / "tumorgrowth.xlsx",
        help="Path to tumorgrowth.xlsx",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Output directory (default: same as xlsx)",
    )
    parser.add_argument(
        "--uncompressed",
        action="store_true",
        help="Also write tshs_tumor_umimic.csv (not only .xz)",
    )
    args = parser.parse_args(argv)
    convert(args.xlsx, args.dest, write_uncompressed=args.uncompressed)


if __name__ == "__main__":
    main()
