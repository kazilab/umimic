#!/usr/bin/env python3
"""Download publicly available datasets for U-MIMIC.

Usage:
    python download_datasets.py --all
    python download_datasets.py --dataset bestdr
    python download_datasets.py --dataset phenopop
    python download_datasets.py --dataset tshs_tumor
    python download_datasets.py --dataset hafner_gr
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download a file from URL to destination path."""
    print(f"  Downloading {desc or url}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  -> Saved to {dest}")
        return dest
    except Exception as e:
        print(f"  ERROR: Failed to download {url}: {e}")
        raise


def download_bestdr() -> None:
    """Download BESTDR breast cancer longitudinal cell count data.

    Source: https://github.com/olliemcdonald/bestdr
    Data: R package data objects containing longitudinal cell counts
          for 8 breast cancer cell lines treated with 25 drugs.

    Since the data is embedded in an R package, we clone the repo
    and extract the data files. If R is available, we convert to CSV.
    """
    dest = DATA_DIR / "bestdr"
    dest.mkdir(exist_ok=True)

    repo_url = "https://github.com/olliemcdonald/bestdr.git"
    clone_dir = dest / "bestdr_repo"

    if clone_dir.exists():
        print("  BESTDR repo already cloned, skipping...")
    else:
        print("  Cloning BESTDR repository...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
                check=True, capture_output=True, text=True,
            )
            print("  -> Cloned successfully")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  ERROR: git clone failed: {e}")
            print("  Trying direct download of data files...")
            _download_bestdr_direct(dest)
            return

    # Try to extract data using R if available
    r_script = """
    library(bestdr)
    data_dir <- commandArgs(trailingOnly=TRUE)[1]

    # Extract the built-in datasets
    data(package="bestdr")

    # Try to access and export the breast cancer cell count data
    # The package likely has data objects - export them all
    for (ds_name in data(package="bestdr")$results[,"Item"]) {
        tryCatch({
            data(list=ds_name)
            obj <- get(ds_name)
            if (is.data.frame(obj)) {
                write.csv(obj, file.path(data_dir, paste0(ds_name, ".csv")),
                          row.names=FALSE)
                cat(paste("Exported:", ds_name, "\\n"))
            }
        }, error=function(e) {
            cat(paste("Skipped:", ds_name, "-", e$message, "\\n"))
        })
    }
    """

    try:
        subprocess.run(
            ["Rscript", "-e", r_script, str(dest)],
            check=True, capture_output=True, text=True, timeout=120,
        )
        print("  -> R data extraction successful")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("  R not available or extraction failed. Data files are in the cloned repo.")
        print(f"  Look for .rda/.RData files in: {clone_dir / 'data'}")
        _create_bestdr_placeholder(dest)


def _download_bestdr_direct(dest: Path) -> None:
    """Download key data files directly from GitHub."""
    base = "https://raw.githubusercontent.com/olliemcdonald/bestdr/main/data"
    try:
        # Try to list and download .rda files from the data directory
        download_file(
            f"{base}/../inst/extdata/README.md",
            dest / "source_readme.md",
            "BESTDR data readme",
        )
    except Exception:
        pass
    _create_bestdr_placeholder(dest)


def _create_bestdr_placeholder(dest: Path) -> None:
    """Create a placeholder CSV showing expected BESTDR format."""
    placeholder = dest / "bestdr_format_example.csv"
    with open(placeholder, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hours", "viable_count", "concentration", "cell_line", "drug", "replicate_id"])
        writer.writerow(["# Real data: clone https://github.com/olliemcdonald/bestdr"])
        writer.writerow(["# and extract with R: library(bestdr); data(package='bestdr')"])
    print(f"  -> Created format example at {placeholder}")


def download_phenopop() -> None:
    """Download PhenoPop stochastic Ba/F3 cell line data.

    Source: https://github.com/chenyuwu233/PhenoPop_stochastic
    Data: CSV files with live-cell imaging cell counts for Ba/F3 cells
          (imatinib sensitive and resistant) at 11 concentrations.
    """
    dest = DATA_DIR / "phenopop"
    dest.mkdir(exist_ok=True)

    repo_url = "https://github.com/chenyuwu233/PhenoPop_stochastic.git"
    clone_dir = dest / "phenopop_repo"

    if clone_dir.exists():
        print("  PhenoPop repo already cloned, skipping...")
    else:
        print("  Cloning PhenoPop repository...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
                check=True, capture_output=True, text=True,
            )
            print("  -> Cloned successfully")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  ERROR: git clone failed: {e}")
            return

    # Copy the in vitro experiment data to a convenient location
    src_data = clone_dir / "In vitro experiment"
    if src_data.exists():
        converted_dir = dest / "converted"
        converted_dir.mkdir(exist_ok=True)
        _convert_phenopop_data(src_data, converted_dir)
    else:
        # Try alternative paths
        for candidate in ["data", "Data", "in_vitro_experiment"]:
            alt = clone_dir / candidate
            if alt.exists():
                print(f"  Found data at: {alt}")
                break
        print(f"  Data directory structure: {list(clone_dir.iterdir())}")


def _convert_phenopop_data(src_dir: Path, dest_dir: Path) -> None:
    """Convert PhenoPop raw data files to standardized CSV format."""
    import numpy as np

    print("  Converting PhenoPop data to U-MIMIC format...")
    all_rows = []

    for data_file in sorted(src_dir.glob("*.csv")) + sorted(src_dir.glob("*.txt")):
        try:
            # PhenoPop data: rows = time points, columns = replicates
            raw = np.genfromtxt(data_file, delimiter=",", skip_header=0)
            if raw.ndim < 2:
                raw = np.genfromtxt(data_file, delimiter="\t", skip_header=0)

            # Try to extract concentration from filename
            name = data_file.stem
            conc = _extract_concentration(name)

            if raw.ndim == 2:
                n_timepoints, n_replicates = raw.shape
                # Assume equal time spacing (typical: 4h intervals)
                times = np.arange(n_timepoints) * 4.0

                for rep_idx in range(n_replicates):
                    for t_idx in range(n_timepoints):
                        all_rows.append({
                            "hours": times[t_idx],
                            "viable_count": raw[t_idx, rep_idx],
                            "concentration": conc,
                            "replicate_id": f"{name}_rep{rep_idx}",
                            "source_file": data_file.name,
                        })
        except Exception as e:
            print(f"    Warning: Could not parse {data_file.name}: {e}")

    if all_rows:
        outfile = dest_dir / "phenopop_baf3_combined.csv"
        with open(outfile, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["hours", "viable_count", "concentration",
                                                    "replicate_id", "source_file"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  -> Converted {len(all_rows)} data points to {outfile}")
    else:
        print("  -> No data files found to convert. Check repo structure manually.")


def _extract_concentration(filename: str) -> float:
    """Try to extract drug concentration from a filename."""
    import re
    # Try patterns like "1uM", "0.3uM", "conc_1", etc.
    patterns = [
        r"(\d+\.?\d*)\s*[uU][mM]",
        r"conc[_=]?(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*nM",
    ]
    for pat in patterns:
        m = re.search(pat, filename)
        if m:
            return float(m.group(1))
    return 0.0


def download_tshs_tumor() -> None:
    """Prepare TSHS tumor growth xenograft dataset.

    Source / citation (required)::

        Daskalakis C. Tumor Growth Dataset. TSHS Resources Portal, 2016.
        Available at https://www.causeweb.org/tshs/tumor-growth/.

    The colour-coded Excel workbook is the authoritative source of measurement
    QC flags. If ``tumorgrowth.xlsx`` is present, convert it with
    :mod:`umimic.data.public.convert_tshs` (excludes the two orange clear-error
    cells). Automatic HTTP download is best-effort only.
    """
    dest = DATA_DIR / "tshs_tumor"
    dest.mkdir(exist_ok=True)

    xlsx = dest / "tumorgrowth.xlsx"
    product = dest / "tshs_tumor_umimic.csv.xz"
    if product.exists():
        print(f"  TSHS product already present: {product}")
        print("  Citation: Daskalakis C. Tumor Growth Dataset. TSHS Resources Portal, 2016.")
        print("            https://www.causeweb.org/tshs/tumor-growth/")
        return

    if xlsx.exists():
        print("  Converting local tumorgrowth.xlsx (preferred; preserves QC flags)...")
        from umimic.data.public.convert_tshs import convert

        convert(xlsx, dest)
        return

    # Best-effort CSV download (no colour QC); prefer manual xlsx when possible
    urls = [
        "https://www.causeweb.org/tshs/datasets/Tumor%20Growth.csv",
        "https://www.causeweb.org/tshs/datasets/tumor_growth.csv",
        "https://www.causeweb.org/tshs/datasets/Tumor_Growth.csv",
    ]
    downloaded = False
    for url in urls:
        try:
            outpath = dest / "tumor_growth_raw.csv"
            download_file(url, outpath, "TSHS tumor growth data")
            downloaded = True
            break
        except Exception:
            continue

    if not downloaded:
        print("  Could not auto-download TSHS data.")
        print("  Place tumorgrowth.xlsx in", dest)
        print("  then run: python -m umimic.data.public.convert_tshs")
        print("  Source: https://www.causeweb.org/tshs/tumor-growth/")
        return

    print(
        "  WARNING: CSV download has no orange/green/red QC markup. "
        "Prefer converting tumorgrowth.xlsx so clear-error cells are excluded."
    )
    _convert_tshs_data(dest)


def _convert_tshs_data(dest: Path) -> None:
    """Legacy wide/long CSV conversion when only a raw CSV is available."""
    raw_path = dest / "tumor_growth_raw.csv"
    if not raw_path.exists():
        return

    try:
        import pandas as pd

        df = pd.read_csv(raw_path)
        print(f"  Raw columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")

        outpath = dest / "tshs_tumor_umimic.csv"
        id_cols = []
        time_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(k in col_lower for k in ["group", "treatment", "mouse", "id", "arm"]):
                id_cols.append(col)
            elif any(k in col_lower for k in ["day", "time", "week", "vol", "size"]):
                time_cols.append(col)

        if time_cols:
            rows = []
            for _, row in df.iterrows():
                group_id = (
                    "_".join(str(row[c]) for c in id_cols) if id_cols else str(row.name)
                )
                for tcol in time_cols:
                    val = row[tcol]
                    if pd.notna(val):
                        import re

                        m = re.search(r"(\d+\.?\d*)", tcol)
                        day = float(m.group(1)) if m else 0
                        rows.append(
                            {
                                "hours": day * 24,
                                "day": day,
                                "tumor_volume": float(val),
                                "concentration": 0.0,
                                "group_id": group_id,
                                "treatment": "unknown",
                                "mouse_id": group_id,
                                "quality_flag": "ok",
                            }
                        )
            if rows:
                pd.DataFrame(rows).to_csv(outpath, index=False)
                print(f"  -> Converted to U-MIMIC format: {outpath}")
                return

        df.to_csv(outpath, index=False)
        print(f"  -> Saved as-is to {outpath} (check column names)")

    except ImportError:
        print("  pandas not available, skipping conversion. Raw file saved.")
    except Exception as e:
        print(f"  Conversion error: {e}. Raw file saved at {raw_path}")


def download_hafner_gr() -> None:
    """Download Hafner/Niepel GR metrics breast cancer dataset from Dryad.

    Source: https://datadryad.org/dataset/doi:10.5061/dryad.03n60
    Data: CSV files with cell count measurements, GR values, and dose-response
          parameters for 71 breast cancer cell lines x 107 drugs.
    """
    dest = DATA_DIR / "hafner_gr"
    dest.mkdir(exist_ok=True)

    # Dryad dataset DOI - the actual download URL may vary
    print("  The Hafner/Niepel GR dataset is hosted on Dryad.")
    print("  Due to Dryad's download structure, automatic download may not work.")
    print("  Manual download instructions:")
    print("    1. Go to: https://datadryad.org/dataset/doi:10.5061/dryad.03n60")
    print("    2. Download the zip archive")
    print(f"    3. Extract to: {dest}")
    print()

    # Try direct download from known Dryad URLs
    try:
        url = "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.03n60/download"
        outpath = dest / "hafner_gr_raw.zip"
        download_file(url, outpath, "Hafner GR dataset (zip)")

        # Extract zip
        with zipfile.ZipFile(outpath, "r") as zf:
            zf.extractall(dest)
        print(f"  -> Extracted to {dest}")

        # Convert to U-MIMIC format
        _convert_hafner_data(dest)

    except Exception as e:
        print(f"  Auto-download failed: {e}")
        _create_hafner_placeholder(dest)


def _convert_hafner_data(dest: Path) -> None:
    """Convert Hafner GR data to U-MIMIC standardized format."""
    try:
        import pandas as pd

        # Look for the main data file (usually named something like DS0_*)
        csv_files = list(dest.glob("**/*.csv")) + list(dest.glob("**/*.tsv"))
        if not csv_files:
            print("  No CSV/TSV files found in extracted data.")
            return

        print(f"  Found data files: {[f.name for f in csv_files]}")

        for csv_file in csv_files:
            sep = "\t" if csv_file.suffix == ".tsv" else ","
            df = pd.read_csv(csv_file, sep=sep, nrows=5)
            print(f"    {csv_file.name}: columns={list(df.columns)[:10]}")

    except ImportError:
        print("  pandas not available. Extract and explore files manually.")
    except Exception as e:
        print(f"  Conversion error: {e}")


def _create_hafner_placeholder(dest: Path) -> None:
    """Create placeholder with download instructions."""
    info = dest / "DOWNLOAD_INSTRUCTIONS.txt"
    with open(info, "w") as f:
        f.write("Hafner/Niepel GR Metrics Dataset\n")
        f.write("================================\n\n")
        f.write("Download from: https://datadryad.org/dataset/doi:10.5061/dryad.03n60\n")
        f.write("Paper: Hafner et al., Scientific Data 4:170166 (2017)\n\n")
        f.write("After downloading, extract the zip file here.\n")
        f.write("Then use: from umimic.data.public_datasets import load_hafner_gr\n")
    print(f"  -> Created download instructions at {info}")


DATASETS = {
    "bestdr": ("BESTDR Breast Cancer Longitudinal Data", download_bestdr),
    "phenopop": ("PhenoPop Ba/F3 Live-Cell Imaging Data", download_phenopop),
    "tshs_tumor": ("TSHS Xenograft Tumor Growth Data", download_tshs_tumor),
    "hafner_gr": ("Hafner/Niepel GR Metrics Data", download_hafner_gr),
}


def main():
    parser = argparse.ArgumentParser(
        description="Download public datasets for U-MIMIC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available datasets:
  bestdr      - BESTDR breast cancer longitudinal cell counts (GitHub)
  phenopop    - PhenoPop Ba/F3 live-cell imaging data (GitHub)
  tshs_tumor  - TSHS xenograft tumor growth curves (TSHS)
  hafner_gr   - Hafner/Niepel GR metrics from Dryad

Examples:
  python download_datasets.py --all
  python download_datasets.py --dataset bestdr phenopop
        """,
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument(
        "--dataset", nargs="+", choices=list(DATASETS.keys()),
        help="Specific dataset(s) to download",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")

    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for key, (desc, _) in DATASETS.items():
            print(f"  {key:15s} - {desc}")
        return

    if not args.all and not args.dataset:
        parser.print_help()
        return

    targets = list(DATASETS.keys()) if args.all else args.dataset

    for name in targets:
        desc, func = DATASETS[name]
        print(f"\n{'='*60}")
        print(f"Downloading: {desc}")
        print(f"{'='*60}")
        try:
            func()
        except Exception as e:
            print(f"  FAILED: {e}")
            print("  Continuing with next dataset...")

    print(f"\n{'='*60}")
    print("Download complete! Data stored in:", DATA_DIR)
    print("Use: from umimic.data.public_datasets import load_<dataset_name>")


if __name__ == "__main__":
    main()
