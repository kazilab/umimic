#!/usr/bin/env python3
"""Convert PhenoPop MATLAB data to CSV format for U-MIMIC.

Source: Wu et al., PLoS Comp Bio 2024
Data: Ba/F3 cells (imatinib sensitive and resistant) under imatinib treatment.

The .mat files contain 3D arrays of shape (replicates, concentrations, timepoints):
  - BF_11, BF_12, BF_21, BF_41: Mixed populations (sensitive:resistant ratios 1:1, 1:2, 2:1, 4:1)
  - MONOCLONAL_DATA.mat: Pure sensitive and resistant populations
  - Conc: 11 imatinib concentrations (0 to 5 uM)
  - Time: 14 time points (0 to 39 hours, every 3h)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat


DATA_DIR = Path(__file__).parent / "phenopop"
REPO_DIR = DATA_DIR / "phenopop_repo" / "In vitro experiment"
OUT_DIR = DATA_DIR / "converted"


def convert_mixture_data():
    """Convert mixed-population data to CSV."""
    # All mixture .mat files contain the same data arrays
    mat = loadmat(str(REPO_DIR / "BF_11.mat"))

    concentrations = mat["Conc"].flatten()  # 11 concentrations
    times = mat["Time"].flatten().astype(float)  # 14 time points (hours)

    # Mixture ratios: sensitive:resistant
    mixtures = {
        "BF_11": {"label": "mix_1to1", "sensitive_ratio": 0.5},
        "BF_12": {"label": "mix_1to2", "sensitive_ratio": 1/3},
        "BF_21": {"label": "mix_2to1", "sensitive_ratio": 2/3},
        "BF_41": {"label": "mix_4to1", "sensitive_ratio": 4/5},
    }

    rows = []
    for key, info in mixtures.items():
        data = mat[key]  # shape: (replicates, concentrations, timepoints)
        n_rep, n_conc, n_time = data.shape

        for rep_idx in range(n_rep):
            for conc_idx in range(n_conc):
                for t_idx in range(n_time):
                    count = data[rep_idx, conc_idx, t_idx]
                    if np.isnan(count):
                        continue
                    rows.append({
                        "hours": times[t_idx],
                        "viable_count": count,
                        "concentration_uM": concentrations[conc_idx],
                        "population": info["label"],
                        "sensitive_ratio": info["sensitive_ratio"],
                        "replicate_id": f"{info['label']}_rep{rep_idx}",
                        "replicate_idx": rep_idx,
                    })

    outfile = OUT_DIR / "phenopop_mixture_data.csv"
    _write_csv(outfile, rows, [
        "hours", "viable_count", "concentration_uM", "population",
        "sensitive_ratio", "replicate_id", "replicate_idx",
    ])
    print(f"Mixture data: {len(rows)} observations -> {outfile}")
    return rows


def convert_monoclonal_data():
    """Convert monoclonal (pure population) data to CSV."""
    mat = loadmat(str(REPO_DIR / "MONOCLONAL_DATA.mat"))

    # Reuse concentrations and times from BF_11.mat
    ref = loadmat(str(REPO_DIR / "BF_11.mat"))
    concentrations = ref["Conc"].flatten()
    times = ref["Time"].flatten().astype(float)

    populations = {
        "SENSITIVE_500_BF": {"label": "sensitive", "initial_cells": 500},
        "SENSITIVE_1000_BF": {"label": "sensitive", "initial_cells": 1000},
        "RESISTANT_250_BF": {"label": "resistant", "initial_cells": 250},
        "RESISTANT_500_BF": {"label": "resistant", "initial_cells": 500},
    }

    rows = []
    for key, info in populations.items():
        data = mat[key]  # shape: (replicates, concentrations, timepoints)
        n_rep, n_conc, n_time = data.shape

        for rep_idx in range(n_rep):
            for conc_idx in range(n_conc):
                for t_idx in range(n_time):
                    count = data[rep_idx, conc_idx, t_idx]
                    if np.isnan(count):
                        continue
                    rows.append({
                        "hours": times[t_idx],
                        "viable_count": count,
                        "concentration_uM": concentrations[conc_idx],
                        "population": info["label"],
                        "initial_cells": info["initial_cells"],
                        "replicate_id": f"{key}_rep{rep_idx}",
                        "replicate_idx": rep_idx,
                    })

    outfile = OUT_DIR / "phenopop_monoclonal_data.csv"
    _write_csv(outfile, rows, [
        "hours", "viable_count", "concentration_uM", "population",
        "initial_cells", "replicate_id", "replicate_idx",
    ])
    print(f"Monoclonal data: {len(rows)} observations -> {outfile}")
    return rows


def create_summary():
    """Create a summary of the converted data."""
    summary = OUT_DIR / "data_summary.txt"
    with open(summary, "w") as f:
        f.write("PhenoPop Ba/F3 Imatinib Dataset Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write("Source: Wu et al., PLoS Comp Bio 2024\n")
        f.write("Cell line: Ba/F3 (murine pro-B)\n")
        f.write("Drug: Imatinib\n\n")

        ref = loadmat(str(REPO_DIR / "BF_11.mat"))
        concs = ref["Conc"].flatten()
        times = ref["Time"].flatten()

        f.write(f"Concentrations (uM): {concs.tolist()}\n")
        f.write(f"Time points (hours): {times.tolist()}\n\n")

        f.write("Mixture populations (sensitive:resistant):\n")
        f.write("  BF_11: 1:1 (50% sensitive)\n")
        f.write("  BF_12: 1:2 (33% sensitive)\n")
        f.write("  BF_21: 2:1 (67% sensitive)\n")
        f.write("  BF_41: 4:1 (80% sensitive)\n")
        f.write(f"  Replicates per condition: {ref['BF_11'].shape[0]}\n\n")

        f.write("Monoclonal populations:\n")
        mono = loadmat(str(REPO_DIR / "MONOCLONAL_DATA.mat"))
        for key in sorted(mono.keys()):
            if not key.startswith("_"):
                f.write(f"  {key}: shape={mono[key].shape}\n")

    print(f"Summary: {summary}")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    """Write rows to CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not REPO_DIR.exists():
        print(f"PhenoPop repo not found at {REPO_DIR}")
        print("Run: python download_datasets.py --dataset phenopop")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Converting PhenoPop data to CSV format...")
    print()
    convert_mixture_data()
    convert_monoclonal_data()
    create_summary()
    print("\nDone! Files saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
