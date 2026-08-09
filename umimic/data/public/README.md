# Public datasets for U-MIMIC

This directory holds **catalog metadata**, **download/convert scripts**, and
**compact long-form products** (`*_umimic.csv.xz`). Multi-gigabyte raw sources
(NCI DOSERESP, full PhenoPop git clone, Dryad dumps) are **not** shipped; fetch
them into a cache with the download script when you need to rebuild products.

## Layout

```text
public/
  catalog/datasets.json     # citations, licenses, column roles, QC notes
  download_datasets.py
  convert_phenopop.py
  convert_tshs.py
  tshs_tumor/
    tshs_tumor_umimic.csv.xz   # canonical product (2 clear-error values removed)
    EXCLUDED_VALUES.csv
    DATA_NOTES.md
    tumorgrowth.xlsx           # colour-coded source (QC flags)
  phenopop/
    phenopop_umimic.csv.xz
  hafner_gr/
    hafner_gr_umimic.csv.xz
  nci60/
    nci60_umimic.csv.xz        # optional compressed extract
  bestdr/
    bestdr_umimic.csv.xz       # HCT116 + cisplatin live-cell counts
    DATA_NOTES.md
  whiting2025_barcode/
    whiting2025_barcode_population_umimic.csv.xz   # tiny bulk pop sizes
    DATA_NOTES.md
```

Set ``UMIMIC_DATA_ROOT`` to point loaders at an external cache directory with
the same subfolder names if you prefer data outside the package tree.

## Quick start

```bash
# Rebuild TSHS product from the Excel workbook (excludes orange clear-errors)
python -m umimic.data.public.convert_tshs

# Download other datasets into this tree / your data root
python -m umimic.data.public.download_datasets --dataset phenopop
python -m umimic.data.public.download_datasets --all
```

```python
from umimic.data import (
    load_tshs_tumor,
    load_phenopop,
    load_nci60,
    list_available_datasets,
)
from umimic.data.public_datasets import load_catalog, dataset_citation

print(list_available_datasets()["tshs_tumor"]["citation"])
ds = load_tshs_tumor()  # 37 mice, volume modality

# NCI-60 endpoint screening (filter tightly — full extract is ~2.5M rows)
nci = load_nci60(nsc="cisplatin", cell_line="MCF7")
# honest GI%/PTC only (no count reconstruction):
nci_gi = load_nci60(nsc=119875, panel="Breast", representation="endpoint_metrics")

# BESTDR live-cell imaging counts (HCT116 + cisplatin)
from umimic.data import load_bestdr, load_whiting2025_barcode
bd = load_bestdr(plate="dose_response")

# Whiting 2025 bulk barcode population sizes (not imaging; tiny extract)
w = load_whiting2025_barcode(cell_line="HCT")
```

## TSHS citation (required)

> Daskalakis C. Tumor Growth Dataset. TSHS Resources Portal, 2016.  
> Available at https://www.causeweb.org/tshs/tumor-growth/.

Permission was granted to use and redistribute for the purposes described in
U-MIMIC with that citation. Two clearly wrong orange-highlighted values are
removed from the distributed product; red/green QC flags are retained in the
`quality_flag` column. Full notes: `tshs_tumor/DATA_NOTES.md`.

## What not to commit / ship

| Path | Why |
|------|-----|
| `nci60/DOSERESP.csv` | ~2.3 GB raw bulk |
| `phenopop/phenopop_repo/` | Full MATLAB clone (~150 MB) |
| `hafner_gr/dryad_data/`, `nmeth_supp*` | Raw Dryad / supplements |
| Uncompressed `*_umimic.csv` next to `.xz` | Redundant |
| `__pycache__/`, `.DS_Store` | Build junk |

See the repository `.gitignore`.
