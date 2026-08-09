# Whiting et al. (2025) — bulk barcode population sizes only

## Citation (required)

Whiting FJH, et al. Quantitative measurement of phenotype dynamics during
cancer drug resistance evolution using genetic barcoding.
*Nature Communications* (2025).
https://doi.org/10.1038/s41467-025-59479-x  
Supplementary Data 2 population tables (HCTbc / SW6bc).

## What this product is

**Tiny bulk population time series only** (not imaging, not scRNA, not full
barcode count matrices):

| cell_line | replicates | time points (days) |
|-----------|------------|--------------------|
| HCT116_barcode (HCTbc) | 4 | 10, 29, 67, 102 |
| SW620_barcode (SW6bc) | 4 | 10, 22, 37, 85 |

- `hours = day * 24` for U-MIMIC time-series conventions.
- `population_size` is total population at each sample day.
- `concentration` is set to 0; the experiment uses a **pulsed treatment
  schedule** (see Supp Data 3 treatment timings), not a constant in-well dose.

## What is deliberately excluded

- Full barcode abundance tables (HCTbc ~14k barcodes; SW6bc ~688k rows)
- scRNA / UMAP tables from Supplementary Data 6–7
- Model posterior sheets

Re-download the Nature Communications supplements if you need those layers.
