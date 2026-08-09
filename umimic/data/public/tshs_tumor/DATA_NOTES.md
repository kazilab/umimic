# TSHS Tumor Growth — distribution notes

## Citation (required)

> Daskalakis C. Tumor Growth Dataset. TSHS Resources Portal, 2016.  
> Available at https://www.causeweb.org/tshs/tumor-growth/.

Permission was granted to use and distribute this dataset for U-MIMIC
research and teaching with that citation and no additional conditions.

## Nature of the data

Although based on a real xenograft experiment, **some measurements were
tweaked** for teaching/illustration. Measurement times are **irregular**
(typically every 2–3 days; weekends/holidays cause gaps). Volumes come from
**caliper** measurements and are inherently noisy; modest day-to-day
non-monotonic wiggles are expected.

## Quality coding (from the source Excel)

| Excel mark | `quality_flag` | Distributed product |
|------------|----------------|---------------------|
| (none) | `ok` | Retained |
| Red font | `nonmonotonic_red` | Retained — modest non-monotonicity / normal error |
| Green fill | `suspicious_green` | Retained — larger fluctuation; possible real shrinkage |
| Orange fill | `exclude_clear_error` | **Removed** — clearly wrong |

### Values excluded from `tshs_tumor_umimic.csv.xz`

| mouse_id | treatment | day | volume | reason |
|----------|-----------|-----|--------|--------|
| 202 | DRUG | 3 | 623.7 | Orange cell; clearly inconsistent with neighbours |
| 302 | RADIATION | 26 | 734.4 | Orange cell; clearly inconsistent with neighbours |

See also `EXCLUDED_VALUES.csv`. Re-run conversion with:

```bash
python -m umimic.data.public.convert_tshs
```

## Files in this directory

| File | Role |
|------|------|
| `tshs_tumor_umimic.csv.xz` | Canonical long-form product (orange values removed) |
| `EXCLUDED_VALUES.csv` | The two removed points (audit trail) |
| `tumorgrowth.xlsx` | Original colour-coded workbook (source of QC flags) |
| `DATA_NOTES.md` | This file |
