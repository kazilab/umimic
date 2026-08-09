# BESTDR subset — HCT116 + cisplatin (live-cell imaging counts)

## Citation

McDonald et al., BESTDR (birth–death dose–response) package and associated
manuscript (bioRxiv / journal as published). Source repository:
https://github.com/olliemcdonald/bestdr

This product is a **compact extract** of the public HCT116 cisplatin live-cell
imaging count tables (not the full multi-line R package data object).

## Content

| Field | Description |
|-------|-------------|
| hours | Elapsed time from imaging start (4 h sampling) |
| viable_count | Live cells (`intersect_live_cells`) |
| dead_count | Dead cells (`intersect_dead_cells`) |
| total_count | Total segmented cells |
| concentration | Cisplatin concentration (µM as in source) |
| well, img | Well and FOV index from the imaging plate |
| seed | Seeding / construct label from source |
| plate | `replicates` (0 & 12.5 µM) or `dose_response` (full ladder) |
| quality_flag | `ok` or `negative_live_clipped` (segmentation glitches set to 0) |

Series key for loaders: `replicate_id` = plate × well × img × seed × concentration.

## QC

A small number of source rows have **negative live counts** (imaging/segmentation
artefacts). Those values are clipped to 0 and flagged.
