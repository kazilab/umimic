"""Loaders for publicly available cancer drug response datasets.

Each loader returns an ExperimentalDataset compatible with U-MIMIC's
inference pipeline. Prefer compressed long-form products
(``*_umimic.csv.xz``). Metadata and citations live in
``public/catalog/datasets.json``.

Use the download script to re-fetch raw sources into a local cache::

    python -m umimic.data.public.download_datasets --dataset tshs_tumor

Supported datasets:
    - BESTDR: HCT116 + cisplatin live-cell imaging counts (compact extract)
    - PhenoPop: Ba/F3 live-cell imaging under imatinib
    - TSHS Tumor Growth: Xenograft tumor volumes (4 treatment arms)
    - Hafner/Niepel GR: 71 breast cancer lines x 107 drugs (endpoint + GR)
    - NCI-60: Growth inhibition endpoint screening (filtered load)
    - Whiting 2025: Tiny bulk barcode population sizes (not imaging)
"""

from __future__ import annotations

import json
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from umimic.data.schemas import TimeSeriesData, ExperimentalDataset


def _resolve_default_data_root() -> Path:
    """Resolve default data root.

    Resolution order:
    1) ``UMIMIC_DATA_ROOT`` env var if provided.
    2) Package-local ``umimic/data/public`` directory.
    """
    env_root = os.getenv("UMIMIC_DATA_ROOT")
    if env_root:
        resolved = Path(env_root).expanduser().resolve()
        if not resolved.exists():
            warnings.warn(
                f"UMIMIC_DATA_ROOT is set to '{resolved}' but the directory "
                f"does not exist. Falling back to package-local data directory.",
                stacklevel=2,
            )
            return Path(__file__).resolve().parent / "public"
        return resolved

    return Path(__file__).resolve().parent / "public"


DATA_ROOT = _resolve_default_data_root()
CATALOG_PATH = Path(__file__).resolve().parent / "public" / "catalog" / "datasets.json"

# TSHS citation required by the data owner for redistribution / use.
TSHS_CITATION = (
    "Daskalakis C. Tumor Growth Dataset. TSHS Resources Portal, 2016. "
    "Available at https://www.causeweb.org/tshs/tumor-growth/."
)

NCI60_CITATION = (
    "NCI Developmental Therapeutics Program (DTP). NCI-60 screening data. "
    "https://dtp.cancer.gov/databases_tools/bulk_data.htm"
)

BESTDR_CITATION = (
    "McDonald et al. BESTDR: birth–death models for live-cell drug-response "
    "imaging. https://github.com/olliemcdonald/bestdr"
)

WHITING2025_CITATION = (
    "Whiting FJH, et al. Quantitative measurement of phenotype dynamics during "
    "cancer drug resistance evolution using genetic barcoding. "
    "Nature Communications (2025). https://doi.org/10.1038/s41467-025-59479-x"
)

# Common NSC numbers for a few well-known agents (product has no drug names).
NCI60_COMMON_NSC: dict[str, str] = {
    "cisplatin": "119875",
    "doxorubicin": "123127",
    "paclitaxel": "125973",
    "5-fluorouracil": "19893",
    "5fu": "19893",
    "gemcitabine": "613327",
    "methotrexate": "740",
    "vinblastine": "49842",
    "etoposide": "141540",
    "topotecan": "609699",
}


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    """Load ``public/catalog/datasets.json`` (metadata only)."""
    if not CATALOG_PATH.exists():
        return {"version": 0, "datasets": {}}
    with CATALOG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def dataset_citation(dataset_id: str) -> str | None:
    """Return the preferred citation string for a catalogued dataset."""
    entry = load_catalog().get("datasets", {}).get(dataset_id)
    if not entry:
        return None
    return entry.get("citation")


def _find_data_dir(dataset_name: str, data_root: Path | None = None) -> Path:
    """Locate the directory for a given dataset.

    Raises:
        FileNotFoundError: With actionable instructions when the dataset
            directory is missing.
    """
    root = Path(data_root) if data_root else DATA_ROOT
    d = root / dataset_name

    if not d.exists():
        available = [
            p.name for p in root.iterdir() if p.is_dir()
        ] if root.exists() else []
        available_msg = (
            f"  Available datasets in {root}: {', '.join(available)}"
            if available
            else f"  Data root directory does not exist: {root}"
        )

        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found at: {d}\n\n"
            f"To download it, run:\n"
            f"  python -m umimic.data.public.download_datasets --dataset {dataset_name}\n\n"
            f"Or set the UMIMIC_DATA_ROOT environment variable to point to your "
            f"data directory.\n\n"
            f"{available_msg}"
        )
    return d


def _resolve_umimic_product(dataset_dir: Path, stem: str) -> Path | None:
    """Prefer ``{stem}.csv.xz``, then ``{stem}.csv``, under *dataset_dir*."""
    for name in (f"{stem}.csv.xz", f"{stem}.csv"):
        path = dataset_dir / name
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------------------
# BESTDR: Longitudinal cell counts (live-cell imaging)
# ---------------------------------------------------------------------------

def load_bestdr(
    data_root: Path | None = None,
    cell_line: str | None = None,
    drug: str | None = None,
    *,
    plate: str | None = None,
    include_dead: bool = True,
) -> ExperimentalDataset:
    """Load BESTDR live-cell imaging count trajectories.

    The shipped product is a compact HCT116 + cisplatin extract
    (``bestdr/bestdr_umimic.csv.xz``) derived from public BESTDR tables:
    longitudinal live/dead/total counts every 4 h, multi-well and multi-FOV.

    Args:
        data_root: Override data root (default: package ``public/``).
        cell_line: Substring filter (default product is HCT116).
        drug: Substring filter (default product is cisplatin).
        plate: ``"replicates"``, ``"dose_response"``, or None for both.
        include_dead: If True (default), attach ``dead_count`` as a second
            modality alongside ``cell_counts`` (live).

    Returns:
        ExperimentalDataset with one series per FOV × concentration
        (``replicate_id``).

    Notes:
        Source rows with negative live counts (segmentation artefacts) were
        clipped to 0 and flagged ``negative_live_clipped`` in the product.
        Cite BESTDR / McDonald et al. when publishing.
    """
    d = _find_data_dir("bestdr", data_root)

    product = _resolve_umimic_product(d, "bestdr_umimic")
    if product is not None:
        return _load_bestdr_product(
            product,
            cell_line=cell_line,
            drug=drug,
            plate=plate,
            include_dead=include_dead,
        )

    # Fallback: any long CSVs under the directory (user export)
    csv_files = sorted(d.glob("**/*.csv")) + sorted(d.glob("**/*.csv.xz"))
    csv_files = [p for p in csv_files if "EXCLUDED" not in p.name]
    if not csv_files:
        raise FileNotFoundError(
            f"No BESTDR product found in {d}. Expected bestdr_umimic.csv.xz.\n"
            "Place the compact extract there, or export from the BESTDR R package.\n"
            "Source: https://github.com/olliemcdonald/bestdr"
        )

    return _load_generic_cell_count_csv(
        csv_files,
        dataset_name="bestdr",
        cell_line_filter=cell_line,
        drug_filter=drug,
    )


def _load_bestdr_product(
    path: Path,
    *,
    cell_line: str | None,
    drug: str | None,
    plate: str | None,
    include_dead: bool,
) -> ExperimentalDataset:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to load BESTDR data. Install pandas: pip install pandas"
        ) from exc

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "hours" not in cols or "viable_count" not in cols:
        raise ValueError(
            f"{path} is not a BESTDR umimic product (need hours, viable_count)."
        )

    if cell_line and "cell_line" in cols:
        key = cell_line.lower()
        df = df[df[cols["cell_line"]].astype(str).str.lower().str.contains(key, regex=False)]
    if drug and "drug" in cols:
        key = drug.lower()
        df = df[df[cols["drug"]].astype(str).str.lower().str.contains(key, regex=False)]
    if plate and "plate" in cols:
        key = plate.lower()
        df = df[df[cols["plate"]].astype(str).str.lower() == key]

    if df.empty:
        raise FileNotFoundError(
            f"No BESTDR rows matched filters "
            f"(cell_line={cell_line!r}, drug={drug!r}, plate={plate!r}) in {path}."
        )

    group_c = cols.get("replicate_id", cols.get("fov_id"))
    time_c = cols["hours"]
    live_c = cols["viable_count"]
    dead_c = cols.get("dead_count")
    conc_c = cols.get("concentration")

    series: list[TimeSeriesData] = []
    for gid, sub in df.groupby(group_c, sort=False):
        sub = sub.sort_values(time_c)
        if sub.duplicated(subset=[time_c]).any():
            sub = sub.drop_duplicates(subset=[time_c], keep="first")
        times = sub[time_c].to_numpy(dtype=float)
        if times.size < 2 or np.any(np.diff(times) <= 0):
            continue
        obs: dict[str, np.ndarray] = {
            "cell_counts": sub[live_c].to_numpy(dtype=float),
        }
        units = {"cell_counts": "cells (live, imaging)"}
        if include_dead and dead_c is not None:
            obs["dead_count"] = sub[dead_c].to_numpy(dtype=float)
            units["dead_count"] = "cells (dead, imaging)"

        conc = float(sub[conc_c].iloc[0]) if conc_c is not None else None
        meta: dict[str, Any] = {
            "dataset": "bestdr",
            "citation": BESTDR_CITATION,
            "source_file": path.name,
        }
        for key in ("cell_line", "drug", "plate", "well", "img", "seed", "fov_id", "quality_flag"):
            actual = cols.get(key)
            if actual is not None and actual in sub.columns:
                meta[key] = str(sub[actual].iloc[0])
        if "quality_flag" in cols:
            flags = sorted({str(v) for v in sub[cols["quality_flag"]].unique()})
            meta["quality_flags_in_series"] = flags

        series.append(
            TimeSeriesData(
                times=times,
                observations=obs,
                concentration=conc,
                group_id=str(gid),
                replicate_id=str(gid),
                units=units,
                metadata=meta,
            )
        )

    return ExperimentalDataset(series=series, name="bestdr_hct116_cisplatin", context="in_vitro")


# ---------------------------------------------------------------------------
# Whiting 2025 Nat Commun — bulk barcode population sizes (tiny product)
# ---------------------------------------------------------------------------

def load_phenopop_reference(kind: str = "fits", data_root: Path | None = None):
    """Load the PhenoPop authors' published fits, for benchmarking against.

    Wu et al. (2024) ship the results of their MATLAB analysis: per-mixture
    optima, AIC values and bootstrap confidence intervals for three competing
    models. Comparing U-MIMIC against these needs no MATLAB run.

    Args:
        kind: ``"fits"`` for point estimates, AIC and objective values;
            ``"intervals"`` for bootstrap CIs on the mixture proportion and the
            two subpopulation growth rates.
        data_root: Override data root.

    Returns:
        A pandas DataFrame. See ``convert_phenopop_reference.py`` for column
        semantics and for why only element 0 of the parameter vectors is
        interpreted.

    Models are ``sto`` (stochastic birth-death, the paper's contribution),
    ``dyn`` (deterministic mean model) and ``hl`` (the original PhenoPop
    Hill-based deconvolution).

    Note:
        The estimated mixture proportions do **not** match the nominal seeding
        ratios -- BF_11 is nominally 1:1 but estimated near 0.30. That is the
        authors' own published result, not a conversion artefact, so treat the
        nominal ratio as the intended design rather than as ground truth.
    """
    if kind not in ("fits", "intervals"):
        raise ValueError(
            f"kind must be 'fits' or 'intervals', got {kind!r}."
        )
    d = _find_data_dir("phenopop", data_root)
    name = f"phenopop_reference_{kind}_umimic"
    product = _resolve_umimic_product(d, name)
    if product is None:
        raise FileNotFoundError(
            f"PhenoPop reference product '{name}' not found in {d}.\n"
            "Rebuild with: python -m umimic.data.public.convert_phenopop_reference"
        )
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to load PhenoPop reference results."
        ) from exc
    return pd.read_csv(product)


def load_csc_dip_reference(data_root: Path | None = None):
    """Load the CSC/drug-induced-plasticity authors' published AIC values.

    Wu et al. (2025) test for therapy-induced resistance by model selection.
    Their repository ships the resulting information criteria but **not the
    measurements behind them**, so their conclusions can be compared while
    their fit cannot be reproduced -- see the ``csc_dip`` catalog entry for the
    list of missing inputs.

    Args:
        data_root: Override data root.

    Returns:
        A pandas DataFrame with columns ``experiment``, ``context``, ``model``,
        ``criterion`` (AIC or AICc), ``value``, ``source_file``. Models are
        ``DIP``/``nDIP`` (with and without drug-induced plasticity) optionally
        crossed with ``Asy``/``nAsy`` (asymmetric division).
    """
    d = _find_data_dir("csc_dip", data_root)
    product = _resolve_umimic_product(d, "csc_dip_reference_aic_umimic")
    if product is None:
        raise FileNotFoundError(
            f"CSC/DIP reference product not found in {d}.\n"
            "Rebuild with: python -m umimic.data.public.convert_csc_dip_reference"
        )
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to load CSC/DIP reference results."
        ) from exc
    return pd.read_csv(product)


def load_whiting2025_lineage(
    data_root: Path | None = None,
    cell_line: str | None = None,
    min_count: int = 0,
):
    """Load Whiting et al. (2025) per-barcode lineage abundances.

    This is the lineage-tracing measurement the study is built on: clone-level
    abundances for barcoded HCT116 and SW620 under pulsed 5-FU, from
    Supplementary Data 2. The companion :func:`load_whiting2025_barcode` returns
    only the bulk population sizes, which cannot support any clone-level
    analysis on their own.

    Args:
        data_root: Override data root.
        cell_line: Substring filter (``"HCT"`` / ``"SW6"``).
        min_count: Drop rows below this count. The default of 0 keeps zeros,
            which matter: a barcode absent at a timepoint is an observed
            extinction, not a missing measurement, and silently dropping those
            rows biases any estimate of clonal turnover.

    Returns:
        A pandas DataFrame with columns ``cell_line``, ``barcode``,
        ``timepoint``, ``plate``, ``sample``, ``count``. A DataFrame rather
        than an ExperimentalDataset because these are clone abundances indexed
        by sequencing timepoint, not a per-series time course, and umimic has
        no lineage-aware observation model to consume them yet.

    Notes:
        ``timepoint`` is the DT index (1-4) of the sequencing sample and
        ``plate`` the replicate plate (P1/P2); ``sample`` keeps the original
        ``DT<n>_P<m>`` label. Library sizes differ substantially between
        samples, so compare relative frequencies rather than raw counts.
    """
    d = _find_data_dir("whiting2025_barcode", data_root)
    product = _resolve_umimic_product(d, "whiting2025_barcode_lineage_umimic")
    if product is None:
        raise FileNotFoundError(
            f"Whiting 2025 lineage product not found in {d}.\n"
            "Expected whiting2025_barcode_lineage_umimic.csv.xz.\n"
            "Rebuild it with: python -m umimic.data.public.convert_whiting2025\n"
            "Source: https://doi.org/10.1038/s41467-025-59479-7"
        )

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to load Whiting 2025 lineage data. "
            "Install pandas: pip install pandas"
        ) from exc

    df = pd.read_csv(product)
    if cell_line is not None:
        key = cell_line.lower()
        df = df[df["cell_line"].str.lower().str.contains(key)]
        if df.empty:
            raise ValueError(
                f"No Whiting 2025 lineage rows match cell_line={cell_line!r}. "
                "Available: HCT116_barcode, SW620_barcode."
            )
    if min_count > 0:
        df = df[df["count"] >= min_count]
    return df.reset_index(drop=True)


def load_whiting2025_barcode(
    data_root: Path | None = None,
    cell_line: str | None = None,
) -> ExperimentalDataset:
    """Load Whiting et al. (2025) bulk population trajectories (barcode study).

    This is **not imaging data**. It is a minimal extract of bulk population
    size vs time for barcoded HCT116 and SW620 lines under pulsed treatment
    (Nature Communications Supplementary Data 2 population tables only).

    Args:
        data_root: Override data root.
        cell_line: Substring filter (``"HCT"`` / ``"SW6"`` / full product id).

    Returns:
        ExperimentalDataset with modality ``cell_counts`` = population size,
        one series per biological replicate (4 per line).

    Notes:
        Time is stored in hours (``day * 24``). Concentration is 0 because the
        experiment uses a multi-cycle pulsed schedule, not a constant dose —
        see ``whiting2025_barcode/DATA_NOTES.md``. The per-barcode lineage
        matrices are shipped separately -- see :func:`load_whiting2025_lineage`.
        scRNA/scWGS tables are intentionally not shipped.
    """
    d = _find_data_dir("whiting2025_barcode", data_root)
    product = _resolve_umimic_product(d, "whiting2025_barcode_population_umimic")
    if product is None:
        raise FileNotFoundError(
            f"Whiting 2025 population product not found in {d}.\n"
            "Expected whiting2025_barcode_population_umimic.csv.xz.\n"
            "Source: https://doi.org/10.1038/s41467-025-59479-7"
        )

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to load Whiting 2025 data. Install pandas: pip install pandas"
        ) from exc

    df = pd.read_csv(product)
    if cell_line is not None:
        key = cell_line.lower()
        df = df[df["cell_line"].astype(str).str.lower().str.contains(key, regex=False)]
    if df.empty:
        raise FileNotFoundError(
            f"No Whiting 2025 rows matched cell_line={cell_line!r} in {product}."
        )

    series: list[TimeSeriesData] = []
    for gid, sub in df.groupby("group_id", sort=False):
        sub = sub.sort_values("hours")
        series.append(
            TimeSeriesData(
                times=sub["hours"].to_numpy(dtype=float),
                observations={
                    "cell_counts": sub["population_size"].to_numpy(dtype=float),
                },
                concentration=0.0,
                group_id=str(gid),
                units={"cell_counts": "cells (bulk population size)"},
                metadata={
                    "dataset": "whiting2025_barcode",
                    "citation": WHITING2025_CITATION,
                    "cell_line": str(sub["cell_line"].iloc[0]),
                    "replicate": int(sub["replicate"].iloc[0])
                    if "replicate" in sub.columns
                    else None,
                    "note": (
                        "Bulk population size from barcoding study; not live-cell "
                        "imaging. Treatment is pulsed (see DATA_NOTES.md)."
                    ),
                    "source_file": product.name,
                },
            )
        )

    return ExperimentalDataset(
        series=series,
        name="whiting2025_barcode_population",
        context="in_vitro",
    )


# ---------------------------------------------------------------------------
# PhenoPop: Ba/F3 live-cell imaging
# ---------------------------------------------------------------------------

def load_phenopop(
    data_root: Path | None = None,
    population: str | None = None,
) -> ExperimentalDataset:
    """Load PhenoPop Ba/F3 live-cell imaging data.

    Args:
        data_root: Override data directory.
        population: Filter by population type ("sensitive", "resistant", "mixed").

    Returns:
        ExperimentalDataset with one TimeSeriesData per replicate.

    Resolution order:
        1. ``phenopop_umimic.csv.xz`` / ``.csv`` (canonical product)
        2. ``converted/`` intermediate CSVs
        3. ``.mat`` files in a cloned ``phenopop_repo`` cache
    """
    d = _find_data_dir("phenopop", data_root)

    product = _resolve_umimic_product(d, "phenopop_umimic")
    if product is not None:
        # Product rows share replicate_id across concentrations; series key is
        # (replicate_id, concentration).
        ds = _load_long_csv(
            product,
            dataset_name="phenopop_baf3",
            context="in_vitro",
            time_col="hours",
            count_col="viable_count",
            conc_col="concentration",
            group_col="replicate_id",
            composite_group_cols=("replicate_id", "concentration"),
        )
        if population is not None:
            key = population.lower()
            ds = ExperimentalDataset(
                series=[
                    s for s in ds.series
                    if key in str(s.metadata.get("population", s.group_id or "")).lower()
                    or key in str(s.group_id or "").lower()
                ],
                name=ds.name,
                context=ds.context,
            )
        return ds

    # Intermediate conversion outputs
    mono_csv = d / "converted" / "phenopop_monoclonal_data.csv"
    mix_csv = d / "converted" / "phenopop_mixture_data.csv"

    if mono_csv.exists() or mix_csv.exists():
        return _load_phenopop_csv(d / "converted", population)

    combined = d / "converted" / "phenopop_baf3_combined.csv"
    if combined.exists():
        return _load_long_csv(
            combined,
            dataset_name="phenopop_baf3",
            context="in_vitro",
        )

    # Optional raw cache (not shipped with the package)
    repo_data = d / "phenopop_repo" / "In vitro experiment"
    if repo_data.exists():
        return _load_phenopop_mat(repo_data, population)

    raise FileNotFoundError(
        f"PhenoPop data not found in {d}. "
        "Run: python -m umimic.data.public.download_datasets --dataset phenopop\n"
        "Then: python -m umimic.data.public.convert_phenopop"
    )


def _load_phenopop_csv(
    converted_dir: Path,
    population_filter: str | None = None,
) -> ExperimentalDataset:
    """Load PhenoPop data from converted CSV files."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to load PhenoPop CSV data.")

    series = []

    for csv_name in ["phenopop_monoclonal_data.csv", "phenopop_mixture_data.csv"]:
        csv_path = converted_dir / csv_name
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)

        # Apply population filter
        if population_filter and "population" in df.columns:
            if population_filter.lower() in ["sensitive", "resistant"]:
                df = df[df["population"].str.contains(population_filter, case=False)]
            elif population_filter.lower() in ["mixed", "mixture", "mix"]:
                df = df[df["population"].str.contains("mix", case=False)]

        if df.empty:
            continue

        # Group by replicate_id and concentration
        group_cols = ["replicate_id", "concentration_uM"]
        available_cols = [c for c in group_cols if c in df.columns]
        if not available_cols:
            continue

        for group_key, sub in df.groupby(available_cols):
            sub = sub.sort_values("hours")
            times = sub["hours"].values.astype(float)
            counts = sub["viable_count"].values.astype(float)

            conc = sub["concentration_uM"].iloc[0] if "concentration_uM" in sub.columns else 0.0
            rep_id = sub["replicate_id"].iloc[0] if "replicate_id" in sub.columns else "unknown"
            pop = sub["population"].iloc[0] if "population" in sub.columns else "unknown"

            ts = TimeSeriesData(
                times=times,
                observations={"cell_counts": counts},
                concentration=float(conc),
                group_id=f"{rep_id}_C{conc}",
                metadata={
                    "dataset": "phenopop",
                    "population": str(pop),
                    "drug": "imatinib",
                    "cell_line": "Ba/F3",
                },
            )
            series.append(ts)

    return ExperimentalDataset(
        series=series,
        name="phenopop_baf3",
        context="in_vitro",
    )


def _load_phenopop_mat(
    data_dir: Path,
    population_filter: str | None = None,
) -> ExperimentalDataset:
    """Load PhenoPop data directly from MATLAB .mat files."""
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError(
            "scipy is required to load .mat files. "
            "Alternatively, run: python data/public/convert_phenopop.py"
        )

    series = []

    # Load mixture data
    mat_path = data_dir / "BF_11.mat"
    if mat_path.exists() and (population_filter is None or "mix" in (population_filter or "").lower()
                               or population_filter is None):
        mat = loadmat(str(mat_path))
        concs = mat["Conc"].flatten()
        times = mat["Time"].flatten().astype(float)

        mixtures = {"BF_11": "mix_1to1", "BF_12": "mix_1to2",
                     "BF_21": "mix_2to1", "BF_41": "mix_4to1"}

        for key, label in mixtures.items():
            if key not in mat:
                continue
            data = mat[key]
            n_rep, n_conc, n_time = data.shape

            for rep_idx in range(n_rep):
                for conc_idx in range(n_conc):
                    counts = data[rep_idx, conc_idx, :]
                    if np.all(np.isnan(counts)):
                        continue
                    ts = TimeSeriesData(
                        times=times,
                        observations={"cell_counts": counts},
                        concentration=float(concs[conc_idx]),
                        group_id=f"{label}_rep{rep_idx}_C{concs[conc_idx]}",
                        metadata={"dataset": "phenopop", "population": label},
                    )
                    series.append(ts)

    # Load monoclonal data
    mono_path = data_dir / "MONOCLONAL_DATA.mat"
    if mono_path.exists():
        mono = loadmat(str(mono_path))
        ref = loadmat(str(mat_path)) if mat_path.exists() else {}
        concs = ref.get("Conc", np.zeros((1, 11))).flatten()
        times = ref.get("Time", np.arange(14) * 3).flatten().astype(float)

        pop_map = {
            "SENSITIVE_500_BF": "sensitive",
            "SENSITIVE_1000_BF": "sensitive",
            "RESISTANT_250_BF": "resistant",
            "RESISTANT_500_BF": "resistant",
        }

        for key, pop_label in pop_map.items():
            if key not in mono:
                continue
            if population_filter and population_filter.lower() not in pop_label:
                continue

            data = mono[key]
            n_rep, n_conc, n_time = data.shape
            for rep_idx in range(n_rep):
                for conc_idx in range(min(n_conc, len(concs))):
                    counts = data[rep_idx, conc_idx, :]
                    if np.all(np.isnan(counts)):
                        continue
                    ts = TimeSeriesData(
                        times=times,
                        observations={"cell_counts": counts},
                        concentration=float(concs[conc_idx]),
                        group_id=f"{key}_rep{rep_idx}_C{concs[conc_idx]}",
                        metadata={"dataset": "phenopop", "population": pop_label},
                    )
                    series.append(ts)

    return ExperimentalDataset(
        series=series,
        name="phenopop_baf3",
        context="in_vitro",
    )


# ---------------------------------------------------------------------------
# TSHS Tumor Growth
# ---------------------------------------------------------------------------

def load_tshs_tumor(
    data_root: Path | None = None,
    treatment_group: str | None = None,
    *,
    include_suspicious: bool = True,
    include_nonmonotonic: bool = True,
) -> ExperimentalDataset:
    """Load TSHS xenograft tumor growth data.

    Args:
        data_root: Override data directory.
        treatment_group: Filter by arm name fragment (e.g. ``"control"``,
            ``"drug"``, ``"radiation"``, ``"drug+radiation"`` / ``"combo"``).
        include_suspicious: Keep Excel green-highlighted points
            (``quality_flag=suspicious_green``). Default True (owner advice).
        include_nonmonotonic: Keep Excel red-font non-monotonic points.
            Default True (treated as ordinary caliper error).

    Returns:
        ExperimentalDataset with one TimeSeriesData per mouse, modality
        ``volume``.

    Notes:
        Cite: Daskalakis C. Tumor Growth Dataset. TSHS Resources Portal, 2016.
        Available at https://www.causeweb.org/tshs/tumor-growth/.

        Two clearly erroneous orange-highlighted values (623.7 on mouse 202
        day 3; 734.4 on mouse 302 day 26) are **removed** from the distributed
        product. See ``tshs_tumor/DATA_NOTES.md`` and ``EXCLUDED_VALUES.csv``.
    """
    d = _find_data_dir("tshs_tumor", data_root)

    product = _resolve_umimic_product(d, "tshs_tumor_umimic")
    if product is None:
        # Legacy names
        for legacy in ("tumor_growth_umimic.csv", "tumor_growth_raw.csv"):
            candidate = d / legacy
            if candidate.exists() and candidate.stat().st_size > 200:
                product = candidate
                break

    if product is None:
        raise FileNotFoundError(
            f"TSHS tumor growth data not found in {d}.\n"
            "Place tshs_tumor_umimic.csv.xz there, or convert the workbook:\n"
            "  python -m umimic.data.public.convert_tshs\n"
            "Source: https://www.causeweb.org/tshs/tumor-growth/"
        )

    return _load_tshs_long(
        product,
        group_filter=treatment_group,
        include_suspicious=include_suspicious,
        include_nonmonotonic=include_nonmonotonic,
    )


def _load_tshs_long(
    path: Path,
    group_filter: str | None = None,
    *,
    include_suspicious: bool = True,
    include_nonmonotonic: bool = True,
) -> ExperimentalDataset:
    """Parse TSHS long-form product (csv or csv.xz)."""
    try:
        import pandas as pd
    except ImportError:
        return _load_tshs_numpy(path, group_filter)

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    # Drop placeholder / comment-only stubs
    if df.empty or (
        "hours" in cols
        and df[cols["hours"]].astype(str).str.startswith("#").all()
    ):
        raise FileNotFoundError(
            f"{path} looks like a placeholder, not real TSHS data. "
            "Run: python -m umimic.data.public.convert_tshs"
        )

    if "hours" not in cols or "tumor_volume" not in cols:
        return _load_tshs_wide(df, path, group_filter)

    # QC filters (orange clear-errors already removed from the product)
    if "quality_flag" in cols:
        qcol = cols["quality_flag"]
        drop: set[str] = {"exclude_clear_error"}
        if not include_suspicious:
            drop.add("suspicious_green")
        if not include_nonmonotonic:
            drop.add("nonmonotonic_red")
        df = df[~df[qcol].astype(str).isin(drop)]

    treat_col = cols.get("treatment")
    if group_filter and treat_col is not None:
        key = group_filter.lower().replace(" ", "")
        if key in ("combo", "drug+rad", "drug+radiation", "drugradiation"):
            key = "drug+radiation"
        mask = df[treat_col].astype(str).str.lower().str.replace(" ", "").str.contains(
            key, regex=False
        )
        df = df[mask]

    group_c = cols.get("group_id", cols.get("mouse_id"))
    time_c = cols["hours"]
    vol_c = cols["tumor_volume"]
    series: list[TimeSeriesData] = []
    groups = df[group_c].unique() if group_c in df.columns else ["all"]

    for gid in groups:
        sub = (
            df[df[group_c] == gid].sort_values(time_c)
            if group_c in df.columns
            else df.sort_values(time_c)
        )
        if len(sub) < 2:
            continue
        meta: dict[str, Any] = {
            "dataset": "tshs_tumor",
            "citation": TSHS_CITATION,
        }
        if treat_col is not None:
            meta["treatment_group"] = str(sub[treat_col].iloc[0])
        if "mouse_id" in cols:
            meta["mouse_id"] = str(sub[cols["mouse_id"]].iloc[0])
        if "quality_flag" in cols:
            meta["quality_flags"] = sorted(
                {str(v) for v in sub[cols["quality_flag"]].unique()}
            )
        series.append(
            TimeSeriesData(
                times=sub[time_c].to_numpy(dtype=float),
                observations={"volume": sub[vol_c].to_numpy(dtype=float)},
                concentration=0.0,
                group_id=str(gid),
                units={"volume": "mm^3"},
                metadata=meta,
            )
        )

    return ExperimentalDataset(
        series=series,
        name="tshs_tumor_growth",
        context="in_vivo",
    )


def _load_tshs_wide(
    df,
    path: Path,
    group_filter: str | None = None,
) -> ExperimentalDataset:
    """Fallback: wide-format CSV (columns = measurement days)."""
    import re

    import pandas as pd

    group_col = None
    for candidate in ["Group", "group", "Treatment", "treatment", "Grp", "Arm"]:
        if candidate in df.columns:
            group_col = candidate
            break

    id_col = None
    for candidate in ["Mouse", "ID", "Subject", "Animal", "mouse_id"]:
        if candidate in df.columns:
            id_col = candidate
            break

    time_cols = []
    for col in df.columns:
        if col in (group_col, id_col):
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() > 0:
            time_cols.append(col)

    if not time_cols:
        warnings.warn(f"Could not identify time columns in {path}", stacklevel=2)
        return ExperimentalDataset(series=[], name="tshs_tumor_growth", context="in_vivo")

    day_map: dict[str, float] = {}
    for col in time_cols:
        m = re.search(r"(\d+\.?\d*)", str(col))
        day_map[col] = float(m.group(1)) if m else float(time_cols.index(col))

    series: list[TimeSeriesData] = []
    for idx, row in df.iterrows():
        group = str(row[group_col]) if group_col else "unknown"
        if group_filter and group_filter.lower() not in group.lower():
            continue
        mouse_id = str(row[id_col]) if id_col else f"mouse_{idx}"
        times_h: list[float] = []
        volumes: list[float] = []
        for col in time_cols:
            val = row[col]
            if pd.notna(val):
                try:
                    volumes.append(float(val))
                    times_h.append(day_map[col] * 24.0)
                except (ValueError, TypeError):
                    continue
        if len(times_h) < 2:
            continue
        series.append(
            TimeSeriesData(
                times=np.asarray(times_h, dtype=float),
                observations={"volume": np.asarray(volumes, dtype=float)},
                concentration=0.0,
                group_id=f"mouse_{mouse_id}",
                units={"volume": "mm^3"},
                metadata={
                    "treatment_group": group,
                    "dataset": "tshs_tumor",
                    "citation": TSHS_CITATION,
                },
            )
        )

    return ExperimentalDataset(
        series=series,
        name="tshs_tumor_growth",
        context="in_vivo",
    )


def _load_tshs_numpy(path: Path, group_filter: str | None = None) -> ExperimentalDataset:
    """Fallback TSHS loader without pandas -- not implemented.

    This previously returned one all-zero series per row, ignoring both the
    file contents and ``group_filter``. Zero-volume tumours measured at a
    single time point are not a degraded answer, they are a fabricated one,
    and nothing downstream can tell them from real data.
    """
    raise ImportError(
        f"pandas is required to load the TSHS tumor-growth data at {path}. "
        "The wide-format parsing (variable per-study group and day columns) "
        "has no numpy-only implementation. Install pandas: pip install pandas"
    )


# ---------------------------------------------------------------------------
# Hafner/Niepel GR Metrics
# ---------------------------------------------------------------------------

def load_hafner_gr(
    data_root: Path | None = None,
    cell_line: str | None = None,
    drug: str | None = None,
) -> ExperimentalDataset:
    """Load Hafner/Niepel GR metrics breast cancer dataset.

    Args:
        data_root: Override data directory.
        cell_line: Filter by cell line name.
        drug: Filter by drug name.

    Returns:
        ExperimentalDataset with dose-response data (Tz + 72h endpoint).

    Prefers the compact ``hafner_gr_umimic.csv.xz`` product. Raw Dryad /
    Nature Methods supplement trees are optional caches only.
    """
    d = _find_data_dir("hafner_gr", data_root)

    product = _resolve_umimic_product(d, "hafner_gr_umimic")
    if product is not None:
        ds = _load_long_csv(
            product,
            dataset_name="hafner_gr",
            context="in_vitro",
            time_col="hours",
            count_col="viable_count",
            conc_col="concentration",
            group_col="replicate_id",
        )
        if cell_line is not None or drug is not None:
            kept = []
            for s in ds.series:
                cl = str(s.metadata.get("cell_line", "")).lower()
                dr = str(s.metadata.get("drug", "")).lower()
                if cell_line and cell_line.lower() not in cl:
                    continue
                if drug and drug.lower() not in dr:
                    continue
                kept.append(s)
            ds = ExperimentalDataset(series=kept, name=ds.name, context=ds.context)
        return ds

    # Fallback: only top-level / product-like files, not nested raw dumps
    csv_files = sorted(d.glob("*.csv")) + sorted(d.glob("*.tsv"))
    csv_files += sorted(d.glob("*.csv.xz"))
    if not csv_files:
        raise FileNotFoundError(
            f"No Hafner product found in {d}. Expected hafner_gr_umimic.csv.xz.\n"
            "Download from: https://datadryad.org/dataset/doi:10.5061/dryad.03n60\n"
            "or run: python -m umimic.data.public.download_datasets --dataset hafner_gr"
        )

    return _load_hafner_files(csv_files, cell_line, drug)


def _load_hafner_files(
    csv_files: list[Path],
    cell_line_filter: str | None = None,
    drug_filter: str | None = None,
) -> ExperimentalDataset:
    """Parse Hafner GR data files into ExperimentalDataset."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to load Hafner GR data.")

    series = []
    for fpath in csv_files:
        sep = "\t" if fpath.suffix == ".tsv" else ","
        try:
            df = pd.read_csv(fpath, sep=sep)
        except Exception:
            continue

        cols_lower = {c.lower(): c for c in df.columns}

        # Look for key columns
        cell_col = None
        for c in ["cell_line", "cellline", "cell line", "CellLine"]:
            if c in cols_lower:
                cell_col = cols_lower[c]
                break

        drug_col = None
        for c in ["agent", "drug", "perturbagen", "compound", "Drug"]:
            if c in cols_lower:
                drug_col = cols_lower[c]
                break

        conc_col = None
        for c in ["concentration", "conc", "dose", "Concentration"]:
            if c in cols_lower:
                conc_col = cols_lower[c]
                break

        count_col = None
        for c in ["cell_count", "cellcount", "value", "ctg", "cell_count__time0",
                   "cell_count__ctrl", "relative_cell_count"]:
            if c in cols_lower:
                count_col = cols_lower[c]
                break

        gr_col = None
        for c in ["gr_value", "grvalue", "gr", "GRvalue"]:
            if c in cols_lower:
                gr_col = cols_lower[c]
                break

        if conc_col is None:
            continue

        # Apply filters
        if cell_line_filter and cell_col:
            df = df[df[cell_col].str.contains(cell_line_filter, case=False, na=False)]
        if drug_filter and drug_col:
            df = df[df[drug_col].str.contains(drug_filter, case=False, na=False)]

        if df.empty:
            continue

        # Group by cell line + drug combination
        group_cols = [c for c in [cell_col, drug_col] if c is not None]
        if not group_cols:
            group_cols = [df.columns[0]]

        for group_key, sub_df in df.groupby(group_cols):
            if isinstance(group_key, str):
                gid = group_key
            else:
                gid = "_".join(str(k) for k in group_key)

            concs = sub_df[conc_col].values.astype(float)
            sort_idx = np.argsort(concs)
            concs = concs[sort_idx]

            observations = {}
            if count_col and count_col in sub_df.columns:
                observations["cell_counts"] = sub_df[count_col].values.astype(float)[sort_idx]
            if gr_col and gr_col in sub_df.columns:
                observations["gr_value"] = sub_df[gr_col].values.astype(float)[sort_idx]

            if not observations:
                continue

            # For endpoint data, represent as dose points (time=72h for all)
            ts = TimeSeriesData(
                times=np.full(len(concs), 72.0),  # 72h endpoint
                observations=observations,
                concentrations=concs,
                group_id=gid,
                metadata={
                    "dataset": "hafner_gr",
                    "source_file": fpath.name,
                    "data_type": "dose_response_endpoint",
                },
            )
            series.append(ts)

    return ExperimentalDataset(
        series=series,
        name="hafner_gr_metrics",
        context="in_vitro",
    )


# ---------------------------------------------------------------------------
# NCI-60 Growth Inhibition
# ---------------------------------------------------------------------------

def resolve_nci60_nsc(nsc: str | int | None) -> str | None:
    """Map a drug alias or NSC code to the string NSC used in the product.

    Accepts numeric NSC (``119875``), strings (``"119875"``), or a few common
    names (``"cisplatin"`` → ``"119875"``). Unknown names are returned uppercased
    as-is so callers can pass raw NSC strings.
    """
    if nsc is None:
        return None
    text = str(nsc).strip()
    if not text:
        return None
    key = text.lower().replace(" ", "").replace("_", "-")
    if key in NCI60_COMMON_NSC:
        return NCI60_COMMON_NSC[key]
    # bare integer / numeric string
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _resolve_nci60_product(
    data_path: str | Path | None = None,
    data_root: Path | None = None,
) -> Path:
    """Locate ``nci60_umimic.csv.xz`` (or a user-supplied long-form table)."""
    if data_path is not None:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"NCI-60 data path not found: {path}")
        return path

    root = Path(data_root) if data_root is not None else DATA_ROOT
    d = root / "nci60"
    product = _resolve_umimic_product(d, "nci60_umimic")
    if product is not None:
        return product
    raise FileNotFoundError(
        f"NCI-60 product not found under {d}. Expected nci60_umimic.csv.xz.\n"
        "Source: https://dtp.cancer.gov/databases_tools/bulk_data.htm"
    )


def _read_nci60_filtered(
    path: Path,
    *,
    nsc: str | None,
    cell_line: str | None,
    panel: str | None,
    chunksize: int = 200_000,
):
    """Stream the product and keep rows matching filters (pandas DataFrame)."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to load NCI-60 data. Install pandas: pip install pandas"
        ) from exc

    pieces = []
    nsc_s = str(nsc) if nsc is not None else None
    cell_key = cell_line.lower() if cell_line else None
    panel_key = panel.lower() if panel else None

    for chunk in pd.read_csv(path, chunksize=chunksize):
        if "NSC" not in chunk.columns:
            raise ValueError(
                f"{path} is not an NCI-60 umimic product (missing NSC column). "
                f"Columns: {list(chunk.columns)}"
            )
        sub = chunk
        if nsc_s is not None:
            sub = sub[sub["NSC"].astype(str).str.replace(r"\.0$", "", regex=True) == nsc_s]
        if cell_key is not None:
            sub = sub[sub["cell_line"].astype(str).str.lower().str.contains(cell_key, regex=False)]
        if panel_key is not None:
            sub = sub[sub["panel"].astype(str).str.lower().str.contains(panel_key, regex=False)]
        if not sub.empty:
            pieces.append(sub)

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def load_nci60(
    data_path: str | Path | None = None,
    cell_line: str | None = None,
    *,
    data_root: Path | None = None,
    nsc: str | int | None = None,
    panel: str | None = None,
    representation: str = "reconstructed_counts",
    n0: float = 10_000.0,
    assay_hours: float = 48.0,
    aggregate: str = "mean",
    max_series: int | None = 5_000,
    min_endpoint_count: float = 1.0,
) -> ExperimentalDataset:
    """Load NCI-60 DTP dose-response screening data into U-MIMIC series.

    The shipped product (``nci60/nci60_umimic.csv.xz``) is long-form
    **endpoint** screening: one growth-inhibition / percent-of-control value
    per experiment × cell line × NSC × concentration. It is **not** a true
    multi-time longitudinal count assay.

    Parameters
    ----------
    data_path:
        Optional path to ``nci60_umimic.csv.xz`` (or equivalent long table).
        Defaults to the package product under ``DATA_ROOT/nci60/``.
    cell_line:
        Substring filter on cell line (case-insensitive), e.g. ``"MCF7"``.
    nsc:
        NSC compound code or a known alias (``"cisplatin"`` → ``119875``).
        See :data:`NCI60_COMMON_NSC`.
    panel:
        Substring filter on tissue panel (e.g. ``"Breast"``).
    representation:
        How to embed endpoints into :class:`TimeSeriesData`:

        * ``"reconstructed_counts"`` (default) — build a **two-point** series
          ``(t=0, N0)`` → ``(t=assay_hours, N_end)`` with
          ``N_end = max(N0 * percent_of_control / 100, min_endpoint_count)``.
          Suitable for simple net-growth / dose-response fits. **Assumes** a
          nominal seed ``N0`` and assay duration; PTC is relative to untreated
          control at the endpoint, not a true absolute count trajectory.
        * ``"endpoint_metrics"`` — single time point at ``assay_hours`` with
          modalities ``growth_inhibition_pct`` and ``percent_of_control``
          (honest endpoint storage; not directly a birth-death count series).
    n0, assay_hours:
        Seed size and assay length used only for ``reconstructed_counts``.
        Defaults (10 000 cells, 48 h) match common NCI-60 practice and the
        package notebooks; they are **not** measured per row in the product.
    aggregate:
        ``"mean"`` (default) averages GI% and PTC across ``experiment_id`` at
        the same (cell_line, NSC, concentration_log10M). ``"none"`` keeps each
        experiment as its own series.
    max_series:
        Safety cap on the number of returned series (default 5000). Set to
        ``None`` only when filters already bound the query tightly.
    min_endpoint_count:
        Floor for reconstructed endpoint counts (avoids non-positive values
        when PTC ≤ 0 under strong cytotoxicity).

    Returns
    -------
    ExperimentalDataset
        One series per (cell line × NSC × concentration [× experiment]),
        ``context="in_vitro"``. Metadata includes citation, NSC, panel, GI%,
        PTC, and reconstruction assumptions when applicable.

    Notes
    -----
    * **Filters required in practice:** the full extract is ~2.5M rows. Always
      pass ``nsc`` and/or ``cell_line`` / ``panel``. Loading without filters
      is allowed only if ``max_series`` is set (still streams the whole file).
    * Concentration column is **µM** (``10**concentration_log10M * 1e6``).
    * Cite NCI DTP when publishing results derived from these data.

    Examples
    --------
    >>> ds = load_nci60(nsc="cisplatin", cell_line="MCF7")  # doctest: +SKIP
    >>> ds = load_nci60(nsc=119875, panel="Breast", representation="endpoint_metrics")  # doctest: +SKIP
    """
    if representation not in ("reconstructed_counts", "endpoint_metrics"):
        raise ValueError(
            "representation must be 'reconstructed_counts' or 'endpoint_metrics', "
            f"got {representation!r}"
        )
    if aggregate not in ("mean", "none"):
        raise ValueError(f"aggregate must be 'mean' or 'none', got {aggregate!r}")
    if n0 <= 0:
        raise ValueError(f"n0 must be positive, got {n0}")
    if assay_hours <= 0:
        raise ValueError(f"assay_hours must be positive, got {assay_hours}")

    nsc_resolved = resolve_nci60_nsc(nsc)
    if nsc_resolved is None and cell_line is None and panel is None:
        warnings.warn(
            "load_nci60 called without nsc/cell_line/panel filters. "
            "Streaming the full ~2.5M-row extract; results are capped by "
            f"max_series={max_series}. Prefer filtering (e.g. nsc='cisplatin').",
            stacklevel=2,
        )

    path = _resolve_nci60_product(data_path, data_root)
    df = _read_nci60_filtered(
        path, nsc=nsc_resolved, cell_line=cell_line, panel=panel
    )
    if df.empty:
        raise FileNotFoundError(
            "No NCI-60 rows matched the filters "
            f"(nsc={nsc_resolved!r}, cell_line={cell_line!r}, panel={panel!r}) "
            f"in {path}."
        )

    # Normalise types
    df = df.copy()
    df["NSC"] = df["NSC"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["concentration_log10M"] = df["concentration_log10M"].astype(float)
    df["growth_inhibition_pct"] = df["growth_inhibition_pct"].astype(float)
    df["percent_of_control"] = df["percent_of_control"].astype(float)
    if "concentration" in df.columns:
        df["concentration"] = df["concentration"].astype(float)
    else:
        df["concentration"] = (10.0 ** df["concentration_log10M"]) * 1e6

    group_cols = ["cell_line", "NSC", "concentration_log10M"]
    if aggregate == "none":
        group_cols = ["experiment_id"] + group_cols

    series: list[TimeSeriesData] = []
    for keys, sub in df.groupby(group_cols, sort=False):
        # pandas returns a scalar key when grouping by one column; always tuple here.
        key_t = keys if isinstance(keys, tuple) else (keys,)
        if aggregate == "mean":
            cell_s, nsc_s, clog = str(key_t[0]), str(key_t[1]), float(key_t[2])
            gi = float(sub["growth_inhibition_pct"].mean())
            ptc = float(sub["percent_of_control"].mean())
            conc_uM = float(sub["concentration"].mean())
            n_exp = (
                int(sub["experiment_id"].nunique())
                if "experiment_id" in sub.columns
                else 1
            )
            exp_id = None
        else:
            exp_id = str(key_t[0])
            cell_s, nsc_s, clog = str(key_t[1]), str(key_t[2]), float(key_t[3])
            gi = float(sub["growth_inhibition_pct"].iloc[0])
            ptc = float(sub["percent_of_control"].iloc[0])
            conc_uM = float(sub["concentration"].iloc[0])
            n_exp = 1

        panel_s = str(sub["panel"].iloc[0]) if "panel" in sub.columns else ""
        panel_code = str(sub["panel_code"].iloc[0]) if "panel_code" in sub.columns else ""

        meta: dict[str, Any] = {
            "dataset": "nci60",
            "citation": NCI60_CITATION,
            "NSC": nsc_s,
            "cell_line": cell_s,
            "panel": panel_s,
            "panel_code": panel_code,
            "growth_inhibition_pct": gi,
            "percent_of_control": ptc,
            "concentration_log10M": clog,
            "concentration_uM": conc_uM,
            "n_experiments_averaged": n_exp,
            "representation": representation,
            "source_file": path.name,
        }
        if exp_id is not None:
            meta["experiment_id"] = exp_id

        group_id = f"{cell_s}_NSC{nsc_s}_c{conc_uM:.4g}"
        if exp_id is not None:
            group_id = f"{exp_id}_{group_id}"

        if representation == "reconstructed_counts":
            n_end = max(n0 * ptc / 100.0, float(min_endpoint_count))
            meta.update(
                {
                    "reconstruction": {
                        "n0": n0,
                        "assay_hours": assay_hours,
                        "formula": "N_end = max(N0 * percent_of_control / 100, min_endpoint_count)",
                        "warning": (
                            "Endpoint screening reconstructed as a 2-point count "
                            "series. N0 and assay_hours are nominal assumptions, "
                            "not measured per well. PTC is relative to untreated "
                            "control at the endpoint."
                        ),
                    }
                }
            )
            ts = TimeSeriesData(
                times=np.array([0.0, float(assay_hours)]),
                observations={"cell_counts": np.array([float(n0), float(n_end)])},
                concentration=conc_uM,
                group_id=group_id,
                units={"cell_counts": "cells (reconstructed)"},
                metadata=meta,
            )
        else:
            ts = TimeSeriesData(
                times=np.array([float(assay_hours)]),
                observations={
                    "growth_inhibition_pct": np.array([gi]),
                    "percent_of_control": np.array([ptc]),
                },
                concentration=conc_uM,
                group_id=group_id,
                units={
                    "growth_inhibition_pct": "%",
                    "percent_of_control": "% of untreated control",
                },
                metadata=meta,
            )
        series.append(ts)

        if max_series is not None and len(series) >= max_series:
            warnings.warn(
                f"load_nci60 truncated to max_series={max_series}. "
                "Tighten nsc/cell_line/panel filters for a complete subset.",
                stacklevel=2,
            )
            break

    # Stable order: line, NSC, concentration
    series.sort(
        key=lambda s: (
            str(s.metadata.get("cell_line", "")),
            str(s.metadata.get("NSC", "")),
            float(s.concentration if s.concentration is not None else 0.0),
            str(s.group_id or ""),
        )
    )

    return ExperimentalDataset(
        series=series,
        name="nci60_dtp",
        context="in_vitro",
    )


# ---------------------------------------------------------------------------
# Generic CSV helpers
# ---------------------------------------------------------------------------

def _load_long_csv(
    path: Path,
    dataset_name: str = "",
    context: str = "in_vitro",
    time_col: str = "hours",
    count_col: str = "viable_count",
    conc_col: str = "concentration",
    group_col: str = "replicate_id",
    composite_group_cols: tuple[str, ...] | None = None,
) -> ExperimentalDataset:
    """Load a long-format CSV (or ``.csv.xz``) into ExperimentalDataset."""
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except ImportError:
        # Numpy fallback
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
        ts = TimeSeriesData(
            times=data[time_col].astype(float),
            observations={"cell_counts": data[count_col].astype(float)},
        )
        return ExperimentalDataset(series=[ts], name=dataset_name, context=context)

    # Flexible column matching
    cols = {c.lower(): c for c in df.columns}
    time_c = cols.get(time_col.lower(), cols.get("hours", cols.get("time", df.columns[0])))
    count_c = cols.get(count_col.lower(), cols.get("viable_count",
               cols.get("cell_count", cols.get("count"))))
    conc_c = cols.get(conc_col.lower(), cols.get("concentration", cols.get("conc")))
    group_c = cols.get(group_col.lower(), cols.get("replicate_id",
               cols.get("well", cols.get("group_id"))))

    series = []
    if composite_group_cols:
        resolved = []
        for name in composite_group_cols:
            actual = cols.get(name.lower())
            if actual is None or actual not in df.columns:
                resolved = []
                break
            resolved.append(actual)
        if resolved:
            group_iter = df.groupby(resolved, sort=False)
        elif group_c and group_c in df.columns:
            group_iter = ((gid, df[df[group_c] == gid]) for gid in df[group_c].unique())
        else:
            group_iter = (("all", df),)
    elif group_c and group_c in df.columns:
        group_iter = ((gid, df[df[group_c] == gid]) for gid in df[group_c].unique())
    else:
        group_iter = (("all", df),)

    for gid, sub_raw in group_iter:
        sub = sub_raw.sort_values(time_c)
        # Drop exact duplicate time stamps within a series (keep first).
        if sub.duplicated(subset=[time_c]).any():
            sub = sub.drop_duplicates(subset=[time_c], keep="first")
        times = sub[time_c].to_numpy(dtype=float)
        if times.size < 1:
            continue
        if times.size >= 2 and np.any(np.diff(times) <= 0):
            warnings.warn(
                f"Skipping non-monotonic series {gid!r} in {path.name}",
                stacklevel=2,
            )
            continue

        observations = {}
        if count_c and count_c in sub.columns:
            observations["cell_counts"] = sub[count_c].to_numpy(dtype=float)

        # Also check for volume, BLI, etc.
        for modality_key, candidates in [
            ("volume", ["tumor_volume", "volume", "vol"]),
            ("bli", ["bli", "bioluminescence", "photon_flux"]),
            ("gr_value", ["gr_value", "gr"]),
        ]:
            for cand in candidates:
                actual = cols.get(cand)
                if actual and actual in sub.columns:
                    observations[modality_key] = sub[actual].to_numpy(dtype=float)
                    break

        if not observations:
            continue

        conc = None
        if conc_c and conc_c in sub.columns:
            conc = float(sub[conc_c].iloc[0])

        meta: dict[str, Any] = {"dataset": dataset_name}
        for extra in (
            "population",
            "cell_line",
            "drug",
            "treatment",
            "mouse_id",
            "initial_cells",
            "sensitive_ratio",
            "source",
            "quality_flag",
            "replicate_id",
        ):
            actual = cols.get(extra)
            if actual and actual in sub.columns:
                meta[extra] = str(sub[actual].iloc[0])

        if isinstance(gid, tuple):
            group_id = "_".join(str(x) for x in gid)
        else:
            group_id = str(gid)

        ts = TimeSeriesData(
            times=times,
            observations=observations,
            concentration=conc,
            group_id=group_id,
            metadata=meta,
        )
        series.append(ts)

    return ExperimentalDataset(series=series, name=dataset_name, context=context)


def _load_generic_cell_count_csv(
    csv_files: list[Path],
    dataset_name: str = "",
    cell_line_filter: str | None = None,
    drug_filter: str | None = None,
) -> ExperimentalDataset:
    """Load cell count data from one or more CSV files."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to load this dataset.")

    all_series = []

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue

        cols = {c.lower(): c for c in df.columns}

        # Apply filters
        for filter_val, candidates in [
            (cell_line_filter, ["cell_line", "cellline", "cell"]),
            (drug_filter, ["drug", "agent", "compound", "treatment"]),
        ]:
            if filter_val:
                for cand in candidates:
                    actual = cols.get(cand)
                    if actual and actual in df.columns:
                        df = df[df[actual].str.contains(filter_val, case=False, na=False)]
                        break

        if df.empty:
            continue

        # Find time and count columns
        time_c = None
        for cand in ["hours", "time", "time_hours", "t"]:
            if cand in cols:
                time_c = cols[cand]
                break
        if time_c is None:
            continue

        count_c = None
        for cand in ["viable_count", "cell_count", "count", "n_cells", "cells"]:
            if cand in cols:
                count_c = cols[cand]
                break
        if count_c is None:
            continue

        conc_c = None
        for cand in ["concentration", "conc", "dose"]:
            if cand in cols:
                conc_c = cols[cand]
                break

        group_c = None
        for cand in ["replicate_id", "well", "rep", "group_id", "sample"]:
            if cand in cols:
                group_c = cols[cand]
                break

        groups = df[group_c].unique() if group_c and group_c in df.columns else ["all"]

        for gid in groups:
            if group_c and group_c in df.columns:
                sub = df[df[group_c] == gid].sort_values(time_c)
            else:
                sub = df.sort_values(time_c)

            times = sub[time_c].values.astype(float)
            counts = sub[count_c].values.astype(float)

            conc = None
            if conc_c and conc_c in sub.columns:
                conc = float(sub[conc_c].iloc[0])

            meta: dict[str, Any] = {"dataset": dataset_name, "source_file": fpath.name}
            for extra in ["cell_line", "drug", "cellline", "agent"]:
                actual = cols.get(extra)
                if actual and actual in sub.columns:
                    meta[extra] = str(sub[actual].iloc[0])

            ts = TimeSeriesData(
                times=times,
                observations={"cell_counts": counts},
                concentration=conc,
                group_id=str(gid),
                metadata=meta,
            )
            all_series.append(ts)

    return ExperimentalDataset(
        series=all_series,
        name=dataset_name,
        context="in_vitro",
    )


def _extract_concentration_from_name(name: str) -> float:
    """Extract drug concentration from a filename."""
    import re
    patterns = [
        r"(\d+\.?\d*)\s*[uU][mM]",
        r"conc[_=]?(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*nM",
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return float(m.group(1))
    return 0.0


# ---------------------------------------------------------------------------
# Convenience: list and describe available datasets
# ---------------------------------------------------------------------------

def list_available_datasets() -> dict[str, dict[str, str]]:
    """List available public datasets and their product / catalog status."""
    catalog = load_catalog().get("datasets", {})
    datasets: dict[str, dict[str, str]] = {}

    fallback = {
        "bestdr": {
            "description": "BESTDR HCT116 + cisplatin live-cell imaging counts",
            "source": "https://github.com/olliemcdonald/bestdr",
            "type": "in_vitro, time-series",
            "citation": BESTDR_CITATION,
        },
        "phenopop": {
            "description": "PhenoPop Ba/F3 live-cell imaging (imatinib)",
            "source": "https://github.com/chenyuwu233/PhenoPop_stochastic",
            "type": "in_vitro, time-series",
        },
        "tshs_tumor": {
            "description": "TSHS xenograft tumor growth (4 treatment arms)",
            "source": "https://www.causeweb.org/tshs/tumor-growth/",
            "type": "in_vivo, tumor volume",
            "citation": TSHS_CITATION,
        },
        "hafner_gr": {
            "description": "Hafner/Niepel GR metrics (71 lines x 107 drugs)",
            "source": "https://datadryad.org/dataset/doi:10.5061/dryad.03n60",
            "type": "in_vitro, dose-response endpoint",
        },
        "nci60": {
            "description": "NCI-60 growth inhibition bulk extract",
            "source": "https://dtp.cancer.gov/databases_tools/bulk_data.htm",
            "type": "in_vitro, dose-response",
            "citation": NCI60_CITATION,
        },
        "whiting2025_barcode": {
            "description": "Whiting 2025 bulk barcode population sizes (tiny extract)",
            "source": "https://doi.org/10.1038/s41467-025-59479-x",
            "type": "in_vitro, population time-series",
            "citation": WHITING2025_CITATION,
        },
    }

    names = sorted(set(fallback) | set(catalog))
    for name in names:
        entry = catalog.get(name, {})
        info = dict(fallback.get(name, {}))
        if entry:
            info["description"] = entry.get("title", info.get("description", name))
            info["source"] = entry.get("source_url", info.get("source", ""))
            info["type"] = entry.get("context", info.get("type", ""))
            if entry.get("citation"):
                info["citation"] = entry["citation"]
            if entry.get("product"):
                info["product"] = entry["product"]
        d = DATA_ROOT / name
        product_rel = entry.get("product") if entry else None
        product_path = (DATA_ROOT / product_rel) if product_rel else None
        has_product = bool(
            (product_path is not None and product_path.exists())
            or _resolve_umimic_product(d, f"{name}_umimic") is not None
            or (
                name == "tshs_tumor"
                and _resolve_umimic_product(d, "tshs_tumor_umimic") is not None
            )
            or (
                name == "whiting2025_barcode"
                and _resolve_umimic_product(
                    d, "whiting2025_barcode_population_umimic"
                )
                is not None
            )
            or (
                name == "bestdr"
                and _resolve_umimic_product(d, "bestdr_umimic") is not None
            )
        )
        info["product_present"] = "yes" if has_product else "no"
        info["downloaded"] = info["product_present"]
        info["path"] = str(d)
        datasets[name] = info

    return datasets
