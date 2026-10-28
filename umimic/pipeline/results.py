"""Results serialization and comparison."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from umimic.types import InferenceResult


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def dataset_fingerprint(series) -> str:
    """Identify the actual observations, exposure labels and series identities."""
    records = [
        {
            "times": item.times,
            "observations": item.observations,
            "concentration": item.concentration,
            "concentrations": item.concentrations,
            "group_id": item.group_id,
            "replicate_id": item.replicate_id,
            "initial_condition": {
                key: item.metadata[key]
                for key in ("initial_cells", "initial_fractions")
                if key in item.metadata
            },
        }
        for item in series
    ]
    payload = json.dumps(records, sort_keys=True, cls=NumpyEncoder, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_result(result: InferenceResult, path: str | Path) -> None:
    """Save inference result to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "schema_version": 1,
        "method": result.method,
        "context": result.context,
        "point_estimates": result.point_estimates,
        "metadata": result.metadata,
    }

    if result.mle is not None:
        data["mle"] = {
            "parameters": result.mle.parameters,
            "log_likelihood": result.mle.log_likelihood,
            "aic": result.mle.aic,
            "bic": result.mle.bic,
            "se": result.mle.se,
            "converged": result.mle.converged,
            "hessian": result.mle.hessian,
            "n_evaluations": result.mle.n_evaluations,
        }

    if result.mcmc is not None:
        data["mcmc"] = {
            "posterior_mean": result.mcmc.posterior_mean(),
            "posterior_std": result.mcmc.posterior_std(),
            "credible_interval_95": result.mcmc.credible_interval(0.05),
            "n_samples": result.mcmc.n_samples,
            "n_chains": result.mcmc.n_chains,
            "diagnostics": result.mcmc.diagnostics,
            "samples": result.mcmc.samples,
            "log_likelihood_trace": result.mcmc.log_likelihood_trace,
            "log_posterior_trace": result.mcmc.log_posterior_trace,
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def load_result(path: str | Path) -> dict:
    """Load inference result from JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare_results(
    results: dict[str, InferenceResult],
) -> dict[str, Any]:
    """Compare multiple inference results side by side.

    Args:
        results: Dict mapping label -> InferenceResult.

    Returns:
        Comparison summary with parameter estimates and fit statistics.
    """
    comparison = {"parameters": {}, "fit_statistics": {}}

    all_params = set()
    for label, res in results.items():
        all_params.update(res.point_estimates.keys())

    for param in sorted(all_params):
        comparison["parameters"][param] = {}
        for label, res in results.items():
            est = res.point_estimates.get(param)
            if est is not None:
                comparison["parameters"][param][label] = est

    for label, res in results.items():
        stats = {}
        if res.mle is not None:
            stats["log_likelihood"] = res.mle.log_likelihood
            stats["aic"] = res.mle.aic
            stats["bic"] = res.mle.bic
        comparison["fit_statistics"][label] = stats

    return comparison
