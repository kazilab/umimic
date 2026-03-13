"""Results serialization and comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from umimic.types import InferenceResult, MLEResult, MCMCResult


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


def save_result(result: InferenceResult, path: str | Path) -> None:
    """Save inference result to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
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
        }

    if result.mcmc is not None:
        data["mcmc"] = {
            "posterior_mean": result.mcmc.posterior_mean(),
            "posterior_std": result.mcmc.posterior_std(),
            "credible_interval_95": result.mcmc.credible_interval(0.05),
            "n_samples": result.mcmc.n_samples,
            "n_chains": result.mcmc.n_chains,
            "diagnostics": result.mcmc.diagnostics,
        }

    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def load_result(path: str | Path) -> dict:
    """Load inference result from JSON file."""
    with open(path) as f:
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
