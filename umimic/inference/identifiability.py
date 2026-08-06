"""Practical identifiability diagnostics.

A model can fit well and still be telling you nothing about most of its
parameters: if the data cannot distinguish two parameter combinations, the
optimiser will return one of them, and the posterior will return the prior.
That failure is silent -- the fit converges, the curve looks right, and the
reported estimates carry the confidence of the prior rather than the data.

This module quantifies it. It builds the sensitivity of the predicted
observations to each parameter, on a log scale so that parameters with
different units are comparable, and inspects the singular values. Directions
with negligible singular values are combinations the data cannot constrain.

This is *practical* identifiability at a point in parameter space with a given
design, not structural identifiability. A parameter can be structurally
identifiable and still unrecoverable from a particular experiment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IdentifiabilityReport:
    """Result of a practical-identifiability analysis."""

    param_names: list[str]
    singular_values: np.ndarray
    #: Per-parameter score in [0, 1]: the fraction of that parameter's
    #: direction lying in the identifiable subspace. 1 means fully
    #: constrained by the data, 0 means invisible to it.
    scores: np.ndarray
    #: Right singular vectors of the near-null directions.
    null_directions: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    threshold: float = 1e-6

    @property
    def n_identifiable(self) -> int:
        """Number of parameter combinations the data can constrain."""
        if self.singular_values.size == 0:
            return 0
        return int(np.sum(self.singular_values > self.singular_values[0] * self.threshold))

    @property
    def condition_number(self) -> float:
        sv = self.singular_values
        if sv.size == 0 or sv[-1] <= 0:
            return float("inf")
        return float(sv[0] / sv[-1])

    def unidentifiable(self, cutoff: float = 0.1) -> list[str]:
        """Parameters the data effectively cannot constrain."""
        return [
            name
            for name, score in zip(self.param_names, self.scores)
            if score < cutoff
        ]

    def summary(self) -> str:
        """Human-readable report."""
        lines = [
            f"Practical identifiability: {self.n_identifiable} of "
            f"{len(self.param_names)} directions constrained "
            f"(condition number {self.condition_number:.2e})",
        ]
        order = np.argsort(self.scores)
        for i in order:
            flag = "  <-- unidentifiable" if self.scores[i] < 0.1 else ""
            lines.append(f"  {self.param_names[i]:>16s}  {self.scores[i]:.3f}{flag}")
        weak = self.unidentifiable()
        if weak:
            lines.append(
                "These parameters are set by the prior, not the data: "
                + ", ".join(weak)
            )
        return "\n".join(lines)


def analyze_identifiability(
    predict,
    params: dict[str, float],
    param_names: list[str] | None = None,
    *,
    step: float = 0.01,
    threshold: float = 1e-6,
) -> IdentifiabilityReport:
    """Assess which parameters a given design can actually constrain.

    Args:
        predict: Callable mapping a parameter dict to a 1-D array of predicted
            observations. Use whatever the experiment measures -- log counts,
            a marker fraction, or several modalities concatenated.
        params: Parameter values at which to evaluate. Practical
            identifiability is local, so use a plausible operating point.
        param_names: Subset to analyse. Defaults to every key in `params`.
        step: Relative finite-difference step on each parameter.
        threshold: Singular values below `threshold * max` count as null.

    Returns:
        An :class:`IdentifiabilityReport`.

    Example:
        >>> report = analyze_identifiability(predict, params)   # doctest: +SKIP
        >>> print(report.summary())                             # doctest: +SKIP
    """
    names = list(param_names if param_names is not None else params)
    if not names:
        raise ValueError("No parameters to analyse.")

    baseline = np.asarray(predict(params), dtype=float).ravel()
    if not np.all(np.isfinite(baseline)):
        raise ValueError("Baseline prediction contains non-finite values.")

    # Sensitivities with respect to log-parameters, so columns are comparable
    # across parameters measured in different units.
    sensitivity = np.zeros((baseline.size, len(names)))
    for j, name in enumerate(names):
        value = params[name]
        if value == 0:
            logger.warning(
                "Parameter %s is exactly zero; log-sensitivity is undefined "
                "and it is reported as unidentifiable.", name,
            )
            continue
        perturbed = dict(params)
        perturbed[name] = value * (1.0 + step)
        shifted = np.asarray(predict(perturbed), dtype=float).ravel()
        sensitivity[:, j] = (shifted - baseline) / np.log1p(step)

    _, singular_values, vt = np.linalg.svd(sensitivity, full_matrices=False)

    if singular_values[0] <= 0:
        scores = np.zeros(len(names))
        null = vt
    else:
        keep = singular_values > singular_values[0] * threshold
        # How much of each parameter's unit vector lies in the identifiable
        # span: the column norm of the retained right singular vectors.
        scores = np.sqrt((vt[keep] ** 2).sum(axis=0)) if keep.any() else np.zeros(len(names))
        null = vt[~keep]

    return IdentifiabilityReport(
        param_names=names,
        singular_values=singular_values,
        scores=scores,
        null_directions=null,
        threshold=threshold,
    )


def likelihood_identifiability(
    likelihood,
    theta: np.ndarray,
    **kwargs,
) -> IdentifiabilityReport:
    """Convenience wrapper for a :class:`~umimic.inference.likelihood.ModelLikelihood`.

    Predicts the expected observation for every modality the likelihood uses,
    concatenated, so the report reflects the actual measurement design rather
    than counts alone.
    """
    params = likelihood.theta_to_params(np.asarray(theta, dtype=float))

    def predict(p: dict[str, float]) -> np.ndarray:
        rate_set = likelihood._build_rate_set(p)
        pieces = []
        for conc, series in likelihood._conc_groups.items():
            times = likelihood._group_times[conc]
            if len(times) < 2:
                continue
            t_sol, means, _ = likelihood._solve_forward(
                rate_set, conc, times, likelihood._initial_state(series[0], rate_set)
            )
            for data in series:
                idx = np.clip(np.searchsorted(t_sol, data.times), 0, len(t_sol) - 1)
                latent = np.maximum(means[idx], 0.0)
                for modality in likelihood._active_modalities:
                    if not data.has_modality(modality):
                        continue
                    model = likelihood._modality_models[modality]
                    # Same points the likelihood scores: the anchor used to
                    # set the initial condition is not a free observation.
                    mask = likelihood.scored_mask(data, modality)
                    pieces.append(
                        np.array(
                            [model.expected_value(s) for s in latent[mask]],
                            dtype=float,
                        )
                    )
        stacked = np.concatenate(pieces) if pieces else np.zeros(0)
        # Work on a log scale where the observable is positive, matching the
        # multiplicative noise these modalities carry.
        return np.log(np.maximum(stacked, 1e-300))

    return analyze_identifiability(predict, params, likelihood.param_names, **kwargs)
