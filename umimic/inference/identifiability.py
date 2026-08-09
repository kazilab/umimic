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
    *,
    step: float = 0.01,
    threshold: float = 1e-6,
    **kwargs,
) -> IdentifiabilityReport:
    """Practical identifiability for a :class:`~umimic.inference.likelihood.ModelLikelihood`.

    Uses the curvature of the log-likelihood -- the observed Fisher
    information ``-d2 l / d log(theta)^2`` -- rather than the sensitivity of
    the predicted *mean*.

    That distinction matters. A mean-based analysis can only see parameters
    that move ``expected_value``, so every observation-noise parameter
    (``overdispersion``, ``sigma_log_bli``, ``sigma_v``,
    ``biomarker_precision``) gets an exactly zero column and is reported
    "unidentifiable" no matter how well the data pins it down. Those
    parameters appear in most of the shipped parameter sets and are typically
    among the best determined, so the old verdict was not conservative, it was
    wrong -- and it told users to drop identified parameters.

    Evaluate this at or near a maximum. The observed information is only
    positive semi-definite at a stationary point; away from one the
    log-likelihood curves upward along some directions, those eigenvalues come
    out negative, and they are clipped to zero here (no information rather than
    negative information). The scores stay meaningful, but ``n_identifiable``
    undercounts, so prefer a fitted theta -- e.g. ``MLEResult.parameters`` --
    over a hand-picked one.

    Args:
        likelihood: The ModelLikelihood to analyse.
        theta: Parameter vector to analyse at. Identifiability is local, so
            use a fitted or otherwise plausible operating point.
        step: Relative finite-difference step on each parameter.
        threshold: Eigenvalues below ``threshold * max`` count as null.

    Returns:
        An :class:`IdentifiabilityReport`. Its ``singular_values`` are the
        square roots of the information eigenvalues, so the condition number
        keeps the same meaning as in :func:`analyze_identifiability`.
    """
    if kwargs:
        raise TypeError(
            f"Unexpected arguments {sorted(kwargs)}. This function now takes "
            "`step` and `threshold` only; it no longer forwards to "
            "analyze_identifiability."
        )

    theta = np.asarray(theta, dtype=float)
    names = list(likelihood.param_names)
    n = len(names)

    # Work in log-parameters so columns are comparable across units. A
    # parameter sitting at exactly zero has no log scale; it is excluded and
    # reported as unassessable rather than silently scored 0.
    if np.any(theta == 0):
        zeros = [names[i] for i in np.flatnonzero(theta == 0)]
        logger.warning(
            "Parameter(s) %s are exactly zero, so log-scale curvature is "
            "undefined and they are reported as unidentifiable.", zeros,
        )

    def ll_at(log_theta: np.ndarray) -> float:
        value = likelihood(np.exp(log_theta))
        return float(value)

    with np.errstate(divide="ignore"):
        log_theta0 = np.log(np.abs(theta))
    log_theta0 = np.where(theta == 0, 0.0, log_theta0)

    h = np.log1p(step)
    base = ll_at(log_theta0)
    if not np.isfinite(base):
        raise ValueError(
            "Log-likelihood is not finite at the requested point, so its "
            "curvature is undefined."
        )

    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ei[i] = h
            ej = np.zeros(n)
            ej[j] = h
            if i == j:
                f_p = ll_at(log_theta0 + ei)
                f_m = ll_at(log_theta0 - ei)
                value = (f_p - 2.0 * base + f_m) / h**2
            else:
                f_pp = ll_at(log_theta0 + ei + ej)
                f_pm = ll_at(log_theta0 + ei - ej)
                f_mp = ll_at(log_theta0 - ei + ej)
                f_mm = ll_at(log_theta0 - ei - ej)
                value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h**2)
            if not np.isfinite(value):
                # A non-finite stencil point means the likelihood is undefined
                # nearby (a bound, or a failed solve). Treating that direction
                # as carrying no information is the conservative reading.
                value = 0.0
            hessian[i, j] = hessian[j, i] = value

    fisher = -(hessian + hessian.T) / 2.0
    # Parameters pinned at zero carry no log-scale information.
    for i in np.flatnonzero(theta == 0):
        fisher[i, :] = 0.0
        fisher[:, i] = 0.0

    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    # A negative eigenvalue means the point is not a maximum along that
    # direction; it carries no usable information either way.
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    vt = eigenvectors[:, order].T

    singular_values = np.sqrt(eigenvalues)
    if singular_values[0] <= 0:
        scores = np.zeros(n)
        null = vt
    else:
        keep = singular_values > singular_values[0] * threshold
        scores = (
            np.sqrt((vt[keep] ** 2).sum(axis=0)) if keep.any() else np.zeros(n)
        )
        null = vt[~keep]

    return IdentifiabilityReport(
        param_names=names,
        singular_values=singular_values,
        scores=scores,
        null_directions=null,
        threshold=threshold,
    )
