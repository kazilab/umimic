"""Cross-context transfer learning: in vitro posteriors -> in vivo priors.

Use mechanistic parameters learned in vitro, under controlled conditions, as
informative priors for in vivo inference, where data is sparse and noisy.

What may be transferred unchanged
---------------------------------
Only parameters that are **context-invariant**, i.e. dimensionless shape
parameters of the concentration-response relationship:

- ``hill_death``  -- the steepness of the response, a property of the
  target-engagement curve rather than of the medium.
- ``emax_death``  -- the maximal fractional effect.

Parameters carrying units of concentration or time are *not* invariant, and
are not transferred by default:

- ``ec50_death`` is a **concentration**. In vitro it is free drug in medium
  (essentially unbound in 10% FBS); the in vivo model's ``C(t)`` comes from
  ``PKConfig`` and is a *total plasma* concentration. For a compound that is
  90-99% protein bound the in vivo EC50 on a total-concentration scale is
  10-100x the in vitro value.
- ``b0``, ``d0_P`` are rates (1/hour). Xenograft doubling times are routinely
  3-10x slower than the matched 2D culture.

Transferring those unscaled produces a *narrow* prior centred on a wrong
value, which is the worst available failure mode: with sparse in vivo data a
tight biased prior dominates the likelihood rather than being overwhelmed by
it. This module therefore refuses to invent a scale factor. Supply one
explicitly via ``scale_factors`` when you have compound-specific information
(a free-fraction ratio ``fu_vitro / fu_vivo`` for EC50-type parameters, a
measured growth-rate ratio for kinetic ones), and the parameter becomes
transferable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from umimic.inference.priors import PriorSpec
from umimic.types import InferenceResult

logger = logging.getLogger(__name__)

#: Dimensionless shape parameters, transferable across contexts as-is.
CONTEXT_INVARIANT_PARAMS = ("emax_death", "hill_death")

#: Parameters whose units make them context-dependent, mapped to the reason.
CONTEXT_DEPENDENT_PARAMS = {
    "ec50_death": (
        "a concentration: in vitro it is free drug in medium, in vivo the PK "
        "model supplies total plasma concentration (protein binding typically "
        "shifts this 10-100x)"
    ),
    "ec50_birth": (
        "a concentration; see ec50_death. Protein binding and tissue "
        "partitioning both apply"
    ),
    "b0": (
        "a rate (1/hour): xenograft doubling times are routinely 3-10x slower "
        "than the matched 2D culture"
    ),
    "d0_P": "a rate (1/hour); see b0",
    "d0_Q": "a rate (1/hour); see b0",
    "d0_R": "a rate (1/hour); see b0",
}


@dataclass
class TransferResult:
    """Result of cross-context transfer learning."""

    invitro_summary: dict[str, Any]
    transferred_priors: PriorSpec
    transfer_params: list[str]
    #: Parameters requested but not transferred, mapped to the reason.
    skipped: dict[str, str] = field(default_factory=dict)
    invivo_result: InferenceResult | None = None


class TransferLearning:
    """Use in vitro posteriors as informative priors for in vivo inference.

    Workflow:
    1. Fit in vitro data -> posterior samples for PD parameters
    2. Fit parametric distributions to posterior marginals
    3. Use fitted distributions as priors for in vivo inference
    4. In vivo inference only needs to learn PK + observation params

    By default only the dimensionless shape parameters are transferred; see
    the module docstring for why, and use ``scale_factors`` to transfer a
    unit-carrying parameter with an explicit conversion.
    """

    def __init__(
        self,
        invitro_result: InferenceResult,
        transfer_params: list[str] | None = None,
        shrinkage: float = 1.0,
        scale_factors: dict[str, float] | None = None,
    ):
        """
        Args:
            invitro_result: Inference result from in vitro experiment.
            transfer_params: Which parameters to transfer. Defaults to the
                context-invariant shape parameters only. Naming a
                context-dependent parameter here requires a matching entry in
                ``scale_factors``; otherwise it is skipped with a warning
                rather than transferred on a wrong scale.
            shrinkage: How tightly the transferred prior hugs the in vitro
                estimate, in ``(0, 1]``.

                - ``1.0``: use the posterior (or MLE SE) width as-is.
                - ``s < 1``: inflate prior SD by ``1/s`` (weaker prior).
                - ``s <= 0`` or ``s > 1``: rejected (would divide by zero or
                  shrink beyond the data-supported width without a separate
                  model of bias).
            scale_factors: Multiplicative in vitro -> in vivo conversions,
                e.g. ``{"ec50_death": 30.0}`` for a compound whose in vivo
                total-concentration EC50 is 30x the in vitro free-drug value,
                or ``{"b0": 0.25}`` for a xenograft growing at a quarter of the
                culture rate. Applied to the prior's location; the relative
                width is preserved.
        """
        if not np.isfinite(shrinkage) or not (0.0 < shrinkage <= 1.0):
            raise ValueError(
                f"shrinkage must lie in (0, 1], got {shrinkage}. "
                "Use 1.0 for full transfer strength; values closer to 0 "
                "inflate prior variance (weaker information)."
            )
        self.invitro_result = invitro_result
        self.transfer_params = list(
            transfer_params
            if transfer_params is not None
            else CONTEXT_INVARIANT_PARAMS
        )
        self.shrinkage = float(shrinkage)

        self.scale_factors = dict(scale_factors or {})
        for name, factor in self.scale_factors.items():
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError(
                    f"scale_factors[{name!r}] must be finite and positive, got "
                    f"{factor!r}."
                )

        #: Populated by build_priors: parameter -> why it was not transferred.
        self.skipped: dict[str, str] = {}

    def _scale_for(self, name: str) -> float | None:
        """Conversion factor for `name`, or None if it must not be transferred.

        Returns 1.0 for context-invariant parameters, the user's factor when
        supplied, and None for a unit-carrying parameter with no factor -- the
        case where transferring would assert a relationship the package has no
        basis for.
        """
        if name in self.scale_factors:
            return self.scale_factors[name]
        if name in CONTEXT_DEPENDENT_PARAMS:
            return None
        return 1.0

    def build_priors(self) -> PriorSpec:
        """Convert in vitro posteriors to parametric priors.

        Fits a lognormal to each posterior marginal, applies any explicit in
        vitro -> in vivo scale factor to its location, and optionally inflates
        the width by the shrinkage factor. Parameters that cannot be
        transferred are recorded in ``self.skipped`` and logged, never dropped
        silently.
        """
        from scipy import stats

        self.skipped = {}
        spec = PriorSpec()

        samples = (
            self.invitro_result.mcmc.samples
            if self.invitro_result.mcmc is not None
            else None
        )

        for name in self.transfer_params:
            scale_factor = self._scale_for(name)
            if scale_factor is None:
                reason = (
                    f"{name} is {CONTEXT_DEPENDENT_PARAMS[name]}, so its in "
                    "vitro value does not carry over unchanged. Pass "
                    f"scale_factors={{{name!r}: <in vivo / in vitro ratio>}} "
                    "to transfer it explicitly."
                )
                self.skipped[name] = reason
                logger.warning("Not transferring %s. %s", name, reason)
                continue

            if samples is not None:
                fitted = self._fit_from_samples(samples, name)
            else:
                fitted = self._fit_from_point_estimate(name)

            if fitted is None:
                continue

            sigma, location = fitted
            spec.add(name, stats.lognorm(s=sigma, scale=location * scale_factor))

        return spec

    def _fit_from_samples(
        self, samples: dict[str, np.ndarray], name: str
    ) -> tuple[float, float] | None:
        """Lognormal (sigma, median) fitted to a posterior marginal."""
        if name not in samples:
            self.skipped[name] = "absent from the in vitro posterior samples"
            logger.warning(
                "Not transferring %s: it is not among the in vitro posterior "
                "samples (%s).", name, sorted(samples),
            )
            return None

        flat = np.asarray(samples[name]).reshape(-1)
        positive = flat[flat > 0]
        if positive.size < 10:
            self.skipped[name] = (
                f"only {positive.size} positive posterior draws; too few to fit "
                "a lognormal"
            )
            logger.warning("Not transferring %s: %s.", name, self.skipped[name])
            return None

        log_samples = np.log(positive)
        sigma = float(np.std(log_samples)) / self.shrinkage
        return max(sigma, 0.05), float(np.exp(np.mean(log_samples)))

    def _fit_from_point_estimate(self, name: str) -> tuple[float, float] | None:
        """Lognormal (sigma, median) around an MLE point estimate."""
        estimates = self.invitro_result.point_estimates
        if name not in estimates:
            self.skipped[name] = "absent from the in vitro point estimates"
            logger.warning("Not transferring %s: %s.", name, self.skipped[name])
            return None

        val = float(estimates[name])
        if val <= 0:
            self.skipped[name] = (
                f"in vitro estimate is {val:g}; a lognormal prior needs a "
                "positive location"
            )
            logger.warning("Not transferring %s: %s.", name, self.skipped[name])
            return None

        se = 0.1 * val  # 10% relative uncertainty when no SE is available
        mle = self.invitro_result.mle
        if mle is not None and mle.se and name in mle.se and mle.se[name] is not None:
            se = float(mle.se[name])

        se /= self.shrinkage  # inflate if shrinkage < 1
        # Lognormal shape s ~ CV for moderate spreads; floor keeps a usable
        # prior when SE is tiny or zero.
        return max(se / max(abs(val), 1e-6), 0.05), val

    def summarize_transfer(self) -> dict[str, Any]:
        """Summarize what was transferred and the prior specifications."""
        priors = self.build_priors()
        summary: dict[str, Any] = {
            # Count what was actually transferred. Reporting
            # len(self.transfer_params) counted requests, not results, so a
            # summary could claim three transfers having made one.
            "n_params_transferred": len(priors.distributions),
            "n_params_requested": len(self.transfer_params),
            "shrinkage": self.shrinkage,
            "scale_factors": dict(self.scale_factors),
            "skipped": dict(self.skipped),
            "parameters": {},
        }

        for name in self.transfer_params:
            if name in priors.distributions:
                dist = priors.distributions[name]
                summary["parameters"][name] = {
                    # Both are reported because they differ: scale= sets the
                    # median, and the lognormal mean is median*exp(s^2/2),
                    # which is what dist.mean() returns.
                    "prior_median": float(dist.median()),
                    "prior_mean": float(dist.mean()),
                    "prior_std": float(dist.std()),
                    "prior_type": "lognormal",
                    "scale_factor": self._scale_for(name),
                }

        return summary

    def to_result(self) -> TransferResult:
        """Bundle the transfer into a :class:`TransferResult`.

        `TransferResult` was previously exported as public API with nothing in
        the package able to construct one.
        """
        priors = self.build_priors()
        return TransferResult(
            invitro_summary=self.summarize_transfer(),
            transferred_priors=priors,
            transfer_params=list(self.transfer_params),
            skipped=dict(self.skipped),
        )
