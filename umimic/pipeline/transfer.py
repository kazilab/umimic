"""Cross-context transfer learning: in vitro posteriors -> in vivo priors.

The key U-MIMIC innovation: use mechanistic parameters learned in vitro
(under controlled conditions) as informative priors for in vivo inference
(where data is sparse and noisy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from umimic.inference.priors import PriorSpec
from umimic.types import InferenceResult, MCMCResult


@dataclass
class TransferResult:
    """Result of cross-context transfer learning."""

    invitro_summary: dict[str, Any]
    transferred_priors: PriorSpec
    transfer_params: list[str]
    invivo_result: InferenceResult | None = None


class TransferLearning:
    """Use in vitro posteriors as informative priors for in vivo inference.

    Workflow:
    1. Fit in vitro data -> posterior samples for PD parameters
    2. Fit parametric distributions to posterior marginals
    3. Use fitted distributions as priors for in vivo inference
    4. In vivo inference only needs to learn PK + observation params
    """

    def __init__(
        self,
        invitro_result: InferenceResult,
        transfer_params: list[str] | None = None,
        shrinkage: float = 1.0,
    ):
        """
        Args:
            invitro_result: Inference result from in vitro experiment.
            transfer_params: Which parameters to transfer (default: PD params).
            shrinkage: Shrinkage factor (0-1). 1.0 = use full posterior as prior.
                      <1.0 = inflate variance (less informative).
        """
        self.invitro_result = invitro_result
        self.transfer_params = transfer_params or [
            "b0", "d0_P", "emax_death", "ec50_death", "hill_death",
        ]
        self.shrinkage = shrinkage

    def build_priors(self) -> PriorSpec:
        """Convert in vitro posteriors to parametric priors.

        Fits a lognormal distribution to each posterior marginal,
        optionally inflating variance by the shrinkage factor.
        """
        if self.invitro_result.mcmc is not None:
            samples = self.invitro_result.mcmc.samples
        else:
            # If only MLE, create narrow priors around point estimates
            from scipy import stats
            spec = PriorSpec()
            for name in self.transfer_params:
                if name in self.invitro_result.point_estimates:
                    val = self.invitro_result.point_estimates[name]
                    se = 0.1 * val  # 10% relative uncertainty
                    if (
                        self.invitro_result.mle is not None
                        and self.invitro_result.mle.se
                        and name in self.invitro_result.mle.se
                    ):
                        se = self.invitro_result.mle.se[name]

                    se /= self.shrinkage  # inflate if shrinkage < 1
                    spec.add(
                        name,
                        stats.lognorm(
                            s=max(se / max(val, 1e-6), 0.1),
                            scale=val,
                        ),
                    )
            return spec

        # Full posterior transfer
        from scipy import stats

        spec = PriorSpec()
        for name in self.transfer_params:
            if name in samples:
                flat = samples[name].flatten()
                flat = flat[flat > 0]
                if len(flat) < 10:
                    continue

                log_samples = np.log(flat)
                mu = np.mean(log_samples)
                sigma = np.std(log_samples)

                # Apply shrinkage: inflate variance
                sigma /= self.shrinkage

                sigma = max(sigma, 0.05)  # minimum uncertainty
                spec.add(name, stats.lognorm(s=sigma, scale=np.exp(mu)))

        return spec

    def summarize_transfer(self) -> dict[str, Any]:
        """Summarize what was transferred and the prior specifications."""
        priors = self.build_priors()
        summary = {
            "n_params_transferred": len(self.transfer_params),
            "shrinkage": self.shrinkage,
            "parameters": {},
        }

        for name in self.transfer_params:
            if name in priors.distributions:
                dist = priors.distributions[name]
                summary["parameters"][name] = {
                    "prior_mean": float(dist.mean()),
                    "prior_std": float(dist.std()),
                    "prior_type": "lognormal",
                }

        return summary
