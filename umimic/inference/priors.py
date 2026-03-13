"""Prior specification utilities for Bayesian inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class PriorSpec:
    """Specification of prior distributions for model parameters.

    Each parameter maps to a scipy.stats distribution.
    """

    distributions: dict[str, stats.rv_continuous | stats.rv_frozen] = field(
        default_factory=dict
    )

    def add(self, name: str, dist: stats.rv_frozen) -> None:
        """Add a prior for a parameter."""
        self.distributions[name] = dist

    def log_prior(self, params: dict[str, float]) -> float:
        """Compute total log-prior for a parameter dict."""
        total = 0.0
        for name, value in params.items():
            if name in self.distributions:
                lp = self.distributions[name].logpdf(value)
                if np.isfinite(lp):
                    total += lp
                else:
                    return -np.inf
        return total

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, float]:
        """Draw one sample from the prior."""
        result = {}
        for name, dist in self.distributions.items():
            result[name] = float(dist.rvs(random_state=rng))
        return result

    @property
    def param_names(self) -> list[str]:
        return list(self.distributions.keys())

    @classmethod
    def default_invitro(cls) -> PriorSpec:
        """Default weakly informative priors for in vitro parameters."""
        spec = cls()
        spec.add("b0", stats.lognorm(s=0.5, scale=0.04))
        spec.add("d0_P", stats.lognorm(s=0.5, scale=0.01))
        spec.add("emax_death", stats.halfnorm(scale=0.1))
        spec.add("ec50_death", stats.lognorm(s=1.0, scale=1.0))
        spec.add("hill_death", stats.lognorm(s=0.3, scale=1.5))
        spec.add("u_PQ", stats.lognorm(s=0.5, scale=0.005))
        spec.add("u_QP", stats.lognorm(s=0.5, scale=0.003))
        spec.add("overdispersion", stats.lognorm(s=0.5, scale=10.0))
        return spec

    @classmethod
    def from_posterior(
        cls, posterior_samples: dict[str, np.ndarray], transfer_params: list[str]
    ) -> PriorSpec:
        """Build informative priors from posterior samples (for transfer learning).

        Fits a lognormal distribution to each posterior marginal.
        """
        spec = cls()
        for name in transfer_params:
            if name in posterior_samples:
                samples = posterior_samples[name].flatten()
                samples = samples[samples > 0]
                if len(samples) > 10:
                    log_samples = np.log(samples)
                    mu = np.mean(log_samples)
                    sigma = np.std(log_samples)
                    sigma = max(sigma, 0.1)  # minimum spread
                    spec.add(name, stats.lognorm(s=sigma, scale=np.exp(mu)))
        return spec
