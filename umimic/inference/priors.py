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
    def default_resistance(cls) -> PriorSpec:
        """Priors for a model carrying a resistant compartment.

        Matches :data:`~umimic.inference.likelihood.RESISTANCE_PARAM_NAMES`.
        R's division rate is centred slightly below P's (a modest fitness
        cost) and its residual drug sensitivity on a half-normal concentrated
        near zero, which encodes "resistant unless the data say otherwise"
        without forbidding partial sensitivity.
        """
        spec = cls.default_invitro()
        spec.add("b0_R", stats.lognorm(s=0.5, scale=0.036))
        spec.add("d0_R", stats.lognorm(s=0.5, scale=0.01))
        spec.add("emax_death_R", stats.halfnorm(scale=0.02))
        spec.add("u_PR", stats.lognorm(s=1.0, scale=1e-4))
        return spec

    @classmethod
    def default_persister(cls) -> PriorSpec:
        """Priors for the P -> Q -> R persister route.

        Matches :data:`~umimic.inference.likelihood.PERSISTER_PARAM_NAMES`.
        Persisters divide slowly, are largely refractory to the drug, and
        convert to resistance at a low drug-induced rate.
        """
        spec = cls.default_resistance()
        spec.add("b0_Q", stats.lognorm(s=0.8, scale=0.002))
        spec.add("d0_Q", stats.lognorm(s=0.5, scale=0.003))
        spec.add("emax_death_Q", stats.halfnorm(scale=0.005))
        spec.add("u_QR", stats.lognorm(s=1.5, scale=1e-6))
        spec.add("induced_QR", stats.halfnorm(scale=2e-4))
        return spec

    @classmethod
    def default_mechanism(cls) -> PriorSpec:
        """Priors covering both cytotoxic and cytostatic drug action.

        Matches :data:`umimic.inference.likelihood.MECHANISM_PARAM_NAMES`. The
        birth-modulation Emax is bounded on [0, 1] because it is a fractional
        reduction of the baseline division rate, whereas the death Emax is an
        additive rate increase.
        """
        spec = cls.default_invitro()
        spec.add("emax_birth", stats.uniform(loc=0.0, scale=1.0))
        spec.add("ec50_birth", stats.lognorm(s=1.0, scale=1.0))
        spec.add("hill_birth", stats.lognorm(s=0.3, scale=1.5))
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
