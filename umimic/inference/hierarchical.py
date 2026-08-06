"""Hierarchical Bayesian model for pooling across replicates/animals.

Borrows strength across:
- Replicate wells (in vitro)
- Animals (in vivo)
- Cell lines or tumor models
- Batches/experiments

This is essential for sparse in vivo data where individual trajectories
have limited information.

Sample layout matches :mod:`umimic.inference.mcmc`: draws are returned as
``(n_walkers, n_draws)``, with log-likelihood and log-posterior stored
separately, so the same diagnostics apply.
"""

from __future__ import annotations

import logging

import numpy as np

from umimic.data.schemas import ExperimentalDataset
from umimic.dynamics.states import ModelTopology
from umimic.inference.likelihood import ModelLikelihood
from umimic.inference.priors import PriorSpec
from umimic.observations.base import ObservationModel
from umimic.observations.multimodal import MultimodalObservation
from umimic.types import MCMCResult

logger = logging.getLogger(__name__)

def _log_half_cauchy(scale: float) -> float:
    """Log density of a half-Cauchy(0, 1), up to an additive constant.

    The dropped constant log(2/pi) does not affect MCMC, which only ever uses
    density ratios.
    """
    return -np.log1p(scale**2)


class HierarchicalModel:
    """Hierarchical Bayesian model for multi-group inference.

    Structure:
        Population level: theta_pop ~ prior
        Group level: theta_i ~ LogNormal(log(theta_pop), tau)  for each group i
        Observation level: y_ij ~ p(y | theta_i)  for observation j in group i

    Uses emcee with a custom hierarchical log-posterior.
    """

    def __init__(
        self,
        dataset: ExperimentalDataset,
        topology: ModelTopology,
        shared_params: list[str],
        random_effect_params: list[str],
        priors: PriorSpec | None = None,
        observation_model: ObservationModel | MultimodalObservation | None = None,
        mode: str = "moment",
        rng: np.random.Generator | int | None = None,
    ):
        """
        Args:
            dataset: Collection of time-series data (one per group/replicate).
            topology: Model topology.
            shared_params: Parameters shared across all groups (e.g. EC50, Hill).
            random_effect_params: Parameters that vary by group (e.g. b0, d0).
            priors: Priors for population-level parameters. Must cover every
                shared and random-effect parameter.
            observation_model: Observation model handed to each group's
                likelihood. Without it only cell counts contribute, so a
                multimodal dataset would be fitted on one modality.
            mode: Forward model, "moment" (LNA, supplies process variance) or
                "ode". Previously hardcoded to "ode", which discarded the
                variance signature.
            rng: Generator or seed. Supplying one makes the fit reproducible.
        """
        if not shared_params and not random_effect_params:
            raise ValueError("At least one parameter must be shared or group-varying.")
        overlap = set(shared_params) & set(random_effect_params)
        if overlap:
            raise ValueError(
                f"Parameters {sorted(overlap)} are both shared and group-varying."
            )
        if dataset.n_series == 0:
            raise ValueError("Dataset contains no series.")

        self.dataset = dataset
        self.topology = topology
        self.shared_params = list(shared_params)
        self.random_effect_params = list(random_effect_params)
        self.priors = priors or PriorSpec.default_invitro()
        self.mode = mode
        self.rng = np.random.default_rng(rng)
        self.n_groups = dataset.n_series

        # Every parameter needs a prior: falling back to a hardcoded value
        # would silently invent an initialisation the user never chose.
        missing = [
            name
            for name in self.shared_params + self.random_effect_params
            if name not in self.priors.distributions
        ]
        if missing:
            raise ValueError(
                f"No prior supplied for parameter(s) {missing}. Every shared "
                "and random-effect parameter needs one."
            )

        # Group labels must be unique or later groups overwrite earlier ones
        # in the returned samples.
        self.group_labels = self._unique_group_labels()

        all_params = self.shared_params + self.random_effect_params
        self.group_likelihoods = [
            ModelLikelihood(
                topology=topology,
                data=series,
                param_names=all_params,
                mode=mode,
                observation_model=observation_model,
            )
            for series in dataset.series
        ]

    def _unique_group_labels(self) -> list[str]:
        """Stable, unique label per series for naming output samples."""
        labels: list[str] = []
        seen: dict[str, int] = {}
        for i, series in enumerate(self.dataset.series):
            base = series.replicate_id or series.group_id or f"group_{i}"
            if base in seen:
                seen[base] += 1
                base = f"{base}#{seen[base]}"
            else:
                seen[base] = 0
            labels.append(base)
        return labels

    @property
    def n_dim(self) -> int:
        """Length of the full parameter vector."""
        return (
            len(self.shared_params)
            + 2 * len(self.random_effect_params)
            + self.n_groups * len(self.random_effect_params)
        )

    def _split(self, theta_full: np.ndarray):
        """Split the flat vector into its hierarchical components."""
        n_shared = len(self.shared_params)
        n_re = len(self.random_effect_params)
        shared = theta_full[:n_shared]
        pop_mean = theta_full[n_shared : n_shared + n_re]
        pop_sd = theta_full[n_shared + n_re : n_shared + 2 * n_re]
        group_re = theta_full[n_shared + 2 * n_re :].reshape(self.n_groups, n_re)
        return shared, pop_mean, pop_sd, group_re

    def log_prior(self, theta_full: np.ndarray) -> float:
        """Population priors, half-Cauchy scales, and random-effect density."""
        shared, pop_mean, pop_sd, group_re = self._split(theta_full)

        # Shared parameters are only required to be non-negative: several of
        # them (emax_death, transition rates, induction terms) are legitimately
        # zero, meaning "no effect", and the priors on them are half-normal.
        # The random-effect components must be strictly positive because the
        # LogNormal below takes their log -- that is the model's definition,
        # not a bound imposed on top of it.
        if (
            np.any(shared < 0)
            or np.any(pop_mean <= 0)
            or np.any(pop_sd <= 0)
            or np.any(group_re <= 0)
        ):
            return -np.inf

        shared_dict = dict(zip(self.shared_params, shared))
        pop_dict = dict(zip(self.random_effect_params, pop_mean))
        log_prior = self.priors.log_prior({**shared_dict, **pop_dict})
        if not np.isfinite(log_prior):
            return -np.inf

        # Weakly informative half-Cauchy on each between-group scale.
        log_prior += sum(_log_half_cauchy(sd) for sd in pop_sd)

        # theta_i ~ LogNormal(log(pop_mean), pop_sd), up to a constant.
        log_mu = np.log(pop_mean)
        log_re = np.log(group_re)
        z = (log_re - log_mu) / pop_sd
        log_prior += float(
            np.sum(-0.5 * z**2 - np.log(pop_sd) - log_re)
        )
        return float(log_prior)

    def log_likelihood(self, theta_full: np.ndarray) -> float:
        """Summed observation log-likelihood across groups."""
        shared, _, _, group_re = self._split(theta_full)
        total = 0.0
        for i in range(self.n_groups):
            theta_i = np.concatenate([shared, group_re[i]])
            value = self.group_likelihoods[i](theta_i)
            if not np.isfinite(value):
                return -np.inf
            total += value
        return float(total)

    def _log_posterior_blobs(self, theta_full: np.ndarray) -> tuple[float, float]:
        """Log-posterior with the log-likelihood carried as an emcee blob."""
        log_prior = self.log_prior(theta_full)
        if not np.isfinite(log_prior):
            return -np.inf, -np.inf
        log_lik = self.log_likelihood(theta_full)
        if not np.isfinite(log_lik):
            return -np.inf, -np.inf
        return log_prior + log_lik, log_lik

    def _log_posterior(self, theta_full: np.ndarray) -> float:
        return self._log_posterior_blobs(theta_full)[0]

    def _initial_walkers(self, n_walkers: int) -> np.ndarray:
        """Draw walker starts from the priors, using the seeded generator."""
        n_re = len(self.random_effect_params)
        p0 = np.zeros((n_walkers, self.n_dim))

        for w in range(n_walkers):
            draw = self.priors.sample(self.rng)
            jitter = lambda v: v * (1.0 + 0.1 * self.rng.standard_normal())  # noqa: E731
            values = [jitter(draw[name]) for name in self.shared_params]
            values += [jitter(draw[name]) for name in self.random_effect_params]
            # Between-group scales start small and positive.
            values += [
                0.1 + 0.05 * abs(self.rng.standard_normal()) for _ in range(n_re)
            ]
            for _ in range(self.n_groups):
                values += [jitter(draw[name]) for name in self.random_effect_params]
            p0[w] = values

        return np.abs(p0) + 1e-6

    def fit(
        self,
        n_samples: int = 2000,
        n_warmup: int = 1000,
        n_walkers: int | None = None,
    ) -> MCMCResult:
        """Fit the hierarchical model using emcee.

        Returns:
            MCMCResult with population- and group-level posterior samples,
            each shaped ``(n_walkers, n_draws)``.
        """
        try:
            import emcee
        except ImportError:
            raise ImportError("emcee required for hierarchical inference")

        ndim = self.n_dim
        if n_walkers is None:
            n_walkers = max(2 * ndim + 2, 32)
        if n_walkers < 2 * ndim:
            raise ValueError(
                f"emcee needs at least {2 * ndim} walkers for {ndim} dimensions, "
                f"got {n_walkers}."
            )

        p0 = self._initial_walkers(n_walkers)
        sampler = emcee.EnsembleSampler(
            n_walkers,
            ndim,
            self._log_posterior_blobs,
            blobs_dtype=[("log_likelihood", float)],
        )
        sampler.random_state = np.random.RandomState(
            int(self.rng.integers(0, 2**32 - 1))
        ).get_state()

        state = sampler.run_mcmc(p0, n_warmup, progress=False)
        sampler.reset()
        sampler.run_mcmc(state, n_samples, progress=False)

        # Keep the walker axis: flattening it destroys the information R-hat
        # and ESS need, and emcee walkers are not independent chains.
        chain = sampler.get_chain(flat=False)  # (n_draws, n_walkers, ndim)

        samples: dict[str, np.ndarray] = {}
        for idx, name in enumerate(self._parameter_labels()):
            samples[name] = np.ascontiguousarray(chain[:, :, idx].T)

        blobs = sampler.get_blobs(flat=False)
        log_lik = (
            np.ascontiguousarray(blobs["log_likelihood"].T)
            if blobs is not None
            else None
        )

        return MCMCResult(
            samples=samples,
            log_likelihood_trace=log_lik,
            log_posterior_trace=np.ascontiguousarray(
                sampler.get_log_prob(flat=False).T
            ),
            n_chains=n_walkers,
            n_samples=n_samples,
            diagnostics={
                "backend": "emcee_hierarchical",
                "n_groups": self.n_groups,
                "acceptance": float(np.mean(sampler.acceptance_fraction)),
                "walkers_are_independent_chains": False,
                "group_labels": list(self.group_labels),
            },
        )

    def _parameter_labels(self) -> list[str]:
        """Names for each entry of the full parameter vector, in order."""
        labels = list(self.shared_params)
        labels += [f"pop_mean_{n}" for n in self.random_effect_params]
        labels += [f"pop_sd_{n}" for n in self.random_effect_params]
        for label in self.group_labels:
            labels += [f"{n}_{label}" for n in self.random_effect_params]
        return labels
