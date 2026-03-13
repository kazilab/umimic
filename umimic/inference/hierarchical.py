"""Hierarchical Bayesian model for pooling across replicates/animals.

Borrows strength across:
- Replicate wells (in vitro)
- Animals (in vivo)
- Cell lines or tumor models
- Batches/experiments

This is essential for sparse in vivo data where individual trajectories
have limited information.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from umimic.inference.likelihood import ModelLikelihood
from umimic.inference.priors import PriorSpec
from umimic.data.schemas import TimeSeriesData, ExperimentalDataset
from umimic.dynamics.states import ModelTopology
from umimic.types import MCMCResult


class HierarchicalModel:
    """Hierarchical Bayesian model for multi-group inference.

    Structure:
        Population level: theta_pop ~ prior
        Group level: theta_i ~ N(theta_pop, tau^2)  for each group i
        Observation level: y_ij ~ p(y | theta_i)    for each observation j in group i

    Uses emcee with a custom hierarchical log-posterior.
    """

    def __init__(
        self,
        dataset: ExperimentalDataset,
        topology: ModelTopology,
        shared_params: list[str],
        random_effect_params: list[str],
        priors: PriorSpec | None = None,
    ):
        """
        Args:
            dataset: Collection of time-series data (one per group/replicate).
            topology: Model topology.
            shared_params: Parameters shared across all groups (e.g., EC50, Hill).
            random_effect_params: Parameters that vary by group (e.g., b0, d0).
            priors: Prior specification for population-level parameters.
        """
        self.dataset = dataset
        self.topology = topology
        self.shared_params = shared_params
        self.random_effect_params = random_effect_params
        self.priors = priors or PriorSpec.default_invitro()
        self.n_groups = dataset.n_series

        # Build individual likelihoods
        self.group_likelihoods = []
        all_params = shared_params + random_effect_params
        for series in dataset.series:
            ll = ModelLikelihood(
                topology=topology,
                data=series,
                param_names=all_params,
                mode="ode",
            )
            self.group_likelihoods.append(ll)

    def _log_posterior(self, theta_full: np.ndarray) -> float:
        """Compute hierarchical log-posterior.

        Parameter vector layout:
        [shared_params, pop_mean_RE, pop_sd_RE, group1_RE, group2_RE, ...]
        """
        n_shared = len(self.shared_params)
        n_re = len(self.random_effect_params)
        n_groups = self.n_groups

        # Extract components
        shared = theta_full[:n_shared]
        pop_mean = theta_full[n_shared : n_shared + n_re]
        pop_sd = theta_full[n_shared + n_re : n_shared + 2 * n_re]
        group_re = theta_full[n_shared + 2 * n_re :].reshape(n_groups, n_re)

        # Check positivity
        if np.any(shared <= 0) or np.any(pop_mean <= 0) or np.any(pop_sd <= 0):
            return -np.inf
        if np.any(group_re <= 0):
            return -np.inf

        # Population-level prior
        log_post = 0.0
        shared_dict = {name: shared[i] for i, name in enumerate(self.shared_params)}
        pop_dict = {name: pop_mean[i] for i, name in enumerate(self.random_effect_params)}
        all_pop = {**shared_dict, **pop_dict}
        log_post += self.priors.log_prior(all_pop)

        # Half-Cauchy prior on population SDs
        for sd in pop_sd:
            # Half-Cauchy(0, 1) on log-scale
            log_post += -np.log(1 + sd**2)

        # Group-level: log-normal random effects
        for i in range(n_groups):
            for j in range(n_re):
                # theta_i ~ LogNormal(log(pop_mean), pop_sd)
                log_re = np.log(group_re[i, j])
                log_mu = np.log(pop_mean[j])
                log_post += -0.5 * ((log_re - log_mu) / pop_sd[j]) ** 2
                log_post += -np.log(pop_sd[j]) - np.log(group_re[i, j])

        # Observation-level likelihood per group
        for i in range(n_groups):
            theta_i = np.concatenate([shared, group_re[i]])
            ll_i = self.group_likelihoods[i](theta_i)
            if not np.isfinite(ll_i):
                return -np.inf
            log_post += ll_i

        return log_post

    def fit(
        self,
        n_samples: int = 2000,
        n_warmup: int = 1000,
        n_walkers: int | None = None,
    ) -> MCMCResult:
        """Fit the hierarchical model using emcee.

        Returns:
            MCMCResult with population-level and group-level posterior samples.
        """
        try:
            import emcee
        except ImportError:
            raise ImportError("emcee required for hierarchical inference")

        n_shared = len(self.shared_params)
        n_re = len(self.random_effect_params)
        n_groups = self.n_groups
        ndim = n_shared + 2 * n_re + n_groups * n_re

        if n_walkers is None:
            n_walkers = max(2 * ndim + 2, 32)

        # Initialize walkers
        p0 = np.zeros((n_walkers, ndim))
        for w in range(n_walkers):
            sample = self.priors.sample()
            idx = 0
            for name in self.shared_params:
                p0[w, idx] = sample.get(name, 0.01) * (1 + 0.1 * np.random.randn())
                idx += 1
            for name in self.random_effect_params:
                val = sample.get(name, 0.01)
                p0[w, idx] = val * (1 + 0.1 * np.random.randn())  # pop mean
                idx += 1
            for name in self.random_effect_params:
                p0[w, idx] = 0.1 + 0.05 * abs(np.random.randn())  # pop sd
                idx += 1
            for g in range(n_groups):
                for name in self.random_effect_params:
                    val = sample.get(name, 0.01)
                    p0[w, idx] = val * (1 + 0.1 * np.random.randn())
                    idx += 1

        p0 = np.abs(p0) + 1e-6

        sampler = emcee.EnsembleSampler(n_walkers, ndim, self._log_posterior)

        # Burn-in
        state = sampler.run_mcmc(p0, n_warmup, progress=False)
        sampler.reset()

        # Production
        sampler.run_mcmc(state, n_samples, progress=False)
        flat = sampler.get_chain(flat=True)

        # Extract named samples
        samples = {}
        idx = 0
        for name in self.shared_params:
            samples[name] = flat[:, idx]
            idx += 1
        for name in self.random_effect_params:
            samples[f"pop_mean_{name}"] = flat[:, idx]
            idx += 1
        for name in self.random_effect_params:
            samples[f"pop_sd_{name}"] = flat[:, idx]
            idx += 1
        for g in range(n_groups):
            for name in self.random_effect_params:
                gid = self.dataset.series[g].group_id or f"group_{g}"
                samples[f"{name}_{gid}"] = flat[:, idx]
                idx += 1

        return MCMCResult(
            samples=samples,
            n_chains=n_walkers,
            n_samples=n_samples,
            diagnostics={
                "backend": "emcee_hierarchical",
                "n_groups": n_groups,
                "acceptance": float(np.mean(sampler.acceptance_fraction)),
            },
        )
