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
        condition_on_first: bool = True,
        anchor_modality: str = "cell_counts",
        initial_fractions=None,
        max_abs_z: float = 8.0,
        max_log_deviation: float = 5.0,
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
            condition_on_first: Forwarded to each group's ModelLikelihood.
            anchor_modality: Forwarded to each group's ModelLikelihood.
            initial_fractions: Forwarded to each group's ModelLikelihood.
            max_abs_z: Truncation on the standardized group effects. Keeps the
                sampler out of regions where theta_i overflows into a stiff,
                very slow, and uninformative likelihood evaluation.
            max_log_deviation: Truncation on |pop_sd * z|, i.e. on
                |log(theta_i / pop_mean)|. This is the quantity that actually
                controls how extreme a group parameter can get; bounding
                pop_sd alone does not, because a moderate scale times a
                moderate z still reaches e^80. The default allows a group to
                sit within e^5 (~148x) of the population median, which is
                already generous for biological replicates.

        The last three used to be left at their defaults, so a hierarchical fit
        could not be configured to match a ModelLikelihood or ParticleMCMC fit
        of the same data -- which made the two silently non-comparable.
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
        if not max_abs_z > 0:
            raise ValueError(f"max_abs_z must be positive, got {max_abs_z}.")
        if not max_log_deviation > 0:
            raise ValueError(
                f"max_log_deviation must be positive, got {max_log_deviation}."
            )
        self.max_abs_z = float(max_abs_z)
        self.max_log_deviation = float(max_log_deviation)
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
                condition_on_first=condition_on_first,
                anchor_modality=anchor_modality,
                initial_fractions=initial_fractions,
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
        """Split the flat sampling vector into its hierarchical components.

        The sampled coordinates are ``[shared, pop_mean, log_pop_sd, z]``:
        the between-group scale is sampled on the log scale and the group
        effects as standardized normals. See :meth:`group_values` for the
        transform back to natural units.
        """
        n_shared = len(self.shared_params)
        n_re = len(self.random_effect_params)
        shared = theta_full[:n_shared]
        pop_mean = theta_full[n_shared : n_shared + n_re]
        log_pop_sd = theta_full[n_shared + n_re : n_shared + 2 * n_re]
        z = theta_full[n_shared + 2 * n_re :].reshape(self.n_groups, n_re)
        return shared, pop_mean, log_pop_sd, z

    @staticmethod
    def group_values(
        pop_mean: np.ndarray, pop_sd: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """Non-centered map from standardized effects to group parameters.

        ``theta_i = pop_mean * exp(pop_sd * z_i)``, which is exactly
        ``theta_i ~ LogNormal(log(pop_mean), pop_sd)`` with the dependence on
        ``pop_sd`` moved out of the sampled coordinates.

        The centered form -- sampling ``theta_i`` directly alongside
        ``pop_sd`` -- is Neal's funnel: as ``pop_sd -> 0`` the feasible region
        for the group effects collapses, and emcee's stretch move (affine
        invariant, but not funnel-aware) cannot traverse the neck. It does not
        fail visibly; it returns a converged-looking chain whose ``pop_sd``
        posterior is biased toward whatever the sampler could reach. Sparse in
        vivo data -- this class's stated use case -- is exactly where
        ``pop_sd`` is weakly informed and the funnel dominates.
        """
        return pop_mean * np.exp(pop_sd * z)

    def log_prior(self, theta_full: np.ndarray) -> float:
        """Population priors, half-Cauchy scales, and standardized effects."""
        shared, pop_mean, log_pop_sd, z = self._split(theta_full)

        # Shared parameters are only required to be non-negative: several of
        # them (emax_death, transition rates, induction terms) are legitimately
        # zero, meaning "no effect", and the priors on them are half-normal.
        # pop_mean must be strictly positive because it is a lognormal median.
        # pop_sd needs no positivity check any more: it is exp(log_pop_sd), so
        # it is positive by construction, and the boundary at 0 is reachable in
        # the limit rather than being a hard wall the sampler bounces off.
        if np.any(shared < 0) or np.any(pop_mean <= 0):
            return -np.inf
        if not np.all(np.isfinite(log_pop_sd)) or not np.all(np.isfinite(z)):
            return -np.inf

        # Truncate the non-centered coordinates. Both bounds cut regions with
        # negligible prior mass, and both exist for a concrete reason:
        #
        # z: theta_i = pop_mean * exp(pop_sd * z), so a walker at large |z|
        #   produces astronomically large rates. Those are finite, so they pass
        #   every downstream check, and the ODE solver then grinds through a
        #   stiff system for seconds per evaluation before the likelihood
        #   finally comes back useless. |z| > 8 is 8 SD from a standard normal.
        #
        # pop_sd * z: the half-Cauchy is heavy-tailed and the log
        #   parameterization contributes +log(pop_sd) from the Jacobian, which
        #   nearly cancels the tail penalty -- so nothing stops a walker
        #   drifting to a huge scale. Bounding pop_sd alone is not enough:
        #   theta_i depends on the *product*, and a scale of e^10 with z = 8
        #   still gives exp(80) ~ 5e34. That is finite, so it passes every
        #   downstream check, and the ODE solver then spends seconds on a stiff
        #   system per evaluation. Bounding |log(theta_i / pop_mean)| directly
        #   is what keeps the likelihood cheap.
        if np.any(np.abs(z) > self.max_abs_z):
            return -np.inf

        pop_sd = np.exp(log_pop_sd)
        if np.any(np.abs(pop_sd * z) > self.max_log_deviation):
            return -np.inf

        shared_dict = dict(zip(self.shared_params, shared))
        pop_dict = dict(zip(self.random_effect_params, pop_mean))
        log_prior = self.priors.log_prior({**shared_dict, **pop_dict})
        if not np.isfinite(log_prior):
            return -np.inf

        # Weakly informative half-Cauchy on each between-group scale, plus the
        # Jacobian d(pop_sd)/d(log_pop_sd) = pop_sd for sampling it on the log
        # scale.
        log_prior += sum(_log_half_cauchy(sd) for sd in pop_sd)
        log_prior += float(np.sum(log_pop_sd))

        # z_i ~ Normal(0, 1). The lognormal density of theta_i and its Jacobian
        # are absorbed by the deterministic transform in `group_values`, so
        # this standard normal is the whole random-effect contribution.
        log_prior += float(np.sum(-0.5 * z**2))
        return float(log_prior)

    def log_likelihood(self, theta_full: np.ndarray) -> float:
        """Summed observation log-likelihood across groups."""
        shared, pop_mean, log_pop_sd, z = self._split(theta_full)
        group_re = self.group_values(pop_mean, np.exp(log_pop_sd), z)
        if np.any(group_re <= 0) or not np.all(np.isfinite(group_re)):
            return -np.inf
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
        """Draw walker starts from the priors, using the seeded generator.

        Only the shared and population-mean blocks are drawn from the priors
        and forced positive. `log_pop_sd` and `z` live on the whole real line,
        so taking their absolute value -- as the old centered version did for
        the entire vector -- would fold half of each start distribution back on
        itself and bias the between-group scale upward.
        """
        n_shared = len(self.shared_params)
        n_re = len(self.random_effect_params)
        p0 = np.zeros((n_walkers, self.n_dim))

        for w in range(n_walkers):
            draw = self.priors.sample(self.rng)
            jitter = lambda v: v * (1.0 + 0.1 * self.rng.standard_normal())  # noqa: E731
            positive = [jitter(draw[name]) for name in self.shared_params]
            positive += [jitter(draw[name]) for name in self.random_effect_params]
            p0[w, : n_shared + n_re] = np.abs(positive) + 1e-6

            # Between-group scales start around 0.1, on the log scale.
            p0[w, n_shared + n_re : n_shared + 2 * n_re] = np.log(0.1) + (
                0.2 * self.rng.standard_normal(n_re)
            )
            # Standardized group effects start near the population mean.
            p0[w, n_shared + 2 * n_re :] = 0.1 * self.rng.standard_normal(
                self.n_groups * n_re
            )

        return p0

    def to_natural(self, theta_full: np.ndarray) -> np.ndarray:
        """Map one sampling vector to the natural scale users see.

        `pop_sd` is exponentiated and the standardized effects become group
        parameters, so reported samples keep the same names and units as
        before this class was reparameterized.
        """
        shared, pop_mean, log_pop_sd, z = self._split(theta_full)
        pop_sd = np.exp(log_pop_sd)
        group = self.group_values(pop_mean, pop_sd, z)
        return np.concatenate([shared, pop_mean, pop_sd, group.reshape(-1)])

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

        # Sampling happens in non-centered coordinates; report natural ones, so
        # `pop_sd_*` is a scale and `<param>_<group>` is a group parameter,
        # exactly as the labels claim.
        natural = np.empty_like(chain)
        for i in range(chain.shape[0]):
            for j in range(chain.shape[1]):
                natural[i, j] = self.to_natural(chain[i, j])

        samples: dict[str, np.ndarray] = {}
        for idx, name in enumerate(self._parameter_labels()):
            samples[name] = np.ascontiguousarray(natural[:, :, idx].T)

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
