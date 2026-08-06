"""Sequential Monte Carlo / Particle Filter for robust inference.

The 'robust mode' inference backend for U-MIMIC. Used when populations
are small, dynamics are highly nonlinear, or the normal approximation
in moment-based inference breaks down.

Correctness notes
-----------------
* Particles are propagated over *absolute* experiment time, so a time-varying
  PK exposure is evaluated at the real clock time rather than at an offset.
* Weights are updated recursively (previous weight times the new likelihood)
  and normalized with logsumexp. This is required because resampling is
  adaptive: on steps where no resampling occurs the previous weights still
  carry information.
* Complete particle degeneracy returns ``-inf`` rather than silently resetting
  to uniform weights, and simulator event-limit truncation invalidates the
  estimate instead of passing unnoticed.
* The PMCMC proposal is a lognormal random walk, whose asymmetry is corrected
  by the Hastings term ``sum(log theta' - log theta)``.
* Proposals and particle-filter likelihood estimates draw from independent RNG
  streams, as pseudo-marginal correctness requires.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.special import logsumexp

from umimic.data.schemas import TimeSeriesData
from umimic.dynamics.gillespie import GillespieSimulator
from umimic.dynamics.rates import RateSet
from umimic.dynamics.states import CellType, ModelTopology
from umimic.inference.likelihood import (
    KNOWN_PARAM_NAMES,
    OBSERVATION_PARAM_NAMES,
    PARAMETER_SETS,
    build_rate_set,
    missing_state_params,
    resolve_initial_fractions,
)
from umimic.observations.base import ObservationModel
from umimic.observations.multimodal import MultimodalObservation
from umimic.types import MCMCResult

if TYPE_CHECKING:
    from umimic.inference.priors import PriorSpec

logger = logging.getLogger(__name__)


class ParticleDegeneracyError(RuntimeError):
    """Raised when every particle has zero likelihood."""


class ParticleFilter:
    """Bootstrap particle filter for state estimation.

    Each particle represents a possible latent state trajectory.
    Particles are propagated forward using the stochastic simulator
    and re-weighted based on observation likelihood.
    """

    def __init__(
        self,
        rate_set: RateSet,
        topology: ModelTopology,
        exposure_fn: Callable[[float], float],
        observation_model: ObservationModel | MultimodalObservation,
        n_particles: int = 500,
        rng: np.random.Generator | None = None,
        max_events: int = 100_000,
        ess_fraction: float = 0.5,
        obs_params: dict[str, float] | None = None,
    ):
        self.rate_set = rate_set
        self.topology = topology
        self.exposure_fn = exposure_fn
        self.obs_model = observation_model
        self.n_particles = n_particles
        self.rng = rng or np.random.default_rng(42)
        self.max_events = int(max_events)
        self.ess_fraction = float(ess_fraction)
        # Observation-model parameters (overdispersion, sigma_log_bli, ...).
        # Without these the observation model silently falls back to its
        # construction-time values, so any such parameter being *sampled*
        # would never touch the likelihood and would return its prior.
        self.obs_params = dict(obs_params) if obs_params else None
        # Modality a single (non-composite) observation model measures.
        self._single_modality = (
            getattr(observation_model, "modality_name", None) or "cell_counts"
        )

    def _observation_log_likelihood(
        self,
        data: TimeSeriesData,
        k: int,
        particle: np.ndarray,
        skip_modality: str | None = None,
    ) -> float:
        """Log-likelihood of every available modality at time index k.

        ``skip_modality`` drops a single modality at this index, used for the
        anchor observation that already set the initial condition.
        """
        if isinstance(self.obs_model, MultimodalObservation):
            observations = {}
            for modality in data.modalities:
                if modality == skip_modality:
                    continue
                if modality in self.obs_model.models:
                    value = data.observations[modality][k]
                    observations[modality] = None if np.isnan(value) else value
            if not observations:
                return 0.0
            return self.obs_model.log_likelihood(
                observations, particle, self.obs_params
            )

        # A bare model reads its own modality. Hardcoding "cell_counts" here
        # meant a lone BLIObservation was handed cell counts and scored them
        # on a lognormal signal scale -- or, with no counts present, scored
        # nothing and returned a flat likelihood.
        modality = self._single_modality
        if skip_modality == modality:
            return 0.0
        value = data.observations.get(modality)
        if value is None or np.isnan(value[k]):
            return 0.0
        return self.obs_model.log_likelihood(
            float(value[k]), particle, self.obs_params
        )

    def filter(
        self,
        data: TimeSeriesData,
        initial_state: np.ndarray,
        anchor_modality: str | None = None,
        anchor_index: int | None = None,
    ) -> dict:
        """Run the particle filter through observed data.

        Args:
            data: Observed time-series data.
            initial_state: Initial cell state vector.
            anchor_modality: Modality whose observation produced
                `initial_state`, if any.
            anchor_index: Time index of that observation.

        Passing the anchor excludes that one measurement from the weights.
        Scoring it would use the same number twice -- once to place the
        particle cloud and once to judge it -- which tightens the initial
        state beyond what the data supports and puts the marginal likelihood
        on a different scale from :class:`~umimic.inference.likelihood.
        ModelLikelihood`, whose default also excludes the anchor. Leave both
        as None when `initial_state` comes from somewhere other than the data.

        Returns:
            Dict with filtered state estimates, weights, and log-likelihood.
            ``marginal_log_likelihood`` is -inf on total degeneracy.
        """
        n = self.n_particles
        n_states = self.topology.n_states
        times = np.asarray(data.times, dtype=float)

        # Initialize particles at the initial state with Poisson dispersion.
        particles = np.tile(np.asarray(initial_state, float), (n, 1))
        positive = particles > 0
        particles[positive] = self.rng.poisson(particles[positive])

        # Work in log-space throughout; start from uniform weights.
        log_weights = np.full(n, -np.log(n))
        marginal_ll = 0.0
        degenerate = False
        truncated = False

        filtered_means = np.zeros((len(times), n_states))
        filtered_vars = np.zeros((len(times), n_states))

        # One simulator, reused: rebuilding reactions per particle per step is
        # pure overhead.
        simulator = GillespieSimulator(
            self.rate_set,
            self.topology,
            self.exposure_fn,
            self.rng,
        )

        for k in range(len(times)):
            # Propagation step over the ABSOLUTE interval [t_{k-1}, t_k], so
            # exposure is evaluated at true experiment time.
            if k > 0:
                t_prev = float(times[k - 1])
                t_now = float(times[k])
                dt = t_now - t_prev

                # Shift the clock: the simulator integrates on [0, dt] but the
                # exposure must be sampled at t_prev + s.
                def shifted(s, _t0=t_prev):
                    return self.exposure_fn(_t0 + s)

                simulator.exposure_fn = shifted

                new_particles = np.zeros_like(particles)
                for i in range(n):
                    result = simulator.simulate(
                        particles[i],
                        dt,
                        t_record=np.array([dt]),
                        max_events=self.max_events,
                    )
                    if result.metadata.get("truncated"):
                        truncated = True
                    for j, ct in enumerate(self.topology.active_states):
                        new_particles[i, j] = result.populations[ct.name][-1]
                particles = new_particles

            # Weight update: previous weight multiplied by the new likelihood.
            # At the anchor index the anchor modality contributes nothing, so
            # the initial cloud keeps the spread the initial-state prior gave
            # it instead of being re-tightened by its own seed value.
            skip = anchor_modality if k == anchor_index else None
            incremental = np.array(
                [
                    self._observation_log_likelihood(data, k, particles[i], skip)
                    for i in range(n)
                ]
            )
            log_weights = log_weights + incremental

            log_norm = logsumexp(log_weights)
            if not np.isfinite(log_norm):
                # Every particle has zero likelihood: the estimate is not
                # merely imprecise, it is undefined.
                degenerate = True
                marginal_ll = -np.inf
                logger.debug(
                    "Particle filter degenerated at time index %d (t=%.4g).",
                    k,
                    times[k],
                )
                filtered_means[k:] = np.nan
                filtered_vars[k:] = np.nan
                break

            # Incremental marginal likelihood contribution.
            marginal_ll += float(log_norm)
            log_weights = log_weights - log_norm

            weights = np.exp(log_weights)
            filtered_means[k] = np.average(particles, weights=weights, axis=0)
            filtered_vars[k] = np.average(
                (particles - filtered_means[k]) ** 2, weights=weights, axis=0
            )

            # Systematic resampling if ESS is low.
            ess = 1.0 / float(np.sum(weights**2))
            if ess < self.ess_fraction * n:
                indices = self._systematic_resample(weights)
                particles = particles[indices]
                log_weights = np.full(n, -np.log(n))

        if truncated:
            logger.warning(
                "Particle propagation hit the simulator event limit (%d); the "
                "marginal likelihood estimate is biased and is reported as "
                "invalid.",
                self.max_events,
            )
            marginal_ll = -np.inf

        return {
            "filtered_means": filtered_means,
            "filtered_vars": filtered_vars,
            "marginal_log_likelihood": float(marginal_ll),
            "final_particles": particles,
            "final_weights": np.exp(log_weights),
            "degenerate": degenerate,
            "truncated": truncated,
        }

    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        """Systematic resampling for particle filter."""
        n = len(weights)
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0
        u = (self.rng.uniform() + np.arange(n)) / n
        indices = np.searchsorted(cumsum, u)
        return np.clip(indices, 0, n - 1)


class ParticleMCMC:
    """Particle Markov Chain Monte Carlo (PMCMC).

    Combines the particle filter (for an unbiased marginal likelihood
    estimate) with Metropolis-Hastings (for parameter inference).

    This is the full 'robust mode' for parameter estimation when
    the moment-based approach fails.
    """

    def __init__(
        self,
        topology: ModelTopology,
        data: TimeSeriesData,
        observation_model: ObservationModel | MultimodalObservation,
        priors: PriorSpec,
        n_particles: int = 200,
        rng: np.random.Generator | None = None,
        param_names: list[str] | None = None,
        anchor_modality: str | None = None,
        condition_on_first: bool = True,
        initial_fractions=None,
    ):
        """
        Args:
            anchor_modality: Modality whose first observation sets the initial
                condition, matching ModelLikelihood's contract. Defaults to
                the modality a single observation model measures, or to
                "cell_counts" for a composite.
            condition_on_first: Exclude that anchor observation from the
                likelihood, as ModelLikelihood does by default. Set False to
                treat the initial state as fixed and score every observation.
            initial_fractions: How the initial count is split across states,
                with the same meaning as in ModelLikelihood. The default puts
                every cell in P, which is a structural assumption rather than
                a neutral one -- it biases the estimated transition rates and
                the early trajectory.
        """
        self.topology = topology
        self.data = data
        self.obs_model = observation_model
        self.priors = priors
        self.n_particles = n_particles
        self.param_names = param_names
        if anchor_modality is None:
            anchor_modality = (
                "cell_counts"
                if isinstance(observation_model, MultimodalObservation)
                else getattr(observation_model, "modality_name", None)
                or "cell_counts"
            )
        self.anchor_modality = anchor_modality
        self.condition_on_first = condition_on_first
        self.initial_fractions = initial_fractions

        # Resolved once so a topology without P fails at construction rather
        # than inside the likelihood, where it would be caught and turned into
        # a -inf that looks like an ordinary rejection.
        self._seed_index = (
            topology.state_index(CellType.P)
            if topology.has_state(CellType.P)
            else 0
        )
        self._default_fractions = np.zeros(topology.n_states)
        self._default_fractions[self._seed_index] = 1.0
        self._has_fraction_spec = initial_fractions is not None or (
            "initial_fractions" in (data.metadata or {})
        )

        if param_names is not None:
            self._validate_param_names(param_names)

        # Pseudo-marginal MCMC requires the likelihood estimator's randomness
        # to be independent of the proposal randomness; sharing one stream
        # correlates acceptance decisions with the noise in the estimate.
        base = rng or np.random.default_rng(42)
        self._proposal_rng, self._filter_rng = base.spawn(2)

    def _validate_param_names(self, names: list[str]) -> None:
        """Reject parameters the forward model would never read.

        A name outside the known set is sampled, priced by its prior, and
        then dropped -- its posterior is its prior, and the parameters that
        *are* wired absorb the misfit. That failure is invisible in a trace
        plot, so it has to be an error rather than a warning.
        """
        unknown = [n for n in names if n not in KNOWN_PARAM_NAMES]
        if unknown:
            raise ValueError(
                f"Parameter(s) {unknown} are not read by the forward model or "
                "any observation model, so sampling them would return the "
                f"prior. Known parameters: {sorted(KNOWN_PARAM_NAMES)}."
            )

        problems = missing_state_params(self.topology, set(names))
        if problems:
            raise ValueError(
                "The topology contains states the parameter vector does not "
                f"describe: {'; '.join(problems)}. Such a state inherits the "
                "shared birth rate and has no death rate, making it an "
                "immortal fully-fit clone during inference. Use one of the "
                f"parameter sets {sorted(PARAMETER_SETS)} or pass an explicit "
                "param_names covering every active state."
            )

    def _boundary_scales(self, param_names: list[str]) -> dict[str, float]:
        """Step scales for parameters whose prior puts mass at exactly zero.

        A lognormal random walk lives on ``log theta``, so it can never reach
        zero and mixes ever more slowly as it approaches it: a parameter whose
        true value is "no effect" sends the chain drifting toward -inf, and the
        reported posterior then depends on how long the chain ran. Parameters
        that *can* be zero -- Emax terms, induction rates, anything with a
        half-normal or uniform(0, .) prior -- therefore get a random walk on
        the linear scale instead, reflected at zero so the boundary is
        reachable and the density there is respected.

        Detection is by the prior, not by name: a finite log-density at zero
        is exactly the statement "zero is a value this parameter can take".
        A lognormal prior has -inf there and keeps the log walk.
        """
        scales: dict[str, float] = {}
        for name in param_names:
            dist = self.priors.distributions.get(name)
            if dist is None:
                continue
            try:
                if not np.isfinite(float(dist.logpdf(0.0))):
                    continue
                scale = float(dist.std())
            except (ValueError, TypeError, AttributeError):
                continue
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            scales[name] = scale
        return scales

    def sample(
        self,
        n_samples: int = 1000,
        n_warmup: int = 500,
        proposal_scale: float = 0.01,
        initial_params: dict[str, float] | None = None,
    ) -> MCMCResult:
        """Run PMCMC sampling.

        Args:
            n_samples: Number of post-warmup samples.
            n_warmup: Number of warmup iterations.
            proposal_scale: Random-walk scale. Applied to log theta for
                strictly positive parameters, and to theta itself (in units of
                the prior SD) for parameters whose prior admits zero.
            initial_params: Starting parameter values.

        Returns:
            MCMCResult with posterior samples, shaped (1, n_samples).
        """
        if initial_params is None:
            initial_params = self.priors.sample(self._proposal_rng)

        # Fix a deterministic parameter order rather than relying on dict order.
        param_names = list(self.param_names or sorted(initial_params))
        missing = [p for p in param_names if p not in initial_params]
        if missing:
            raise ValueError(f"initial_params is missing {missing}.")
        # Names may come from the priors rather than the constructor, so they
        # are checked here too.
        self._validate_param_names(param_names)

        current = {name: float(initial_params[name]) for name in param_names}
        current_ll = self._particle_filter_ll(current)
        current_lp = self.priors.log_prior(current)

        if not np.isfinite(current_ll + current_lp):
            raise ValueError(
                "The initial parameter values have zero posterior density "
                "(log-likelihood="
                f"{current_ll}, log-prior={current_lp}). Supply "
                "initial_params inside the prior support."
            )

        boundary_scales = self._boundary_scales(param_names)

        samples = {name: [] for name in param_names}
        ll_trace = []
        lp_trace = []
        n_accept = 0

        total_iter = n_warmup + n_samples
        for iteration in range(total_iter):
            # Lognormal random walk: log theta' = log theta + N(0, s).
            proposed = {}
            log_hastings = 0.0
            for name in param_names:
                if name in boundary_scales:
                    # Reflected linear-scale walk: theta' = |theta + N(0, s)|.
                    # Folding at zero keeps the proposal symmetric --
                    # q(theta'|theta) = phi(theta'-theta) + phi(theta'+theta)
                    # is unchanged under swapping the two -- so it contributes
                    # nothing to the Hastings term.
                    step = self._proposal_rng.normal(
                        0, proposal_scale * boundary_scales[name]
                    )
                    proposed[name] = float(abs(current[name] + step))
                    continue

                val = max(current[name], 1e-300)
                log_val = np.log(val)
                log_prop = log_val + self._proposal_rng.normal(0, proposal_scale)
                proposed[name] = float(np.exp(log_prop))
                # q(theta|theta')/q(theta'|theta) = theta'/theta for this
                # proposal; omitting it biases the chain toward small values.
                log_hastings += log_prop - log_val

            prop_lp = self.priors.log_prior(proposed)
            if np.isfinite(prop_lp):
                prop_ll = self._particle_filter_ll(proposed)
                log_alpha = (
                    (prop_ll + prop_lp) - (current_ll + current_lp) + log_hastings
                )
                if np.log(self._proposal_rng.uniform()) < log_alpha:
                    current = proposed
                    current_ll = prop_ll
                    current_lp = prop_lp
                    n_accept += 1

            if iteration >= n_warmup:
                for name in param_names:
                    samples[name].append(current[name])
                ll_trace.append(current_ll)
                lp_trace.append(current_ll + current_lp)

        return MCMCResult(
            # Shape (n_chains=1, n_draws) so diagnostics see the chain axis.
            samples={
                k: np.asarray(v, dtype=float)[np.newaxis, :]
                for k, v in samples.items()
            },
            log_likelihood_trace=np.asarray(ll_trace, dtype=float)[np.newaxis, :],
            log_posterior_trace=np.asarray(lp_trace, dtype=float)[np.newaxis, :],
            n_chains=1,
            n_samples=n_samples,
            diagnostics={
                "acceptance_rate": n_accept / total_iter,
                "backend": "particle_mcmc",
                "n_particles": self.n_particles,
                "param_names": param_names,
                # Which parameters were sampled on the reflected linear scale
                # because their prior admits zero.
                "boundary_params": sorted(boundary_scales),
            },
        )

    def _anchor(self) -> tuple[float, int | None]:
        """Initial count and the index of the observation it came from.

        Mirrors ModelLikelihood._initial_state: the first *non-missing* value
        of the anchor modality, falling back to metadata.
        """
        values = self.data.observations.get(self.anchor_modality)
        if values is not None and np.size(values):
            present = np.flatnonzero(~np.isnan(values))
            if present.size:
                index = int(present[0])
                return max(float(values[index]), 1.0), index
        n0 = float(self.data.metadata.get("initial_cells", 100.0))
        return max(n0, 1.0), None

    def _particle_filter_ll(self, params: dict[str, float]) -> float:
        """Run the particle filter and return the marginal log-likelihood."""
        rate_set = build_rate_set(params)
        conc = self.data.concentration if self.data.concentration is not None else 0.0
        def exposure_fn(t, _c=conc):
            return _c

        pf = ParticleFilter(
            rate_set,
            self.topology,
            exposure_fn,
            self.obs_model,
            self.n_particles,
            self._filter_rng,
            obs_params={
                k: v for k, v in params.items() if k in OBSERVATION_PARAM_NAMES
            },
        )

        n0, anchor_index = self._anchor()
        # "stable" needs the rates, which is why this is resolved per
        # evaluation rather than once at construction.
        if self._has_fraction_spec:
            fractions = resolve_initial_fractions(
                self.topology, self.data, self.initial_fractions, rate_set
            )
        else:
            fractions = self._default_fractions
        initial_state = n0 * fractions

        try:
            result = pf.filter(
                self.data,
                initial_state,
                anchor_modality=(
                    self.anchor_modality if self.condition_on_first else None
                ),
                anchor_index=anchor_index if self.condition_on_first else None,
            )
        except (ValueError, RuntimeError) as exc:
            # Genuine model/data errors must not be silently turned into a
            # rejection; log them so a systematically broken model is visible.
            logger.warning("Particle filter failed for %s: %s", params, exc)
            return -np.inf

        return result["marginal_log_likelihood"]
