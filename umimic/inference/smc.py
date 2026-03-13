"""Sequential Monte Carlo / Particle Filter for robust inference.

The 'robust mode' inference backend for U-MIMIC. Used when populations
are small, dynamics are highly nonlinear, or the normal approximation
in moment-based inference breaks down.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from umimic.dynamics.states import ModelTopology
from umimic.dynamics.rates import RateSet
from umimic.dynamics.gillespie import GillespieSimulator
from umimic.observations.base import ObservationModel
from umimic.data.schemas import TimeSeriesData
from umimic.types import MCMCResult


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
        observation_model: ObservationModel,
        n_particles: int = 500,
        rng: np.random.Generator | None = None,
    ):
        self.rate_set = rate_set
        self.topology = topology
        self.exposure_fn = exposure_fn
        self.obs_model = observation_model
        self.n_particles = n_particles
        self.rng = rng or np.random.default_rng(42)

    def filter(
        self,
        data: TimeSeriesData,
        initial_state: np.ndarray,
    ) -> dict:
        """Run the particle filter through observed data.

        Args:
            data: Observed time-series data.
            initial_state: Initial cell state vector.

        Returns:
            Dict with filtered state estimates, weights, and log-likelihood.
        """
        n = self.n_particles
        n_states = self.topology.n_states
        times = data.times

        # Initialize particles at initial state (with small perturbation)
        particles = np.tile(initial_state, (n, 1)).astype(float)
        # Add Poisson noise to initial conditions
        for i in range(n):
            for j in range(n_states):
                if particles[i, j] > 0:
                    particles[i, j] = max(
                        self.rng.poisson(particles[i, j]), 0
                    )

        weights = np.ones(n) / n
        marginal_ll = 0.0

        filtered_means = np.zeros((len(times), n_states))
        filtered_vars = np.zeros((len(times), n_states))

        for k in range(len(times)):
            # Propagation step: simulate each particle forward
            if k > 0:
                dt = times[k] - times[k - 1]
                new_particles = np.zeros_like(particles)
                for i in range(n):
                    sim = GillespieSimulator(
                        self.rate_set, self.topology, self.exposure_fn, self.rng
                    )
                    result = sim.simulate(
                        particles[i],
                        dt,
                        t_record=np.array([dt]),
                        max_events=100_000,
                    )
                    for j, ct in enumerate(self.topology.active_states):
                        new_particles[i, j] = result.populations[ct.name][-1]
                particles = new_particles

            # Weight update step: compute observation likelihood
            if "cell_counts" in data.observations:
                obs = data.observations["cell_counts"][k]
                log_weights = np.zeros(n)
                for i in range(n):
                    log_weights[i] = self.obs_model.log_likelihood(
                        obs, particles[i]
                    )

                # Normalize in log-space
                max_lw = np.max(log_weights)
                if np.isfinite(max_lw):
                    log_weights -= max_lw
                    weights = np.exp(log_weights)
                    total_w = np.sum(weights)
                    if total_w > 0:
                        weights /= total_w
                        marginal_ll += max_lw + np.log(total_w) - np.log(n)
                    else:
                        weights = np.ones(n) / n
                else:
                    weights = np.ones(n) / n

            # Record filtered state
            filtered_means[k] = np.average(particles, weights=weights, axis=0)
            filtered_vars[k] = np.average(
                (particles - filtered_means[k]) ** 2, weights=weights, axis=0
            )

            # Systematic resampling if ESS is low
            ess = 1.0 / np.sum(weights**2)
            if ess < n / 2:
                indices = self._systematic_resample(weights)
                particles = particles[indices]
                weights = np.ones(n) / n

        return {
            "filtered_means": filtered_means,
            "filtered_vars": filtered_vars,
            "marginal_log_likelihood": marginal_ll,
            "final_particles": particles,
            "final_weights": weights,
        }

    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        """Systematic resampling for particle filter."""
        n = len(weights)
        cumsum = np.cumsum(weights)
        u = (self.rng.uniform() + np.arange(n)) / n
        indices = np.searchsorted(cumsum, u)
        return np.clip(indices, 0, n - 1)


class ParticleMCMC:
    """Particle Markov Chain Monte Carlo (PMCMC).

    Combines the particle filter (for marginal likelihood estimation)
    with Metropolis-Hastings (for parameter inference).

    This is the full 'robust mode' for parameter estimation when
    the moment-based approach fails.
    """

    def __init__(
        self,
        topology: ModelTopology,
        data: TimeSeriesData,
        observation_model: ObservationModel,
        priors: "PriorSpec",
        n_particles: int = 200,
        rng: np.random.Generator | None = None,
    ):
        self.topology = topology
        self.data = data
        self.obs_model = observation_model
        self.priors = priors
        self.n_particles = n_particles
        self.rng = rng or np.random.default_rng(42)

    def _build_rate_set_from_params(self, params: dict[str, float]) -> RateSet:
        """Construct RateSet from parameter dict (same as in likelihood.py)."""
        from umimic.dynamics.rates import EmaxHill

        death_mod = None
        if params.get("emax_death", 0) > 0:
            death_mod = EmaxHill(
                emax=params["emax_death"],
                ec50=params.get("ec50_death", 1.0),
                hill=params.get("hill_death", 1.5),
            )

        from umimic.dynamics.states import CellType

        return RateSet(
            birth_base=params.get("b0", 0.04),
            death_base={
                CellType.P: params.get("d0_P", 0.01),
                CellType.Q: params.get("d0_P", 0.01) * 0.5,
            },
            death_modulation={CellType.P: death_mod} if death_mod else {},
            transition_base={
                (CellType.P, CellType.Q): params.get("u_PQ", 0.005),
                (CellType.Q, CellType.P): params.get("u_QP", 0.003),
            },
        )

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
            proposal_scale: Scale of random walk proposal.
            initial_params: Starting parameter values.

        Returns:
            MCMCResult with posterior samples.
        """
        if initial_params is None:
            initial_params = self.priors.sample(self.rng)

        param_names = list(initial_params.keys())
        current = initial_params.copy()
        current_ll = self._particle_filter_ll(current)
        current_lp = self.priors.log_prior(current)

        samples = {name: [] for name in param_names}
        ll_trace = []
        n_accept = 0

        total_iter = n_warmup + n_samples
        for iteration in range(total_iter):
            # Propose new parameters (random walk on log-scale)
            proposed = {}
            for name, val in current.items():
                log_val = np.log(max(val, 1e-10))
                log_prop = log_val + self.rng.normal(0, proposal_scale)
                proposed[name] = np.exp(log_prop)

            # Evaluate proposed
            prop_lp = self.priors.log_prior(proposed)
            if not np.isfinite(prop_lp):
                # Reject: out of prior support
                if iteration >= n_warmup:
                    for name in param_names:
                        samples[name].append(current[name])
                    ll_trace.append(current_ll)
                continue

            prop_ll = self._particle_filter_ll(proposed)

            # Metropolis-Hastings acceptance
            log_alpha = (prop_ll + prop_lp) - (current_ll + current_lp)
            if np.log(self.rng.uniform()) < log_alpha:
                current = proposed
                current_ll = prop_ll
                current_lp = prop_lp
                n_accept += 1

            if iteration >= n_warmup:
                for name in param_names:
                    samples[name].append(current[name])
                ll_trace.append(current_ll)

        return MCMCResult(
            samples={k: np.array(v) for k, v in samples.items()},
            log_likelihood_trace=np.array(ll_trace),
            n_chains=1,
            n_samples=n_samples,
            diagnostics={
                "acceptance_rate": n_accept / total_iter,
                "backend": "particle_mcmc",
                "n_particles": self.n_particles,
            },
        )

    def _particle_filter_ll(self, params: dict[str, float]) -> float:
        """Run particle filter and return marginal log-likelihood."""
        try:
            rate_set = self._build_rate_set_from_params(params)
            conc = self.data.concentration or 0.0
            exposure_fn = lambda t: conc

            pf = ParticleFilter(
                rate_set, self.topology, exposure_fn,
                self.obs_model, self.n_particles, self.rng,
            )

            n0 = self.data.observations.get("cell_counts", np.array([100]))[0]
            initial_state = np.zeros(self.topology.n_states)
            initial_state[0] = max(n0, 1.0)

            result = pf.filter(self.data, initial_state)
            return result["marginal_log_likelihood"]
        except Exception:
            return -np.inf
