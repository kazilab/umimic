"""Approximate stochastic simulation via tau-leaping."""

from __future__ import annotations

from typing import Callable

import numpy as np

from umimic.dynamics.states import ModelTopology
from umimic.dynamics.rates import RateSet
from umimic.dynamics.gillespie import build_reactions
from umimic.types import SimulationResult, EnsembleResult


class TauLeapingSimulator:
    """Tau-leaping approximate stochastic simulation.

    Faster than exact Gillespie for large populations. Approximates
    the number of events in each time step as Poisson-distributed.
    Falls back to smaller steps when populations are small.
    """

    def __init__(
        self,
        rate_set: RateSet,
        topology: ModelTopology,
        exposure_fn: Callable[[float], float],
        tau: float = 0.1,
        rng: np.random.Generator | None = None,
    ):
        self.rate_set = rate_set
        self.topology = topology
        self.exposure_fn = exposure_fn
        self.tau = tau
        self.rng = rng or np.random.default_rng()
        self.reactions = build_reactions(rate_set, topology)

    def simulate(
        self,
        x0: np.ndarray,
        t_max: float,
        t_record: np.ndarray | None = None,
    ) -> SimulationResult:
        """Run one tau-leaping trajectory.

        Args:
            x0: Initial state vector.
            t_max: Maximum simulation time.
            t_record: Times at which to record state.

        Returns:
            SimulationResult with populations at recorded times.
        """
        if t_record is None:
            t_record = np.linspace(0, t_max, 100)

        n_states = len(x0)
        state = x0.astype(float).copy()
        t = 0.0

        recorded = np.zeros((len(t_record), n_states))
        rec_idx = 0

        while rec_idx < len(t_record) and t_record[rec_idx] <= t:
            recorded[rec_idx] = state
            rec_idx += 1

        while t < t_max:
            conc = self.exposure_fn(t)
            total = float(np.sum(np.maximum(state, 0)))

            # Compute propensities
            propensities = np.array(
                [r.propensity_fn(state, conc, total) for r in self.reactions]
            )
            propensities = np.maximum(propensities, 0.0)

            # Adaptive tau: reduce step if populations are small
            a0 = np.sum(propensities)
            if a0 <= 0:
                while rec_idx < len(t_record):
                    recorded[rec_idx] = state
                    rec_idx += 1
                break

            min_pop = float(np.min(state[state > 0])) if np.any(state > 0) else 0
            adaptive_tau = self.tau
            if min_pop > 0 and min_pop < 10:
                adaptive_tau = min(self.tau, 0.5 / a0)  # smaller steps for small pops

            t_next = t + adaptive_tau

            # Sample number of each reaction in this interval (Poisson)
            n_fires = np.array(
                [
                    self.rng.poisson(max(0, p * adaptive_tau))
                    for p in propensities
                ]
            )

            # Apply all reactions
            delta = np.zeros(n_states)
            for i, n in enumerate(n_fires):
                if n > 0:
                    delta += n * self.reactions[i].stoichiometry

            state = np.maximum(state + delta, 0)

            # Record at requested times
            while rec_idx < len(t_record) and t_record[rec_idx] <= t_next:
                recorded[rec_idx] = state
                rec_idx += 1

            t = t_next

        # Fill remaining
        while rec_idx < len(t_record):
            recorded[rec_idx] = state
            rec_idx += 1

        populations = {}
        for i, ct in enumerate(self.topology.active_states):
            populations[ct.name] = recorded[:, i]

        return SimulationResult(
            times=t_record,
            populations=populations,
            metadata={"method": "tau_leaping", "tau": self.tau},
        )

    def simulate_ensemble(
        self,
        x0: np.ndarray,
        t_max: float,
        t_record: np.ndarray | None = None,
        n_trajectories: int = 100,
    ) -> EnsembleResult:
        """Run multiple independent tau-leaping trajectories."""
        if t_record is None:
            t_record = np.linspace(0, t_max, 100)

        trajectories = []
        for _ in range(n_trajectories):
            result = self.simulate(x0, t_max, t_record)
            trajectories.append(result.populations)

        return EnsembleResult(times=t_record, trajectories=trajectories)
