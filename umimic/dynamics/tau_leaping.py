"""Approximate stochastic simulation via tau-leaping.

Approximation contract
----------------------
Tau-leaping treats propensities as constant over a leap of length tau and
draws the number of firings of each reaction from a Poisson distribution. It
is an approximation to the SSA that becomes exact as tau -> 0.

This implementation uses the critical-reaction partition of Cao, Gillespie and
Petzold (J. Chem. Phys. 124, 044109, 2006): reactions that are close to
exhausting one of their reactants are classed as *critical* and fired one at a
time by exact SSA, while the remaining reactions are leaped. A leap that would
still drive a population negative is rejected and retried with tau halved, so
populations are never clipped at zero. Below `ssa_threshold` total cells the
simulator delegates to the exact SSA entirely.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from umimic.dynamics.gillespie import GillespieSimulator, build_reactions
from umimic.dynamics.rates import RateSet
from umimic.dynamics.states import ModelTopology
from umimic.types import EnsembleResult, SimulationResult

logger = logging.getLogger(__name__)


class TauLeapingSimulator:
    """Tau-leaping approximate stochastic simulation.

    Faster than exact Gillespie for large populations. Falls back to the exact
    SSA when populations are small, and handles near-exhaustion reactions
    exactly so that no reaction can consume more cells than exist.
    """

    def __init__(
        self,
        rate_set: RateSet,
        topology: ModelTopology,
        exposure_fn: Callable[[float], float],
        tau: float = 0.1,
        rng: np.random.Generator | None = None,
        n_critical: int = 10,
        ssa_threshold: float = 100.0,
    ):
        """
        Args:
            rate_set: Reaction rates.
            topology: Model topology.
            exposure_fn: Drug concentration as a function of time.
            tau: Nominal leap size (hours). Upper bound on the adaptive step.
            rng: Random generator.
            n_critical: A reaction is critical if fewer than this many firings
                would exhaust one of its reactants.
            ssa_threshold: Below this total population, use the exact SSA.
        """
        if not tau > 0:
            raise ValueError(f"tau must be positive, got {tau}.")
        self.rate_set = rate_set
        self.topology = topology
        self.exposure_fn = exposure_fn
        self.tau = float(tau)
        self.rng = rng or np.random.default_rng()
        self.n_critical = int(n_critical)
        self.ssa_threshold = float(ssa_threshold)
        self.reactions = build_reactions(rate_set, topology)
        self._stoich = np.array([r.stoichiometry for r in self.reactions], dtype=float)

    def _critical_mask(
        self, state: np.ndarray, propensities: np.ndarray
    ) -> np.ndarray:
        """Reactions whose reactants are close to exhaustion."""
        crit = np.zeros(len(self.reactions), dtype=bool)
        for j in range(len(self.reactions)):
            if propensities[j] <= 0:
                continue
            consumed = self._stoich[j] < 0
            if not np.any(consumed):
                continue
            # Firings available before some reactant runs out.
            capacity = np.min(
                np.floor(state[consumed] / np.abs(self._stoich[j][consumed]))
            )
            if capacity < self.n_critical:
                crit[j] = True
        return crit

    def simulate(
        self,
        x0: np.ndarray,
        t_max: float,
        t_record: np.ndarray | None = None,
    ) -> SimulationResult:
        """Run one tau-leaping trajectory.

        Args:
            x0: Initial state vector.
            t_max: Maximum simulation time. The trajectory never advances past
                this time; the final leap is truncated to land on it.
            t_record: Times at which to record state.

        Returns:
            SimulationResult with populations at recorded times.
        """
        if t_record is None:
            t_record = np.linspace(0, t_max, 100)
        t_record = np.asarray(t_record, dtype=float)

        # Small populations: the leap approximation is not valid, use exact SSA.
        if float(np.sum(np.maximum(x0, 0))) < self.ssa_threshold:
            ssa = GillespieSimulator(
                self.rate_set, self.topology, self.exposure_fn, self.rng
            )
            result = ssa.simulate(x0, t_max, t_record)
            result.metadata["method"] = "tau_leaping(ssa_fallback)"
            result.metadata["ssa_threshold"] = self.ssa_threshold
            return result

        n_states = len(x0)
        state = np.maximum(x0.astype(float).copy(), 0.0)
        t = 0.0

        recorded = np.zeros((len(t_record), n_states))
        rec_idx = 0
        while rec_idx < len(t_record) and t_record[rec_idx] <= t:
            recorded[rec_idx] = state
            rec_idx += 1

        n_leaps = 0
        n_rejected = 0
        n_critical_fired = 0
        tau_min = np.inf

        while t < t_max:
            conc = self.exposure_fn(t)
            total = self.topology.density_total(state)
            propensities = np.array(
                [r.propensity_fn(state, conc, total) for r in self.reactions]
            )
            propensities = np.maximum(propensities, 0.0)
            a0 = float(np.sum(propensities))

            if a0 <= 0:
                break

            crit = self._critical_mask(state, propensities)
            a_crit = float(np.sum(propensities[crit]))

            # Candidate leap for the non-critical reactions.
            tau_leap = min(self.tau, t_max - t)

            # Time to the next critical (exactly simulated) reaction.
            if a_crit > 0:
                tau_crit = float(self.rng.exponential(1.0 / a_crit))
            else:
                tau_crit = np.inf

            fire_critical = tau_crit < tau_leap
            step = min(tau_leap, tau_crit)
            step = min(step, t_max - t)
            if step <= 0:
                break

            # Propose the leap, halving on any negative population.
            accepted = False
            while not accepted:
                delta = np.zeros(n_states)
                non_crit = ~crit
                if np.any(non_crit):
                    n_fires = self.rng.poisson(propensities[non_crit] * step)
                    delta += n_fires @ self._stoich[non_crit]

                if fire_critical and a_crit > 0:
                    # Exactly one critical reaction fires, chosen proportionally.
                    p = propensities[crit] / a_crit
                    which = np.flatnonzero(crit)[
                        self.rng.choice(len(p), p=p)
                    ]
                    delta += self._stoich[which]

                candidate = state + delta
                if np.all(candidate >= 0):
                    accepted = True
                    if fire_critical and a_crit > 0:
                        n_critical_fired += 1
                else:
                    # Never clip: reject and retry with a shorter leap.
                    n_rejected += 1
                    step /= 2.0
                    fire_critical = False
                    if step < 1e-12:
                        logger.warning(
                            "Tau-leaping could not find a non-negative leap at "
                            "t=%.4g; falling back to a single SSA event.",
                            t,
                        )
                        # Defensive: unreachable while the propensities agree
                        # with their stoichiometries, because the first
                        # rejection clears `fire_critical` and the Poisson
                        # counts collapse to zero as `step` halves, so the
                        # candidate converges on the current (non-negative)
                        # state. Reaching here therefore means a propensity is
                        # positive in a state that cannot support the reaction.
                        #
                        # Draw only among reactions that can actually fire.
                        # Clipping at zero instead would silently repair that
                        # inconsistency, and the trajectory would no longer be
                        # a sample from the model at all -- while the module
                        # docstring promises populations are never clipped.
                        feasible = np.array(
                            [
                                bool(np.all(state + s >= 0))
                                for s in self._stoich
                            ]
                        )
                        usable = propensities * feasible
                        a_usable = float(np.sum(usable))
                        if a_usable <= 0:
                            raise RuntimeError(
                                f"At t={t:.6g} the total propensity is "
                                f"{a0:.6g} but no reaction can fire from state "
                                f"{state.tolist()}: every channel with positive "
                                "propensity would drive a population negative. "
                                "This means a propensity function disagrees "
                                "with its stoichiometry."
                            )
                        cumsum = np.cumsum(usable)
                        which = int(
                            np.searchsorted(cumsum, self.rng.uniform(0, a_usable))
                        )
                        which = min(which, len(self.reactions) - 1)
                        candidate = state + self._stoich[which]
                        # Waiting time from the same total the channel was
                        # drawn against. Using a0 here while choosing against
                        # a_usable would pair an SSA holding time with a
                        # different reaction set -- the two halves of one event
                        # must come from one propensity vector. (They coincide
                        # whenever the model is consistent, since an infeasible
                        # channel then has zero propensity.)
                        step = float(self.rng.exponential(1.0 / a_usable))
                        step = min(step, t_max - t)
                        accepted = True

            t_next = min(t + step, t_max)
            tau_min = min(tau_min, step)

            # Record BEFORE applying the leap: within [t, t_next) the trajectory
            # is represented by the state at t, matching the SSA convention.
            while rec_idx < len(t_record) and t_record[rec_idx] < t_next:
                recorded[rec_idx] = state
                rec_idx += 1

            state = candidate
            t = t_next
            n_leaps += 1

        # Remaining record slots (including t_max itself) take the final state.
        while rec_idx < len(t_record):
            recorded[rec_idx] = state
            rec_idx += 1

        populations = {}
        for i, ct in enumerate(self.topology.active_states):
            populations[ct.name] = recorded[:, i]

        if n_rejected:
            logger.debug(
                "Tau-leaping rejected %d proposed leaps to keep populations "
                "non-negative.",
                n_rejected,
            )

        return SimulationResult(
            times=t_record,
            populations=populations,
            metadata={
                "method": "tau_leaping",
                "tau": self.tau,
                "n_leaps": n_leaps,
                "n_rejected_leaps": n_rejected,
                "n_critical_events": n_critical_fired,
                "min_step": None if tau_min is np.inf else float(tau_min),
                "approximation": (
                    "Poisson leap with critical-reaction SSA; exact as tau -> 0"
                ),
            },
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
