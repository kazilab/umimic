"""Exact stochastic simulation via the Gillespie algorithm (SSA)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.rates import RateSet
from umimic.types import SimulationResult, EnsembleResult


@dataclass
class Reaction:
    """A single reaction in the CTMC."""

    name: str
    stoichiometry: np.ndarray  # change vector applied to state
    propensity_fn: Callable[[np.ndarray, float, float], float]
    # propensity_fn(state, concentration, total_cells) -> rate


def build_reactions(
    rate_set: RateSet,
    topology: ModelTopology,
) -> list[Reaction]:
    """Build the reaction list from a RateSet and ModelTopology.

    Each possible event (birth, death, transition, clearance) becomes a Reaction.
    """
    n = topology.n_states
    reactions = []
    idx = {ct: i for i, ct in enumerate(topology.active_states)}

    # Division reactions: P -> P + P (net: +1 in P)
    for ct in topology.division_states:
        i = idx[ct]
        stoich = np.zeros(n)
        stoich[i] = 1  # net gain of 1

        K = topology.carrying_capacity if topology.density_dependent else None

        def make_birth_prop(cell_idx, K_val=K):
            def prop(state, conc, total):
                return rate_set.birth_rate(conc, total, K_val) * max(state[cell_idx], 0)
            return prop

        reactions.append(
            Reaction(
                name=f"birth_{ct.name}",
                stoichiometry=stoich.copy(),
                propensity_fn=make_birth_prop(i),
            )
        )

    # Death reactions: X -> A (or X -> empty if A not tracked)
    for ct in topology.death_states:
        if ct == CellType.A:
            continue
        i = idx[ct]
        stoich = np.zeros(n)
        stoich[i] = -1
        if topology.has_state(CellType.A):
            stoich[idx[CellType.A]] = 1

        def make_death_prop(cell_type, cell_idx):
            def prop(state, conc, total):
                return rate_set.death_rate(cell_type, conc) * max(state[cell_idx], 0)
            return prop

        reactions.append(
            Reaction(
                name=f"death_{ct.name}",
                stoichiometry=stoich.copy(),
                propensity_fn=make_death_prop(ct, i),
            )
        )

    # Apoptotic clearance: A -> empty
    if topology.has_state(CellType.A):
        a_idx = idx[CellType.A]
        stoich = np.zeros(n)
        stoich[a_idx] = -1

        def clearance_prop(state, conc, total):
            return rate_set.clearance_rate * max(state[a_idx], 0)

        reactions.append(
            Reaction(
                name="clearance_A",
                stoichiometry=stoich.copy(),
                propensity_fn=clearance_prop,
            )
        )

    # State transitions: src -> tgt
    for src, tgt in topology.transitions:
        i_src = idx[src]
        i_tgt = idx[tgt]
        stoich = np.zeros(n)
        stoich[i_src] = -1
        stoich[i_tgt] = 1

        def make_trans_prop(source, target, src_idx):
            def prop(state, conc, total):
                return rate_set.transition_rate(source, target, conc) * max(
                    state[src_idx], 0
                )
            return prop

        reactions.append(
            Reaction(
                name=f"trans_{src.name}_{tgt.name}",
                stoichiometry=stoich.copy(),
                propensity_fn=make_trans_prop(src, tgt, i_src),
            )
        )

    return reactions


class GillespieSimulator:
    """Exact stochastic simulation (Gillespie direct method).

    Simulates the continuous-time Markov jump process for cell population
    dynamics with time-varying drug exposure.
    """

    def __init__(
        self,
        rate_set: RateSet,
        topology: ModelTopology,
        exposure_fn: Callable[[float], float],
        rng: np.random.Generator | None = None,
    ):
        self.rate_set = rate_set
        self.topology = topology
        self.exposure_fn = exposure_fn
        self.rng = rng or np.random.default_rng()
        self.reactions = build_reactions(rate_set, topology)

    def simulate(
        self,
        x0: np.ndarray,
        t_max: float,
        t_record: np.ndarray | None = None,
        max_events: int = 10_000_000,
    ) -> SimulationResult:
        """Run one trajectory of the Gillespie algorithm.

        Args:
            x0: Initial state vector.
            t_max: Maximum simulation time.
            t_record: Times at which to record state (interpolated).
            max_events: Safety limit on number of events.

        Returns:
            SimulationResult with populations at recorded times.
        """
        if t_record is None:
            t_record = np.linspace(0, t_max, 100)

        n_states = len(x0)
        state = x0.astype(float).copy()
        t = 0.0

        # Pre-allocate recording arrays
        recorded = np.zeros((len(t_record), n_states))
        rec_idx = 0

        # Record initial state for any t_record <= 0
        while rec_idx < len(t_record) and t_record[rec_idx] <= t:
            recorded[rec_idx] = state
            rec_idx += 1

        n_events = 0
        while t < t_max and n_events < max_events:
            conc = self.exposure_fn(t)
            total = float(np.sum(np.maximum(state, 0)))

            # Compute propensities
            propensities = np.array(
                [r.propensity_fn(state, conc, total) for r in self.reactions]
            )
            propensities = np.maximum(propensities, 0.0)
            a0 = np.sum(propensities)

            if a0 <= 0:
                # No more events possible (all cells dead/gone)
                while rec_idx < len(t_record):
                    recorded[rec_idx] = state
                    rec_idx += 1
                break

            # Time to next event (exponential)
            tau = self.rng.exponential(1.0 / a0)
            t_next = t + tau

            # Record state at any t_record between t and t_next
            while rec_idx < len(t_record) and t_record[rec_idx] <= t_next:
                recorded[rec_idx] = state
                rec_idx += 1

            if t_next > t_max:
                break

            # Choose which reaction fires
            cumsum = np.cumsum(propensities)
            u = self.rng.uniform(0, a0)
            reaction_idx = np.searchsorted(cumsum, u)
            reaction_idx = min(reaction_idx, len(self.reactions) - 1)

            # Apply reaction
            state = state + self.reactions[reaction_idx].stoichiometry
            state = np.maximum(state, 0)  # prevent negative counts
            t = t_next
            n_events += 1

        # Fill any remaining recording slots
        while rec_idx < len(t_record):
            recorded[rec_idx] = state
            rec_idx += 1

        populations = {}
        for i, ct in enumerate(self.topology.active_states):
            populations[ct.name] = recorded[:, i]

        return SimulationResult(
            times=t_record,
            populations=populations,
            metadata={"n_events": n_events, "method": "gillespie"},
        )

    def simulate_ensemble(
        self,
        x0: np.ndarray,
        t_max: float,
        t_record: np.ndarray | None = None,
        n_trajectories: int = 100,
        max_events_per: int = 10_000_000,
    ) -> EnsembleResult:
        """Run multiple independent trajectories.

        Args:
            x0: Initial state vector.
            t_max: Maximum simulation time.
            t_record: Times at which to record.
            n_trajectories: Number of independent trajectories.
            max_events_per: Max events per trajectory.

        Returns:
            EnsembleResult with all trajectories.
        """
        if t_record is None:
            t_record = np.linspace(0, t_max, 100)

        trajectories = []
        for _ in range(n_trajectories):
            result = self.simulate(x0, t_max, t_record, max_events_per)
            trajectories.append(result.populations)

        return EnsembleResult(
            times=t_record,
            trajectories=trajectories,
        )
