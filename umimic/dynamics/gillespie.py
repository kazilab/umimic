"""Exact stochastic simulation via the Gillespie algorithm (SSA).

Exposure contract
-----------------
The direct method assumes propensities are constant between events, which
holds only for constant or piecewise-constant drug exposure. With a
continuously varying PK profile the propensities change *during* the waiting
time, and freezing them at the interval start is not exact.

This module therefore selects its method from the exposure profile:

* constant exposure -> Gillespie direct method (exact, fast);
* time-varying exposure -> Extrande thinning (Voliotis et al., 2016), which
  is exact for time-dependent propensities given a valid upper bound on the
  total propensity over each look-ahead window.

Pass ``exposure_mode="constant"`` to assert constancy and reject anything
else, or ``"direct"`` to force the frozen-propensity method knowing it is
approximate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.rates import RateSet
from umimic.types import SimulationResult, EnsembleResult

logger = logging.getLogger(__name__)


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

        def make_birth_prop(cell_idx, cell_type, K_val=K):
            def prop(state, conc, total):
                return rate_set.birth_rate(
                    conc, total, K_val, cell_type=cell_type
                ) * max(state[cell_idx], 0)
            return prop

        reactions.append(
            Reaction(
                name=f"birth_{ct.name}",
                stoichiometry=stoich.copy(),
                propensity_fn=make_birth_prop(i, ct),
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
    """Exact stochastic simulation of the cell-population jump process.

    Uses the Gillespie direct method under constant exposure and Extrande
    thinning under time-varying exposure. See the module docstring for the
    exposure contract.
    """

    def __init__(
        self,
        rate_set: RateSet,
        topology: ModelTopology,
        exposure_fn: Callable[[float], float],
        rng: np.random.Generator | None = None,
        exposure_mode: str = "auto",
        lookahead: float = 1.0,
        bound_samples: int = 16,
        bound_safety: float = 1.05,
    ):
        """
        Args:
            rate_set: Reaction rates.
            topology: Model topology.
            exposure_fn: Drug concentration as a function of time.
            rng: Random generator.
            exposure_mode: "auto" (detect), "constant" (assert and reject
                non-constant profiles), "thinning" (force Extrande), or
                "direct" (force the approximate frozen-propensity method).
            lookahead: Extrande look-ahead window (hours).
            bound_samples: Grid points used to bound propensities over a window.
            bound_safety: Multiplicative safety factor on that bound.
        """
        self.rate_set = rate_set
        self.topology = topology
        self.exposure_fn = exposure_fn
        self.rng = rng or np.random.default_rng()
        self.reactions = build_reactions(rate_set, topology)
        self.lookahead = float(lookahead)
        self.bound_samples = int(bound_samples)
        self.bound_safety = float(bound_safety)

        if exposure_mode not in ("auto", "constant", "thinning", "direct"):
            raise ValueError(
                f"Unknown exposure_mode {exposure_mode!r}; expected 'auto', "
                "'constant', 'thinning' or 'direct'."
            )
        self.exposure_mode = exposure_mode

    def _is_constant_exposure(self, t_max: float, n_probe: int = 33) -> bool:
        """Probe the exposure profile for time dependence."""
        probes = np.linspace(0.0, max(t_max, 1e-9), n_probe)
        values = np.array([float(self.exposure_fn(t)) for t in probes])
        return bool(np.allclose(values, values[0], rtol=1e-12, atol=1e-12))

    def _resolve_method(self, t_max: float) -> tuple[str, bool]:
        """Choose the simulation method and whether it is exact.

        Returns ``(method, exact)``. The direct method freezes propensities at
        the interval start, so it is exact only for a constant exposure. That
        distinction has to reach the caller: an "exact" flag on a trajectory
        sampled with frozen propensities under a moving PK curve invites the
        user to treat an approximation as ground truth.
        """
        if self.exposure_mode == "direct":
            # Explicitly requested, so the constancy probe is the only way to
            # know whether the result is exact.
            return "direct", self._is_constant_exposure(t_max)
        if self.exposure_mode == "thinning":
            return "thinning", True

        constant = self._is_constant_exposure(t_max)
        if self.exposure_mode == "constant":
            if not constant:
                raise ValueError(
                    "exposure_mode='constant' requires a constant exposure "
                    "profile, but exposure_fn varies with time. Use "
                    "exposure_mode='thinning' for an exact time-varying "
                    "simulation."
                )
            return "direct", True
        # "auto" only picks direct after confirming the exposure is constant.
        return ("direct", True) if constant else ("thinning", True)

    def _propensities(self, state: np.ndarray, t: float) -> np.ndarray:
        conc = self.exposure_fn(t)
        total = self.topology.density_total(state)
        a = np.array([r.propensity_fn(state, conc, total) for r in self.reactions])
        return np.maximum(a, 0.0)

    def _propensity_bound(
        self, state: np.ndarray, t: float, horizon: float
    ) -> float:
        """Upper bound on total propensity over [t, t + horizon] at fixed state.

        Propensities are linear in the state, which is constant across the
        window, so bounding them reduces to bounding the rate coefficients.
        These are sampled on a grid; `bound_safety` guards against
        under-resolving a sharp PK peak.
        """
        probes = np.linspace(t, t + horizon, self.bound_samples)
        worst = np.zeros(len(self.reactions))
        for tp in probes:
            worst = np.maximum(worst, self._propensities(state, float(tp)))
        return float(np.sum(worst) * self.bound_safety)

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
        t_record = np.asarray(t_record, dtype=float)

        method, exact = self._resolve_method(t_max)
        if not exact:
            logger.warning(
                "exposure_mode='direct' was requested with a time-varying "
                "exposure: propensities are frozen at each interval start, so "
                "this trajectory is an approximation. Use "
                "exposure_mode='thinning' for an exact sample."
            )

        n_states = len(x0)
        state = np.maximum(x0.astype(float).copy(), 0.0)
        t = 0.0

        # Pre-allocate recording arrays
        recorded = np.zeros((len(t_record), n_states))
        rec_idx = 0

        # Record initial state for any t_record <= 0
        while rec_idx < len(t_record) and t_record[rec_idx] <= t:
            recorded[rec_idx] = state
            rec_idx += 1

        n_events = 0
        n_rejected = 0
        truncated = False
        extinct = False

        while t < t_max:
            if n_events >= max_events:
                truncated = True
                break

            if method == "thinning":
                # Extrande: bound the propensity over a look-ahead window and
                # accept a candidate event with probability a0(t)/B.
                horizon = min(self.lookahead, t_max - t)
                B = self._propensity_bound(state, t, horizon)
                if B <= 0:
                    extinct = True
                    break

                tau = float(self.rng.exponential(1.0 / B))
                if tau > horizon:
                    # No event in this window; advance to its end.
                    t_next = t + horizon
                    while rec_idx < len(t_record) and t_record[rec_idx] <= t_next:
                        recorded[rec_idx] = state
                        rec_idx += 1
                    t = t_next
                    continue

                t_cand = t + tau
                propensities = self._propensities(state, t_cand)
                a0 = float(np.sum(propensities))

                while rec_idx < len(t_record) and t_record[rec_idx] <= t_cand:
                    recorded[rec_idx] = state
                    rec_idx += 1

                t = t_cand
                if self.rng.uniform() * B > a0:
                    # Thinning rejection: time advances, state does not.
                    n_rejected += 1
                    continue
            else:
                propensities = self._propensities(state, t)
                a0 = float(np.sum(propensities))

                if a0 <= 0:
                    extinct = True
                    break

                tau = float(self.rng.exponential(1.0 / a0))
                t_next = t + tau

                # Record state at any t_record between t and t_next
                while rec_idx < len(t_record) and t_record[rec_idx] <= t_next:
                    recorded[rec_idx] = state
                    rec_idx += 1

                if t_next > t_max:
                    t = t_max
                    break
                t = t_next

            # Choose which reaction fires
            cumsum = np.cumsum(propensities)
            u = self.rng.uniform(0, float(cumsum[-1]))
            reaction_idx = int(np.searchsorted(cumsum, u))
            reaction_idx = min(reaction_idx, len(self.reactions) - 1)

            # Apply reaction. Propensities vanish when a reactant is absent, so
            # a correct SSA never produces a negative count; assert rather than
            # silently clip, which would hide a malformed propensity.
            new_state = state + self.reactions[reaction_idx].stoichiometry
            if np.any(new_state < 0):
                raise RuntimeError(
                    f"Reaction {self.reactions[reaction_idx].name!r} drove a "
                    f"population negative (state={state}, result={new_state}). "
                    "This indicates an inconsistent propensity function."
                )
            state = new_state
            n_events += 1

        # Fill any remaining recording slots
        while rec_idx < len(t_record):
            recorded[rec_idx] = state
            rec_idx += 1

        if truncated:
            logger.warning(
                "Gillespie simulation hit the event limit (%d) at t=%.4g < "
                "t_max=%.4g; the trajectory is truncated and its tail is not a "
                "valid sample.",
                max_events,
                t,
                t_max,
            )

        populations = {}
        for i, ct in enumerate(self.topology.active_states):
            populations[ct.name] = recorded[:, i]

        return SimulationResult(
            times=t_record,
            populations=populations,
            metadata={
                "n_events": n_events,
                "method": f"gillespie:{method}",
                "exact": exact,
                "truncated": truncated,
                "t_reached": float(t),
                "extinct": extinct,
                "n_thinning_rejections": n_rejected,
            },
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
