"""Cell state definitions and model topology."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class CellType(Enum):
    """Enumeration of cell states in the branching process."""

    P = "proliferating"
    Q = "quiescent"
    A = "apoptotic"
    R = "resistant"


# Standard state ordering for array representation
STATE_ORDER = [CellType.P, CellType.Q, CellType.A, CellType.R]


@dataclass
class StateVector:
    """Snapshot of cell population state at a given time."""

    populations: dict[CellType, float]
    time: float = 0.0

    @property
    def viable(self) -> float:
        """Viable cells: P + Q + R."""
        return sum(
            v for k, v in self.populations.items() if k != CellType.A
        )

    @property
    def total(self) -> float:
        return sum(self.populations.values())

    def as_array(self, order: list[CellType] | None = None) -> np.ndarray:
        order = order or STATE_ORDER
        return np.array([self.populations.get(ct, 0.0) for ct in order])

    @classmethod
    def from_array(
        cls, arr: np.ndarray, time: float = 0.0, order: list[CellType] | None = None
    ) -> StateVector:
        order = order or STATE_ORDER
        pops = {ct: float(arr[i]) for i, ct in enumerate(order) if i < len(arr)}
        return cls(populations=pops, time=time)

    def __repr__(self) -> str:
        parts = [f"{ct.name}={v:.1f}" for ct, v in self.populations.items() if v > 0]
        return f"StateVector(t={self.time:.2f}, {', '.join(parts)})"


@dataclass
class ModelTopology:
    """Defines the structure of the cell-state model.

    Specifies which states are active, what transitions are allowed,
    and optional extensions (density dependence). Phase-type dwell times are
    reserved but not implemented; see `linear_chain_stages`.
    """

    active_states: list[CellType] = field(
        default_factory=lambda: [CellType.P, CellType.Q]
    )
    transitions: list[tuple[CellType, CellType]] = field(
        default_factory=lambda: [(CellType.P, CellType.Q), (CellType.Q, CellType.P)]
    )
    division_states: list[CellType] = field(
        default_factory=lambda: [CellType.P]
    )
    death_states: list[CellType] = field(
        default_factory=lambda: [CellType.P, CellType.Q]
    )
    density_dependent: bool = False
    carrying_capacity: float | None = None
    # Whether apoptotic cells occupy space in the density term. Counting
    # corpses suppresses division until they are cleared, which produces
    # artefactual growth inhibition after a cytotoxic pulse; the default is to
    # let only viable cells compete for space.
    density_counts_apoptotic: bool = False
    # NOT IMPLEMENTED. Reserved for phase-type (Erlang) dwell times; no solver
    # reads it yet, so all states have exponential dwell times.
    linear_chain_stages: dict[CellType, int] | None = None

    # NOTE: apoptotic clearance lives on RateSet.clearance_rate, not here.
    # A duplicate field on the topology was a second, unread source of truth:
    # setting it changed nothing while looking as though it had. Assigning it
    # now raises -- see __setattr__.

    #: Attributes removed because they were never read by any solver, mapped
    #: to the setting that actually takes effect.
    _RELOCATED_ATTRIBUTES = {
        "apoptotic_clearance_rate": (
            "RateSet.clearance_rate (the simulators read clearance from the "
            "RateSet, never from the topology)"
        ),
    }

    def __setattr__(self, name: str, value) -> None:
        """Block assignment to settings that would silently do nothing."""
        if name in self._RELOCATED_ATTRIBUTES:
            raise AttributeError(
                f"ModelTopology.{name} has been removed because nothing read "
                f"it; setting it had no effect on any simulation. Set "
                f"{self._RELOCATED_ATTRIBUTES[name]} instead."
            )
        super().__setattr__(name, value)

    @property
    def n_states(self) -> int:
        return len(self.active_states)

    @property
    def density_mask(self) -> np.ndarray:
        """Indicator of the states that contribute to the density term.

        Returned as a float vector so it doubles as the gradient of the
        crowding total with respect to the state, which the moment-equation
        Jacobian needs.
        """
        return np.array(
            [
                0.0
                if (ct == CellType.A and not self.density_counts_apoptotic)
                else 1.0
                for ct in self.active_states
            ],
            dtype=float,
        )

    def density_total(self, state: np.ndarray) -> float:
        """Population competing for space at the given state."""
        return float(self.density_mask @ np.maximum(np.asarray(state, float), 0.0))

    def state_index(self, ct: CellType) -> int:
        return self.active_states.index(ct)

    def has_state(self, ct: CellType) -> bool:
        return ct in self.active_states

    @classmethod
    def two_state(cls) -> ModelTopology:
        """Simple P-Q model (proliferating + quiescent)."""
        return cls(
            active_states=[CellType.P, CellType.Q],
            transitions=[(CellType.P, CellType.Q), (CellType.Q, CellType.P)],
            division_states=[CellType.P],
            death_states=[CellType.P, CellType.Q],
        )

    @classmethod
    def three_state(cls) -> ModelTopology:
        """P-Q-A model (proliferating + quiescent + apoptotic)."""
        return cls(
            active_states=[CellType.P, CellType.Q, CellType.A],
            transitions=[(CellType.P, CellType.Q), (CellType.Q, CellType.P)],
            division_states=[CellType.P],
            death_states=[CellType.P, CellType.Q],
        )

    @classmethod
    def persister_resistance(
        cls,
        include_reversion: bool = False,
        include_direct_pr: bool = False,
    ) -> ModelTopology:
        """P <-> Q -> R topology with a persister route into resistance.

        The P -> R edge in :meth:`four_state` says resistance arises directly
        from cycling sensitive cells, skipping the drug-tolerant intermediate
        that the experimental literature places at the centre of acquired
        resistance to targeted therapy. This topology adds the persister
        route Q -> R instead.

        Args:
            include_reversion: Add R -> Q. Off by default. Reversion is a
                meaningful per-cell process only when R is an epigenetically
                stable state; for a genetic R, apparent resensitisation is
                competitive dilution, already carried by the rates. It is also
                close to unidentifiable from aggregate observables, since a
                Q -> R -> Q round trip barely changes viable counts.
            include_direct_pr: Also keep the direct P -> R edge, for models
                where resistance can arise without passing through Q.
        """
        transitions = [
            (CellType.P, CellType.Q),
            (CellType.Q, CellType.P),
            (CellType.Q, CellType.R),
        ]
        if include_reversion:
            transitions.append((CellType.R, CellType.Q))
        if include_direct_pr:
            transitions.append((CellType.P, CellType.R))

        return cls(
            active_states=[CellType.P, CellType.Q, CellType.A, CellType.R],
            transitions=transitions,
            # Persisters are slow-cycling, not strictly arrested, so Q divides.
            division_states=[CellType.P, CellType.Q, CellType.R],
            death_states=[CellType.P, CellType.Q, CellType.R],
        )

    @classmethod
    def four_state(cls) -> ModelTopology:
        """P-Q-A-R model (full model with resistant state).

        R divides. A resistant subpopulation that cannot proliferate is not a
        resistant clone: it is an absorbing sink that can only be filled by
        P -> R and can never outgrow the sensitive population under treatment.
        Resistance is then expressed through the rates rather than the
        topology -- R carries its own birth rate and is not subject to the
        drug's birth or death modulation (see
        :meth:`umimic.dynamics.rates.RateSet.resistant_clone`).
        """
        return cls(
            active_states=[CellType.P, CellType.Q, CellType.A, CellType.R],
            transitions=[
                (CellType.P, CellType.Q),
                (CellType.Q, CellType.P),
                (CellType.P, CellType.R),
            ],
            division_states=[CellType.P, CellType.R],
            death_states=[CellType.P, CellType.Q, CellType.R],
        )
