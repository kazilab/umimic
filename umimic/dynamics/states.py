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
    and optional extensions (density dependence, phase-type dwell times).
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
    linear_chain_stages: dict[CellType, int] | None = None
    apoptotic_clearance_rate: float = 0.1  # rate at which dead cells are cleared

    @property
    def n_states(self) -> int:
        return len(self.active_states)

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
    def four_state(cls) -> ModelTopology:
        """P-Q-A-R model (full model with resistant state)."""
        return cls(
            active_states=[CellType.P, CellType.Q, CellType.A, CellType.R],
            transitions=[
                (CellType.P, CellType.Q),
                (CellType.Q, CellType.P),
                (CellType.P, CellType.R),
            ],
            division_states=[CellType.P],
            death_states=[CellType.P, CellType.Q, CellType.R],
        )
