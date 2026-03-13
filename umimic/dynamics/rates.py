"""Dose-response functions and rate parameterization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from umimic.dynamics.states import CellType


class DoseResponseFunction(ABC):
    """Abstract base for concentration -> effect modulation."""

    @abstractmethod
    def __call__(self, concentration: float | np.ndarray) -> float | np.ndarray:
        ...

    @abstractmethod
    def param_names(self) -> list[str]:
        ...


@dataclass
class EmaxHill(DoseResponseFunction):
    """Emax/Hill dose-response: m(C) = Emax * C^Hill / (EC50^Hill + C^Hill).

    Returns a value in [0, Emax] that increases with concentration.
    """

    emax: float = 1.0
    ec50: float = 1.0
    hill: float = 1.0

    def __call__(self, c: float | np.ndarray) -> float | np.ndarray:
        c = np.asarray(c, dtype=float)
        c_safe = np.maximum(c, 0.0)
        return self.emax * c_safe**self.hill / (self.ec50**self.hill + c_safe**self.hill)

    def param_names(self) -> list[str]:
        return ["emax", "ec50", "hill"]


@dataclass
class FourParameterLogistic(DoseResponseFunction):
    """4-parameter logistic: m(C) = bottom + (top - bottom) / (1 + (C/EC50)^hill).

    Commonly used for sigmoidal dose-response curves.
    """

    top: float = 1.0
    bottom: float = 0.0
    ec50: float = 1.0
    hill: float = 1.0

    def __call__(self, c: float | np.ndarray) -> float | np.ndarray:
        c = np.asarray(c, dtype=float)
        c_safe = np.maximum(c, 1e-30)
        return self.bottom + (self.top - self.bottom) / (
            1.0 + (c_safe / self.ec50) ** self.hill
        )

    def param_names(self) -> list[str]:
        return ["top", "bottom", "ec50", "hill"]


@dataclass
class ConstantRate(DoseResponseFunction):
    """Constant (no drug modulation): m(C) = value."""

    value: float = 0.0

    def __call__(self, c: float | np.ndarray) -> float | np.ndarray:
        return np.full_like(np.asarray(c, dtype=float), self.value)

    def param_names(self) -> list[str]:
        return ["value"]


@dataclass
class RateSet:
    """Complete set of concentration-modulated rates for the CTMC.

    Mechanistic rate parameterization:
    - birth_rate(C) = b0 * (1 - mb(C))  -- drug reduces division
    - death_rate(C) = d0 + md(C)         -- drug increases death
    - transition_rate(C) = u0 * (1 + mt(C))  -- drug may modulate transitions

    Parameter naming convention (inference ↔ model mapping):
        Inference vector    RateSet field             Description
        ──────────────────  ────────────────────────  ──────────────────────
        b0                  birth_base                Baseline birth rate
        d0_P                death_base[CellType.P]    Baseline death (P)
        d0_Q                death_base[CellType.Q]    Baseline death (Q)
        emax_death          death_modulation Emax     Max drug death effect
        ec50_death          death_modulation EC50     Half-max concentration
        hill_death          death_modulation Hill     Hill coefficient
        u_PQ                transition_base[(P, Q)]   P→Q transition rate
        u_QP                transition_base[(Q, P)]   Q→P transition rate
        u_PR                transition_base[(P, R)]   P→R transition rate
        overdispersion      (observation model)       NegBin overdispersion
    """

    # Birth rate (inference name: b0)
    birth_base: float = 0.04  # 1/hour (doubling time ~17h)
    birth_modulation: DoseResponseFunction | None = None

    # Death rates per cell type (inference names: d0_P, d0_Q)
    death_base: dict[CellType, float] = field(
        default_factory=lambda: {CellType.P: 0.01, CellType.Q: 0.005}
    )
    death_modulation: dict[CellType, DoseResponseFunction | None] = field(
        default_factory=dict
    )

    # Transition rates between states (inference names: u_PQ, u_QP, u_PR)
    transition_base: dict[tuple[CellType, CellType], float] = field(
        default_factory=lambda: {
            (CellType.P, CellType.Q): 0.005,
            (CellType.Q, CellType.P): 0.003,
        }
    )
    transition_modulation: dict[
        tuple[CellType, CellType], DoseResponseFunction | None
    ] = field(default_factory=dict)

    # Apoptotic clearance
    clearance_rate: float = 0.1  # rate of dead cell removal

    def birth_rate(
        self,
        concentration: float,
        total_cells: float = 0.0,
        carrying_capacity: float | None = None,
    ) -> float:
        """Effective birth rate at given concentration.

        b(C) = b0 * (1 - mb(C)) * density_correction
        """
        b = self.birth_base
        if self.birth_modulation is not None:
            b *= 1.0 - float(self.birth_modulation(concentration))
        if carrying_capacity is not None and carrying_capacity > 0:
            b *= max(0.0, 1.0 - total_cells / carrying_capacity)
        return max(0.0, b)

    def death_rate(self, cell_type: CellType, concentration: float) -> float:
        """Effective death rate for a cell type at given concentration.

        d(C) = d0 + md(C)
        """
        d = self.death_base.get(cell_type, 0.0)
        mod = self.death_modulation.get(cell_type)
        if mod is not None:
            d += float(mod(concentration))
        return max(0.0, d)

    def transition_rate(
        self, source: CellType, target: CellType, concentration: float
    ) -> float:
        """Effective transition rate from source to target at given concentration.

        u(C) = u0 * (1 + mt(C))
        """
        key = (source, target)
        u = self.transition_base.get(key, 0.0)
        mod = self.transition_modulation.get(key)
        if mod is not None:
            u *= 1.0 + float(mod(concentration))
        return max(0.0, u)

    def net_growth_rate(self, concentration: float) -> float:
        """Net growth rate at given concentration: b(C) - d_P(C)."""
        b = self.birth_rate(concentration)
        d = self.death_rate(CellType.P, concentration)
        return b - d

    def all_rates_at(
        self,
        concentration: float,
        state: np.ndarray,
        topology: "ModelTopology",
    ) -> dict[str, float]:
        """Compute all rates given concentration and current state.

        Returns a dict with named rates for inspection.
        """
        total = float(np.sum(state))
        K = topology.carrying_capacity if topology.density_dependent else None
        rates = {
            "birth": self.birth_rate(concentration, total, K),
        }
        for ct in topology.death_states:
            rates[f"death_{ct.name}"] = self.death_rate(ct, concentration)
        for src, tgt in topology.transitions:
            rates[f"trans_{src.name}_{tgt.name}"] = self.transition_rate(
                src, tgt, concentration
            )
        rates["clearance"] = self.clearance_rate
        return rates

    @classmethod
    def cytotoxic_drug(
        cls,
        b0: float = 0.04,
        d0: float = 0.01,
        emax_death: float = 0.05,
        ec50_death: float = 1.0,
        hill_death: float = 1.5,
    ) -> RateSet:
        """Create a RateSet for a purely cytotoxic drug (increases death only)."""
        return cls(
            birth_base=b0,
            birth_modulation=None,
            death_base={CellType.P: d0, CellType.Q: d0 * 0.5},
            death_modulation={
                CellType.P: EmaxHill(emax=emax_death, ec50=ec50_death, hill=hill_death),
            },
        )

    @classmethod
    def cytostatic_drug(
        cls,
        b0: float = 0.04,
        d0: float = 0.01,
        emax_birth: float = 0.8,
        ec50_birth: float = 1.0,
        hill_birth: float = 1.5,
    ) -> RateSet:
        """Create a RateSet for a purely cytostatic drug (reduces birth only)."""
        return cls(
            birth_base=b0,
            birth_modulation=EmaxHill(
                emax=emax_birth, ec50=ec50_birth, hill=hill_birth
            ),
            death_base={CellType.P: d0, CellType.Q: d0 * 0.5},
        )
