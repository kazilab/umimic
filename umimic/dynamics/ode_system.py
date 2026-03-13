"""Deterministic ODE system for cell population mean-field dynamics."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.rates import RateSet
from umimic.types import SimulationResult


class CellDynamicsODE:
    """Deterministic ODE system for cell population dynamics.

    Models the mean-field (deterministic) evolution of the cell state vector:

        dP/dt = b(C)*P - dP(C)*P - sum(uPj)*P + sum(ujP)*j
        dQ/dt = uPQ*P - uQP*Q - dQ(C)*Q
        dA/dt = sum(di*Xi) - clearance*A     (accumulates dead cells)
        dR/dt = uPR*P - dR(C)*R

    where C = C(t) is the drug concentration from the exposure profile.
    """

    def __init__(
        self,
        rate_set: RateSet,
        topology: ModelTopology,
        exposure_fn: Callable[[float], float],
    ):
        self.rate_set = rate_set
        self.topology = topology
        self.exposure_fn = exposure_fn
        self._state_idx = {ct: i for i, ct in enumerate(topology.active_states)}

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Right-hand side of the ODE system."""
        n = len(self.topology.active_states)
        dydt = np.zeros(n)
        c = self.exposure_fn(t)
        total = float(np.sum(np.maximum(y, 0.0)))
        K = (
            self.topology.carrying_capacity
            if self.topology.density_dependent
            else None
        )

        for i, ct in enumerate(self.topology.active_states):
            pop_i = max(y[i], 0.0)

            # Division (only for proliferating states)
            if ct in self.topology.division_states:
                b = self.rate_set.birth_rate(c, total, K)
                dydt[i] += b * pop_i

            # Death
            if ct in self.topology.death_states:
                d = self.rate_set.death_rate(ct, c)
                dydt[i] -= d * pop_i
                # If apoptotic state tracked, add to A
                if (
                    ct != CellType.A
                    and self.topology.has_state(CellType.A)
                ):
                    a_idx = self._state_idx[CellType.A]
                    dydt[a_idx] += d * pop_i

            # Apoptotic clearance
            if ct == CellType.A:
                dydt[i] -= self.rate_set.clearance_rate * pop_i

            # Outgoing transitions
            for src, tgt in self.topology.transitions:
                if src == ct:
                    rate = self.rate_set.transition_rate(src, tgt, c)
                    j = self._state_idx[tgt]
                    dydt[i] -= rate * pop_i
                    dydt[j] += rate * pop_i

        return dydt

    def solve(
        self,
        y0: np.ndarray,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None = None,
        method: str = "RK45",
        **kwargs,
    ) -> SimulationResult:
        """Solve the ODE system.

        Args:
            y0: Initial state vector (one value per active state).
            t_span: (t_start, t_end) time interval.
            t_eval: Times at which to record the solution.
            method: ODE solver method ('RK45', 'BDF', etc.).

        Returns:
            SimulationResult with time points and population trajectories.
        """
        sol = solve_ivp(
            self.rhs,
            t_span,
            y0,
            t_eval=t_eval,
            method=method,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10,
            **kwargs,
        )

        if not sol.success:
            # Retry with stiff solver
            sol = solve_ivp(
                self.rhs,
                t_span,
                y0,
                t_eval=t_eval,
                method="BDF",
                dense_output=True,
                rtol=1e-8,
                atol=1e-10,
                **kwargs,
            )

        populations = {}
        for i, ct in enumerate(self.topology.active_states):
            populations[ct.name] = np.maximum(sol.y[i], 0.0)

        return SimulationResult(
            times=sol.t,
            populations=populations,
            metadata={"method": method, "success": sol.success, "nfev": sol.nfev},
        )

    def solve_dose_response(
        self,
        y0: np.ndarray,
        t_span: tuple[float, float],
        concentrations: list[float],
        t_eval: np.ndarray | None = None,
    ) -> dict[float, SimulationResult]:
        """Solve for multiple constant concentrations (in vitro dose-response).

        Args:
            y0: Initial state vector.
            t_span: Time interval.
            concentrations: List of drug concentrations to simulate.
            t_eval: Times at which to record the solution.

        Returns:
            Dict mapping concentration -> SimulationResult.
        """
        results = {}
        for conc in concentrations:
            # Override exposure function with constant concentration
            original_fn = self.exposure_fn
            self.exposure_fn = lambda t, c=conc: c
            results[conc] = self.solve(y0, t_span, t_eval)
            self.exposure_fn = original_fn
        return results


def build_ode_system(
    rate_set: RateSet,
    topology: ModelTopology,
    exposure_fn: Callable[[float], float] | None = None,
    constant_concentration: float | None = None,
) -> CellDynamicsODE:
    """Convenience builder for the ODE system.

    Provide either exposure_fn (for time-varying in vivo) or
    constant_concentration (for in vitro).
    """
    if exposure_fn is None:
        if constant_concentration is not None:
            exposure_fn = lambda t: constant_concentration
        else:
            exposure_fn = lambda t: 0.0

    return CellDynamicsODE(rate_set, topology, exposure_fn)
