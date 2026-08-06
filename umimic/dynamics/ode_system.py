"""Deterministic ODE system for cell population mean-field dynamics."""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.rates import RateSet
from umimic.types import SimulationResult

logger = logging.getLogger(__name__)


class CellDynamicsODE:
    """Deterministic ODE system for cell population dynamics.

    Models the mean-field (deterministic) evolution of the cell state vector:

        dP/dt = b(C)*P - dP(C)*P - sum(uPj)*P + sum(ujP)*j
        dQ/dt = uPQ*P - uQP*Q - dQ(C)*Q
        dA/dt = sum(di*Xi) - clearance*A     (accumulates dead cells)
        dR/dt = bR(C)*R + uPR*P + uQR*Q - dR(C)*R - uRQ*R

    where C = C(t) is the drug concentration from the exposure profile.
    """

    def __init__(
        self,
        rate_set: RateSet,
        topology: ModelTopology,
        exposure_fn: Callable[[float], float],
        rate_multiplier_fn: Callable[[float, str], float] | None = None,
    ):
        self.rate_set = rate_set
        self.topology = topology
        self.exposure_fn = exposure_fn
        self.rate_multiplier_fn = rate_multiplier_fn
        self._state_idx = {ct: i for i, ct in enumerate(topology.active_states)}

    def _multiplier(self, t: float, key: str) -> float:
        """Return optional time-varying multiplier for a named rate."""
        if self.rate_multiplier_fn is None:
            return 1.0
        mult = float(self.rate_multiplier_fn(t, key))
        if not np.isfinite(mult) or mult < 0:
            return 1.0
        return mult

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Right-hand side of the ODE system."""
        n = len(self.topology.active_states)
        dydt = np.zeros(n)
        c = self.exposure_fn(t)
        # Only space-occupying cells crowd out division; apoptotic corpses do
        # not, unless the topology says otherwise.
        total = self.topology.density_total(y)
        K = (
            self.topology.carrying_capacity
            if self.topology.density_dependent
            else None
        )

        for i, ct in enumerate(self.topology.active_states):
            pop_i = max(y[i], 0.0)

            # Division (only for proliferating states)
            if ct in self.topology.division_states:
                b = self.rate_set.birth_rate(c, total, K, cell_type=ct)
                b *= self._multiplier(t, "birth")
                dydt[i] += b * pop_i

            # Death
            if ct in self.topology.death_states:
                d = self.rate_set.death_rate(ct, c)
                d *= self._multiplier(t, f"death:{ct.name}")
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
                    rate *= self._multiplier(t, f"transition:{src.name}->{tgt.name}")
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
            logger.warning("ODE solve with %s failed, retrying with BDF: %s", method, sol.message)
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
            if not sol.success:
                raise RuntimeError(f"ODE integration failed after BDF retry: {sol.message}")

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
        concentrations: Sequence[float],
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
            self.exposure_fn = self._make_constant_exposure_fn(conc)
            results[conc] = self.solve(y0, t_span, t_eval)
            self.exposure_fn = original_fn
        return results

    @staticmethod
    def _make_constant_exposure_fn(c: float) -> Callable[[float], float]:
        """Build a constant-concentration exposure function."""
        return lambda t: c


def build_ode_system(
    rate_set: RateSet,
    topology: ModelTopology,
    exposure_fn: Callable[[float], float] | None = None,
    constant_concentration: float | None = None,
    rate_multiplier_fn: Callable[[float, str], float] | None = None,
) -> CellDynamicsODE:
    """Convenience builder for the ODE system.

    Provide either exposure_fn (for time-varying in vivo) or
    constant_concentration (for in vitro).
    """
    if exposure_fn is None:
        fixed = constant_concentration if constant_concentration is not None else 0.0

        def exposure_fn(t, _c=fixed):
            return _c

    return CellDynamicsODE(
        rate_set,
        topology,
        exposure_fn,
        rate_multiplier_fn=rate_multiplier_fn,
    )
