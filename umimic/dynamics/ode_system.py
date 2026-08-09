"""Deterministic ODE system for cell population mean-field dynamics."""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

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

            # Death. The apoptotic compartment is excluded: cells there are
            # already dead, and A leaves only via clearance below. Applying a
            # death rate to A as well would make this ODE a different model
            # from the SSA and the LNA, which both skip A here
            # (build_reactions and MomentODE._precompute_structure), and would
            # break the corpse balance A tracks.
            if ct in self.topology.death_states and ct != CellType.A:
                d = self.rate_set.death_rate(ct, c)
                d *= self._multiplier(t, f"death:{ct.name}")
                dydt[i] -= d * pop_i
                # If apoptotic state tracked, add to A
                if self.topology.has_state(CellType.A):
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

    def _linear_generator(
        self, t_span: tuple[float, float], t_eval: np.ndarray
    ) -> np.ndarray | None:
        """Return the constant generator A with dy/dt = A y, or None.

        The mean-field system is linear in the state whenever growth is
        density-independent, and it is time-invariant whenever the exposure and
        any rate multipliers are constant over the interval. Under both it has
        the closed form ``y(t) = expm(A t) y0``, so integrating it numerically
        is wasted work -- and, because the constant-exposure case is stiff at
        high drug concentrations, actively harmful: RK45 fails over to BDF and
        a single fit can run for tens of minutes.

        Linearity and time-invariance are *verified against the real RHS*
        rather than assumed. ``rhs`` is linear with no constant term, so column
        j is ``rhs`` applied to the j-th basis vector; the result is then
        checked at interior and end times against a probe state. A
        time-varying exposure (in vivo PK), a density-dependent topology or a
        time-varying rate multiplier all fail that check and fall back to the
        integrator.
        """
        if self.topology.density_dependent or self.rate_multiplier_fn is not None:
            return None

        n = len(self.topology.active_states)
        t0, t1 = float(t_span[0]), float(t_span[1])

        A = np.empty((n, n))
        for j in range(n):
            basis = np.zeros(n)
            basis[j] = 1.0
            A[:, j] = self.rhs(t0, basis)
        if not np.all(np.isfinite(A)):
            return None

        # Distinct, strictly positive probe so a state-dependent or
        # time-dependent term cannot cancel by coincidence.
        probe = np.linspace(1.0, 2.0, n)
        check_times = {t0, t1, 0.5 * (t0 + t1)}
        if t_eval is not None and len(t_eval):
            check_times.add(float(t_eval[len(t_eval) // 2]))
        for t in check_times:
            if not np.allclose(self.rhs(t, probe), A @ probe, rtol=1e-10, atol=1e-12):
                return None
        return A

    def _solve_linear(
        self, A: np.ndarray, y0: np.ndarray, t0: float, t_eval: np.ndarray
    ) -> np.ndarray:
        """Propagate y0 through the constant generator A at t_eval."""
        taus = np.asarray(t_eval, dtype=float) - t0
        y = np.empty((len(y0), len(taus)))

        # A uniform grid -- which every in vitro plate in this package uses --
        # needs one matrix exponential and a matrix-vector product per step,
        # instead of one exponential per recorded time.
        steps = np.diff(taus)
        if len(taus) > 2 and np.allclose(steps, steps[0], rtol=1e-12, atol=1e-12):
            state = expm(A * taus[0]) @ y0 if taus[0] != 0.0 else np.asarray(y0, float)
            step = expm(A * steps[0])
            y[:, 0] = state
            for k in range(1, len(taus)):
                state = step @ state
                y[:, k] = state
            return y

        for k, tau in enumerate(taus):
            y[:, k] = y0 if tau == 0.0 else expm(A * tau) @ y0
        return y

    def solve(
        self,
        y0: np.ndarray,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None = None,
        method: str = "RK45",
        linear_fast_path: bool = True,
        **kwargs,
    ) -> SimulationResult:
        """Solve the ODE system.

        Args:
            y0: Initial state vector (one value per active state).
            t_span: (t_start, t_end) time interval.
            t_eval: Times at which to record the solution.
            method: ODE solver method ('RK45', 'BDF', etc.).
            linear_fast_path: Use the exact matrix-exponential solution when
                the system is linear and time-invariant over `t_span`. Set
                False to force numerical integration (used by the tests that
                cross-check the two paths).

        Returns:
            SimulationResult with time points and population trajectories.
        """
        if linear_fast_path and t_eval is not None and not kwargs:
            t_eval_arr = np.asarray(t_eval, dtype=float)
            A = self._linear_generator(t_span, t_eval_arr)
            if A is not None:
                y = self._solve_linear(A, np.asarray(y0, float),
                                       float(t_span[0]), t_eval_arr)
                populations = {
                    ct.name: np.maximum(y[i], 0.0)
                    for i, ct in enumerate(self.topology.active_states)
                }
                return SimulationResult(
                    times=t_eval_arr,
                    populations=populations,
                    metadata={"method": "expm", "success": True,
                              "nfev": len(y0) + 4},
                )

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
