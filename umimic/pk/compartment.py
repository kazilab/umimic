"""Pharmacokinetic compartment models for in vivo drug exposure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from umimic.pk.dosing import DosingSchedule


@dataclass
class OneCompartmentPK:
    """One-compartment PK model.

    IV bolus: dC/dt = -ke * C
    Oral: dA/dt = -ka * A;  dC/dt = ka*A/Vd - ke*C

    Parameters:
        vd: Volume of distribution (L or L/kg)
        ke: Elimination rate constant (1/h)
        ka: Absorption rate constant (1/h), for oral dosing
    """

    vd: float = 10.0  # L
    ke: float = 0.1   # 1/h (half-life ~7h)
    ka: float | None = None  # oral absorption (None = IV)

    def solve(
        self,
        dosing: DosingSchedule,
        t_eval: np.ndarray,
    ) -> np.ndarray:
        """Compute concentration-time profile.

        Args:
            dosing: Dosing schedule.
            t_eval: Times at which to evaluate concentration.

        Returns:
            Array of concentrations at t_eval.
        """
        if dosing.is_constant:
            return np.full_like(t_eval, dosing.constant_concentration, dtype=float)

        t_max = float(t_eval[-1])

        if self.ka is None:
            # IV: analytical superposition of bolus doses
            C = np.zeros_like(t_eval, dtype=float)
            for dose in dosing.doses:
                mask = t_eval >= dose.time
                dt = t_eval[mask] - dose.time
                C0 = dose.amount / self.vd
                C[mask] += C0 * np.exp(-self.ke * dt)
            return C
        else:
            # Oral: ODE solution with absorption compartment
            return self._solve_oral_ode(dosing, t_eval, t_max)

    def _solve_oral_ode(
        self, dosing: DosingSchedule, t_eval: np.ndarray, t_max: float
    ) -> np.ndarray:
        """Solve oral PK via ODE integration."""
        dose_times = {d.time: d.amount for d in dosing.doses}

        def rhs(t, y):
            A_gut, C = y
            dA = -self.ka * A_gut
            dC = self.ka * A_gut / self.vd - self.ke * C
            return [dA, dC]

        # Event-driven integration: restart at each dose time
        all_times = sorted(set([0.0] + [d.time for d in dosing.doses] + [t_max]))
        y = np.array([0.0, 0.0])  # [gut amount, concentration]
        C_result = np.zeros_like(t_eval, dtype=float)

        for i in range(len(all_times) - 1):
            t_start = all_times[i]
            t_end = all_times[i + 1]

            # Add dose at t_start if applicable
            if t_start in dose_times:
                y[0] += dose_times[t_start]

            # Find t_eval points in this interval
            mask = (t_eval >= t_start) & (t_eval <= t_end)
            t_seg = t_eval[mask]

            if len(t_seg) > 0:
                sol = solve_ivp(
                    rhs, (t_start, t_end), y, t_eval=t_seg,
                    method="RK45", rtol=1e-8, atol=1e-10,
                )
                C_result[mask] = sol.y[1]
                y = sol.y[:, -1]
            else:
                sol = solve_ivp(
                    rhs, (t_start, t_end), y,
                    method="RK45", rtol=1e-8, atol=1e-10,
                )
                y = sol.y[:, -1]

        return C_result

    @property
    def half_life(self) -> float:
        """Elimination half-life (hours)."""
        return np.log(2) / self.ke


@dataclass
class TwoCompartmentPK:
    """Two-compartment PK model with central and peripheral compartments.

    dC1/dt = -(ke + k12)*C1 + k21*C2*(V2/V1) + input(t)/V1
    dC2/dt = k12*C1*(V1/V2) - k21*C2

    Parameters:
        vc: Central compartment volume (L)
        vp: Peripheral compartment volume (L)
        cl: Clearance (L/h)
        q: Intercompartmental clearance (L/h)
        ka: Absorption rate for oral dosing (1/h)
    """

    vc: float = 10.0   # L
    vp: float = 20.0   # L
    cl: float = 1.0    # L/h
    q: float = 0.5     # L/h
    ka: float | None = None

    @property
    def ke(self) -> float:
        return self.cl / self.vc

    @property
    def k12(self) -> float:
        return self.q / self.vc

    @property
    def k21(self) -> float:
        return self.q / self.vp

    def solve(
        self,
        dosing: DosingSchedule,
        t_eval: np.ndarray,
    ) -> np.ndarray:
        """Compute concentration-time profile for central compartment."""
        if dosing.is_constant:
            return np.full_like(t_eval, dosing.constant_concentration, dtype=float)

        t_max = float(t_eval[-1])
        dose_times = {d.time: d.amount for d in dosing.doses}

        n_compartments = 3 if self.ka is not None else 2

        def rhs(t, y):
            if n_compartments == 3:
                A_gut, C1, C2 = y
                dA = -self.ka * A_gut
                input_rate = self.ka * A_gut / self.vc
            else:
                C1, C2 = y
                input_rate = 0.0
                dA = 0.0

            dC1 = -(self.ke + self.k12) * C1 + self.k21 * C2 * (self.vp / self.vc) + input_rate
            dC2 = self.k12 * C1 * (self.vc / self.vp) - self.k21 * C2

            if n_compartments == 3:
                return [dA, dC1, dC2]
            return [dC1, dC2]

        all_times = sorted(set([0.0] + [d.time for d in dosing.doses] + [t_max]))
        y = np.zeros(n_compartments)
        C_result = np.zeros_like(t_eval, dtype=float)

        for i in range(len(all_times) - 1):
            t_start = all_times[i]
            t_end = all_times[i + 1]

            if t_start in dose_times:
                if self.ka is not None:
                    y[0] += dose_times[t_start]  # oral: to gut
                else:
                    y[0] += dose_times[t_start] / self.vc  # IV: directly to central

            mask = (t_eval >= t_start) & (t_eval <= t_end)
            t_seg = t_eval[mask]

            if len(t_seg) > 0:
                sol = solve_ivp(
                    rhs, (t_start, t_end), y, t_eval=t_seg,
                    method="RK45", rtol=1e-8, atol=1e-10,
                )
                # Central compartment concentration
                c1_idx = 1 if n_compartments == 3 else 0
                C_result[mask] = sol.y[c1_idx]
                y = sol.y[:, -1]
            else:
                sol = solve_ivp(
                    rhs, (t_start, t_end), y,
                    method="RK45", rtol=1e-8, atol=1e-10,
                )
                y = sol.y[:, -1]

        return np.maximum(C_result, 0.0)
