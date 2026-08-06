"""Pharmacokinetic compartment models for in vivo drug exposure.

Integration contract
--------------------
The internal PK state is always propagated to every dose boundary (dose times
and infusion start/end times), independently of the requested observation
grid. Values at requested times are read off a dense interpolant within each
segment. Concentrations at a shared time are therefore identical whether the
caller asks for a sparse or a dense ``t_eval``.

A dose falling exactly on an evaluation time is applied *before* the value at
that time is recorded, i.e. reported concentrations are post-dose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from umimic.pk.dosing import DosingSchedule

VALID_ROUTES = ("iv_bolus", "iv_infusion", "oral")


def _validate_t_eval(t_eval: np.ndarray) -> np.ndarray:
    """Require a finite, sorted, non-empty evaluation grid."""
    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.ndim != 1 or t_eval.size == 0:
        raise ValueError("t_eval must be a non-empty 1-D array of times.")
    if not np.all(np.isfinite(t_eval)):
        raise ValueError("t_eval must contain only finite values.")
    if np.any(np.diff(t_eval) < 0):
        raise ValueError("t_eval must be sorted in non-decreasing order.")
    return t_eval


def _validate_routes(dosing: DosingSchedule, has_absorption: bool) -> None:
    """Reject dosing routes the model cannot represent."""
    for dose in dosing.doses:
        if dose.route not in VALID_ROUTES:
            raise ValueError(
                f"Unknown dosing route {dose.route!r}; expected one of {VALID_ROUTES}."
            )
        if dose.amount < 0:
            raise ValueError(f"Dose amount must be non-negative, got {dose.amount}.")
        if not np.isfinite(dose.time):
            raise ValueError("Dose times must be finite.")
        if dose.route == "oral" and not has_absorption:
            raise ValueError(
                "Oral dosing requires an absorption rate constant (ka); "
                "this model was constructed with ka=None."
            )
        if dose.route == "iv_infusion" and not dose.duration > 0:
            raise ValueError(
                "An 'iv_infusion' dose requires a positive duration; "
                f"got duration={dose.duration}."
            )


def _breakpoints(dosing: DosingSchedule, t_start: float, t_end: float) -> np.ndarray:
    """All times at which the PK right-hand side or state changes."""
    points = {t_start, t_end}
    for dose in dosing.doses:
        if t_start <= dose.time <= t_end:
            points.add(float(dose.time))
        if dose.route == "iv_infusion":
            stop = dose.time + dose.duration
            if t_start <= stop <= t_end:
                points.add(float(stop))
    return np.array(sorted(points), dtype=float)


def _instant_doses_at(dosing: DosingSchedule, t: float) -> tuple[float, float]:
    """Summed (oral, iv_bolus) amounts administered exactly at time t.

    Simultaneous doses accumulate; they must never overwrite one another.
    """
    oral = sum(d.amount for d in dosing.doses if d.route == "oral" and d.time == t)
    bolus = sum(d.amount for d in dosing.doses if d.route == "iv_bolus" and d.time == t)
    return float(oral), float(bolus)


def _infusion_rate_on(dosing: DosingSchedule, t_mid: float) -> float:
    """Total infusion amount-per-hour active at t_mid (segment interior)."""
    rate = 0.0
    for dose in dosing.doses:
        if dose.route == "iv_infusion" and dose.time <= t_mid < dose.time + dose.duration:
            rate += dose.amount / dose.duration
    return float(rate)


def _integrate_segments(
    dosing: DosingSchedule,
    t_eval: np.ndarray,
    y0: np.ndarray,
    rhs_factory,
    apply_oral,
    apply_bolus,
    readout,
) -> np.ndarray:
    """Propagate PK state across dose boundaries and sample at t_eval.

    Args:
        dosing: Dosing schedule.
        t_eval: Sorted evaluation times.
        y0: Initial state vector.
        rhs_factory: infusion_rate -> rhs(t, y) callable.
        apply_oral: (y, amount) -> None, adds to the absorption compartment.
        apply_bolus: (y, amount) -> None, adds to the central compartment.
        readout: y -> central concentration.

    Returns:
        Concentrations at t_eval.
    """
    dose_times = [d.time for d in dosing.doses]
    t_start = min([0.0, float(t_eval[0])] + dose_times)
    t_end = float(t_eval[-1])

    y = np.asarray(y0, dtype=float).copy()
    out = np.zeros_like(t_eval, dtype=float)
    filled = np.zeros(len(t_eval), dtype=bool)

    breaks = _breakpoints(dosing, t_start, t_end)

    for k, t_bp in enumerate(breaks):
        # Apply instantaneous doses landing exactly on this breakpoint.
        oral, bolus = _instant_doses_at(dosing, float(t_bp))
        if oral:
            apply_oral(y, oral)
        if bolus:
            apply_bolus(y, bolus)

        # Record post-dose values for evaluation times at this breakpoint.
        at_bp = (t_eval == t_bp) & ~filled
        if np.any(at_bp):
            out[at_bp] = readout(y)
            filled |= at_bp

        if k == len(breaks) - 1:
            break

        t_next = float(breaks[k + 1])
        rate = _infusion_rate_on(dosing, 0.5 * (float(t_bp) + t_next))

        # Always integrate the full segment, regardless of the output grid.
        sol = solve_ivp(
            rhs_factory(rate),
            (float(t_bp), t_next),
            y,
            method="LSODA",
            dense_output=True,
            rtol=1e-9,
            atol=1e-12,
        )
        if not sol.success:
            raise RuntimeError(
                f"PK integration failed on segment "
                f"[{t_bp:g}, {t_next:g}]: {sol.message}"
            )

        interior = (t_eval > t_bp) & (t_eval < t_next) & ~filled
        if np.any(interior):
            states = sol.sol(t_eval[interior])
            out[interior] = readout(states)
            filled |= interior

        # Carry the state at the exact segment end, not at the last t_eval point.
        y = sol.sol(t_next)

    # Evaluation times before the integration start (if any) see no drug.
    return np.maximum(out, 0.0)


@dataclass
class OneCompartmentPK:
    """One-compartment PK model.

    IV bolus: dC/dt = -ke * C
    Oral: dA/dt = -ka * A;  dC/dt = ka*A/Vd - ke*C
    IV infusion: dC/dt = rate/Vd - ke*C over the infusion window

    Parameters:
        vd: Volume of distribution (L or L/kg), must be positive
        ke: Elimination rate constant (1/h), must be non-negative
        ka: Absorption rate constant (1/h), for oral dosing
    """

    vd: float = 10.0  # L
    ke: float = 0.1   # 1/h (half-life ~7h)
    ka: float | None = None  # oral absorption (None = IV)

    def __post_init__(self) -> None:
        if not np.isfinite(self.vd) or self.vd <= 0:
            raise ValueError(f"vd must be a positive volume, got {self.vd}.")
        if not np.isfinite(self.ke) or self.ke < 0:
            raise ValueError(f"ke must be a non-negative rate constant, got {self.ke}.")
        if self.ka is not None and (not np.isfinite(self.ka) or self.ka <= 0):
            raise ValueError(f"ka must be a positive rate constant, got {self.ka}.")

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
        t_eval = _validate_t_eval(t_eval)

        if dosing.is_constant:
            return np.full_like(t_eval, dosing.constant_concentration, dtype=float)

        _validate_routes(dosing, has_absorption=self.ka is not None)

        if not dosing.doses:
            return np.zeros_like(t_eval, dtype=float)

        # Pure IV bolus into a one-compartment model has an exact analytical
        # superposition, which is both faster and trivially grid-invariant.
        if all(d.route == "iv_bolus" for d in dosing.doses):
            C = np.zeros_like(t_eval, dtype=float)
            for dose in dosing.doses:
                mask = t_eval >= dose.time
                dt = t_eval[mask] - dose.time
                C[mask] += (dose.amount / self.vd) * np.exp(-self.ke * dt)
            return C

        def rhs_factory(rate):
            def rhs(t, y):
                A_gut, C = y
                ka = self.ka or 0.0
                dA = -ka * A_gut
                dC = ka * A_gut / self.vd + rate / self.vd - self.ke * C
                return [dA, dC]
            return rhs

        def apply_oral(y, amount):
            y[0] += amount

        def apply_bolus(y, amount):
            y[1] += amount / self.vd

        return _integrate_segments(
            dosing,
            t_eval,
            np.zeros(2),
            rhs_factory,
            apply_oral,
            apply_bolus,
            readout=lambda y: np.asarray(y)[1],
        )

    @property
    def half_life(self) -> float:
        """Elimination half-life (hours)."""
        if self.ke == 0:
            return float("inf")
        return float(np.log(2) / self.ke)


@dataclass
class TwoCompartmentPK:
    """Two-compartment PK model with central and peripheral compartments.

    dC1/dt = -(ke + k12)*C1 + k21*C2*(V2/V1) + input(t)/V1
    dC2/dt = k12*C1*(V1/V2) - k21*C2

    Parameters:
        vc: Central compartment volume (L), must be positive
        vp: Peripheral compartment volume (L), must be positive
        cl: Clearance (L/h), must be non-negative
        q: Intercompartmental clearance (L/h), must be non-negative
        ka: Absorption rate for oral dosing (1/h)
    """

    vc: float = 10.0   # L
    vp: float = 20.0   # L
    cl: float = 1.0    # L/h
    q: float = 0.5     # L/h
    ka: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.vc) or self.vc <= 0:
            raise ValueError(f"vc must be a positive volume, got {self.vc}.")
        if not np.isfinite(self.vp) or self.vp <= 0:
            raise ValueError(f"vp must be a positive volume, got {self.vp}.")
        if not np.isfinite(self.cl) or self.cl < 0:
            raise ValueError(f"cl must be a non-negative clearance, got {self.cl}.")
        if not np.isfinite(self.q) or self.q < 0:
            raise ValueError(f"q must be a non-negative clearance, got {self.q}.")
        if self.ka is not None and (not np.isfinite(self.ka) or self.ka <= 0):
            raise ValueError(f"ka must be a positive rate constant, got {self.ka}.")

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
        """Compute concentration-time profile for the central compartment."""
        t_eval = _validate_t_eval(t_eval)

        if dosing.is_constant:
            return np.full_like(t_eval, dosing.constant_concentration, dtype=float)

        _validate_routes(dosing, has_absorption=self.ka is not None)

        if not dosing.doses:
            return np.zeros_like(t_eval, dtype=float)

        def rhs_factory(rate):
            def rhs(t, y):
                A_gut, C1, C2 = y
                ka = self.ka or 0.0
                dA = -ka * A_gut
                input_rate = ka * A_gut / self.vc + rate / self.vc
                dC1 = (
                    -(self.ke + self.k12) * C1
                    + self.k21 * C2 * (self.vp / self.vc)
                    + input_rate
                )
                dC2 = self.k12 * C1 * (self.vc / self.vp) - self.k21 * C2
                return [dA, dC1, dC2]
            return rhs

        def apply_oral(y, amount):
            y[0] += amount

        def apply_bolus(y, amount):
            y[1] += amount / self.vc

        return _integrate_segments(
            dosing,
            t_eval,
            np.zeros(3),
            rhs_factory,
            apply_oral,
            apply_bolus,
            readout=lambda y: np.asarray(y)[1],
        )
