"""Dosing schedule definitions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VALID_ROUTES = ("iv_bolus", "iv_infusion", "oral")


@dataclass
class Dose:
    """A single drug administration event.

    Times and durations are in **hours** (same unit as compartment PK).
    """

    time: float
    amount: float
    route: str = "iv_bolus"  # "iv_bolus", "iv_infusion", "oral"
    duration: float = 0.0  # for infusions (hours)

    def __post_init__(self) -> None:
        if not np.isfinite(self.time):
            raise ValueError(f"Dose time must be finite, got {self.time}.")
        if not np.isfinite(self.amount) or self.amount < 0:
            raise ValueError(
                f"Dose amount must be finite and non-negative, got {self.amount}."
            )
        if self.route not in VALID_ROUTES:
            raise ValueError(
                f"Unknown dosing route {self.route!r}; expected one of "
                f"{VALID_ROUTES}."
            )
        if not np.isfinite(self.duration) or self.duration < 0:
            raise ValueError(
                f"Dose duration must be finite and non-negative, got {self.duration}."
            )
        if self.route == "iv_infusion" and not self.duration > 0:
            raise ValueError(
                "An 'iv_infusion' dose requires a positive duration; "
                f"got duration={self.duration}."
            )


@dataclass
class DosingSchedule:
    """Complete dosing schedule for an experiment."""

    doses: list[Dose] = field(default_factory=list)
    #: When set, exposure is a constant in vitro concentration (no PK).
    fixed_concentration: float | None = None

    def __post_init__(self) -> None:
        if self.fixed_concentration is not None:
            c = self.fixed_concentration
            if not np.isfinite(c) or c < 0:
                raise ValueError(
                    f"fixed_concentration must be finite and non-negative, got {c}."
                )

    @classmethod
    def constant_invitro(cls, concentration: float) -> DosingSchedule:
        """In vitro constant concentration (no PK needed).

        This is a sentinel that tells ExposureProfile to return a constant.
        """
        return cls(doses=[], fixed_concentration=float(concentration))

    @classmethod
    def single_bolus(cls, dose_amount: float, time: float = 0.0) -> DosingSchedule:
        """Single IV bolus at specified time."""
        return cls(doses=[Dose(time=time, amount=dose_amount, route="iv_bolus")])

    @classmethod
    def repeated(
        cls,
        dose_amount: float,
        interval: float,
        n_doses: int,
        route: str = "iv_bolus",
        start_time: float = 0.0,
    ) -> DosingSchedule:
        """Repeated dosing at fixed intervals.

        Args:
            dose_amount: Amount per dose.
            interval: Time between doses (hours).
            n_doses: Total number of doses.
            route: Administration route.
            start_time: Time of first dose.
        """
        doses = [
            Dose(
                time=start_time + i * interval,
                amount=dose_amount,
                route=route,
            )
            for i in range(n_doses)
        ]
        return cls(doses=doses)

    @classmethod
    def oral_repeated(
        cls,
        dose_amount: float,
        interval: float = 24.0,
        n_doses: int = 7,
        start_time: float = 0.0,
    ) -> DosingSchedule:
        """Repeated oral dosing (e.g., daily for 7 days)."""
        return cls.repeated(dose_amount, interval, n_doses, "oral", start_time)

    @property
    def is_constant(self) -> bool:
        """True if this is an in vitro constant concentration."""
        return self.fixed_concentration is not None

    @property
    def constant_concentration(self) -> float | None:
        return self.fixed_concentration

    @property
    def total_dose(self) -> float:
        return sum(d.amount for d in self.doses)

    @property
    def n_doses(self) -> int:
        return len(self.doses)

    @property
    def duration(self) -> float:
        """Time span from first to last dose."""
        if not self.doses:
            return 0.0
        return self.doses[-1].time - self.doses[0].time
