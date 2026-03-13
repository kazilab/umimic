"""Dosing schedule definitions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Dose:
    """A single drug administration event."""

    time: float
    amount: float
    route: str = "iv_bolus"  # "iv_bolus", "iv_infusion", "oral"
    duration: float = 0.0  # for infusions (hours)


@dataclass
class DosingSchedule:
    """Complete dosing schedule for an experiment."""

    doses: list[Dose] = field(default_factory=list)

    @classmethod
    def constant_invitro(cls, concentration: float) -> DosingSchedule:
        """In vitro constant concentration (no PK needed).

        This is a sentinel that tells ExposureProfile to return a constant.
        """
        schedule = cls(doses=[])
        schedule._constant_concentration = concentration
        return schedule

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
        return hasattr(self, "_constant_concentration")

    @property
    def constant_concentration(self) -> float | None:
        return getattr(self, "_constant_concentration", None)

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
