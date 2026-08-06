"""Unified exposure profile: constant (in vitro) or PK-driven (in vivo)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from umimic.pk.dosing import DosingSchedule

if TYPE_CHECKING:
    from umimic.pk.compartment import OneCompartmentPK, TwoCompartmentPK


class ExposureProfile:
    """Unified interface for drug concentration over time.

    In vitro: returns a constant concentration.
    In vivo: computes C(t) from a PK model + dosing schedule.

    This abstraction is the key unification between in vitro and in vivo —
    the dynamics module only needs to call exposure.concentration(t) and
    is agnostic to whether the exposure is constant or time-varying.
    """

    def __init__(
        self,
        pk_model: OneCompartmentPK | TwoCompartmentPK | None = None,
        dosing: DosingSchedule | None = None,
    ):
        self._pk_model = pk_model
        self._dosing = dosing
        self._constant = None

        if dosing is not None and dosing.is_constant:
            self._constant = dosing.constant_concentration

    @classmethod
    def constant(cls, concentration: float) -> ExposureProfile:
        """Create a constant exposure profile (in vitro)."""
        profile = cls(dosing=DosingSchedule.constant_invitro(concentration))
        return profile

    @classmethod
    def from_pk(
        cls,
        pk_model: OneCompartmentPK | TwoCompartmentPK,
        dosing: DosingSchedule,
    ) -> ExposureProfile:
        """Create a PK-driven exposure profile (in vivo)."""
        return cls(pk_model=pk_model, dosing=dosing)

    def concentration(self, t: float | np.ndarray) -> float | np.ndarray:
        """Drug concentration at time t.

        Args:
            t: Time point(s).

        Returns:
            Concentration(s) at the requested time(s).
        """
        if self._constant is not None:
            if isinstance(t, np.ndarray):
                return np.full_like(t, self._constant, dtype=float)
            return self._constant

        if self._pk_model is not None and self._dosing is not None:
            result = self._pk_model.solve(self._dosing, np.atleast_1d(t))
            if not isinstance(t, np.ndarray):
                return float(result[0])
            return result

        # No drug
        if isinstance(t, np.ndarray):
            return np.zeros_like(t, dtype=float)
        return 0.0

    def __call__(self, t: float) -> float:
        """Callable interface for use as exposure_fn in dynamics."""
        result = self.concentration(t)
        if isinstance(result, np.ndarray):
            return float(result[0])
        return float(result)

    @property
    def is_constant(self) -> bool:
        return self._constant is not None

    # `precompute(t_eval)` was removed in 0.0.4. It filled a cache that
    # `concentration` never read, so it promised a speed-up it did not
    # deliver, and wiring it up would have meant silently interpolating a PK
    # curve whose peaks the requested grid may not resolve. Callers wanting a
    # cheap repeated evaluation should sample `concentration` on their own
    # grid and interpolate explicitly, where the error is theirs to see.
