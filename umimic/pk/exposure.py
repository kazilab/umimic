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

    Optional grid cache
    -------------------
    Call :meth:`precompute` with an explicit time grid to evaluate the PK
    once and interpolate thereafter. Peaks that fall between knots are not
    resolved -- that approximation error is the caller's responsibility,
    which is why caching is opt-in rather than automatic.
    """

    def __init__(
        self,
        pk_model: OneCompartmentPK | TwoCompartmentPK | None = None,
        dosing: DosingSchedule | None = None,
    ):
        self._pk_model = pk_model
        self._dosing = dosing
        self._constant = None
        self._cache_t: np.ndarray | None = None
        self._cache_c: np.ndarray | None = None

        if dosing is not None and dosing.is_constant:
            self._constant = dosing.constant_concentration

    @classmethod
    def constant(cls, concentration: float) -> ExposureProfile:
        """Create a constant exposure profile (in vitro)."""
        if not np.isfinite(concentration) or concentration < 0:
            raise ValueError(
                f"Constant concentration must be finite and non-negative, "
                f"got {concentration}."
            )
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

    def precompute(self, t_grid: np.ndarray) -> ExposureProfile:
        """Sample the PK curve on an explicit grid for fast evaluation.

        After this call, :meth:`concentration` and :meth:`__call__` use linear
        interpolation on ``(t_grid, C(t_grid))`` instead of re-solving the PK
        ODE at every query. Constant and zero profiles ignore the grid.

        **Approximation contract.** Linear interpolation under-resolves sharp
        post-bolus peaks and infusion corners that fall between knots. Include
        every dose and infusion start/stop time in ``t_grid`` (and enough
        points between them) if those features matter. Outside
        ``[t_grid[0], t_grid[-1]]`` the PK is solved exactly rather than
        extrapolated, so a query past the end of the grid costs a full solve
        but is never wrong.

        Args:
            t_grid: Strictly sorted 1-D evaluation times (hours).

        Returns:
            ``self``, for chaining.
        """
        t_grid = np.asarray(t_grid, dtype=float)
        if t_grid.ndim != 1 or t_grid.size < 2:
            raise ValueError("t_grid must be a 1-D array with at least two times.")
        if not np.all(np.isfinite(t_grid)):
            raise ValueError("t_grid must contain only finite values.")
        if np.any(np.diff(t_grid) <= 0):
            raise ValueError("t_grid must be strictly increasing.")

        if self._constant is not None:
            self._cache_t = t_grid
            self._cache_c = np.full_like(t_grid, float(self._constant))
            return self

        if self._pk_model is not None and self._dosing is not None:
            self._cache_t = t_grid
            self._cache_c = np.asarray(
                self._pk_model.solve(self._dosing, t_grid), dtype=float
            )
            return self

        self._cache_t = t_grid
        self._cache_c = np.zeros_like(t_grid)
        return self

    def clear_cache(self) -> None:
        """Drop any grid cache; subsequent queries solve the PK exactly again."""
        self._cache_t = None
        self._cache_c = None

    @property
    def has_cache(self) -> bool:
        return self._cache_t is not None and self._cache_c is not None

    def concentration(self, t: float | np.ndarray) -> float | np.ndarray:
        """Drug concentration at time t.

        Args:
            t: Time point(s) in hours.

        Returns:
            Concentration(s) at the requested time(s).
        """
        if self._cache_t is not None and self._cache_c is not None:
            query = np.asarray(t, dtype=float)
            values = np.interp(query, self._cache_t, self._cache_c)

            # Outside the cached span, interpolation has nothing to work with.
            # np.interp would clamp to the end values, which turns a decaying
            # profile into a constant infusion: with a 0-24 h cache, a query at
            # t=1000 h returned the t=24 h concentration instead of ~0. Adaptive
            # ODE solvers routinely probe just past the end of a grid, so this
            # is reached in normal use. Solve those points exactly instead.
            outside = (query < self._cache_t[0]) | (query > self._cache_t[-1])
            if np.any(outside):
                exact = self._concentration_uncached(np.atleast_1d(query)[
                    np.atleast_1d(outside)
                ])
                flat_values = np.atleast_1d(values).astype(float)
                flat_values[np.atleast_1d(outside)] = exact
                values = flat_values.reshape(np.shape(values))

            if np.ndim(t) == 0:
                return float(values)
            return values

        return self._concentration_uncached(t)

    def _concentration_uncached(
        self, t: float | np.ndarray
    ) -> float | np.ndarray:
        """Concentration without consulting the grid cache.

        Uses ``np.ndim(t) == 0`` rather than ``isinstance(t, np.ndarray)`` so
        that a list or a 0-d array is handled by shape, not by type: a list
        previously took the scalar branch and then failed in ``float()``.
        """
        scalar = np.ndim(t) == 0

        if self._constant is not None:
            if scalar:
                return float(self._constant)
            return np.full(np.shape(t), self._constant, dtype=float)

        if self._pk_model is not None and self._dosing is not None:
            result = np.asarray(
                self._pk_model.solve(self._dosing, np.atleast_1d(t)), dtype=float
            )
            if scalar:
                return float(result[0])
            return result.reshape(np.shape(t))

        # No drug
        if scalar:
            return 0.0
        return np.zeros(np.shape(t), dtype=float)

    def __call__(self, t: float) -> float:
        """Callable interface for use as exposure_fn in dynamics.

        Scalar only. This used to do ``float(result[0])`` on whatever came
        back, so an array argument silently collapsed to the concentration at
        its first time point -- every caller in dynamics/ passes this as a bare
        callable, so a vectorized call site would have applied C(t0) at every
        time with no error anywhere. Use :meth:`concentration` for arrays.
        """
        if np.ndim(t) != 0:
            raise TypeError(
                "ExposureProfile.__call__ takes a scalar time; got an array of "
                f"shape {np.shape(t)}. Use .concentration(t) for vectorized "
                "queries -- returning only the first element here would "
                "silently apply one concentration to every time point."
            )
        return float(self.concentration(t))

    @property
    def is_constant(self) -> bool:
        return self._constant is not None
