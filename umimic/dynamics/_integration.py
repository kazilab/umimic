"""Adaptive integration that respects explicit treatment boundaries."""

from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp

from umimic.pk.exposure import exposure_breakpoints


def solve_ivp_exposure(rhs, t_span, y0, *, exposure_fn, t_eval=None, **kwargs):
    """Integrate every dosing segment, independently of the output grid.

    A dormant population has zero derivatives before a delayed dose, so an
    adaptive solver can otherwise step across the complete exposure. At an
    interior segment end evaluate the left limit of the RHS; the next segment
    starts with the post-dose exposure. Population states stay continuous.
    """
    start, end = map(float, t_span)
    points = exposure_breakpoints(exposure_fn, start, end)
    if points.size == 0:
        return solve_ivp(rhs, t_span, y0, t_eval=t_eval, **kwargs)

    direction = 1 if end > start else -1
    if direction < 0:
        points = points[::-1]
    boundaries = np.concatenate(([start], points, [end]))
    requested = None if t_eval is None else np.asarray(t_eval, dtype=float)
    if requested is not None:
        if requested.ndim != 1 or not np.all(np.isfinite(requested)):
            raise ValueError("t_eval must be a finite 1-D array.")
        if np.any(direction * np.diff(requested) <= 0):
            raise ValueError("Values in t_eval are not properly sorted.")
        if np.any((requested < min(start, end)) | (requested > max(start, end))):
            raise ValueError("Values in t_eval are not within t_span.")

    state = np.asarray(y0, dtype=float)
    times, values = [], []
    dense_segments = []
    nfev = 0
    options = dict(kwargs)
    options["dense_output"] = True
    for index, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        interior_end = index < len(boundaries) - 2

        def segment_rhs(t, y):
            sample_time = np.nextafter(right, left) if interior_end and t == right else t
            return rhs(sample_time, y)

        solution = solve_ivp(
            segment_rhs, (left, right), state, **options
        )
        nfev += solution.nfev
        if not solution.success:
            return solution
        state = solution.y[:, -1]
        dense_segments.append((solution.t[-1], solution.sol))
        if requested is None:
            skip = 1 if index else 0
            times.append(solution.t[skip:])
            values.append(solution.y[:, skip:])
        else:
            after_left = direction * (requested - left) >= 0
            if index:
                after_left = direction * (requested - left) > 0
            segment_eval = requested[
                after_left & (direction * (requested - solution.t[-1]) <= 0)
            ]
            times.append(segment_eval)
            values.append(
                solution.sol(segment_eval)
                if len(segment_eval) else np.empty((len(state), 0))
            )
        if solution.status == 1:
            # A terminal event can stop before the first requested output
            # point of this segment. Keep earlier outputs and stop globally.
            break

    segment_ends = direction * np.array([end for end, _ in dense_segments])

    def dense_output(t):
        query = np.asarray(t, dtype=float)
        if query.ndim > 1:
            raise ValueError("Dense output times must be scalar or one-dimensional.")
        flat_query = np.atleast_1d(query)
        indices = np.clip(
            np.searchsorted(segment_ends, direction * flat_query, side="left"),
            0, len(dense_segments) - 1,
        )
        output = np.empty((len(state), len(flat_query)))
        for index in np.unique(indices):
            mask = indices == index
            output[:, mask] = dense_segments[index][1](flat_query[mask])
        return output[:, 0] if query.ndim == 0 else output

    return SimpleNamespace(
        t=np.concatenate(times), y=np.concatenate(values, axis=1),
        success=True, message="Integrated across exposure boundaries.", nfev=nfev,
        sol=dense_output,
    )
