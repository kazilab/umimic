"""Allocate fractional initial mixtures to individual stochastic cells."""

import numpy as np


def integer_initial_state(state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Preserve integer counts or sample a mixture with the same total count."""
    state = np.asarray(state, dtype=float)
    if np.any(~np.isfinite(state)) or np.any(state < 0):
        raise ValueError("Initial populations must be finite and non-negative")
    total = float(state.sum())
    rounded = int(round(total))
    if not np.isclose(total, rounded, rtol=0, atol=1e-8):
        raise ValueError("Stochastic simulation requires an integer total initial population")
    if np.allclose(state, np.rint(state), rtol=0, atol=1e-10):
        return np.rint(state).astype(float)
    return rng.multinomial(rounded, state / total).astype(float)
