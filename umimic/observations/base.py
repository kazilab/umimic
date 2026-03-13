"""Abstract base class for observation models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from umimic.dynamics.states import StateVector


class ObservationModel(ABC):
    """Base class for all observation models.

    Each observation model defines:
    - How to compute the log-likelihood of observed data given latent state
    - How to generate synthetic observations from latent state
    - What parameters it uses
    """

    @abstractmethod
    def log_likelihood(
        self,
        observed: float | np.ndarray,
        latent_state: np.ndarray,
        params: dict | None = None,
        process_variance: float | None = None,
    ) -> float:
        """Log-likelihood of observed data given latent state.

        Args:
            observed: Observed value(s).
            latent_state: Latent cell population state vector [P, Q, A, R].
            params: Optional additional parameters.
            process_variance: Variance from the LNA moment equations (optional).
                When provided, observation models can use the mechanistic variance
                to inform the likelihood (e.g., separating birth/death rates).
        """
        ...

    @abstractmethod
    def sample(
        self,
        latent_state: np.ndarray,
        rng: np.random.Generator,
        params: dict | None = None,
    ) -> float | np.ndarray:
        """Generate a synthetic observation from the latent state.

        Args:
            latent_state: Latent cell population state vector.
            rng: Random number generator.
            params: Optional additional parameters.
        """
        ...

    @abstractmethod
    def param_names(self) -> list[str]:
        """Names of observation model parameters."""
        ...

    def expected_value(self, latent_state: np.ndarray) -> float:
        """Expected observation given latent state (for plotting/diagnostics)."""
        raise NotImplementedError

    def log_likelihood_batch(
        self,
        observed: np.ndarray,
        latent_states: np.ndarray,
        params: dict | None = None,
        process_variances: np.ndarray | None = None,
    ) -> float:
        """Batch log-likelihood across multiple time points.

        Default implementation loops over points. Subclasses can override
        with vectorized implementations for significant speedup.

        Args:
            observed: Array of observed values (n_times,).
            latent_states: Latent states (n_times, n_states).
            params: Optional additional parameters.
            process_variances: Process variances per time point (n_times,).

        Returns:
            Sum of log-likelihoods across all time points.
        """
        ll = 0.0
        for i in range(len(observed)):
            pv = float(process_variances[i]) if process_variances is not None else None
            ll_i = self.log_likelihood(observed[i], latent_states[i], params, pv)
            if not np.isfinite(ll_i):
                return -np.inf
            ll += ll_i
        return ll
