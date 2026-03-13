"""Negative Binomial observation model for cell counts."""

from __future__ import annotations

import numpy as np
from scipy import stats

from umimic.observations.base import ObservationModel


class CellCountObservation(ObservationModel):
    """Negative Binomial observation model for viable cell counts.

    Y ~ NegBin(mean = viable_cells, overdispersion = phi)

    The parameterization uses mean (mu) and overdispersion (phi):
        Var(Y) = mu + mu^2 / phi

    Higher phi -> less overdispersion (approaches Poisson as phi -> inf).
    Lower phi -> more overdispersion.
    """

    def __init__(
        self,
        overdispersion: float = 10.0,
        count_type: str = "viable",
    ):
        """
        Args:
            overdispersion: NegBin overdispersion parameter phi.
            count_type: What to count - "viable" (P+Q+R), "total" (P+Q+A+R),
                        "proliferating" (P only), "dead" (A only).
        """
        self.overdispersion = overdispersion
        self.count_type = count_type

    def _get_mean(self, latent_state: np.ndarray) -> float:
        """Extract the expected count from latent state."""
        if self.count_type == "viable":
            # P + Q + R (everything except A at index 2)
            if len(latent_state) >= 3:
                return max(float(np.sum(latent_state) - latent_state[2]), 1e-6)
            return max(float(np.sum(latent_state)), 1e-6)
        elif self.count_type == "total":
            return max(float(np.sum(latent_state)), 1e-6)
        elif self.count_type == "proliferating":
            return max(float(latent_state[0]), 1e-6)
        elif self.count_type == "dead":
            if len(latent_state) >= 3:
                return max(float(latent_state[2]), 1e-6)
            return 1e-6
        return max(float(np.sum(latent_state)), 1e-6)

    def log_likelihood(
        self,
        observed: float | np.ndarray,
        latent_state: np.ndarray,
        params: dict | None = None,
        process_variance: float | None = None,
    ) -> float:
        """Log-likelihood for cell count observation.

        When process_variance is provided (from LNA moment equations), uses a
        Gaussian likelihood that incorporates the mechanistic variance signature.
        This is critical for identifiability: the process variance depends on
        (b + d) while the mean depends on (b - d), enabling separation of
        birth and death rates from count data alone.

        When process_variance is None, falls back to NegBin(mu, phi).
        """
        mu = self._get_mean(latent_state)

        if process_variance is not None and process_variance > 0:
            # LNA-informed likelihood: Gaussian with mechanistic variance
            # Total variance = process variance + measurement noise (Poisson-level)
            phi = self.overdispersion
            if params and "overdispersion" in params:
                phi = params["overdispersion"]
            # measurement noise: mu^2/phi (extra-Poisson technical noise)
            measurement_var = mu**2 / phi if phi > 0 else 0.0
            total_var = process_variance + measurement_var
            total_var = max(total_var, 1e-6)
            obs_val = float(observed)
            return float(stats.norm.logpdf(obs_val, loc=mu, scale=np.sqrt(total_var)))

        # Fallback: NegBin when no process variance available (ODE mode)
        phi = self.overdispersion
        if params and "overdispersion" in params:
            phi = params["overdispersion"]

        # scipy NegBin parameterization: n = phi, p = phi / (phi + mu)
        n = phi
        p = phi / (phi + mu)
        p = np.clip(p, 1e-10, 1 - 1e-10)

        observed_int = max(int(round(float(observed))), 0)
        return float(stats.nbinom.logpmf(observed_int, n, p))

    def sample(
        self,
        latent_state: np.ndarray,
        rng: np.random.Generator,
        params: dict | None = None,
    ) -> float:
        """Sample a cell count from NegBin(mu, phi)."""
        mu = self._get_mean(latent_state)
        phi = self.overdispersion
        if params and "overdispersion" in params:
            phi = params["overdispersion"]

        n = phi
        p = phi / (phi + mu)
        p = np.clip(p, 1e-10, 1 - 1e-10)

        return float(rng.negative_binomial(n, p))

    def expected_value(self, latent_state: np.ndarray) -> float:
        return self._get_mean(latent_state)

    def log_likelihood_batch(
        self,
        observed: np.ndarray,
        latent_states: np.ndarray,
        params: dict | None = None,
        process_variances: np.ndarray | None = None,
    ) -> float:
        """Vectorized log-likelihood across all time points.

        Evaluates all observations in a single scipy.stats call instead
        of looping per point, avoiding N separate Python→C round-trips.
        """
        # Extract means for all time points at once
        mus = np.array([self._get_mean(s) for s in latent_states])

        phi = self.overdispersion
        if params and "overdispersion" in params:
            phi = params["overdispersion"]

        if process_variances is not None:
            # LNA-informed: vectorized Gaussian likelihood
            valid = process_variances > 0
            if not np.any(valid):
                return 0.0

            measurement_var = mus ** 2 / phi if phi > 0 else np.zeros_like(mus)
            total_var = np.where(valid, process_variances + measurement_var, 1e-6)
            total_var = np.maximum(total_var, 1e-6)

            ll_array = stats.norm.logpdf(
                observed, loc=mus, scale=np.sqrt(total_var)
            )
            if not np.all(np.isfinite(ll_array)):
                return -np.inf
            return float(np.sum(ll_array))

        # Fallback: vectorized NegBin
        n = phi
        p = phi / (phi + mus)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        observed_int = np.maximum(np.rint(observed).astype(int), 0)

        ll_array = stats.nbinom.logpmf(observed_int, n, p)
        if not np.all(np.isfinite(ll_array)):
            return -np.inf
        return float(np.sum(ll_array))

    def param_names(self) -> list[str]:
        return ["overdispersion"]
