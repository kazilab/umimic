"""Tumor volume observation model for in vivo studies."""

from __future__ import annotations

import numpy as np
from scipy import stats

from umimic.observations.base import ObservationModel


class TumorVolumeObservation(ObservationModel):
    """Tumor volume observation model.

    Y_V ~ LogNormal(log(beta * N_viable), sigma_V)

    where:
        beta: cell-to-volume conversion factor (mm^3 per cell)
        N_viable: number of viable cells (P + Q + R)
        sigma_V: measurement noise (log-scale)

    Caliper measurements are inherently noisy due to irregular tumor shape,
    skin thickness, and operator variability.
    """

    def __init__(
        self,
        beta: float = 1e-3,
        sigma_v: float = 0.2,
    ):
        """
        Args:
            beta: Volume per cell (mm^3/cell). Typical: 1e-4 to 1e-2.
            sigma_v: Log-scale standard deviation of volume measurement noise.
        """
        self.beta = beta
        self.sigma_v = sigma_v

    def _get_viable(self, latent_state: np.ndarray) -> float:
        """Extract viable cells from state vector."""
        if len(latent_state) >= 3:
            return max(float(np.sum(latent_state) - latent_state[2]), 1e-6)
        return max(float(np.sum(latent_state)), 1e-6)

    def _expected_volume(self, latent_state: np.ndarray) -> float:
        """Expected tumor volume (mm^3)."""
        return self.beta * self._get_viable(latent_state)

    def log_likelihood(
        self,
        observed: float | np.ndarray,
        latent_state: np.ndarray,
        params: dict | None = None,
        process_variance: float | None = None,
    ) -> float:
        """Log-likelihood under LogNormal volume model."""
        mu_v = self._expected_volume(latent_state)
        sigma = self.sigma_v
        if params and "sigma_v" in params:
            sigma = params["sigma_v"]

        observed = max(float(observed), 1e-6)
        return float(stats.lognorm.logpdf(observed, s=sigma, scale=mu_v))

    def sample(
        self,
        latent_state: np.ndarray,
        rng: np.random.Generator,
        params: dict | None = None,
    ) -> float:
        """Sample a tumor volume measurement."""
        mu_v = self._expected_volume(latent_state)
        return float(rng.lognormal(np.log(mu_v), self.sigma_v))

    def expected_value(self, latent_state: np.ndarray) -> float:
        return self._expected_volume(latent_state)

    def param_names(self) -> list[str]:
        return ["beta", "sigma_v"]
