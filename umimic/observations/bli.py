"""Bioluminescence Imaging (BLI) observation model.

Models the photon flux measured by IVIS imaging:

    Y_BLI = alpha * N_luc * g(C_luc) * Att(depth) * epsilon

where:
    alpha: calibration constant (photons/sec per cell)
    N_luc: viable luciferase-expressing cells
    g(C_luc): luciferin kinetic factor
    Att: tissue optical attenuation
    epsilon: multiplicative lognormal noise
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from umimic.observations.base import ObservationModel
from umimic.pk.luciferin import LuciferinKinetics, TissueAttenuation


class BLIObservation(ObservationModel):
    """BLI observation model with luciferin kinetics and tissue attenuation.

    The key innovation of U-MIMIC over BESTDR for in vivo data: explicitly
    modeling the measurement physics prevents misinterpreting signal changes
    as biological effects.
    """

    def __init__(
        self,
        alpha: float = 1000.0,
        sigma_log: float = 0.3,
        luciferin: LuciferinKinetics | None = None,
        attenuation: TissueAttenuation | None = None,
        imaging_time_post_injection: float | None = None,
    ):
        """
        Args:
            alpha: Photons-per-cell calibration constant.
            sigma_log: Log-scale standard deviation of multiplicative noise.
            luciferin: Luciferin kinetics model (None = assume peak, g=1).
            attenuation: Tissue attenuation model (None = no attenuation).
            imaging_time_post_injection: Time of imaging after luciferin (minutes).
        """
        self.alpha = alpha
        self.sigma_log = sigma_log
        self.luciferin = luciferin
        self.attenuation = attenuation
        self.imaging_time = imaging_time_post_injection

    def _get_viable(self, latent_state: np.ndarray) -> float:
        """Extract viable (luciferase+) cells from state vector."""
        # P + Q + R (everything except A at index 2)
        if len(latent_state) >= 3:
            return max(float(np.sum(latent_state) - latent_state[2]), 1e-6)
        return max(float(np.sum(latent_state)), 1e-6)

    def _expected_signal(
        self,
        latent_state: np.ndarray,
        params: dict | None = None,
    ) -> float:
        """Compute expected BLI signal (without noise)."""
        N_luc = self._get_viable(latent_state)
        signal = self.alpha * N_luc

        # Luciferin kinetic factor
        if self.luciferin is not None and self.imaging_time is not None:
            g = self.luciferin.signal_fraction(self.imaging_time)
            signal *= g

        # Tissue attenuation
        if self.attenuation is not None:
            volume = params.get("tumor_volume") if params else None
            depth = params.get("tumor_depth") if params else None
            att = self.attenuation.attenuation_factor(depth=depth, volume=volume)
            signal *= att

        return max(signal, 1e-6)

    def log_likelihood(
        self,
        observed: float | np.ndarray,
        latent_state: np.ndarray,
        params: dict | None = None,
        process_variance: float | None = None,
    ) -> float:
        """Log-likelihood under LogNormal model.

        Y_BLI ~ LogNormal(log(mu_BLI), sigma_log)
        """
        mu = self._expected_signal(latent_state, params)
        sigma = self.sigma_log
        if params and "sigma_log_bli" in params:
            sigma = params["sigma_log_bli"]

        observed = max(float(observed), 1e-6)
        return float(stats.lognorm.logpdf(observed, s=sigma, scale=mu))

    def sample(
        self,
        latent_state: np.ndarray,
        rng: np.random.Generator,
        params: dict | None = None,
    ) -> float:
        """Sample a BLI observation."""
        mu = self._expected_signal(latent_state, params)
        return float(rng.lognormal(np.log(mu), self.sigma_log))

    def expected_value(self, latent_state: np.ndarray) -> float:
        return self._expected_signal(latent_state)

    def param_names(self) -> list[str]:
        return ["alpha", "sigma_log"]
