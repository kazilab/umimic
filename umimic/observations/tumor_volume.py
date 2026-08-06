"""Tumor volume observation model for in vivo studies."""

from __future__ import annotations

import numpy as np
from scipy import stats

from umimic.dynamics.states import ModelTopology
from umimic.observations.base import (
    EKFUpdate,
    ObservationModel,
    TopologyAwareObservation,
)


class TumorVolumeObservation(TopologyAwareObservation, ObservationModel):
    """Tumor volume observation model.

    ``log(Y_V) ~ Normal(log(beta * N_viable), sigma_V)``

    where:
        beta: cell-to-volume conversion factor (mm^3 per cell)
        N_viable: number of viable cells (all non-apoptotic states)
        sigma_V: measurement noise (log-scale)

    Scale convention: ``beta * N_viable`` is the **median** volume, not the
    mean. The mean is ``beta * N_viable * exp(sigma_v**2 / 2)``.

    Caliper measurements are inherently noisy due to irregular tumor shape,
    skin thickness, and operator variability.
    """

    #: Canonical modality key used by data schemas and configuration.
    modality_name = "volume"

    def __init__(
        self,
        beta: float = 1e-5,
        sigma_v: float = 0.2,
        topology: ModelTopology | None = None,
    ):
        """
        Args:
            beta: Volume per cell (mm^3/cell). Physically plausible values run
                from about 1e-6 (densely packed cells) to 1e-5 (tumour tissue
                including stroma, ~1e5 cells/mm^3). A 100 mm^3 tumour then
                corresponds to roughly 1e7 cells.
            sigma_v: Log-scale standard deviation of volume measurement noise.
            topology: Model topology, used to identify viable states.
        """
        TopologyAwareObservation.__init__(self, topology)
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}.")
        if not np.isfinite(sigma_v) or sigma_v <= 0:
            raise ValueError(f"sigma_v must be positive, got {sigma_v}.")
        self.beta = beta
        self.sigma_v = sigma_v

    def _get_viable(self, latent_state: np.ndarray) -> float:
        """Extract viable cells from state vector."""
        return self._project("viable", latent_state)

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
        """Log-likelihood under the LogNormal volume model.

        An LNA process variance for the viable population is folded into the
        log-scale variance via the delta method, as for BLI, so the
        mechanistic variance informs this modality too.
        """
        mu_v = self._expected_volume(latent_state)
        sigma = self.sigma_v
        if params and "sigma_v" in params:
            sigma = float(params["sigma_v"])
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"sigma_v must be positive, got {sigma}.")

        sigma = self._with_process_variance(sigma, latent_state, process_variance)

        obs_val = float(observed)
        if not np.isfinite(obs_val) or obs_val <= 0:
            raise ValueError(
                f"Tumor volume observations must be finite and strictly "
                f"positive under a lognormal model, got {observed!r}."
            )
        return float(stats.lognorm.logpdf(obs_val, s=sigma, scale=mu_v))

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

    def linearize(
        self,
        observed: float,
        latent_state: np.ndarray,
        params: dict | None = None,
    ) -> EKFUpdate | None:
        """Exact linearization on the log-volume scale.

        The noise is Gaussian in log space, not in mm^3, so the update is
        formed there: ``log Y ~ Normal(log(beta * N), sigma_v)``. Since the
        median is proportional to N, ``d log(h)/dx = H_viable / N`` and no
        approximation beyond the usual EKF linearization is involved.
        """
        obs_val = float(observed)
        if not np.isfinite(obs_val) or obs_val <= 0:
            return None

        x = np.asarray(latent_state, dtype=float)
        H = self.operator("viable", x.shape[-1])
        n_viable = self._get_viable(x)
        if not np.isfinite(n_viable) or n_viable <= 0:
            return None

        sigma = self.sigma_v
        if params and "sigma_v" in params:
            sigma = float(params["sigma_v"])
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"sigma_v must be positive, got {sigma}.")

        return EKFUpdate(
            z=float(np.log(obs_val)),
            z_pred=float(np.log(self.beta * n_viable)),
            H=H / n_viable,
            R=float(sigma**2),
            scale="log",
        )

    def param_names(self) -> list[str]:
        return ["beta", "sigma_v"]
