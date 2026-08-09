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
        lod: float = 0.0,
    ):
        """
        Args:
            beta: Volume per cell (mm^3/cell). Physically plausible values run
                from about 1e-6 (densely packed cells) to 1e-5 (tumour tissue
                including stroma, ~1e5 cells/mm^3). A 100 mm^3 tumour then
                corresponds to roughly 1e7 cells.
            sigma_v: Log-scale standard deviation of volume measurement noise.
            topology: Model topology, used to identify viable states.
            lod: Limit of detection (mm^3). Readings at or below this are
                scored as left-censored, ``log P(Y <= lod)``, rather than as
                exact values -- which is how a non-palpable tumour recorded as
                0 should be treated. 0 disables censoring; a non-positive
                reading is then an error, because a lognormal assigns zero
                probability at or below 0 and censoring there would contribute
                -inf.
        """
        TopologyAwareObservation.__init__(self, topology)
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}.")
        if not np.isfinite(sigma_v) or sigma_v <= 0:
            raise ValueError(f"sigma_v must be positive, got {sigma_v}.")
        if not np.isfinite(lod) or lod < 0:
            raise ValueError(f"lod must be finite and non-negative, got {lod}.")
        self.beta = beta
        self.sigma_v = sigma_v
        self.lod = float(lod)

    def _lod(self, params: dict | None = None) -> float:
        """Limit of detection, overridable per call via params."""
        if params and "volume_lod" in params:
            return float(params["volume_lod"])
        return self.lod

    def _get_viable(self, latent_state: np.ndarray) -> float:
        """Extract viable cells from state vector."""
        return self._project("viable", latent_state)

    def _sigma(self, params: dict | None = None) -> float:
        """Log-scale measurement SD (inference key ``sigma_v``)."""
        sigma = self.sigma_v
        if params and "sigma_v" in params:
            sigma = float(params["sigma_v"])
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"sigma_v must be positive, got {sigma}.")
        return sigma

    def _expected_volume(self, latent_state: np.ndarray) -> float:
        """Expected tumor volume (mm^3); median of the lognormal observation."""
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

        Readings at or below ``lod`` are treated as **left-censored**, not as
        exact zeros: the contribution is ``log P(Y <= lod)``. A pre-palpable
        tumour is normal in an efficacy study, and a lognormal density at zero
        is undefined -- this used to raise and abort the entire fit at the
        first such point.
        """
        mu_v = self._expected_volume(latent_state)
        sigma = self._with_process_variance(
            self._sigma(params), latent_state, process_variance
        )

        obs_val = float(observed)
        if not np.isfinite(obs_val):
            raise ValueError(
                f"Tumor volume observations must be finite, got {observed!r}. "
                "Use NaN for a missing measurement so it is masked out."
            )
        lod = self._lod(params)
        if lod > 0 and obs_val <= lod:
            return float(stats.lognorm.logcdf(lod, s=sigma, scale=mu_v))
        if obs_val <= 0:
            # P(Y <= 0) is exactly 0 under a lognormal, so censoring at zero
            # would contribute -inf and kill the fit just as surely as raising.
            # A recorded zero means "below what the calipers resolve", which is
            # a positive number the user has to supply.
            raise ValueError(
                f"Tumor volume observation {observed!r} is not positive, and "
                "no limit of detection is set (lod=0), so it cannot be scored: "
                "a lognormal puts zero probability at or below 0. Pass "
                "lod=<smallest measurable volume> to treat such readings as "
                "left-censored, or NaN to mark the point missing."
            )
        return float(stats.lognorm.logpdf(obs_val, s=sigma, scale=mu_v))

    def sample(
        self,
        latent_state: np.ndarray,
        rng: np.random.Generator,
        params: dict | None = None,
        process_variance: float | None = None,
    ) -> float:
        """Sample a tumor volume (lognormal; optional process variance)."""
        mu_v = self._expected_volume(latent_state)
        sigma = self._with_process_variance(
            self._sigma(params), latent_state, process_variance
        )
        draw = float(rng.lognormal(np.log(mu_v), sigma))
        # Censor at the detection limit so simulated data has the same
        # shape as real data, and so posterior predictive checks compare
        # like with like against the censored likelihood.
        lod = self._lod(params)
        return 0.0 if lod > 0 and draw <= lod else draw

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

        sigma = self._sigma(params)
        return EKFUpdate(
            z=float(np.log(obs_val)),
            z_pred=float(np.log(self.beta * n_viable)),
            H=H / n_viable,
            R=float(sigma**2),
            scale="log",
        )

    def param_names(self) -> list[str]:
        # beta is a fixed calibration; sigma_v is the free noise key used by
        # ModelLikelihood / OBSERVATION_PARAM_NAMES.
        return ["beta", "sigma_v"]
