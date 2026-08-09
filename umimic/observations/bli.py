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

from umimic.dynamics.states import ModelTopology
from umimic.observations.base import (
    EKFUpdate,
    ObservationModel,
    TopologyAwareObservation,
)
from umimic.pk.luciferin import LuciferinKinetics, TissueAttenuation


class BLIObservation(TopologyAwareObservation, ObservationModel):
    """BLI observation model with luciferin kinetics and tissue attenuation.

    The key innovation of U-MIMIC over BESTDR for in vivo data: explicitly
    modeling the measurement physics prevents misinterpreting signal changes
    as biological effects.
    """

    #: Canonical modality key used by data schemas and configuration.
    modality_name = "bli"

    def __init__(
        self,
        alpha: float = 1000.0,
        sigma_log: float = 0.3,
        luciferin: LuciferinKinetics | None = None,
        attenuation: TissueAttenuation | None = None,
        imaging_time_post_injection: float | None = None,
        topology: ModelTopology | None = None,
        lod: float = 0.0,
    ):
        """
        Args:
            alpha: Photons-per-cell calibration constant.
            sigma_log: Log-scale standard deviation of multiplicative noise.
            luciferin: Luciferin kinetics model (None = assume peak, g=1).
            attenuation: Tissue attenuation model (None = no attenuation).
            imaging_time_post_injection: Time of imaging after luciferin (minutes).
            topology: Model topology, used to identify viable states.
            lod: Limit of detection in the same units as the signal (photon
                flux). Readings at or below it are scored as left-censored
                rather than exact. 0 disables censoring, in which case a
                non-positive reading is an error rather than a silent -inf.
        """
        TopologyAwareObservation.__init__(self, topology)
        if not np.isfinite(lod) or lod < 0:
            raise ValueError(f"lod must be finite and non-negative, got {lod}.")
        self.lod = float(lod)
        if not np.isfinite(sigma_log) or sigma_log <= 0:
            raise ValueError(f"sigma_log must be positive, got {sigma_log}.")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}.")
        self.alpha = alpha
        self.sigma_log = sigma_log
        self.luciferin = luciferin
        self.attenuation = attenuation
        self.imaging_time = imaging_time_post_injection

    def _get_viable(self, latent_state: np.ndarray) -> float:
        """Extract viable (luciferase+) cells from state vector."""
        return self._project("viable", latent_state)

    def _lod(self, params: dict | None = None) -> float:
        """Limit of detection, overridable per call via params."""
        if params and "bli_lod" in params:
            return float(params["bli_lod"])
        return self.lod

    def _sigma(self, params: dict | None = None) -> float:
        """Log-scale measurement SD; inference key is ``sigma_log_bli``.

        Accepts the legacy alias ``sigma_log`` for backwards compatibility.
        """
        sigma = self.sigma_log
        if params:
            if "sigma_log_bli" in params:
                sigma = float(params["sigma_log_bli"])
            elif "sigma_log" in params:
                sigma = float(params["sigma_log"])
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"sigma_log_bli must be positive, got {sigma}.")
        return sigma

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

        # Tissue attenuation.
        #
        # Paired-volume coupling: when a tumour volume measurement is available
        # at the same time point, the caller passes it as params["tumor_volume"]
        # and the attenuation model converts it to an effective optical depth.
        # This is the mechanism by which the volume modality informs the BLI
        # modality; without it the two are only coupled through the shared
        # latent state. params["tumor_depth"] overrides the volume-derived
        # depth when a direct measurement exists.
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
        """Log-likelihood under the LogNormal model.

        ``log(Y_BLI) ~ Normal(log(mu_BLI), sigma_log)``.

        Scale convention: ``mu_BLI`` is the **median** of the observation, not
        its mean. The mean is ``mu_BLI * exp(sigma_log**2 / 2)``. Expected
        values reported by :meth:`expected_value` are medians on this scale.

        When the LNA supplies a process variance for the viable population it
        is folded into the log-scale variance, so the mechanistic variance
        signature informs BLI as well as counts. Because the signal is
        proportional to the viable count, the delta method gives a log-scale
        contribution of ``Var_process / N_viable**2``.
        """
        mu = self._expected_signal(latent_state, params)
        sigma = self._with_process_variance(
            self._sigma(params), latent_state, process_variance
        )

        obs_val = float(observed)
        if not np.isfinite(obs_val):
            raise ValueError(
                f"BLI observations must be finite, got {observed!r}. Use NaN "
                "for a missing measurement so it is masked out."
            )
        lod = self._lod(params)
        if lod > 0 and obs_val <= lod:
            # Left-censored: below background/dark-count. Scoring log P(Y <= lod)
            # keeps the point in the fit instead of aborting on it, which is
            # what a below-threshold IVIS reading used to do.
            return float(stats.lognorm.logcdf(lod, s=sigma, scale=mu))
        if obs_val <= 0:
            raise ValueError(
                f"BLI observation {observed!r} is not positive, and no limit of "
                "detection is set (lod=0), so it cannot be scored: a lognormal "
                "puts zero probability at or below 0. Pass lod=<background "
                "photon flux> to treat such readings as left-censored, or NaN "
                "to mark the point missing."
            )
        return float(stats.lognorm.logpdf(obs_val, s=sigma, scale=mu))

    def log_likelihood_batch(
        self,
        observed: np.ndarray,
        latent_states: np.ndarray,
        params: dict | None = None,
        process_variances: np.ndarray | None = None,
    ) -> float:
        """Batch log-likelihood with **per-time-point** paired covariates.

        Attenuation depends on tumour size, so it must be evaluated at each
        time point's volume. Collapsing a growing tumour to its mean volume
        applies one constant attenuation throughout -- flattening exactly the
        size-dependence the model exists to capture, and introducing a
        systematic tilt (over-correcting early, under-correcting late) that
        can read as a biological trend. Array-valued entries in `params` are
        therefore indexed per point.
        """
        observed = np.asarray(observed, dtype=float)
        states = np.atleast_2d(np.asarray(latent_states, dtype=float))
        n = len(observed)

        per_point_keys = [
            key
            for key, value in (params or {}).items()
            if isinstance(value, np.ndarray) and value.shape == (n,)
        ]
        if not per_point_keys:
            return super().log_likelihood_batch(
                observed, states, params, process_variances
            )

        total = 0.0
        for i in range(n):
            local = dict(params)
            for key in per_point_keys:
                local[key] = float(params[key][i])
            pv = (
                float(process_variances[i])
                if process_variances is not None
                else None
            )
            value = self.log_likelihood(observed[i], states[i], local, pv)
            if not np.isfinite(value):
                return -np.inf
            total += value
        return float(total)

    def sample(
        self,
        latent_state: np.ndarray,
        rng: np.random.Generator,
        params: dict | None = None,
        process_variance: float | None = None,
    ) -> float:
        """Sample a BLI observation (lognormal; optional process variance)."""
        mu = self._expected_signal(latent_state, params)
        sigma = self._with_process_variance(
            self._sigma(params), latent_state, process_variance
        )
        draw = float(rng.lognormal(np.log(mu), sigma))
        # Censor at the detection limit so simulated data has the same
        # shape as real data, and so posterior predictive checks compare
        # like with like against the censored likelihood.
        lod = self._lod(params)
        return 0.0 if lod > 0 and draw <= lod else draw

    def linearize(
        self,
        observed: float,
        latent_state: np.ndarray,
        params: dict | None = None,
    ) -> EKFUpdate | None:
        """Exact linearization on the log-signal scale.

        As for volume, the noise is Gaussian in log space. The expected signal
        carries the luciferin and attenuation factors, but both are constant
        in the state, so they cancel from ``d log(h)/dx = H_viable / N``.
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
            z_pred=float(np.log(self._expected_signal(x, params))),
            H=H / n_viable,
            R=float(sigma**2),
            scale="log",
        )

    def expected_value(self, latent_state: np.ndarray) -> float:
        return self._expected_signal(latent_state)

    def param_names(self) -> list[str]:
        # alpha is a fixed calibration; sigma_log_bli is the free noise key
        # used by ModelLikelihood / OBSERVATION_PARAM_NAMES.
        return ["alpha", "sigma_log_bli"]
