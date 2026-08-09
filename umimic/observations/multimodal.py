"""Composite multimodal observation model.

Combines multiple observation modalities under the assumption of
conditional independence given the latent state:

    p(Y_counts, Y_BLI, Y_volume | X) = p(Y_counts|X) * p(Y_BLI|X) * p(Y_volume|X)

This is the key multimodal fusion capability of U-MIMIC.

That factorization is exact only when X is known. Under the LNA, X is itself
uncertain, and BLI and tumour volume are deterministic functions of the *same*
viable population: a fluctuation in N moves both signals together. Handing the
same process variance to each marginal and multiplying counts one shared
fluctuation once per modality, which over-states the information in the data
and produces posteriors that are too narrow.

:meth:`MultimodalObservation.joint_log_likelihood` therefore integrates the
shared latent once, evaluating a single multivariate normal whose covariance
carries the induced correlation:

    Sigma_obs = J Sigma J^T + diag(sigma_m^2)

with ``J`` the stacked per-modality Jacobians ``d h_m / dx`` supplied by each
model's ``linearize``. Because those Jacobians are full state-space row
vectors, modalities observing different projections (viable vs total) get the
right cross-covariance automatically.
"""

from __future__ import annotations

import logging

import numpy as np

from umimic.observations.base import ObservationModel

logger = logging.getLogger(__name__)


class MultimodalObservation:
    """Composite observation model combining multiple modalities.

    Joint log-likelihood is the sum of individual log-likelihoods
    (conditional independence given latent state).

    Gracefully handles missing modalities: if a modality has no
    observation at a time point, it is simply excluded from the sum.
    """

    def __init__(self, models: dict[str, ObservationModel]):
        """
        Args:
            models: Dict mapping modality name to ObservationModel.
                   Example: {"cell_counts": CellCountObservation(...),
                             "bli": BLIObservation(...),
                             "volume": TumorVolumeObservation(...)}
        """
        self.models = models

    def log_likelihood(
        self,
        observations: dict[str, float | None],
        latent_state: np.ndarray,
        params: dict | None = None,
        process_variance: float | dict[str, float] | None = None,
    ) -> float:
        """Joint log-likelihood across all available modalities.

        Args:
            observations: Dict mapping modality name to observed value.
                         None or missing entries are skipped.
            latent_state: Latent cell population state vector.
            params: Additional parameters.
            process_variance: LNA process variance for the viable population,
                either one scalar shared by every modality or a per-modality
                dict. Dropping it here is what made "variance fusion" a
                property of the orchestrated likelihood path only: a caller
                using this composite directly got measurement noise alone, so
                the mechanistic birth-versus-death signature never reached any
                modality. Leave it None for a filter whose latent states are
                exact particle counts, where there is no LNA variance to fold
                in.
        """
        total = 0.0
        for name, model in self.models.items():
            if name in observations and observations[name] is not None:
                if isinstance(process_variance, dict):
                    variance = process_variance.get(name)
                else:
                    variance = process_variance
                ll = model.log_likelihood(
                    observations[name], latent_state, params, variance
                )
                if np.isfinite(ll):
                    total += ll
                else:
                    return -np.inf
        return total

    def joint_log_likelihood(
        self,
        observations: dict[str, float | None],
        latent_state: np.ndarray,
        covariance: np.ndarray,
        params: dict[str, dict] | None = None,
    ) -> float | None:
        """Joint log-likelihood integrating the shared latent state once.

        Builds one multivariate normal over every modality that supplies a
        Gaussian linearization, with
        ``Sigma_obs = J Sigma J^T + diag(R_m)``. The off-diagonal terms are the
        correlation induced by the modalities sharing a latent population;
        omitting them (i.e. multiplying the marginals) is what double-counts
        the process variance.

        Args:
            observations: Modality name -> observed scalar. Missing, None and
                non-finite entries are skipped.
            latent_state: Latent population state vector.
            covariance: Full (n_states, n_states) LNA state covariance. Not a
                projected scalar: the projection differs per modality and the
                cross-terms need the matrix.
            params: Modality name -> that model's parameter dict.

        Returns:
            The joint log-likelihood, or None when fewer than two modalities
            can be linearized. None means "no fusion applies here": the caller
            should fall back to the exact per-modality likelihoods, which are
            preferable for a single modality because they keep the true
            observation law (negative binomial counts, lognormal BLI) rather
            than a Gaussian approximation to it.
        """
        params = params or {}
        cov = np.asarray(covariance, dtype=float)

        names: list[str] = []
        z: list[float] = []
        z_pred: list[float] = []
        rows: list[np.ndarray] = []
        r_diag: list[float] = []
        scales: list[str] = []

        for name, model in self.models.items():
            value = observations.get(name)
            if value is None or not np.isfinite(value):
                continue
            update = model.linearize(float(value), latent_state, params.get(name))
            if update is None:
                # No Gaussian approximation (e.g. biomarker fractions). Such a
                # modality is not fused here; the caller scores it exactly.
                continue
            names.append(name)
            z.append(update.z)
            z_pred.append(update.z_pred)
            rows.append(np.asarray(update.H, dtype=float).ravel())
            r_diag.append(float(update.R))
            scales.append(update.scale)

        if len(names) < 2:
            return None

        J = np.vstack(rows)
        Sigma = J @ cov @ J.T + np.diag(r_diag)
        Sigma = (Sigma + Sigma.T) / 2.0

        resid = np.asarray(z, dtype=float) - np.asarray(z_pred, dtype=float)
        try:
            chol = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            # A non-PSD fused covariance means the linearization or the LNA
            # covariance is degenerate. Returning -inf would let an optimiser
            # treat this as a merely bad parameter value; None sends the caller
            # back to the exact per-modality path instead.
            logger.warning(
                "Fused observation covariance for %s is not positive definite; "
                "falling back to per-modality likelihoods at this point.",
                names,
            )
            return None

        y = np.linalg.solve(chol, resid)
        quad = float(y @ y)
        log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
        ll = -0.5 * (len(names) * np.log(2 * np.pi) + log_det + quad)

        # The normal above is a density in z. Modalities that linearize on the
        # log scale have z = log y, so returning it as-is would be a density in
        # log y while the exact per-modality path returns one in y. The
        # Jacobian d log y / d y = 1/y makes the two comparable -- necessary
        # because a single fit mixes fused and unfused time points, and because
        # AIC/BIC compare these sums across models.
        log_jacobian = -sum(
            zi for zi, scale in zip(z, scales) if scale == "log"
        )
        return float(ll + log_jacobian)

    def sample(
        self,
        latent_state: np.ndarray,
        rng: np.random.Generator,
        params: dict | None = None,
        process_variance: float | dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Generate synthetic observations from all modalities.

        ``process_variance`` may be a scalar shared by every modality or a
        per-modality dict, matching :meth:`log_likelihood`.
        """
        out: dict[str, float] = {}
        for name, model in self.models.items():
            if isinstance(process_variance, dict):
                variance = process_variance.get(name)
            else:
                variance = process_variance
            out[name] = model.sample(
                latent_state, rng, params, process_variance=variance
            )
        return out

    def expected_values(self, latent_state: np.ndarray) -> dict[str, float]:
        """Expected observation from each modality."""
        result = {}
        for name, model in self.models.items():
            try:
                result[name] = model.expected_value(latent_state)
            except NotImplementedError:
                pass
        return result

    @property
    def modality_names(self) -> list[str]:
        return list(self.models.keys())

    def param_names(self) -> list[str]:
        all_params = []
        for name, model in self.models.items():
            for p in model.param_names():
                all_params.append(f"{name}.{p}")
        return all_params
