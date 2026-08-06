"""Negative Binomial observation model for cell counts."""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

from umimic.dynamics.states import ModelTopology
from umimic.observations.base import (
    EKFUpdate,
    ObservationModel,
    TopologyAwareObservation,
)

logger = logging.getLogger(__name__)


class CellCountObservation(TopologyAwareObservation, ObservationModel):
    """Negative Binomial observation model for viable cell counts.

    Y ~ NegBin(mean = viable_cells, overdispersion = phi)

    The parameterization uses mean (mu) and overdispersion (phi):
        Var(Y) = mu + mu^2 / phi

    Higher phi -> less overdispersion (approaches Poisson as phi -> inf).
    Lower phi -> more overdispersion.
    """

    #: Canonical modality key used by data schemas and configuration.
    modality_name = "cell_counts"

    def __init__(
        self,
        overdispersion: float = 10.0,
        count_type: str = "viable",
        topology: ModelTopology | None = None,
    ):
        """
        Args:
            overdispersion: NegBin overdispersion parameter phi (must be > 0).
            count_type: What to count - "viable" (all non-apoptotic states),
                        "total" (all states), "proliferating" (P), "dead" (A).
            topology: Model topology, used to map count_type onto state
                indices. Strongly recommended; without it the canonical
                [P, Q, A, R] ordering is assumed.
        """
        TopologyAwareObservation.__init__(self, topology)
        if not np.isfinite(overdispersion) or overdispersion <= 0:
            raise ValueError(
                f"overdispersion must be a positive finite value, "
                f"got {overdispersion}."
            )
        self.overdispersion = overdispersion
        self.count_type = count_type

    def _phi(self, params: dict | None) -> float:
        """Resolve the overdispersion parameter, rejecting invalid values."""
        phi = self.overdispersion
        if params and "overdispersion" in params:
            phi = float(params["overdispersion"])
        if not np.isfinite(phi) or phi <= 0:
            raise ValueError(
                f"overdispersion must be a positive finite value, got {phi}."
            )
        return phi

    def _get_mean(self, latent_state: np.ndarray) -> float:
        """Extract the expected count from latent state."""
        return self._project(self.count_type, latent_state)

    #: Below this expected count the Gaussian branch is a poor stand-in for
    #: the discrete, skewed count distribution, and the LNA that supplies the
    #: process variance is itself unreliable.
    SMALL_COUNT_THRESHOLD = 20.0
    _warned_small_counts = False

    def _warn_if_small(self, mu) -> None:
        """Warn once when the Gaussian branch is used at small counts.

        Matching the first two moments does not make a Gaussian a negative
        binomial. At mu = 5 with phi = 10 the Gaussian places ~3.4% of its
        mass below zero, and the count distribution's skew (~0.7) is absent
        entirely. This is also the regime where the linear noise
        approximation supplying the process variance breaks down, so the
        warning flags both at once.
        """
        smallest = float(np.min(mu))
        if smallest >= self.SMALL_COUNT_THRESHOLD or type(self)._warned_small_counts:
            return
        type(self)._warned_small_counts = True
        logger.warning(
            "Moment-mode likelihood evaluated at an expected count of %.3g "
            "(< %g). The Gaussian approximation to the negative binomial is "
            "poor at small counts -- it is symmetric, continuous, and places "
            "non-negligible mass below zero -- and the LNA process variance "
            "is itself unreliable there. Prefer mode='ode' (exact NegBin) or "
            "a particle filter for low-count data.",
            smallest,
            self.SMALL_COUNT_THRESHOLD,
        )

    @staticmethod
    def _total_variance(mu, phi, process_variance):
        """Observation variance, consistent across forward modes.

        Var(Y) = Var_process + mu + mu^2/phi

        The last two terms are the conditional observation variance of a
        negative binomial count given the latent population: `mu` is the
        Poisson counting term and `mu^2/phi` the technical overdispersion.
        Keeping `mu` matters because it is what makes the two forward modes
        the same statistical model: at zero process variance this reduces to
        the NegBin variance used in ODE mode, so switching between MLE (ODE)
        and MCMC (moment) no longer changes the likelihood surface for a
        reason unrelated to the science.
        """
        return process_variance + mu + mu**2 / phi

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
        if not np.isfinite(mu) or mu <= 0:
            return -np.inf

        phi = self._phi(params)
        obs_val = float(observed)
        if not np.isfinite(obs_val) or obs_val < 0:
            raise ValueError(
                f"Cell count observations must be finite and non-negative, "
                f"got {observed!r}."
            )

        if process_variance is not None and process_variance > 0:
            # LNA-informed likelihood: Gaussian with mechanistic variance.
            self._warn_if_small(mu)
            total_var = max(self._total_variance(mu, phi, process_variance), 1e-6)
            return float(stats.norm.logpdf(obs_val, loc=mu, scale=np.sqrt(total_var)))

        # Fallback: NegBin when no process variance available (ODE mode).
        # scipy NegBin parameterization: n = phi, p = phi / (phi + mu)
        p = np.clip(phi / (phi + mu), 1e-10, 1 - 1e-10)
        return float(stats.nbinom.logpmf(max(int(round(obs_val)), 0), phi, p))

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

    def linearize(
        self,
        observed: float,
        latent_state: np.ndarray,
        params: dict | None = None,
    ) -> EKFUpdate:
        """Gaussian approximation to NegBin(mu, phi) on the count scale.

        The observation is already linear in the state, so H is the viable
        operator itself and the only approximation is Gaussian noise with the
        negative binomial's variance mu + mu^2/phi.
        """
        x = np.asarray(latent_state, dtype=float)
        # count_type, not "viable": a model counting only P or only A has a
        # different operator, and reusing the viable one would silently score
        # the wrong compartment.
        H = self.operator(self.count_type, x.shape[-1])
        mu = self._get_mean(x)
        phi = self._phi(params)
        return EKFUpdate(
            z=float(observed),
            z_pred=mu,
            H=H,
            R=float(mu + mu**2 / phi),
        )

    def log_likelihood_batch(
        self,
        observed: np.ndarray,
        latent_states: np.ndarray,
        params: dict | None = None,
        process_variances: np.ndarray | None = None,
    ) -> float:
        """Vectorized log-likelihood across all time points.

        Evaluates all observations in a single scipy.stats call instead of
        looping per point. This is required to be numerically identical to
        summing :meth:`log_likelihood` over the same points, including the
        per-point choice between the Gaussian and NegBin branches.
        """
        observed = np.asarray(observed, dtype=float)
        mus = self._project_batch(self.count_type, latent_states)
        if not np.all(np.isfinite(mus)) or np.any(mus <= 0):
            return -np.inf

        if not np.all(np.isfinite(observed)) or np.any(observed < 0):
            raise ValueError(
                "Cell count observations must be finite and non-negative."
            )

        phi = self._phi(params)

        # Match the scalar path point by point: the Gaussian branch applies
        # only where a positive process variance is available.
        if process_variances is not None:
            pv = np.asarray(process_variances, dtype=float)
            gaussian = pv > 0
        else:
            gaussian = np.zeros(len(mus), dtype=bool)

        ll_array = np.empty(len(mus), dtype=float)

        if np.any(gaussian):
            self._warn_if_small(mus[gaussian])
            total_var = np.maximum(
                self._total_variance(mus[gaussian], phi, pv[gaussian]), 1e-6
            )
            ll_array[gaussian] = stats.norm.logpdf(
                observed[gaussian], loc=mus[gaussian], scale=np.sqrt(total_var)
            )

        if np.any(~gaussian):
            mu_nb = mus[~gaussian]
            p = np.clip(phi / (phi + mu_nb), 1e-10, 1 - 1e-10)
            obs_int = np.maximum(np.rint(observed[~gaussian]).astype(int), 0)
            ll_array[~gaussian] = stats.nbinom.logpmf(obs_int, phi, p)

        if not np.all(np.isfinite(ll_array)):
            return -np.inf
        return float(np.sum(ll_array))

    def param_names(self) -> list[str]:
        return ["overdispersion"]
