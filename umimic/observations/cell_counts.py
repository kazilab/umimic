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
        overdispersion_key: str = "overdispersion",
        moment_family: str = "auto",
    ):
        """
        Args:
            overdispersion: NegBin overdispersion parameter phi (must be > 0).
            count_type: What to count - "viable" (all non-apoptotic states),
                        "total" (all states), "proliferating" (P), "dead" (A).
            topology: Model topology, used to map count_type onto state
                indices. Strongly recommended; without it the canonical
                [P, Q, A, R] ordering is assumed.
            overdispersion_key: Name this model reads its overdispersion from
                in the fitted parameter dict. Two count channels in one
                multimodal fit -- viable cells and dead cells, say -- are
                different measurements with different noise, and the dead
                channel is typically far noisier. Left on the shared default
                they compete for a single phi and the fit splits the
                difference, degrading both. Give the second channel its own
                key (and add it to `param_names`) to estimate them separately.
            moment_family: Distribution used in moment mode once a process
                variance is available. "auto" uses a negative binomial whose
                size parameter is inflated to carry the LNA variance, which
                keeps the likelihood discrete and right-skewed. "gaussian"
                forces the symmetric Gaussian.

                This exists because the two are not interchangeable and the
                package does not use one everywhere: the *fused* multimodal
                path builds a joint Gaussian MVN (there is no tractable
                multivariate negative binomial to carry cross-modality
                correlation), while a single modality goes through this
                method. Comparing a marginal against a joint therefore
                compares two distribution families unless this is set to
                "gaussian" on both sides. Prefer "auto" for fitting; use
                "gaussian" when a diagnostic must line up with the fused path.
        """
        if moment_family not in ("auto", "gaussian"):
            raise ValueError(
                f"moment_family must be 'auto' or 'gaussian', got {moment_family!r}."
            )
        self.moment_family = moment_family
        TopologyAwareObservation.__init__(self, topology)
        if not np.isfinite(overdispersion) or overdispersion <= 0:
            raise ValueError(
                f"overdispersion must be a positive finite value, "
                f"got {overdispersion}."
            )
        self.overdispersion = overdispersion
        self.count_type = count_type
        self.overdispersion_key = overdispersion_key

    def _phi(self, params: dict | None) -> float:
        """Resolve the overdispersion parameter, rejecting invalid values."""
        phi = self.overdispersion
        if params and self.overdispersion_key in params:
            phi = float(params[self.overdispersion_key])
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

    #: Skew of the *total* observation distribution above which a symmetric
    #: likelihood is not an acceptable stand-in. 0.2 is roughly where the
    #: induced bias in a fitted variance term stops being negligible.
    SKEW_THRESHOLD = 0.2

    _warned_small_counts = False
    _warned_skew = False

    @staticmethod
    def _total_skew(mu, phi, process_variance):
        """Skewness of Var_process (Gaussian) convolved with NegBin(mu, phi).

        The LNA term contributes no third central moment, so the skew of the
        sum is the negative binomial's third moment divided by the *total*
        variance to the 3/2. A large process variance therefore genuinely
        Gaussianises the total -- but a large technical overdispersion does
        not, it makes things worse.
        """
        p = np.clip(phi / (phi + mu), 1e-12, 1 - 1e-12)
        var_obs = mu + mu**2 / phi
        m3_obs = (2 - p) / np.sqrt(phi * (1 - p)) * var_obs**1.5
        total = np.maximum(process_variance + var_obs, 1e-12)
        return m3_obs / total**1.5

    def _warn_if_small(self, mu, phi=None, process_variance=None) -> None:
        """Warn once when a symmetric likelihood is a poor stand-in.

        Two distinct failure modes, and the second was previously unguarded.

        Small counts: at mu = 5 with phi = 10 a Gaussian places ~3.4% of its
        mass below zero, and the LNA supplying the process variance is itself
        unreliable there.

        Skewed counts at *large* mu: negative binomial skew is set by phi, not
        by mu -- at mu = 4000 the skew is 0.63 for phi = 10 and 0.016 for
        phi = 1e7. Guarding only on mu stays silent in exactly the regime
        where a symmetric likelihood is worst, and the resulting misfit is
        absorbed by whatever variance term is free -- in moment mode that is
        the mechanistic process variance, so b + d is biased upward and the
        bias grows with sample size rather than averaging out.
        """
        smallest = float(np.min(mu))
        if smallest < self.SMALL_COUNT_THRESHOLD and not type(self)._warned_small_counts:
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

        if phi is None or process_variance is None or type(self)._warned_skew:
            return
        skew = float(np.max(np.abs(self._total_skew(mu, phi, process_variance))))
        if skew <= self.SKEW_THRESHOLD:
            return
        type(self)._warned_skew = True
        logger.warning(
            "Moment-mode observation distribution has skew %.3g (> %g) at "
            "overdispersion phi=%.3g. Skew is set by phi, not by the count "
            "size, so this fires at large counts too. A symmetric likelihood "
            "cannot represent it, and the misfit is absorbed by the fitted "
            "process variance -- biasing birth/death turnover upward, with "
            "the bias growing as more data are added. Raise phi (cleaner "
            "counts) or treat any b+d estimate from this fit as unreliable.",
            skew,
            self.SKEW_THRESHOLD,
            float(np.min(phi)),
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

    @staticmethod
    def _inflated_phi(mu, total_var):
        """NegBin size parameter reproducing `total_var` at mean `mu`.

        Solving mu + mu^2/phi_eff = Var_total gives phi_eff = mu^2 /
        (Var_total - mu). This lets moment mode keep a *count* likelihood --
        discrete, non-negative and right-skewed -- while still carrying the
        LNA process variance, instead of swapping in a Gaussian that matches
        the first two moments and silently drops the third.

        Returns None where the target variance is at or below the Poisson
        floor (Var_total <= mu), which a negative binomial cannot represent.
        """
        excess = total_var - mu
        if np.ndim(excess) == 0:
            return None if excess <= 0 else mu**2 / excess
        phi_eff = np.where(excess > 0, mu**2 / np.where(excess > 0, excess, 1.0), np.inf)
        return phi_eff

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
            # LNA-informed likelihood, as a count distribution. A Gaussian
            # matching the first two moments drops the third, and moment mode
            # then absorbs that misfit into the mechanistic process variance.
            self._warn_if_small(mu, phi, process_variance)
            total_var = max(self._total_variance(mu, phi, process_variance), 1e-6)
            phi_eff = (
                None
                if self.moment_family == "gaussian"
                else self._inflated_phi(mu, total_var)
            )
            if phi_eff is not None and np.isfinite(phi_eff) and phi_eff > 0:
                p_eff = np.clip(phi_eff / (phi_eff + mu), 1e-10, 1 - 1e-10)
                return float(
                    stats.nbinom.logpmf(max(int(round(obs_val)), 0), phi_eff, p_eff)
                )
            # Var_total <= mu: below the Poisson floor, where no negative
            # binomial exists. Fall back to the Gaussian.
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
        process_variance: float | None = None,
    ) -> float:
        """Sample a cell count matching the log-likelihood noise model.

        With process variance: NegBin whose size parameter is inflated to
        carry ``Var = Var_process + mu + mu^2/phi`` -- the same distribution
        the moment-mode log-likelihood scores, so simulated data and fitted
        data remain the same statistical model. Below the Poisson floor,
        where no negative binomial has that variance, a clipped Gaussian.
        Without process variance: NegBin(mu, phi) as in ODE mode.
        """
        mu = self._get_mean(latent_state)
        phi = self._phi(params)

        if process_variance is not None and process_variance > 0:
            self._warn_if_small(mu, phi, process_variance)
            total_var = max(self._total_variance(mu, phi, process_variance), 1e-6)
            phi_eff = (
                None
                if self.moment_family == "gaussian"
                else self._inflated_phi(mu, total_var)
            )
            if phi_eff is not None and np.isfinite(phi_eff) and phi_eff > 0:
                p_eff = np.clip(phi_eff / (phi_eff + mu), 1e-10, 1 - 1e-10)
                return float(rng.negative_binomial(phi_eff, p_eff))
            return float(max(rng.normal(mu, np.sqrt(total_var)), 0.0))

        p = np.clip(phi / (phi + mu), 1e-10, 1 - 1e-10)
        return float(rng.negative_binomial(phi, p))

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
            mu_g = mus[gaussian]
            self._warn_if_small(mu_g, phi, pv[gaussian])
            total_var = np.maximum(
                self._total_variance(mu_g, phi, pv[gaussian]), 1e-6
            )
            # Inflated-NegBin where a negative binomial can carry the target
            # variance, Gaussian only below the Poisson floor. Mirrors the
            # scalar path point by point.
            excess = total_var - mu_g
            nb_ok = (
                np.zeros(len(mu_g), dtype=bool)
                if self.moment_family == "gaussian"
                else excess > 0
            )
            sub = np.empty(len(mu_g), dtype=float)
            if np.any(nb_ok):
                phi_eff = mu_g[nb_ok] ** 2 / excess[nb_ok]
                p_eff = np.clip(phi_eff / (phi_eff + mu_g[nb_ok]), 1e-10, 1 - 1e-10)
                obs_g = np.maximum(np.rint(observed[gaussian][nb_ok]).astype(int), 0)
                sub[nb_ok] = stats.nbinom.logpmf(obs_g, phi_eff, p_eff)
            if np.any(~nb_ok):
                sub[~nb_ok] = stats.norm.logpdf(
                    observed[gaussian][~nb_ok],
                    loc=mu_g[~nb_ok],
                    scale=np.sqrt(total_var[~nb_ok]),
                )
            ll_array[gaussian] = sub

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
