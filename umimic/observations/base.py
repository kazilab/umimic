"""Abstract base class for observation models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from umimic.dynamics.states import STATE_ORDER, CellType, ModelTopology

logger = logging.getLogger(__name__)

# Which cell types contribute to each observable quantity.
_SELECTOR_MEMBERS = {
    "viable": lambda ct: ct != CellType.A,
    "total": lambda ct: True,
    "proliferating": lambda ct: ct == CellType.P,
    "quiescent": lambda ct: ct == CellType.Q,
    "dead": lambda ct: ct == CellType.A,
    "resistant": lambda ct: ct == CellType.R,
}


def observation_operator(
    kind: str,
    topology: ModelTopology | None = None,
    n_states: int | None = None,
) -> np.ndarray:
    """Build the linear observation operator H for an observable quantity.

    The observed mean is ``H @ x`` and the projected process variance is
    ``H @ Sigma @ H.T``. Deriving H from the topology is what keeps the
    mapping correct for non-canonical state sets: a ``[P, Q, R]`` model has
    no apoptotic compartment, so index 2 is *resistant*, not *apoptotic*.

    Args:
        kind: One of the keys of ``_SELECTOR_MEMBERS``.
        topology: Model topology defining the state ordering. Strongly
            preferred; when omitted the canonical ``STATE_ORDER`` prefix is
            assumed, which is only valid for [P], [P,Q], [P,Q,A], [P,Q,A,R].
        n_states: Length of the state vector, required when topology is None.

    Returns:
        Row vector H of length n_states.
    """
    if kind not in _SELECTOR_MEMBERS:
        raise ValueError(
            f"Unknown observable {kind!r}; expected one of "
            f"{sorted(_SELECTOR_MEMBERS)}."
        )

    if topology is not None:
        states = list(topology.active_states)
    else:
        if n_states is None:
            raise ValueError("Either topology or n_states must be provided.")
        if n_states > len(STATE_ORDER):
            raise ValueError(
                f"Cannot infer a state layout for {n_states} states without a "
                "topology; pass topology= to the observation model."
            )
        states = STATE_ORDER[:n_states]
        if kind in ("viable", "dead") and n_states == 3:
            logger.warning(
                "Observation model has no topology and a 3-state vector; "
                "assuming [P, Q, A]. If this model is [P, Q, R] the %s "
                "observable will be wrong. Pass topology= to remove the "
                "ambiguity.",
                kind,
            )

    return np.array(
        [1.0 if _SELECTOR_MEMBERS[kind](ct) else 0.0 for ct in states],
        dtype=float,
    )


@dataclass(frozen=True)
class EKFUpdate:
    """Terms for one scalar Extended Kalman Filter update.

    The filter applies ``z = h(x) + N(0, R)`` with ``H = dh/dx`` evaluated at
    the predicted mean. `scale` names the space the terms live in ("linear"
    or "log") so results can be reported honestly.
    """

    z: float
    z_pred: float
    H: np.ndarray
    R: float
    scale: str = "linear"


class TopologyAwareObservation:
    """Mixin providing a cached, topology-derived observation operator."""

    def __init__(self, topology: ModelTopology | None = None):
        self.topology = topology
        self._operator_cache: dict[tuple[str, int], np.ndarray] = {}

    def set_topology(self, topology: ModelTopology | None) -> None:
        """Attach a topology, clearing any cached operators."""
        self.topology = topology
        self._operator_cache = {}

    def operator(self, kind: str, n_states: int) -> np.ndarray:
        """Cached observation operator H for `kind` at a given state length."""
        key = (kind, n_states)
        cached = self._operator_cache.get(key)
        if cached is None:
            topo = self.topology
            if topo is not None and topo.n_states != n_states:
                # State vector does not match the attached topology; fall back
                # rather than silently applying the wrong operator.
                topo = None
            cached = observation_operator(kind, topo, n_states)
            self._operator_cache[key] = cached
        return cached

    def _project(self, kind: str, latent_state: np.ndarray, floor: float = 1e-6) -> float:
        """H @ x for the requested observable."""
        x = np.asarray(latent_state, dtype=float)
        H = self.operator(kind, x.shape[-1])
        return max(float(H @ np.maximum(x, 0.0)), floor)

    def _project_batch(
        self, kind: str, latent_states: np.ndarray, floor: float = 1e-6
    ) -> np.ndarray:
        """H @ X^T for a stack of states (n_times, n_states)."""
        X = np.atleast_2d(np.asarray(latent_states, dtype=float))
        H = self.operator(kind, X.shape[-1])
        return np.maximum(np.maximum(X, 0.0) @ H, floor)

    def _with_process_variance(
        self,
        sigma_log: float,
        latent_state: np.ndarray,
        process_variance: float | None,
        kind: str = "viable",
    ) -> float:
        """Inflate a log-scale SD by the LNA process variance.

        For a signal proportional to the projected population N, the delta
        method maps an additive variance Var on N to a log-scale variance
        Var / N**2. Ignoring this would leave the mechanistic variance
        informing only the count modality, which is what "multimodal variance
        fusion" is supposed to avoid.
        """
        if process_variance is None or process_variance <= 0:
            return sigma_log
        n = self._project(kind, latent_state)
        if not np.isfinite(n) or n <= 0:
            return sigma_log
        return float(np.sqrt(sigma_log**2 + process_variance / n**2))

    def project_variance(
        self, kind: str, covariance: np.ndarray
    ) -> float | np.ndarray:
        """Project a process covariance onto the observable: H @ Sigma @ H.T.

        Accepts a single (n, n) matrix or a stack of (k, n, n) matrices.
        """
        cov = np.asarray(covariance, dtype=float)
        H = self.operator(kind, cov.shape[-1])
        if cov.ndim == 2:
            return float(H @ cov @ H)
        return np.einsum("i,kij,j->k", H, cov, H)


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
        process_variance: float | None = None,
    ) -> float | np.ndarray:
        """Generate a synthetic observation from the latent state.

        Args:
            latent_state: Latent cell population state vector.
            rng: Random number generator.
            params: Optional additional parameters (same keys as log-likelihood).
            process_variance: Optional LNA process variance for the projected
                population. When provided, sampling uses the same noise model
                as :meth:`log_likelihood` (measurement + process), so posterior
                predictive checks are not overconfident in moment mode.
        """
        ...

    @abstractmethod
    def param_names(self) -> list[str]:
        """Names of observation model parameters."""
        ...

    def expected_value(self, latent_state: np.ndarray) -> float:
        """Expected observation given latent state (for plotting/diagnostics)."""
        raise NotImplementedError

    def linearize(
        self,
        observed: float,
        latent_state: np.ndarray,
        params: dict | None = None,
    ) -> EKFUpdate | None:
        """Local Gaussian linearization for a scalar Kalman update.

        Returning None means this model has no Gaussian approximation, and a
        filter must refuse the modality rather than skip it silently. Models
        with a lognormal observation return terms on the *log* scale, where
        the noise is additive and Gaussian; the filter never needs to know
        which convention a modality uses.
        """
        return None

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
