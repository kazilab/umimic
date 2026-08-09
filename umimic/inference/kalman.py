"""Extended Kalman Filter for moment-based inference (fast mode).

Scope and contract
------------------
The filter propagates the LNA moments between observation times and applies
one scalar update per available observation. Each observation model supplies
its own linearization via :meth:`ObservationModel.linearize`, so every
modality it can approximate as Gaussian -- counts on the linear scale, BLI and
volume on the log scale -- contributes. A modality whose model has no Gaussian
approximation is rejected at construction rather than dropped, because a
filter that silently ignores a modality reports a likelihood for data it never
read.

The observation operator comes from the model's topology-derived operator, not
from an assumed state ordering: a ``[P, Q, R]`` model has no apoptotic
compartment, and a hand-written ``[1, 1, 0, 1]`` row would score the wrong
compartment there.

A failed moment solve invalidates the run. Carrying the previous state forward
and continuing to accumulate likelihood terms would produce a finite number
that no model generated, and optimisers happily walk toward such points.
"""

from __future__ import annotations

import logging

import numpy as np

from umimic.data.schemas import TimeSeriesData
from umimic.dynamics.moment_equations import MomentODE
from umimic.observations.base import (
    ObservationModel,
    TopologyAwareObservation,
)
from umimic.observations.multimodal import MultimodalObservation
from umimic.types import FilterResult

logger = logging.getLogger(__name__)


class ExtendedKalmanFilter:
    """Extended Kalman Filter for U-MIMIC moment-based inference.

    The EKF operates on the moment ODE system:
    - Prediction step: propagate mu and Sigma via moment ODEs between observations
    - Update step: incorporate observation likelihood via EKF update

    This is the computational backbone of 'fast mode' inference.
    """

    def __init__(
        self,
        moment_ode: MomentODE,
        observation_model: ObservationModel | MultimodalObservation,
    ):
        self.moment_ode = moment_ode
        self.obs_model = observation_model
        self.n_states = moment_ode.n

        # Resolve modality -> model, so every configured modality is used
        # rather than only cell counts.
        if isinstance(observation_model, MultimodalObservation):
            self._modality_models = dict(observation_model.models)
        else:
            name = (
                getattr(observation_model, "modality_name", None) or "cell_counts"
            )
            self._modality_models = {name: observation_model}

        # Give every model this filter's topology, so its observation operator
        # matches the state vector being filtered.
        for model in self._modality_models.values():
            if isinstance(model, TopologyAwareObservation):
                model.set_topology(moment_ode.topology)

        unsupported = [
            name
            for name, model in self._modality_models.items()
            if not self._supports_ekf(model)
        ]
        if unsupported:
            raise ValueError(
                f"Observation model(s) for {sorted(unsupported)} provide no "
                "Gaussian linearization, so the EKF cannot use them. Filtering "
                "the remaining modalities would report a likelihood that "
                "silently ignores this data; use a particle filter instead."
            )

    @staticmethod
    def _supports_ekf(model: ObservationModel) -> bool:
        """Whether a model overrides the default (None) linearization."""
        return type(model).linearize is not ObservationModel.linearize

    def _observed_at(
        self,
        data: TimeSeriesData,
        k: int,
        skip_modality: str | None = None,
    ) -> list[tuple[ObservationModel, float]]:
        """Modalities with a finite observation at time index k.

        Linearization deliberately happens in the caller, not here. Each
        modality must linearize against the state left by the preceding
        modality's update; building the whole list up front would evaluate
        every ``z_pred`` at the pre-update mean, which is not the sequential
        decomposition of a joint update.
        """
        observed = []
        for modality, model in self._modality_models.items():
            if modality == skip_modality:
                continue
            if not data.has_modality(modality):
                continue
            value = data.observations[modality][k]
            if not np.isfinite(value):
                # Missing observation: no update, and no contribution. This
                # previously produced a NaN innovation that propagated into
                # the marginal likelihood for every later time point.
                continue
            observed.append((model, float(value)))
        return observed

    def filter(
        self,
        data: TimeSeriesData,
        initial_mu: np.ndarray,
        initial_Sigma: np.ndarray | None = None,
        params: dict | None = None,
        anchor_modality: str | None = None,
        anchor_index: int | None = None,
    ) -> FilterResult:
        """Run the EKF forward filter through all observation times.

        Args:
            data: Observed time-series data.
            initial_mu: Initial mean state vector.
            initial_Sigma: Initial covariance (default: diagonal).
            params: Additional parameters (e.g., overdispersion).
            anchor_modality: Modality whose observation produced `initial_mu`,
                if any.
            anchor_index: Time index of that observation.

        Pass the anchor whenever `initial_mu` was seeded from the data, so the
        same measurement is not used once to place the state and again to
        score it. ModelLikelihood and ParticleFilter both exclude it by
        default; an EKF marginal likelihood used as an optimisation objective
        is not comparable with theirs unless it does too. Leave both None when
        `initial_mu` comes from elsewhere.

        Returns:
            FilterResult with filtered means, covariances, and marginal LL.
            On a failed moment solve the marginal log-likelihood is -inf and
            ``diverged`` is True.
        """
        if initial_Sigma is None:
            initial_Sigma = np.diag(np.maximum(initial_mu, 1.0))

        times = data.times
        n_times = len(times)
        n = self.n_states

        filtered_means = np.zeros((n_times, n))
        filtered_covs = np.zeros((n_times, n, n))
        innovations = np.zeros(n_times)
        marginal_ll = 0.0
        diverged = False

        mu = initial_mu.copy()
        Sigma = initial_Sigma.copy()

        for k in range(n_times):
            # Prediction step: propagate from previous to current time
            if k > 0:
                dt = times[k] - times[k - 1]
                if dt > 0:
                    try:
                        _t_pred, mu_pred, Sigma_pred = self.moment_ode.solve(
                            mu, Sigma,
                            t_span=(times[k - 1], times[k]),
                            t_eval=np.array([times[k]]),
                        )
                        mu = mu_pred[-1]
                        Sigma = Sigma_pred[-1]
                    except (RuntimeError, ValueError, FloatingPointError) as exc:
                        logger.warning(
                            "Moment solve failed between t=%.4g and t=%.4g "
                            "(%s); the filter is invalid from here on and its "
                            "marginal log-likelihood is reported as -inf.",
                            times[k - 1], times[k], exc,
                        )
                        diverged = True
                        marginal_ll = -np.inf
                        filtered_means[k:] = np.nan
                        filtered_covs[k:] = np.nan
                        innovations[k:] = np.nan
                        break

            # Update step: one scalar update per observed modality, applied
            # sequentially. Conditional independence given the latent state
            # makes this equivalent to a joint update with block-diagonal R --
            # but only if each modality is linearized against the state left by
            # the previous one. Linearizing all of them against the pre-update
            # mean uses a stale z_pred and gives both the wrong posterior and
            # the wrong marginal likelihood.
            skip = anchor_modality if k == anchor_index else None
            for model, value in self._observed_at(data, k, skip):
                update = model.linearize(value, mu, params)
                if update is None:
                    continue
                h = np.asarray(update.H, dtype=float).ravel()
                innov = update.z - update.z_pred
                S = float(h @ Sigma @ h) + update.R
                if not np.isfinite(S) or S <= 0:
                    logger.warning(
                        "Non-positive innovation variance at t=%.4g; the "
                        "filter is invalid and reports -inf.", times[k],
                    )
                    diverged = True
                    marginal_ll = -np.inf
                    break

                K = (Sigma @ h) / S

                mu = np.maximum(mu + K * innov, 0.0)

                # Joseph form for numerical stability.
                I_KH = np.eye(n) - np.outer(K, h)
                Sigma = I_KH @ Sigma @ I_KH.T + np.outer(K, K) * update.R
                Sigma = (Sigma + Sigma.T) / 2  # symmetrize

                marginal_ll += -0.5 * (np.log(2 * np.pi * S) + innov**2 / S)
                # Reported innovation is the last one at this time point.
                innovations[k] = innov

            if diverged:
                filtered_means[k:] = np.nan
                filtered_covs[k:] = np.nan
                break

            filtered_means[k] = mu
            filtered_covs[k] = Sigma

        return FilterResult(
            times=times,
            filtered_means=filtered_means,
            filtered_covs=filtered_covs,
            marginal_log_likelihood=float(marginal_ll),
            innovations=innovations,
            diverged=diverged,
        )

    def marginal_log_likelihood(
        self,
        data: TimeSeriesData,
        initial_mu: np.ndarray,
        params: dict | None = None,
        anchor_modality: str | None = None,
        anchor_index: int | None = None,
    ) -> float:
        """Compute marginal log-likelihood for parameter optimization.

        Convenience wrapper that returns just the scalar LL. See :meth:`filter`
        for when the anchor arguments are required.
        """
        result = self.filter(
            data,
            initial_mu,
            params=params,
            anchor_modality=anchor_modality,
            anchor_index=anchor_index,
        )
        return result.marginal_log_likelihood
