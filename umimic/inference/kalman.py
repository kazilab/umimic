"""Extended Kalman Filter for moment-based inference (fast mode)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from umimic.dynamics.moment_equations import MomentODE
from umimic.dynamics.states import ModelTopology
from umimic.dynamics.rates import RateSet
from umimic.observations.base import ObservationModel
from umimic.data.schemas import TimeSeriesData
from umimic.types import FilterResult


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
        observation_model: ObservationModel,
    ):
        self.moment_ode = moment_ode
        self.obs_model = observation_model
        self.n_states = moment_ode.n

    def _observation_jacobian(
        self, latent_state: np.ndarray
    ) -> np.ndarray:
        """Linearized observation matrix H = d(h)/d(x).

        For viable cell count: h(x) = P + Q + R (sum of non-A states).
        H = [1, 1, 0, 1, ...] (1 for each non-A state, 0 for A).
        """
        H = np.zeros((1, self.n_states))
        for i, ct in enumerate(self.moment_ode.topology.active_states):
            from umimic.dynamics.states import CellType
            if ct != CellType.A:
                H[0, i] = 1.0
        return H

    def _observation_noise(
        self, predicted_obs: float, params: dict | None = None
    ) -> np.ndarray:
        """Observation noise covariance R.

        For NegBin(mu, phi): Var = mu + mu^2/phi.
        """
        phi = 10.0
        if params and "overdispersion" in params:
            phi = params["overdispersion"]
        mu = max(predicted_obs, 1.0)
        var = mu + mu**2 / phi
        return np.array([[var]])

    def filter(
        self,
        data: TimeSeriesData,
        initial_mu: np.ndarray,
        initial_Sigma: np.ndarray | None = None,
        params: dict | None = None,
    ) -> FilterResult:
        """Run the EKF forward filter through all observation times.

        Args:
            data: Observed time-series data.
            initial_mu: Initial mean state vector.
            initial_Sigma: Initial covariance (default: diagonal).
            params: Additional parameters (e.g., overdispersion).

        Returns:
            FilterResult with filtered means, covariances, and marginal LL.
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

        mu = initial_mu.copy()
        Sigma = initial_Sigma.copy()

        for k in range(n_times):
            # Prediction step: propagate from previous to current time
            if k > 0:
                dt = times[k] - times[k - 1]
                if dt > 0:
                    try:
                        t_pred, mu_pred, Sigma_pred = self.moment_ode.solve(
                            mu, Sigma,
                            t_span=(times[k - 1], times[k]),
                            t_eval=np.array([times[k]]),
                        )
                        mu = mu_pred[-1]
                        Sigma = Sigma_pred[-1]
                    except Exception:
                        pass  # keep previous state on solver failure

            # Update step: incorporate observation
            if "cell_counts" in data.observations:
                y_obs = data.observations["cell_counts"][k]
                H = self._observation_jacobian(mu)
                y_pred = (H @ mu).item()
                R = self._observation_noise(y_pred, params)

                # Innovation
                innov = y_obs - y_pred
                innovations[k] = innov

                # Innovation covariance
                S = H @ Sigma @ H.T + R

                # Kalman gain
                S_inv = np.linalg.inv(S)
                K = Sigma @ H.T @ S_inv

                # State update
                mu = mu + K.flatten() * innov
                mu = np.maximum(mu, 0)

                # Covariance update (Joseph form for numerical stability)
                I_KH = np.eye(n) - K @ H
                Sigma = I_KH @ Sigma @ I_KH.T + K @ R @ K.T
                Sigma = (Sigma + Sigma.T) / 2  # symmetrize

                # Marginal log-likelihood contribution
                sign, logdet = np.linalg.slogdet(2 * np.pi * S)
                if sign > 0:
                    marginal_ll += -0.5 * (logdet + innov**2 * S_inv.item())

            filtered_means[k] = mu
            filtered_covs[k] = Sigma

        return FilterResult(
            times=times,
            filtered_means=filtered_means,
            filtered_covs=filtered_covs,
            marginal_log_likelihood=marginal_ll,
            innovations=innovations,
        )

    def marginal_log_likelihood(
        self,
        data: TimeSeriesData,
        initial_mu: np.ndarray,
        params: dict | None = None,
    ) -> float:
        """Compute marginal log-likelihood for parameter optimization.

        Convenience wrapper that returns just the scalar LL.
        """
        result = self.filter(data, initial_mu, params=params)
        return result.marginal_log_likelihood
