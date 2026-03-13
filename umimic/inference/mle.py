"""Maximum likelihood estimation via scipy.optimize."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import optimize

from umimic.inference.likelihood import ModelLikelihood
from umimic.inference.priors import PriorSpec
from umimic.types import MLEResult


def _minimize_options(method: str) -> dict[str, float | int]:
    """Return solver-specific scipy.optimize.minimize options."""
    options: dict[str, float | int] = {"maxiter": 1000}
    if method == "Nelder-Mead":
        options["fatol"] = 1e-8
    else:
        options["ftol"] = 1e-8
    return options


class MLEstimator:
    """Maximum likelihood point estimation.

    Supports multiple optimization methods:
    - L-BFGS-B: bounded quasi-Newton (fast, gradient-based)
    - Nelder-Mead: derivative-free simplex (robust)
    - differential_evolution: global optimization (slower, more thorough)

    Also supports MAP estimation (MLE + log-prior) when priors are provided.
    """

    def __init__(
        self,
        likelihood: ModelLikelihood,
        bounds: dict[str, tuple[float, float]] | None = None,
        priors: PriorSpec | None = None,
        method: str = "L-BFGS-B",
    ):
        self.likelihood = likelihood
        self.priors = priors
        self.method = method

        # Default bounds for common parameters
        default_bounds = {
            "b0": (1e-4, 0.2),
            "d0_P": (1e-5, 0.1),
            "d0_Q": (1e-5, 0.1),
            "emax_death": (0.0, 0.5),
            "ec50_death": (1e-3, 100.0),
            "hill_death": (0.3, 5.0),
            "emax_birth": (0.0, 1.0),
            "ec50_birth": (1e-3, 100.0),
            "hill_birth": (0.3, 5.0),
            "u_PQ": (1e-6, 0.05),
            "u_QP": (1e-6, 0.05),
            "u_PR": (1e-6, 0.05),
            "overdispersion": (1.0, 200.0),
        }

        if bounds:
            default_bounds.update(bounds)

        self.bounds = [
            default_bounds.get(name, (1e-6, 100.0))
            for name in likelihood.param_names
        ]

    def _objective(self, theta: np.ndarray) -> float:
        """Negative log-posterior (or neg-log-likelihood if no priors)."""
        ll = self.likelihood(theta)
        if not np.isfinite(ll):
            return 1e20

        if self.priors is not None:
            params = self.likelihood.theta_to_params(theta)
            lp = self.priors.log_prior(params)
            if not np.isfinite(lp):
                return 1e20
            return -(ll + lp)

        return -ll

    def fit(
        self,
        initial_guess: np.ndarray | None = None,
        n_restarts: int = 5,
    ) -> MLEResult:
        """Run optimization with optional multi-start.

        Args:
            initial_guess: Starting parameter vector.
            n_restarts: Number of random restarts to avoid local optima.

        Returns:
            MLEResult with point estimates, standard errors, and fit statistics.
        """
        if initial_guess is None:
            initial_guess = np.array(
                [0.5 * (lo + hi) for lo, hi in self.bounds]
            )

        best_result = None
        best_obj = np.inf

        for restart in range(n_restarts):
            if restart == 0:
                x0 = initial_guess
            else:
                # Random start within bounds
                x0 = np.array(
                    [
                        np.random.uniform(lo, hi)
                        for lo, hi in self.bounds
                    ]
                )

            try:
                if self.method == "differential_evolution":
                    result = optimize.differential_evolution(
                        self._objective,
                        bounds=self.bounds,
                        maxiter=500,
                        tol=1e-6,
                        seed=42 + restart,
                    )
                else:
                    result = optimize.minimize(
                        self._objective,
                        x0,
                        method=self.method,
                        bounds=self.bounds,
                        options=_minimize_options(self.method),
                    )

                if result.fun < best_obj:
                    best_obj = result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is None:
            return MLEResult(
                parameters={},
                log_likelihood=-np.inf,
                aic=np.inf,
                bic=np.inf,
                converged=False,
            )

        # Extract results
        theta_hat = best_result.x
        params = self.likelihood.theta_to_params(theta_hat)
        ll = self.likelihood(theta_hat)

        # Compute Hessian for standard errors
        hessian = None
        se = None
        try:
            hessian = _numerical_hessian(self._objective, theta_hat)
            # Standard errors from inverse Hessian diagonal
            inv_hess = np.linalg.inv(hessian)
            se_values = np.sqrt(np.abs(np.diag(inv_hess)))
            se = {
                name: float(se_values[i])
                for i, name in enumerate(self.likelihood.param_names)
            }
        except Exception:
            pass

        # Information criteria
        k = len(theta_hat)
        n_obs = sum(len(d.times) for d in self.likelihood.data_list)
        aic = 2 * k - 2 * ll
        bic = k * np.log(max(n_obs, 1)) - 2 * ll

        return MLEResult(
            parameters=params,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            hessian=hessian,
            se=se,
            converged=best_result.success,
            n_evaluations=self.likelihood.n_evaluations,
        )


def _numerical_hessian(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute numerical Hessian via central finite differences."""
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)

    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += eps
            x_pp[j] += eps
            x_pm[i] += eps
            x_pm[j] -= eps
            x_mp[i] -= eps
            x_mp[j] += eps
            x_mm[i] -= eps
            x_mm[j] -= eps

            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps**2)
            H[j, i] = H[i, j]

    return H
