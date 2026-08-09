"""Maximum likelihood estimation via scipy.optimize."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from scipy import optimize

from umimic.inference.likelihood import ModelLikelihood
from umimic.inference.priors import PriorSpec
from umimic.types import MLEResult

logger = logging.getLogger(__name__)


def _minimize_options(method: str) -> dict[str, float | int]:
    """Return solver-specific scipy.optimize.minimize options."""
    options: dict[str, float | int] = {"maxiter": 1000}
    if method == "Nelder-Mead":
        options["fatol"] = 1e-8
    else:
        options["ftol"] = 1e-8
    return options


# Numerical guard for a strictly-positive parameter that arrives at exactly 0.
_LOG_FLOOR = 1e-12


class MLEstimator:
    """Maximum likelihood point estimation.

    Supports multiple optimization methods:
    - L-BFGS-B: bounded quasi-Newton (fast, gradient-based)
    - Nelder-Mead: derivative-free simplex (robust)
    - differential_evolution: global optimization (slower, more thorough)

    Also supports MAP estimation (MLE + log-prior) when priors are provided.

    Every rate in this model is strictly positive and the parameter vector
    spans three orders of magnitude (``b0`` ~ 0.05 against ``overdispersion``
    ~ 100), so the optimisation is run on ``log`` parameters by default. That
    is a change of search variable, not of the objective: the value at each
    point is identical, so the point estimate is unchanged and no Jacobian
    term is required. What changes is conditioning -- a 10% move in ``b0`` and
    a 10% move in ``overdispersion`` become the same step length, instead of
    differing by 10^3.

    Args:
        log_space: Optimise log parameters. Set False for the historical
            linear-space behaviour.
        seed: Seeds the multi-start draws. Restarts previously used the global
            numpy RNG, so a fit with ``n_restarts > 1`` was not reproducible.
    """

    def __init__(
        self,
        likelihood: ModelLikelihood,
        bounds: dict[str, tuple[float, float]] | None = None,
        priors: PriorSpec | None = None,
        method: str = "L-BFGS-B",
        log_space: bool = True,
        seed: int | None = None,
    ):
        self.likelihood = likelihood
        self.priors = priors
        self.method = method
        self.log_space = log_space
        self._rng = np.random.default_rng(seed)

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
            "sigma_extrinsic": (1e-3, 2.0),
        }

        if bounds:
            default_bounds.update(bounds)

        self.bounds = [
            default_bounds.get(name, (1e-6, 100.0))
            for name in likelihood.param_names
        ]
        # Log-transform only parameters whose lower bound is strictly
        # positive. A parameter bounded at exactly zero -- the emax terms,
        # where zero means "no drug effect" and is a live hypothesis -- must
        # stay linear. Mapping such a bound onto log(tiny) manufactures tens of
        # units of search space that all decode to the same "no effect" model;
        # the likelihood is flat across it, and a gradient optimiser that
        # drifts in stops there and reports emax at the clamp. The emax terms
        # are also the ones that least need rescaling: they are already of
        # order 0.1-1, unlike b0 ~ 0.05 against overdispersion ~ 100.
        self._log_mask = np.array(
            [self.log_space and lo > 0.0 for lo, _ in self.bounds], dtype=bool
        )
        self._search_bounds = [
            (float(np.log(lo)), float(np.log(hi))) if use_log else (lo, hi)
            for (lo, hi), use_log in zip(self.bounds, self._log_mask)
        ]

    # ------------------------------------------------------------------
    # Search-space transform
    # ------------------------------------------------------------------
    def _to_search(self, theta: np.ndarray) -> np.ndarray:
        u = np.array(theta, dtype=float, copy=True)
        if self._log_mask.any():
            u[self._log_mask] = np.log(
                np.maximum(u[self._log_mask], _LOG_FLOOR)
            )
        return u

    def _from_search(self, u: np.ndarray) -> np.ndarray:
        theta = np.array(u, dtype=float, copy=True)
        if self._log_mask.any():
            theta[self._log_mask] = np.exp(theta[self._log_mask])
        return theta

    def _search_objective(self, u: np.ndarray) -> float:
        return self._objective(self._from_search(u))

    def _default_start(self) -> np.ndarray:
        """Starting vector on the natural scale.

        The prior median where a prior exists, because that is the value the
        prior actually considers typical. The previous default was the
        arithmetic midpoint of the bounds, which put ``ec50_death`` at 50 --
        above the top dose of every dataset in the package, on a likelihood
        that is nearly flat in that direction. Fits started there and stayed
        there.
        """
        start = []
        for i, name in enumerate(self.likelihood.param_names):
            lo, hi = self.bounds[i]
            value = None
            if self.priors is not None and name in self.priors.distributions:
                try:
                    median = float(self.priors.distributions[name].median())
                    if np.isfinite(median) and median > 0:
                        value = median
                except Exception:
                    value = None
            if value is None:
                value = (
                    float(np.sqrt(lo * hi))
                    if self._log_mask[i]
                    else 0.5 * (lo + hi)
                )
            start.append(min(max(value, lo), hi))
        return np.array(start)

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
            initial_guess = self._default_start()

        best_result = None
        best_obj = np.inf

        for restart in range(n_restarts):
            if restart == 0:
                u0 = self._to_search(initial_guess)
            else:
                # Random start, drawn uniformly in the SEARCH space. Drawing
                # uniformly on the natural scale over a bound like
                # (1e-3, 100) puts 99.9% of the mass above 0.1, so restarts
                # never explored small potencies -- the multi-start was not
                # sampling the region where the optimum usually lies.
                u0 = np.array([
                    self._rng.uniform(lo, hi) for lo, hi in self._search_bounds
                ])

            try:
                if self.method == "differential_evolution":
                    result = optimize.differential_evolution(
                        self._search_objective,
                        bounds=self._search_bounds,
                        maxiter=500,
                        tol=1e-6,
                        seed=42 + restart,
                    )
                else:
                    result = optimize.minimize(
                        self._search_objective,
                        u0,
                        method=self.method,
                        bounds=self._search_bounds,
                        options=_minimize_options(self.method),
                    )

                if result.fun < best_obj:
                    best_obj = result.fun
                    best_result = result
            except Exception as e:
                logger.warning("Optimization restart %s failed: %s", restart, e)
                continue

        if best_result is None:
            return MLEResult(
                parameters={},
                log_likelihood=-np.inf,
                aic=np.inf,
                bic=np.inf,
                converged=False,
            )

        # Extract results. The optimiser worked in the search space; every
        # quantity below is reported on the natural scale, as before.
        theta_hat = self._from_search(best_result.x)
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

        # Information criteria. The BIC sample size is the number of points
        # that actually entered the likelihood, not the number of time points:
        # every modality contributes its own term, and the anchor observation
        # consumed by the initial condition is not scored. Counting time points
        # instead undercounts multimodal designs and overcounts conditioned
        # ones, and since the error scales with k it does not cancel from the
        # BIC *differences* used for model comparison.
        k = len(theta_hat)
        n_obs = self.likelihood.n_observations
        aic = 2 * k - 2 * ll
        bic = k * np.log(n_obs) - 2 * ll if n_obs > 0 else np.inf

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
    """Compute numerical Hessian via central finite differences.

    `eps` is a *relative* step, floored so a parameter sitting at zero still
    gets a finite one. A fixed absolute step cannot serve a vector spanning
    ``b0`` ~ 0.05 and ``overdispersion`` ~ 100 at once: 1e-5 is a sane 0.02%
    probe of the former and a 1e-7 relative probe of the latter, which is
    below the noise floor of the likelihood and returns curvature that is
    mostly rounding error.
    """
    n = len(x)
    H = np.zeros((n, n))
    steps = eps * np.maximum(np.abs(np.asarray(x, dtype=float)), 1e-3)

    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += steps[i]
            x_pp[j] += steps[j]
            x_pm[i] += steps[i]
            x_pm[j] -= steps[j]
            x_mp[i] -= steps[i]
            x_mp[j] += steps[j]
            x_mm[i] -= steps[i]
            x_mm[j] -= steps[j]

            H[i, j] = (
                f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)
            ) / (4 * steps[i] * steps[j])
            H[j, i] = H[i, j]

    return H
