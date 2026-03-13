"""MCMC Bayesian inference for U-MIMIC.

Supports multiple backends:
- emcee (ensemble sampler, always available)
- PyMC (NUTS HMC, optional)
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
from typing import Any

import numpy as np

from umimic.inference.likelihood import ModelLikelihood
from umimic.inference.priors import PriorSpec
from umimic.types import MCMCResult


# Module-level references for multiprocessing.
# With the 'spawn' start method (default on macOS), child processes get a
# fresh module import. We use Pool(initializer=_init_worker, initargs=(...))
# to set these in each worker before any work begins.
_global_sampler_ref: MCMCSampler | None = None


def _init_worker(sampler: MCMCSampler) -> None:
    """Initializer called once per worker process to set the shared sampler."""
    global _global_sampler_ref
    _global_sampler_ref = sampler


def _log_posterior_worker(theta: np.ndarray) -> float:
    """Module-level wrapper for parallel walker evaluation."""
    return _global_sampler_ref._log_posterior(theta)


class MCMCSampler:
    """Full Bayesian inference via MCMC sampling.

    The sampler wraps the ModelLikelihood with priors and samples
    from the posterior distribution p(theta | data) proportional to
    p(data | theta) * p(theta).
    """

    def __init__(
        self,
        likelihood: ModelLikelihood,
        priors: PriorSpec,
        backend: str = "emcee",
    ):
        self.likelihood = likelihood
        self.priors = priors
        self.backend = backend

    def _log_posterior(self, theta: np.ndarray) -> float:
        """Log-posterior = log-likelihood + log-prior."""
        params = self.likelihood.theta_to_params(theta)

        # Check prior support (log-prior returns -inf if out of bounds)
        lp = self.priors.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf

        ll = self.likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf

        return ll + lp

    def sample(
        self,
        n_samples: int = 2000,
        n_chains: int = 4,
        n_warmup: int = 1000,
        initial_guess: np.ndarray | None = None,
        n_processes: int = 1,
        **kwargs,
    ) -> MCMCResult:
        """Run MCMC sampling.

        Args:
            n_samples: Number of post-warmup samples per chain.
            n_chains: Number of independent chains (walkers for emcee).
            n_warmup: Number of warmup/burn-in samples.
            initial_guess: Starting point for all chains.
            n_processes: Number of parallel processes for walker evaluation.
                1 (default) = serial execution (safest, works in Jupyter).
                Set > 1 to enable multiprocessing (recommended for scripts).
                None = auto-detect CPU count.

        Returns:
            MCMCResult with posterior samples and diagnostics.
        """
        if self.backend == "emcee":
            return self._sample_emcee(
                n_samples, n_chains, n_warmup, initial_guess, n_processes
            )
        elif self.backend == "pymc":
            return self._sample_pymc(n_samples, n_chains, n_warmup, initial_guess)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _sample_emcee(
        self,
        n_samples: int,
        n_walkers: int,
        n_warmup: int,
        initial_guess: np.ndarray | None,
        n_processes: int = 1,
    ) -> MCMCResult:
        """Run emcee ensemble sampler with optional multiprocessing."""
        try:
            import emcee
        except ImportError:
            raise ImportError(
                "emcee is required for MCMC. Install with: pip install emcee"
            )

        ndim = self.likelihood.n_params

        # emcee needs at least 2*ndim walkers
        n_walkers = max(n_walkers, 2 * ndim + 2)

        # Initialize walkers around initial guess or prior samples
        if initial_guess is None:
            # Sample from prior
            p0 = np.array(
                [
                    list(self.priors.sample().values())
                    for _ in range(n_walkers)
                ]
            )
        else:
            # Small perturbation around initial guess
            p0 = initial_guess[np.newaxis, :] + 1e-3 * np.random.randn(
                n_walkers, ndim
            )
            p0 = np.abs(p0)  # ensure positive

        # Determine parallelism
        if n_processes is None:
            n_processes = max(1, min(os.cpu_count() or 1, n_walkers))
        use_pool = n_processes > 1

        if use_pool:
            try:
                pool = multiprocessing.Pool(
                    processes=n_processes,
                    initializer=_init_worker,
                    initargs=(self,),
                )
                sampler = emcee.EnsembleSampler(
                    n_walkers, ndim, _log_posterior_worker, pool=pool
                )
            except (TypeError, AttributeError, pickle.PicklingError):
                # Fallback to serial if the sampler can't be pickled
                # (e.g. lambda attributes, Jupyter edge cases)
                pool = None
                use_pool = False
                n_processes = 1
                sampler = emcee.EnsembleSampler(
                    n_walkers, ndim, self._log_posterior
                )
        else:
            pool = None
            sampler = emcee.EnsembleSampler(
                n_walkers, ndim, self._log_posterior
            )

        try:
            # Burn-in
            state = sampler.run_mcmc(p0, n_warmup, progress=False)
            sampler.reset()

            # Production
            sampler.run_mcmc(state, n_samples, progress=False)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        # Extract samples: (n_walkers, n_samples, ndim)
        chain = sampler.get_chain(flat=False)  # (n_samples, n_walkers, ndim)
        flat_chain = sampler.get_chain(flat=True)  # (n_samples * n_walkers, ndim)

        samples = {}
        for i, name in enumerate(self.likelihood.param_names):
            samples[name] = flat_chain[:, i]

        # Basic diagnostics
        acceptance = float(np.mean(sampler.acceptance_fraction))
        diagnostics = {
            "acceptance_fraction": acceptance,
            "n_walkers": n_walkers,
            "n_processes": n_processes,
            "backend": "emcee",
        }

        try:
            autocorr = sampler.get_autocorr_time(quiet=True)
            diagnostics["autocorr_time"] = autocorr.tolist()
        except Exception:
            pass

        return MCMCResult(
            samples=samples,
            log_likelihood_trace=sampler.get_log_prob(flat=True),
            n_chains=n_walkers,
            n_samples=n_samples,
            diagnostics=diagnostics,
        )

    def _sample_pymc(
        self,
        n_samples: int,
        n_chains: int,
        n_warmup: int,
        initial_guess: np.ndarray | None,
    ) -> MCMCResult:
        """Run PyMC NUTS sampler."""
        try:
            import pymc as pm
            import pytensor.tensor as pt
        except ImportError:
            raise ImportError(
                "PyMC is required for NUTS sampling. Install with: pip install pymc"
            )

        with pm.Model() as model:
            # Define parameter priors
            theta_vars = {}
            for name in self.likelihood.param_names:
                if name in self.priors.distributions:
                    dist = self.priors.distributions[name]
                    # Map scipy distributions to PyMC
                    mean = float(dist.mean())
                    std = float(dist.std())
                    theta_vars[name] = pm.TruncatedNormal(
                        name, mu=mean, sigma=std, lower=0
                    )
                else:
                    theta_vars[name] = pm.HalfNormal(name, sigma=1.0)

            # Custom likelihood via pm.Potential
            def loglike_op(*args):
                theta = np.array([float(a) for a in args])
                return self.likelihood(theta)

            theta_list = [theta_vars[n] for n in self.likelihood.param_names]
            pm.Potential(
                "custom_likelihood",
                pm.math.log(1.0),  # placeholder; actual LL handled via callbacks
            )

            # Use DEMetropolis (gradient-free) for custom likelihoods
            trace = pm.sample(
                draws=n_samples,
                tune=n_warmup,
                chains=n_chains,
                cores=1,
                step=pm.DEMetropolis(),
                return_inferencedata=False,
                progressbar=False,
            )

        samples = {}
        for name in self.likelihood.param_names:
            if name in trace.varnames:
                samples[name] = trace[name]

        return MCMCResult(
            samples=samples,
            n_chains=n_chains,
            n_samples=n_samples,
            diagnostics={"backend": "pymc"},
        )
