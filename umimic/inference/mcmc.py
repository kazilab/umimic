"""MCMC Bayesian inference for U-MIMIC.

Backends:
- emcee (affine-invariant ensemble sampler) -- the only supported backend.

The PyMC backend was withdrawn in this release: its model attached the data
via a constant ``pm.Potential``, so it sampled the prior regardless of the
observations. See :meth:`MCMCSampler.sample` for details.

Sample layout: posterior draws are returned as ``(n_walkers, n_draws)`` arrays.
emcee walkers are an interacting ensemble, not independent chains; convergence
diagnostics account for this.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle

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


def _log_posterior_worker(theta: np.ndarray) -> tuple[float, float]:
    """Module-level wrapper for parallel walker evaluation."""
    return _global_sampler_ref._log_posterior_blobs(theta)


class MCMCSampler:
    """Full Bayesian inference via MCMC sampling.

    The sampler wraps the ModelLikelihood with priors and samples
    from the posterior distribution p(theta | data) proportional to
    p(data | theta) * p(theta).
    """

    #: Backends that are implemented and validated in this release.
    SUPPORTED_BACKENDS = ("emcee",)

    def __init__(
        self,
        likelihood: ModelLikelihood,
        priors: PriorSpec,
        backend: str = "emcee",
        rng: np.random.Generator | int | None = None,
    ):
        """
        Args:
            likelihood: Model likelihood.
            priors: Prior specification.
            backend: Sampling backend. Only "emcee" is supported.
            rng: Generator or seed. Supplying one makes sampling reproducible.
        """
        self.likelihood = likelihood
        self.priors = priors
        self.backend = backend
        self.rng = np.random.default_rng(rng)

    def _log_prob(self, theta: np.ndarray) -> tuple[float, float]:
        """Return (log_posterior, log_likelihood) at theta."""
        params = self.likelihood.theta_to_params(theta)

        # Check prior support (log-prior returns -inf if out of bounds)
        lp = self.priors.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf, -np.inf

        ll = self.likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf, -np.inf

        return ll + lp, ll

    def _log_posterior(self, theta: np.ndarray) -> float:
        """Log-posterior = log-likelihood + log-prior."""
        return self._log_prob(theta)[0]

    def _log_posterior_blobs(self, theta: np.ndarray) -> tuple[float, float]:
        """Log-posterior with the log-likelihood carried as an emcee blob.

        Keeping the likelihood separate from the posterior is required for
        information criteria and for posterior predictive work; the two must
        not be conflated in a single trace.
        """
        return self._log_prob(theta)

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
        if self.backend == "pymc":
            raise NotImplementedError(
                "The PyMC backend is not available in this release. Its model "
                "attached the data through a constant pm.Potential, so the "
                "sampler targeted the prior and the observations had no "
                "effect on the posterior. Rather than ship a backend that "
                "silently ignores the data, it has been withdrawn pending a "
                "differentiable PyTensor forward model (or a tested PyTensor "
                "wrapper around the numerical likelihood). Use "
                "backend='emcee'."
            )
        raise ValueError(
            f"Unknown backend: {self.backend!r}. Supported backends: "
            f"{list(self.SUPPORTED_BACKENDS)}."
        )

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

        # Initialize walkers around initial guess or prior samples.
        if initial_guess is None:
            # Sample from the prior, ordered by likelihood.param_names rather
            # than by the prior dict's insertion order: theta positions are
            # defined by param_names, and a mismatch silently permutes
            # parameters.
            draws = [self.priors.sample(self.rng) for _ in range(n_walkers)]
            missing = [
                name for name in self.likelihood.param_names
                if name not in draws[0]
            ]
            if missing:
                raise ValueError(
                    f"Priors do not cover parameter(s) {missing}; every entry "
                    "of likelihood.param_names needs a prior to initialize "
                    "walkers."
                )
            p0 = np.array(
                [[d[name] for name in self.likelihood.param_names] for d in draws]
            )
        else:
            initial_guess = np.asarray(initial_guess, dtype=float)
            if initial_guess.shape != (ndim,):
                raise ValueError(
                    f"initial_guess must have shape ({ndim},) matching "
                    f"likelihood.param_names, got {initial_guess.shape}."
                )
            # Small perturbation around initial guess, from the seeded stream.
            p0 = initial_guess[np.newaxis, :] + 1e-3 * self.rng.standard_normal(
                (n_walkers, ndim)
            )
            p0 = np.abs(p0)  # ensure positive

        # Determine parallelism
        if n_processes is None:
            n_processes = max(1, min(os.cpu_count() or 1, n_walkers))
        use_pool = n_processes > 1

        # emcee draws its own randomness; seed it from our generator so that a
        # supplied seed fully determines the run.
        emcee_seed = int(self.rng.integers(0, 2**32 - 1))

        common = dict(
            nwalkers=n_walkers,
            ndim=ndim,
            blobs_dtype=[("log_likelihood", float)],
        )

        if use_pool:
            try:
                pool = multiprocessing.Pool(
                    processes=n_processes,
                    initializer=_init_worker,
                    initargs=(self,),
                )
                sampler = emcee.EnsembleSampler(
                    log_prob_fn=_log_posterior_worker, pool=pool, **common
                )
            except (TypeError, AttributeError, pickle.PicklingError):
                # Fallback to serial if the sampler can't be pickled
                # (e.g. lambda attributes, Jupyter edge cases)
                pool = None
                use_pool = False
                n_processes = 1
                sampler = emcee.EnsembleSampler(
                    log_prob_fn=self._log_posterior_blobs, **common
                )
        else:
            pool = None
            sampler = emcee.EnsembleSampler(
                log_prob_fn=self._log_posterior_blobs, **common
            )

        sampler.random_state = np.random.RandomState(emcee_seed).get_state()

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

        # emcee returns (n_draws, n_walkers, ndim). Keep the walker axis:
        # collapsing it immediately destroys the information needed for R-hat
        # and ESS, and emcee walkers are *not* independent chains.
        chain = sampler.get_chain(flat=False)
        samples = {
            name: np.ascontiguousarray(chain[:, :, i].T)  # (n_walkers, n_draws)
            for i, name in enumerate(self.likelihood.param_names)
        }

        log_post = sampler.get_log_prob(flat=False).T  # (n_walkers, n_draws)
        blobs = sampler.get_blobs(flat=False)
        log_lik = (
            np.ascontiguousarray(blobs["log_likelihood"].T)
            if blobs is not None
            else None
        )

        acceptance = float(np.mean(sampler.acceptance_fraction))
        diagnostics = {
            "acceptance_fraction": acceptance,
            "n_walkers": n_walkers,
            "n_processes": n_processes,
            "backend": "emcee",
            "seed": emcee_seed,
            # emcee's ensemble walkers are correlated by construction; they are
            # reported separately from the notion of independent chains.
            "walkers_are_independent_chains": False,
        }

        try:
            autocorr = sampler.get_autocorr_time(quiet=True)
            diagnostics["autocorr_time"] = autocorr.tolist()
        except Exception:
            pass

        return MCMCResult(
            samples=samples,
            log_likelihood_trace=log_lik,
            log_posterior_trace=np.ascontiguousarray(log_post),
            n_chains=n_walkers,
            n_samples=n_samples,
            diagnostics=diagnostics,
        )
