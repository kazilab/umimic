"""Convergence diagnostics and posterior predictive checks."""

from __future__ import annotations

import numpy as np

from umimic.types import MCMCResult


def compute_rhat(chains: np.ndarray) -> float:
    """Compute Gelman-Rubin R-hat convergence diagnostic.

    Args:
        chains: (n_chains, n_samples) array of MCMC samples.

    Returns:
        R-hat value. Values close to 1.0 indicate convergence.
    """
    if chains.ndim == 1:
        return 1.0  # single chain, can't compute

    n_chains, n_samples = chains.shape
    if n_chains < 2:
        return 1.0

    # Between-chain variance
    chain_means = np.mean(chains, axis=1)
    grand_mean = np.mean(chain_means)
    B = n_samples * np.var(chain_means, ddof=1)

    # Within-chain variance
    chain_vars = np.var(chains, axis=1, ddof=1)
    W = np.mean(chain_vars)

    if W == 0:
        return 1.0

    # Pooled variance estimate
    var_hat = (1 - 1 / n_samples) * W + B / n_samples

    return float(np.sqrt(var_hat / W))


def effective_sample_size(samples: np.ndarray) -> float:
    """Estimate effective sample size (ESS) using autocorrelation.

    Args:
        samples: 1D array of MCMC samples.

    Returns:
        Estimated ESS.
    """
    n = len(samples)
    if n < 10:
        return float(n)

    # Compute autocorrelation via FFT
    x = samples - np.mean(samples)
    fft_x = np.fft.fft(x, n=2 * n)
    acf = np.real(np.fft.ifft(fft_x * np.conj(fft_x))[:n])
    acf /= acf[0]

    # Sum autocorrelation up to first negative pair
    tau = 1.0
    for k in range(1, n // 2):
        if k + 1 < n and acf[k] + acf[k + 1] < 0:
            break
        tau += 2 * acf[k]

    return float(n / max(tau, 1.0))


def summarize_mcmc(result: MCMCResult) -> dict:
    """Compute summary statistics for MCMC result.

    Returns dict with mean, std, ESS, R-hat, and credible intervals
    for each parameter.
    """
    summary = {}
    for name, samples in result.samples.items():
        flat = samples.flatten()
        entry = {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "median": float(np.median(flat)),
            "ci_2.5": float(np.percentile(flat, 2.5)),
            "ci_97.5": float(np.percentile(flat, 97.5)),
            "ess": effective_sample_size(flat),
        }

        # R-hat if multiple chains
        if samples.ndim == 2 and samples.shape[0] > 1:
            entry["rhat"] = compute_rhat(samples)
        else:
            entry["rhat"] = None

        summary[name] = entry

    return summary


def posterior_predictive_check(
    result: MCMCResult,
    likelihood_fn,
    data,
    n_sim: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """Simulate data from the posterior and compare to observed.

    Args:
        result: MCMC result with posterior samples.
        likelihood_fn: ModelLikelihood object.
        data: Observed TimeSeriesData.
        n_sim: Number of posterior predictive simulations.
        rng: Random number generator.

    Returns:
        Dict with simulated datasets and summary statistics.
    """
    rng = rng or np.random.default_rng(42)
    flat_samples = {k: v.flatten() for k, v in result.samples.items()}
    n_total = len(next(iter(flat_samples.values())))

    simulated = []
    indices = rng.choice(n_total, size=min(n_sim, n_total), replace=False)

    for idx in indices:
        theta = np.array(
            [flat_samples[name][idx] for name in likelihood_fn.param_names]
        )
        # Here we would simulate from the model with these params
        # For now, return the log-likelihood at each posterior sample
        ll = likelihood_fn(theta)
        simulated.append(ll)

    return {
        "simulated_ll": np.array(simulated),
        "mean_ll": float(np.mean(simulated)),
        "std_ll": float(np.std(simulated)),
    }
