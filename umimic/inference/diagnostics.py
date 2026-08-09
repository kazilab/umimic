"""Convergence diagnostics and posterior predictive checks.

R-hat and ESS follow Vehtari et al. (2021), "Rank-normalization, folding, and
localization: An improved R-hat for assessing convergence of MCMC": chains are
split in half, rank-normalized, and the potential scale reduction factor is
computed on the normalized ranks. ArviZ is used when available; otherwise an
equivalent implementation in this module is used.

Diagnostics that cannot be computed return ``None``, never 1.0. Reporting
R-hat = 1.0 for a single chain would assert convergence on no evidence.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

from umimic.types import MCMCResult

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only when arviz is installed and healthy
    import arviz as az

    _HAS_ARVIZ = True
except Exception as exc:  # noqa: BLE001 - a broken arviz install must not break us
    az = None
    _HAS_ARVIZ = False
    _ARVIZ_ERROR = exc

_warned_no_arviz = False


def _warn_once_no_arviz() -> None:
    """Warn the first time a diagnostic falls back off ArviZ.

    Not debug-level: which implementation computed a published R-hat is not an
    implementation detail, and a silently broken arviz install is easy to miss.
    Emitted on first use rather than at import so merely importing umimic stays
    quiet.
    """
    global _warned_no_arviz
    if _HAS_ARVIZ or _warned_no_arviz:
        return
    _warned_no_arviz = True
    logger.warning(
        "ArviZ unavailable (%s); using umimic's own R-hat/ESS implementation.",
        _ARVIZ_ERROR,
    )


def _as_2d(chains: np.ndarray) -> np.ndarray | None:
    """Coerce samples to (n_chains, n_draws), or None if not possible."""
    arr = np.asarray(chains, dtype=float)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    if arr.ndim == 2:
        return arr
    return None


def _split_chains(arr: np.ndarray) -> np.ndarray:
    """Split each chain in half, doubling the chain count.

    Splitting detects within-chain non-stationarity that a plain
    between-chain comparison misses.
    """
    n_chains, n_draws = arr.shape
    half = n_draws // 2
    if half < 2:
        return arr
    return np.concatenate([arr[:, :half], arr[:, half : 2 * half]], axis=0)


def _rank_normalize(arr: np.ndarray) -> np.ndarray:
    """Rank-normalize pooled draws to make R-hat robust to heavy tails."""
    flat = arr.reshape(-1)
    ranks = stats.rankdata(flat)
    normalized = stats.norm.ppf((ranks - 0.375) / (len(flat) + 0.25))
    return normalized.reshape(arr.shape)


def compute_rhat(chains: np.ndarray) -> float | None:
    """Split rank-normalized R-hat.

    Args:
        chains: (n_chains, n_draws) array of MCMC samples.

    Returns:
        R-hat, or None when it is undefined (fewer than two chains after
        splitting, too few draws, or zero within-chain variance).
    """
    arr = _as_2d(chains)
    if arr is None:
        return None

    if arr.shape[1] < 4:
        return None

    split = _split_chains(arr)
    if split.shape[0] < 2:
        return None

    if _HAS_ARVIZ:
        try:
            value = float(az.rhat(arr))
            if np.isfinite(value):
                return value
        except Exception:  # pragma: no cover - fall through to local impl
            logger.debug("ArviZ rhat failed; using local implementation.")

    _warn_once_no_arviz()
    normalized = _rank_normalize(split)
    n_chains, n_draws = normalized.shape

    chain_means = normalized.mean(axis=1)
    chain_vars = normalized.var(axis=1, ddof=1)

    W = float(np.mean(chain_vars))
    B = float(n_draws * np.var(chain_means, ddof=1))

    if not np.isfinite(W) or W <= 0:
        return None

    var_hat = (n_draws - 1) / n_draws * W + B / n_draws
    return float(np.sqrt(var_hat / W))


def effective_sample_size(samples: np.ndarray) -> float | None:
    """Bulk effective sample size across chains.

    Args:
        samples: (n_chains, n_draws) or (n_draws,) array.

    Returns:
        Estimated ESS, or None when it cannot be computed.
    """
    arr = _as_2d(samples)
    if arr is None or arr.shape[1] < 4:
        return None

    if _HAS_ARVIZ:
        try:
            value = float(az.ess(arr))
            if np.isfinite(value) and value > 0:
                return value
        except Exception:  # pragma: no cover
            logger.debug("ArviZ ess failed; using local implementation.")

    _warn_once_no_arviz()
    normalized = _rank_normalize(_split_chains(arr))
    n_chains, n_draws = normalized.shape

    # Mean autocovariance across chains, via FFT. Note this is the
    # autocovariance, not each chain's self-normalized autocorrelation:
    # normalizing per chain by its own acov[0] discards the between-chain
    # variance entirely, so chains stuck in different modes each look like
    # clean iid draws and ESS is reported as the full sample size.
    acov_sum = np.zeros(n_draws)
    for c in range(n_chains):
        x = normalized[c] - normalized[c].mean()
        f = np.fft.fft(x, n=2 * n_draws)
        acov = np.real(np.fft.ifft(f * np.conj(f))[:n_draws]) / n_draws
        acov_sum += acov
    mean_acov = acov_sum / n_chains

    # Combine within- and between-chain variance exactly as split-Rhat does,
    # then form rho_t = 1 - (W - mean_acov_t) / var_hat (Vehtari et al. 2021).
    chain_vars = normalized.var(axis=1, ddof=1)
    W = float(np.mean(chain_vars))
    if not np.isfinite(W) or W <= 0:
        return None

    B = float(n_draws * np.var(normalized.mean(axis=1), ddof=1))
    var_hat = (n_draws - 1) / n_draws * W + B / n_draws
    if not np.isfinite(var_hat) or var_hat <= 0:
        return None

    rho = 1.0 - (W - mean_acov) / var_hat
    rho[0] = 1.0

    # Geyer initial positive sequence: sum paired autocorrelations while positive.
    tau = 1.0
    for k in range(1, n_draws - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair <= 0:
            break
        tau += 2.0 * pair

    total = n_chains * n_draws
    if tau <= 0:
        return None
    # ESS above the raw sample size is an artifact of antithetic draws; Stan
    # caps it at N log10(N), and reporting more draws than were taken is
    # never useful here.
    return float(min(total / tau, total * np.log10(max(total, 10))))


def summarize_mcmc(result: MCMCResult) -> dict:
    """Compute summary statistics for an MCMC result.

    Returns a dict with mean, std, ESS, R-hat, and credible intervals for each
    parameter. `rhat` and `ess` are None when undefined.
    """
    summary = {}
    for name, samples in result.samples.items():
        arr = _as_2d(samples)
        flat = np.asarray(samples, dtype=float).reshape(-1)

        entry = {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "median": float(np.median(flat)),
            "ci_2.5": float(np.percentile(flat, 2.5)),
            "ci_97.5": float(np.percentile(flat, 97.5)),
            "ess": effective_sample_size(arr) if arr is not None else None,
            "rhat": compute_rhat(arr) if arr is not None else None,
        }
        if entry["rhat"] is None:
            entry["rhat_note"] = (
                "undefined: needs at least 2 chains (or splittable draws) "
                "with non-zero within-chain variance"
            )
        summary[name] = entry

    return summary


def posterior_predictive_check(
    result: MCMCResult,
    likelihood_fn,
    data=None,
    n_sim: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """Simulate replicate datasets from the posterior and compare to observed.

    For each posterior draw the forward model is solved and synthetic
    observations are generated from the observation model. Observed and
    replicated data are then compared through summary statistics and Bayesian
    p-values. A p-value near 0 or 1 indicates the model cannot reproduce that
    feature of the data.

    Args:
        result: MCMC result with posterior samples.
        likelihood_fn: ModelLikelihood carrying the model and the data.
        data: Unused; the likelihood's own data is used so that replicates
            match the fitted series exactly.
        n_sim: Number of posterior predictive replicates.
        rng: Random number generator.

    Returns:
        Dict with replicated datasets, observed/replicated summaries, and
        Bayesian p-values per modality.
    """
    rng = rng or np.random.default_rng(42)

    flat = result.flat_samples()
    n_total = len(next(iter(flat.values())))
    n_draws = min(n_sim, n_total)
    indices = rng.choice(n_total, size=n_draws, replace=False)

    modalities = likelihood_fn._active_modalities
    models = likelihood_fn._modality_models

    replicated: dict[str, list[np.ndarray]] = {m: [] for m in modalities}
    observed: dict[str, np.ndarray] = {}
    for modality in modalities:
        # Same mask the likelihood scores: the anchor point is excluded when
        # it sets the initial condition, since replicating a point the model
        # was conditioned on reproduces it by construction.
        obs = np.concatenate(
            [
                series.observations[modality][
                    likelihood_fn.scored_mask(series, modality)
                ]
                for series in likelihood_fn.data_list
                if series.has_modality(modality)
            ]
        )
        observed[modality] = obs

    log_liks = []

    for idx in indices:
        theta = np.array(
            [flat[name][idx] for name in likelihood_fn.param_names]
        )
        params = likelihood_fn.theta_to_params(theta)
        rate_set = likelihood_fn._build_rate_set(params)

        draw: dict[str, list[float]] = {m: [] for m in modalities}
        ok = True

        for conc, series in likelihood_fn._conc_groups.items():
            times = likelihood_fn._group_times[conc]
            if len(times) < 2:
                continue
            # Same initial-condition contract as ModelLikelihood: pass the
            # rate set so "stable" (and any rate-dependent) fractions match
            # the fitted model rather than falling back to pure-P.
            initials = [
                likelihood_fn._initial_state(d, rate_set) for d in series
            ]
            shared = all(np.allclose(x, initials[0]) for x in initials)
            try:
                if shared:
                    solutions = [
                        likelihood_fn._solve_forward(
                            rate_set, conc, times, initials[0]
                        )
                    ] * len(series)
                else:
                    solutions = [
                        likelihood_fn._solve_forward(
                            rate_set, conc, times, x0
                        )
                        for x0 in initials
                    ]
            except (RuntimeError, ValueError):
                ok = False
                break

            for s, (t_sol, means, covs) in zip(series, solutions):
                pos = np.searchsorted(t_sol, s.times)
                pos = np.clip(pos, 0, len(t_sol) - 1)
                latent = np.maximum(means[pos], 0.0)
                for modality in modalities:
                    if not s.has_modality(modality):
                        continue
                    mask = likelihood_fn.scored_mask(s, modality)
                    model = models[modality]
                    # Match the likelihood noise model: fold LNA process
                    # variance into replicates when moment mode supplies it.
                    process_vars = None
                    if covs is not None and hasattr(model, "project_variance"):
                        from umimic.inference.likelihood import (
                            MODALITY_OBSERVABLE,
                        )

                        observable = MODALITY_OBSERVABLE.get(modality)
                        if observable is not None:
                            process_vars = np.asarray(
                                model.project_variance(
                                    observable, covs[pos][mask]
                                )
                            )
                    scored_states = latent[mask]
                    for i, state in enumerate(scored_states):
                        pv = (
                            float(process_vars[i])
                            if process_vars is not None
                            else None
                        )
                        draw[modality].append(
                            float(
                                model.sample(
                                    state, rng, params, process_variance=pv
                                )
                            )
                        )

        if not ok:
            continue

        for modality in modalities:
            replicated[modality].append(np.asarray(draw[modality], dtype=float))

        log_liks.append(likelihood_fn(theta))

    summary = {}
    for modality in modalities:
        reps = [r for r in replicated[modality] if r.size == observed[modality].size]
        if not reps:
            continue
        rep = np.vstack(reps)
        obs = observed[modality]

        summary[modality] = {
            "observed_mean": float(np.mean(obs)),
            "replicated_mean": float(np.mean(rep)),
            "observed_std": float(np.std(obs)),
            "replicated_std": float(np.mean(np.std(rep, axis=1))),
            # Bayesian p-values: P(T(y_rep) >= T(y_obs)) under the posterior.
            "p_value_mean": float(
                np.mean(np.mean(rep, axis=1) >= np.mean(obs))
            ),
            "p_value_std": float(np.mean(np.std(rep, axis=1) >= np.std(obs))),
            "replicates": rep,
        }

    return {
        "modalities": summary,
        "log_likelihood": np.asarray(log_liks, dtype=float),
        "n_draws": len(log_liks),
    }
