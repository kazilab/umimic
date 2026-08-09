"""Birth and death rates from clone-size distributions.

Why this module exists
----------------------
Separating birth from death using the *variance of bulk counts* is the route
`mode="moment"` takes, and it is underpowered at any realistic counting noise:
the process variance is ~1.5% of the total observation variance at the package
defaults, so a doubling of turnover is worth ~1 nat across an entire dataset.
See `umimic/inference/SCIENTIFIC_ASSUMPTIONS.md` section 2a.

Clonal data does not have that problem, because the information sits in a
*first-order* observable rather than a second-order one. For a linear
birth-death process started from a single cell, the clone size at time `t`
follows a closed-form law (Kendall 1948):

    alpha = d (e^{rt} - 1) / (b e^{rt} - d)          r = b - d
    beta  = b (e^{rt} - 1) / (b e^{rt} - d)

    P(Z = 0) = alpha
    P(Z = n) = (1 - alpha)(1 - beta) beta^{n-1}       n >= 1

`alpha` is the extinction probability, and it tends to `d/b` for a supercritical
process. So the *fraction of clones that died out* reads off d/b directly,
without ever touching a variance. Two observables -- the zero class and the
geometric decay of the positive class -- determine two unknowns, and the
mapping inverts in closed form (:func:`moments_to_rates`).

What this module does not do
----------------------------
It assumes each clone is founded by one cell, clones are independent, and
rates are constant over the interval. It takes *cell* counts per clone. Read
counts from barcode sequencing are not cell counts: they are a
depth-normalised, noisy, zero-inflated view of them, and the zero class -- the
most informative one -- is exactly where sequencing dropout does its damage.
:func:`fit_clone_sizes` therefore accepts an explicit `detection` probability
so that "not observed" is not silently equated with "extinct"; supply it from
a spike-in or a rarefaction analysis, or treat the result as a bound.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

logger = logging.getLogger(__name__)


@dataclass
class ClonalFit:
    """Birth and death rates estimated from a clone-size distribution."""

    birth_rate: float
    death_rate: float
    #: Extinction probability implied by the fit, P(clone dead by t).
    extinction_prob: float
    #: d/b -- the quantity the zero class pins down most directly.
    death_birth_ratio: float
    n_clones: int
    n_extinct: int
    log_likelihood: float
    converged: bool
    #: Standard errors from the observed information, when computable.
    se: dict[str, float] | None = None

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"ClonalFit(b={self.birth_rate:.5g}, d={self.death_rate:.5g}, "
            f"d/b={self.death_birth_ratio:.4g}, "
            f"extinct={self.n_extinct}/{self.n_clones})"
        )


def clone_size_params(b: float, d: float, t: float) -> tuple[float, float]:
    """Return (alpha, beta) for a linear birth-death process at time `t`.

    `alpha` is P(extinct by t); `beta` is the geometric ratio of the surviving
    size distribution. The critical case b == d is handled by its limit,
    alpha = beta = bt/(1 + bt), rather than by dividing by r = 0.
    """
    if b < 0 or d < 0:
        raise ValueError(f"Rates must be non-negative, got b={b}, d={d}.")
    if t <= 0:
        raise ValueError(f"Time must be positive, got {t}.")

    r = b - d
    if abs(r) * t < 1e-8:                      # critical / near-critical limit
        denom = 1.0 + b * t
        return b * t / denom, b * t / denom

    ert = np.exp(r * t)
    denom = b * ert - d
    if denom <= 0:                             # numerically degenerate
        return np.nan, np.nan
    return d * (ert - 1.0) / denom, b * (ert - 1.0) / denom


def moments_to_rates(alpha: float, beta: float, t: float) -> tuple[float, float]:
    """Invert (alpha, beta) back to (b, d) -- the closed form.

    This is the identifiability statement made concrete: two observables map
    to two rates with no approximation and no iteration.

        r = log((1 - alpha) / (1 - beta)) / t
        b = r beta / ((1 - beta)(e^{rt} - 1))
        d = b - r
    """
    if not (0.0 <= alpha < 1.0) or not (0.0 <= beta < 1.0):
        return np.nan, np.nan
    ratio = (1.0 - alpha) / (1.0 - beta)
    if ratio <= 0:
        return np.nan, np.nan
    r = np.log(ratio) / t
    if abs(r) * t < 1e-8:
        b = beta / ((1.0 - beta) * t)
        return b, b
    b = r * beta / ((1.0 - beta) * (np.exp(r * t) - 1.0))
    return b, b - r


def clone_size_logpmf(sizes: np.ndarray, b: float, d: float, t: float) -> np.ndarray:
    """Log-pmf of clone sizes under a linear birth-death process."""
    sizes = np.asarray(sizes)
    alpha, beta = clone_size_params(b, d, t)
    if not np.isfinite(alpha) or not np.isfinite(beta):
        return np.full(sizes.shape, -np.inf)
    alpha = np.clip(alpha, 1e-300, 1 - 1e-15)
    beta = np.clip(beta, 1e-300, 1 - 1e-15)

    out = np.empty(sizes.shape, dtype=float)
    zero = sizes == 0
    out[zero] = np.log(alpha)
    pos = ~zero
    if np.any(pos):
        out[pos] = (
            np.log1p(-alpha)
            + np.log1p(-beta)
            + (sizes[pos] - 1) * np.log(beta)
        )
    return out


def fit_clone_sizes(
    sizes,
    t: float,
    detection: float = 1.0,
    b0: float = 0.05,
    d0: float = 0.01,
) -> ClonalFit:
    """Estimate birth and death rates from observed clone sizes.

    Args:
        sizes: Observed cells per clone, including zeros for clones that were
            seeded and are no longer present. **The zero class carries most of
            the information about d/b, so omitting it biases the fit toward
            low death.**
        t: Elapsed time.
        detection: Probability that a surviving clone is observed at all.
            With `detection < 1` an observed zero means "extinct OR missed",
            and the likelihood accounts for both:
            P(observe 0) = alpha + (1 - alpha) * (1 - detection).
            Sizes of *detected* clones are otherwise assumed unbiased.
        b0, d0: Starting values.

    Returns:
        ClonalFit. `se` comes from the observed information when the Hessian
        is positive definite.
    """
    sizes = np.asarray(sizes, dtype=float)
    if sizes.ndim != 1 or sizes.size == 0:
        raise ValueError("`sizes` must be a non-empty 1-D array of clone sizes.")
    if np.any(sizes < 0) or np.any(sizes != np.round(sizes)):
        raise ValueError("Clone sizes must be non-negative integers.")
    if not 0.0 < detection <= 1.0:
        raise ValueError(f"detection must be in (0, 1], got {detection}.")

    n_zero = int(np.sum(sizes == 0))
    positive = sizes[sizes > 0]

    def nll(theta: np.ndarray) -> float:
        b, d = np.exp(theta)
        alpha, beta = clone_size_params(b, d, t)
        if not np.isfinite(alpha) or not np.isfinite(beta):
            return 1e12
        alpha = np.clip(alpha, 1e-300, 1 - 1e-15)
        beta = np.clip(beta, 1e-300, 1 - 1e-15)
        # An unobserved clone is extinct, or alive but missed.
        p_zero = alpha + (1.0 - alpha) * (1.0 - detection)
        ll = n_zero * np.log(p_zero)
        if positive.size:
            ll += positive.size * (
                np.log1p(-alpha) + np.log(detection) + np.log1p(-beta)
            )
            ll += float(np.sum(positive - 1.0)) * np.log(beta)
        return -ll if np.isfinite(ll) else 1e12

    best = None
    for bs in (b0, b0 * 2, b0 / 2):
        for ds in (d0, d0 * 3, d0 / 3):
            res = optimize.minimize(
                nll, np.log([bs, ds]), method="Nelder-Mead",
                options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-10},
            )
            if best is None or res.fun < best.fun:
                best = res

    b, d = np.exp(best.x)
    alpha, _ = clone_size_params(b, d, t)

    se = None
    try:                                        # observed information in log space
        eps = 1e-4
        H = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                xpp, xpm, xmp, xmm = (best.x.copy() for _ in range(4))
                xpp[i] += eps; xpp[j] += eps
                xpm[i] += eps; xpm[j] -= eps
                xmp[i] -= eps; xmp[j] += eps
                xmm[i] -= eps; xmm[j] -= eps
                H[i, j] = (nll(xpp) - nll(xpm) - nll(xmp) + nll(xmm)) / (4 * eps**2)
        cov = np.linalg.inv(H)                  # delta method back to natural scale
        se = {
            "birth_rate": float(np.sqrt(abs(cov[0, 0])) * b),
            "death_rate": float(np.sqrt(abs(cov[1, 1])) * d),
        }
    except Exception:                           # pragma: no cover - diagnostics only
        pass

    return ClonalFit(
        birth_rate=float(b),
        death_rate=float(d),
        extinction_prob=float(alpha),
        death_birth_ratio=float(d / b) if b > 0 else np.nan,
        n_clones=int(sizes.size),
        n_extinct=n_zero,
        log_likelihood=float(-best.fun),
        converged=bool(best.success),
        se=se,
    )


def sample_clone_sizes(
    b: float, d: float, t: float, n_clones: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw clone sizes from the exact law -- no simulation required.

    Sampling the closed form rather than running a Gillespie ensemble is exact
    for a linear birth-death process and many orders of magnitude cheaper,
    which is what makes 10^4-10^6 clone studies tractable.
    """
    alpha, beta = clone_size_params(b, d, t)
    if not np.isfinite(alpha) or not np.isfinite(beta):
        raise ValueError(f"Degenerate parameters: b={b}, d={d}, t={t}.")
    survives = rng.random(n_clones) >= alpha
    sizes = np.zeros(n_clones, dtype=np.int64)
    n_alive = int(np.sum(survives))
    if n_alive:
        # Surviving size is Geometric(1 - beta) on {1, 2, ...}.
        sizes[survives] = rng.geometric(1.0 - beta, size=n_alive)
    return sizes
