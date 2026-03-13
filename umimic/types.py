"""Shared type definitions for U-MIMIC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SimulationResult:
    """Result from a forward simulation."""

    times: np.ndarray
    populations: dict[str, np.ndarray]  # state_name -> array of counts at each time
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def viable(self) -> np.ndarray:
        """Total viable cells (P + Q + R, excludes A)."""
        total = np.zeros_like(self.times, dtype=float)
        for name, pop in self.populations.items():
            if name != "A":
                total += pop
        return total

    @property
    def total(self) -> np.ndarray:
        total = np.zeros_like(self.times, dtype=float)
        for pop in self.populations.values():
            total += pop
        return total


@dataclass
class EnsembleResult:
    """Result from an ensemble of stochastic simulations."""

    times: np.ndarray
    trajectories: list[dict[str, np.ndarray]]  # list of population dicts per trajectory
    n_trajectories: int = 0

    def __post_init__(self):
        self.n_trajectories = len(self.trajectories)

    def mean(self) -> dict[str, np.ndarray]:
        """Mean populations across trajectories."""
        if not self.trajectories:
            return {}
        keys = self.trajectories[0].keys()
        return {
            k: np.mean([traj[k] for traj in self.trajectories], axis=0)
            for k in keys
        }

    def std(self) -> dict[str, np.ndarray]:
        """Standard deviation across trajectories."""
        if not self.trajectories:
            return {}
        keys = self.trajectories[0].keys()
        return {
            k: np.std([traj[k] for traj in self.trajectories], axis=0)
            for k in keys
        }

    def variance(self) -> dict[str, np.ndarray]:
        """Variance across trajectories."""
        if not self.trajectories:
            return {}
        keys = self.trajectories[0].keys()
        return {
            k: np.var([traj[k] for traj in self.trajectories], axis=0)
            for k in keys
        }


@dataclass
class FilterResult:
    """Result from Kalman filter forward pass."""

    times: np.ndarray
    filtered_means: np.ndarray  # (n_times, n_states)
    filtered_covs: np.ndarray  # (n_times, n_states, n_states)
    marginal_log_likelihood: float = 0.0
    innovations: np.ndarray | None = None


@dataclass
class MLEResult:
    """Result from maximum likelihood estimation."""

    parameters: dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    hessian: np.ndarray | None = None
    se: dict[str, float] | None = None
    converged: bool = True
    n_evaluations: int = 0


@dataclass
class MCMCResult:
    """Result from MCMC inference."""

    samples: dict[str, np.ndarray]  # param_name -> (n_chains, n_samples)
    log_likelihood_trace: np.ndarray | None = None
    n_chains: int = 1
    n_samples: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def posterior_mean(self) -> dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self.samples.items()}

    def posterior_std(self) -> dict[str, float]:
        return {k: float(np.std(v)) for k, v in self.samples.items()}

    def credible_interval(self, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
        lo = alpha / 2 * 100
        hi = (1 - alpha / 2) * 100
        return {
            k: (float(np.percentile(v, lo)), float(np.percentile(v, hi)))
            for k, v in self.samples.items()
        }


@dataclass
class InferenceResult:
    """Unified inference result container."""

    method: str  # "mle", "mcmc", "smc"
    mle: MLEResult | None = None
    mcmc: MCMCResult | None = None
    context: str = "in_vitro"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def point_estimates(self) -> dict[str, float]:
        if self.mle is not None:
            return self.mle.parameters
        if self.mcmc is not None:
            return self.mcmc.posterior_mean()
        return {}
