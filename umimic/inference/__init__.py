"""Inference entry points for likelihood, optimization, sampling, and diagnostics."""

from .identifiability import (
    IdentifiabilityReport,
    analyze_identifiability,
    likelihood_identifiability,
)
from .diagnostics import (
    compute_rhat,
    effective_sample_size,
    posterior_predictive_check,
    summarize_mcmc,
)
from .hierarchical import HierarchicalModel
from .kalman import ExtendedKalmanFilter
from .likelihood import ModelLikelihood
from .mcmc import MCMCSampler
from .mle import MLEstimator
from .priors import PriorSpec
from .smc import ParticleFilter, ParticleMCMC

__all__ = [
    # Likelihood & Priors
    "ModelLikelihood",
    "PriorSpec",
    # Point Estimation (MLE)
    "MLEstimator",
    # MCMC Sampling
    "MCMCSampler",
    # Kalman Filtering
    "ExtendedKalmanFilter",
    # SMC & Particle MCMC
    "ParticleFilter",
    "ParticleMCMC",
    # Hierarchical Bayesian Models
    "HierarchicalModel",
    # Diagnostics & Checks
    "compute_rhat",
    "effective_sample_size",
    "summarize_mcmc",
    "IdentifiabilityReport",
    "analyze_identifiability",
    "likelihood_identifiability",
    "posterior_predictive_check",
]
