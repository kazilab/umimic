# Scientific assumptions — `umimic.inference`

This note records the statistical and modelling contracts of the inference
layer: what the likelihood actually scores, how parameter vectors reach the
dynamics, and where backends agree or diverge. For the biology of rates and
simulators, see `umimic/dynamics/SCIENTIFIC_ASSUMPTIONS.md`.

**Scope:** `likelihood.py`, `priors.py`, `mle.py`, `mcmc.py`, `kalman.py`,
`smc.py`, `hierarchical.py`, `diagnostics.py`, `identifiability.py`.

---

## 1. Likelihood as the single bridge

All point estimation and sampling that go through `ModelLikelihood` follow the
same pipeline:

1. Map θ → named parameters (`param_names` order is the only source of truth).
2. Build a `RateSet` via the shared `build_rate_set` (not a second mapping).
3. Solve the forward model (`mode="moment"` LNA or `mode="ode"` mean field).
4. Score every **configured and present** modality at each non-missing time.

A second, partial copy of the parameter → rate wiring is treated as a
critical failure mode: names would be sampled but never read, returning their
prior while other parameters absorb the misfit. That is why
`build_rate_set`, `missing_state_params`, and `resolve_initial_fractions` are
module-level and shared with PMCMC.

**Unknown parameter names are rejected at construction** of `ModelLikelihood`
and `ParticleMCMC` (`KNOWN_PARAM_NAMES`). A typo is an error, not a flat
prior.

---

## 2. Parameter sets (what the data can identify)

| Set | Contents | Scientific role |
|-----|----------|-----------------|
| `default` | Cytotoxic death modulation + P/Q transitions + overdispersion | Pure cytotoxic drug; **cannot** separate cytostatic vs cytotoxic action |
| `mechanism` | Default + birth Emax/EC50/Hill | Both mechanisms; **requires** `mode="moment"` — mean confounds \(b-d\), LNA variance scales with \(b+d\) |
| `resistance` | + \(b0_R\), \(d0_R\), residual R death, \(u_{PR}\) | Avoids an immortal, fully fit R clone |
| `persister` | + Q rates, induced Q→R, Q drug sensitivity | Persister route; induction efficacy by default |

**Defaults and fall-backs that are modelling choices, not data:**

- Absent `d0_Q` → fixed at \(0.5\,d0_P\) (warned once).
- Drug-induced transitions (`induced_*`) share cytotoxic EC50/Hill unless
  `ec50_induction` / `hill_induction` are free.
- Per-state drug death (`emax_death_Q`, `emax_death_R`) shares P’s potency and
  differs only in efficacy.
- Q/R with explicit `b0_Q` / `b0_R` do **not** inherit P’s cytostatic birth
  modulation (explicit `None`).

Topology states without their own rate parameters are rejected when they
would become immortal clones (dividing state missing `b0_*`, dying state
missing `d0_*` except documented Q fallback).

---

## 3. Observation and replicate contracts

### Multimodal

Modalities present in the data **and** configured in the observation model
contribute under conditional independence given the latent state:

\[
\log L = \sum_{\text{series}}\;\sum_{\text{modality}}\;\sum_{\text{times}}
  \ell(y_{m,t}\mid x_t,\phi).
\]

Missing values (NaN) are skipped. Process variance from the LNA, when
available, is projected as \(H\Sigma H^\top\) onto that modality’s observable
(not a raw sum of covariance entries).

### Replicates

Series that share a concentration share one forward solve on the **union** of
their observation times. Each replicate is scored at its own times by exact
grid membership (no nearest-neighbour snap). Different measured initial
counts force separate solves.

### Initial condition and the anchor

The first non-missing observation of `anchor_modality` (default
`cell_counts`) sets the total initial count and, with
`condition_on_first=True` (default), is **excluded** from the likelihood.
Using the same number both to condition and to score double-counts
information.

How that count is split across states (`initial_fractions`):

| Spec | Meaning |
|------|---------|
| `None` / `"proliferating"` | All cells in P — historical default, **structural**, biases early growth and transition rates |
| `"stable"` | Relaxed phenotype mix at the series concentration (needs rates) |
| dict / array | Explicit fractions (normalised) |

Per-series override: `metadata["initial_fractions"]`.

`n_observations` (AIC/BIC sample size) counts only scored points: non-missing
values in active modalities, minus anchors when conditioned.

---

## 4. Forward modes

| Mode | Mean | Process noise in LL | When to use |
|------|------|---------------------|-------------|
| `moment` | LNA \(\mu(t)\) | Yes (\(D\), projected) | Default; mechanism separation; bulk culture |
| `ode` | Deterministic ODE | No | Large \(N\), mean-only fits; **not** for `mechanism` set |

Failed solves or rate-law rejections (invalid Hill parameters, etc.) return
\(-\infty\), not an exception mid-optimiser.

---

## 5. Priors (`PriorSpec`)

- `log_prior` is the sum of scipy log-pdfs for names present in both the
  parameter dict and `distributions`. Names in θ with **no** prior entry are
  treated as flat (improper on the half-line after non-negativity checks in
  the likelihood). MCMC initialisation requires every free name to have a
  prior so walkers are well-defined.
- Defaults (`default_invitro`, `default_mechanism`, `default_resistance`,
  `default_persister`) are weakly informative lognormals / half-normals /
  uniforms matched to the parameter sets above.
- `from_posterior` fits lognormals to positive marginals for transfer
  learning (minimum \(\sigma\) floor).

---

## 6. Point estimation (`MLEstimator`)

- Objective: \(-\log L\) (MLE) or \(-\bigl(\log L + \log\pi\bigr)\) (MAP).
- Multi-start local search or differential evolution.
- AIC / BIC use scored \(n\) and free \(k = \dim\theta\).
- Standard errors from a numerical Hessian of the optimised objective
  (classical for MLE; for MAP this is a local Gaussian posterior scale, not
  a frequentist SE). Fixed finite-difference step can be unstable across
  orders-of-magnitude rate scales.

---

## 7. MCMC (`MCMCSampler`)

- Backend: **emcee** only. PyMC was withdrawn: an earlier model attached data
  via a constant `pm.Potential` and sampled the prior regardless of
  observations.
- Target: \(\log L(\theta) + \log\pi(\theta)\); log-likelihood stored
  separately as emcee blobs for information criteria and PPC.
- Sample layout: `(n_walkers, n_draws)` per parameter. Ensemble walkers are
  **not** independent chains; diagnostics are labelled accordingly.
- Walker count is raised to at least \(2\dim+2\) when needed.
- Initialisation follows `param_names` order (never prior dict insertion
  order).

---

## 8. Extended Kalman filter (`ExtendedKalmanFilter`)

“Fast mode” filtering on LNA moments:

1. **Predict** \(\mu,\Sigma\) between observation times with `MomentODE`.
2. **Update** once per available modality via `ObservationModel.linearize`
   (Gaussian approximation); Joseph form for \(\Sigma\).
3. Marginal LL = sum of Gaussian innovation terms.

**Contracts:**

- Modalities without a Gaussian linearization are **rejected at
  construction** (no silent drop).
- Failed moment solve → `diverged=True`, marginal LL \(=-\infty\).
- Pass `anchor_modality` / `anchor_index` when \(\mu_0\) was seeded from data
  so the anchor is not scored twice (parity with `ModelLikelihood`).

The EKF likelihood is **not** the same objective as the NegBin / lognormal
batch likelihood in `ModelLikelihood`; treat them as related but distinct.

---

## 9. Particle filter and PMCMC (`smc.py`)

“Robust mode” for small \(N\), strong nonlinearity, or LNA breakdown.

**Bootstrap filter**

- Particles: Gillespie SSA on absolute experiment time (exposure shifted so
  PK is not evaluated on a relative clock).
- Weights: recursive log-space update; logsumexp normalisation; systematic
  resampling when ESS is low.
- Complete degeneracy or SSA event-limit truncation → marginal LL \(=-\infty\).

**PMCMC**

- Metropolis–Hastings with unbiased (noisy) marginal LL from the particle
  filter.
- Proposal RNG and filter RNG are **independent streams** (pseudo-marginal
  correctness).
- Strictly positive parameters: lognormal random walk with Hastings
  correction \(\sum(\log\theta'-\log\theta)\).
- Parameters whose prior is finite at zero (e.g. half-normal Emax): reflected
  linear random walk (symmetric; boundary reachable).

Same rate builder, state checks, anchor, and `obs_params` forwarding as
`ModelLikelihood`.

---

## 10. Hierarchical model

Partial pooling across series:

\[
\begin{aligned}
\theta_{\mathrm{shared}} &\sim \pi,\\
\mu_p &\sim \pi,\quad
\tau_p &\sim \mathrm{HalfCauchy},\\
\theta_{i,p} &\sim \mathrm{LogNormal}(\log\mu_p,\,\tau_p),\\
y_i &\sim p(y\mid \theta_{\mathrm{shared}},\theta_i).
\end{aligned}
\]

Shared parameters may be zero (no effect); random-effect components are
strictly positive. Dimension grows with groups × random-effect parameters;
emcee needs many walkers. Default forward mode is `"moment"`.

---

## 11. Diagnostics and identifiability

**R-hat / ESS** (Vehtari et al. style: split, rank-normalise): undefined
cases return `None`, never a fake 1.0. Single long chains can still yield
split R-hat after splitting draws in half.

**Posterior predictive check**

- Draws θ from the posterior, resimulates observations from the observation
  model on the **same scored mask** as the likelihood (anchor excluded when
  conditioned).
- Initial states use the fitted rate set, so `initial_fractions="stable"`
  matches the fit (not a pure-P fallback).
- Per-concentration series with different initials get separate solves, as in
  the likelihood.
- Bayesian p-values near 0 or 1 flag systematic misfit for that summary.

**Practical identifiability**

- Local sensitivity of predicted (log) observations to log-parameters at a
  point θ; SVD of the sensitivity matrix.
- Scores in \([0,1]\) measure how much of each parameter lies in the
  identifiable subspace. Near-zero score ⇒ data do not constrain that
  direction; the posterior will track the prior.
- This is **not** structural identifiability. Zero parameters have undefined
  log-sensitivity and are reported unidentifiable.
- The likelihood helper uses the scored mask (anchor excluded).

---

## 12. Checklist before trusting a fit

1. Parameter set matches topology (resistance / persister when R or Q→R is
   present).
2. `mechanism` set only with `mode="moment"`.
3. Priors cover every free name if sampling; unknown names are errors.
4. Initial fractions: `"stable"` or explicit for passaged cultures; pure-P is
   a modelling claim.
5. Multimodal data: configure every modality you want scored.
6. MCMC: inspect acceptance, ESS, R-hat; remember walkers ≠ independent
   chains.
7. PPC: extreme p-values mean the observation model or dynamics cannot
   reproduce the data feature.
8. Identifiability report: do not over-interpret parameters with score ≈ 0.
9. EKF vs batch LL vs PMCMC: different approximations; do not mix objectives
   in model comparison without care.
10. Hierarchical: check that group labels and shared vs RE split match the
    experimental design.

---

## 13. What this layer deliberately does *not* claim

- Global structural identifiability of multitype branching models.
- That MLE standard errors are reliable under flat ridges or boundary
  parameters.
- That emcee has mixed because the mean trajectory looks good.
- Pharmacometric NLME with full covariate models (hierarchical here is a
  lighter partial-pooling scaffold).
- A differentiable / HMC path through the numerical likelihood (PyMC/NUTS
  not supported).

---

## References (conceptual)

- Multitype branching / LNA moments as a fast forward model for inference.
- Bootstrap particle filter; particle marginal Metropolis–Hastings
  (pseudo-marginal MCMC).
- Extended Kalman filter on continuous–discrete state space models.
- Vehtari et al. (2021), rank-normalized split R-hat and ESS.
- Practical identifiability via sensitivity / Fisher information geometry
  at an operating point.

Implementation tests live under `tests/test_inference/` and
`tests/test_numerics/` (semantics, parity across backends, acceptance).
