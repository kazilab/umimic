# Changelog

All notable changes to U-MIMIC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.4]

> **Results from this release will differ from 0.0.2.** Several numerical
> defects were corrected. Any quantitative result produced with 0.0.2 or
> earlier should be regenerated. The affected areas are listed under
> "Corrected results" below.

### Corrected modelling assumptions

- **The resistant state could not proliferate.** `R` was absent from
  `division_states` and had no `death_base` entry, making it a non-dividing,
  non-dying sink reachable only through `P -> R`. A resistant population could
  never sustain or expand itself, so no four-state run could reproduce the
  outgrowth of a resistant clone under treatment. `R` now divides, has its own
  death rate, and does not inherit the drug's birth or death modulation; see
  `RateSet.resistant_clone()` and the `resistant_fitness_cost` setting.
- **Carrying capacity counted apoptotic cells.** The density term summed every
  state, so corpses suppressed division until they were cleared. After a
  cytotoxic pulse this produced artefactual growth inhibition -- in a test case
  a population at 30% of carrying capacity *shrank*. Only space-occupying
  states now enter the crowding total (`density_counts_apoptotic=False`), and
  the moment-equation Jacobian uses the matching density gradient.
- **`net_growth_rate` was not the growth rate.** It returned `b(C) - d_P(C)`,
  ignoring transitions out of P and death in every other state. For the
  default P/Q model that is 0.030 /h against a true asymptotic rate of
  0.0254 /h -- doubling times of 23 h versus 27 h. Added
  `RateSet.asymptotic_growth_rate()` (dominant eigenvalue of the low-density
  rate matrix) and `doubling_time()`; `net_growth_rate` remains but documents
  what it is not. The `b0 = 0.04  # doubling time ~17h` comment described the
  division time, not the net doubling time, and has been corrected.
- **Cytotoxic kill was silently confined to P.** `cytotoxic_drug()` and
  `mixed_drug()` modulated death for P only, leaving quiescent cells at
  baseline death under any dose. That is a defensible model of a
  cell-cycle-specific agent but was never stated. It is now the explicit
  `quiescent_sensitivity` argument (default 0, i.e. Q refractory) with the
  trade-off documented.
- **The LNA variance signature reached only cell counts.** BLI and tumour
  volume accepted `process_variance` and discarded it, so multimodal
  "variance fusion" fused nothing outside the count modality. Both now fold it
  into the log-scale variance via the delta method (`Var / N**2`).
- **The likelihood changed statistical model with the solver.** ODE mode used
  NegBin with variance `mu + mu^2/phi`; moment mode used a Gaussian with
  `process + mu^2/phi`, dropping the Poisson counting term. MLE and MCMC could
  therefore disagree for reasons unrelated to the science. The Gaussian branch
  now uses `process + mu + mu^2/phi`, which reduces exactly to the NegBin
  variance at zero process variance.
- **Every cell started in P.** Both simulation and inference put the whole
  initial count in the proliferating state, biasing transition-rate estimates
  and early growth. `ModelLikelihood(initial_fractions=...)` now accepts
  `"stable"` (the relaxed state distribution, via
  `RateSet.stable_state_fractions()`), an explicit mapping, or the previous
  all-P default, which is documented as a structural assumption.
- **Tissue attenuation double-counted path length.** Depth was taken as
  `reference_depth + radius` and used in a point-source exponential, which
  evaluates attenuation at the tumour centroid. Because `exp` is convex the
  cells nearest the surface dominate the real signal, so this overstates loss,
  and increasingly so as the tumour grows -- a growing tumour could look like
  progressive cell loss. Attenuation is now averaged over the emitting sphere
  in closed form, evaluated by a series expansion at small optical radius to
  avoid catastrophic cancellation (agrees with a 40-digit reference to 1e-9
  across nine orders of magnitude).
- **Ki-67 counted only P.** The marker labels cycling cells, so with a
  proliferating resistant compartment `P / viable` undercounts the
  proliferative fraction by exactly the R population. It is now summed over
  the topology's division states.
- **Biomarkers reported 50% positive at extinction**, and
  `biomarker_type="custom"` silently returned a constant 0.5. An extinct
  population now yields NaN and contributes nothing to the likelihood, and
  unsupported biomarker types are rejected at construction.
- **The toy MAPK/AKT drug term pointed the wrong way.** `d_mapk = drive * c`
  made drug *increase* pathway activity, and since rate coupling multiplies by
  `1 + effect`, higher concentration then increased proliferation -- backwards
  for the inhibitors most oncology agents are. The node now relaxes toward a
  drug-shifted set point with an explicit `direction`, defaulting to
  `"inhibitory"`, and activity is bounded in [0, 1]. The model remains a
  plumbing scaffold and says so.
- **Default tumour volume scale was ~100x off.** `volume_beta = 1e-3` mm^3/cell
  implies about 1000 cells/mm^3; tumour tissue is nearer 1e5-1e6 cells/mm^3.
  The default is now `1e-5` mm^3/cell, so a 100 mm^3 tumour corresponds to
  ~1e7 cells. Any absolute volume-to-cell-number calibration, and any BLI
  attenuation that depends on it, changes accordingly.

### Corrected results

- **Linear noise approximation means.** `MomentODE` used the Jacobian as the
  mean drift (`dmu/dt = J @ mu`). This is correct only for linear models; with
  density dependence it integrates `r*mu*(1 - 2*mu/K)`, so the mean saturated at
  **K/2 instead of K** (a ~48% error at carrying capacity). The drift and the
  Jacobian are now computed separately: `dmu/dt = f(t, mu)` and
  `dSigma/dt = J Sigma + Sigma J^T + D`.
- **Pharmacokinetics depended on the output grid.** The internal PK state was
  carried forward from the last requested `t_eval` point rather than from the
  end of each dose interval, so the same model and dosing gave concentrations
  differing by ~29% between sparse and dense grids. Integration now always runs
  to every dose boundary and samples requested times from a dense interpolant.
- **Simultaneous doses were silently dropped.** Doses were keyed by time in a
  dict, so two doses at the same time kept only the last. They are now summed.
- **Multimodal inference used only cell counts.** BLI, tumor volume, and
  biomarker observations never entered the likelihood. All configured
  modalities now contribute. `Experiment.fit()` also passes the observation
  models it builds to the likelihood; previously the orchestrator (and so the
  CLI) dropped them and fitted counts alone even when BLI and volume were
  configured.
- **The LNA variance signature was off on the default path.** The forward
  model was tied to the inference mode (`"ode" if mode == "mle"`), so the
  default MLE route discarded the process variance that separates birth from
  death. `inference.forward_mode` now controls this independently and defaults
  to `"moment"`. The default parameter set remains cytotoxic-only and is
  documented as such; `inference.parameter_set: mechanism` selects a set with
  birth-modulation terms (and matching priors) that can distinguish cytostatic
  from cytotoxic action.
- **Process variance was mis-projected.** The likelihood used the sum of all
  covariance entries (the variance of the *total* population) for a viable-cell
  observation. It now projects through the observation operator, `H Sigma H^T`.
- **Apoptotic state was hard-coded to index 2.** For a `[P, Q, R]` topology
  this counted resistant cells as dead. State indices are now derived from
  `ModelTopology`.
- **Configured states were replaced by their count.** `states: [P, Q, R]` built
  `[P, Q, A]` because the topology was chosen by the number of states.
- **Configured clearance never reached the simulation**, which used the
  `RateSet` default instead.
- **Tau-leaping** could advance past `t_max`, recorded post-leap states at
  requested times, and clipped negative populations after over-consuming cells.
- **Particle filter** propagated particles on a local clock (so time-varying
  exposure was evaluated at the wrong time), overwrote weights instead of
  updating them recursively, and reset to uniform weights on total degeneracy.
- **PMCMC** omitted the Hastings term for its lognormal random-walk proposal
  and shared one RNG stream between proposals and likelihood estimates.
- **Luciferin peak time** returned 0 whenever `ka < ke`.

### Phenotype plasticity and resistance modelling

- **Drug-induced transitions are now expressible.** The transition law was
  purely multiplicative, `u = u0 * (1 + m)`, so a route absent without
  treatment (`u0 = 0`) stayed zero at every dose -- `0 * anything = 0`. The
  law is now `u = u0 * f(C) * (1 + m(C)) + a(C)`, where the additive
  `transition_induction` term creates genuinely de novo treatment-induced
  transitions. This is what the persister-to-resistant route requires.
- **Transition suppression has a proper representation.** `HillFoldChange` is
  a non-negative multiplier that may rise or fall with concentration, so
  "drug blocks Q -> P resensitisation" is a fold-change rather than a
  negative-valued modulation. Modulator-direction validation now exempts
  fold-changes and points users at them; previously the only way to write
  suppression was the sign hack the validator (correctly) rejected.
- **Per-phenotype rate profiles.** `PhenotypeRateProfile` and
  `TransitionRateProfile` group each phenotype's baseline fitness and drug
  response, replacing four parallel dicts that had to be kept in step by
  hand. `RateSet.from_profiles()` builds from them and, given a topology,
  **refuses to let a dividing state inherit P's drug response** -- previously
  a state with no profile silently acquired the sensitivity a resistant
  compartment is supposed to lack.
- **`ModelTopology.persister_resistance()`** adds the P <-> Q -> R topology.
  The existing `four_state()` routes resistance directly from cycling
  sensitive cells (P -> R), skipping the drug-tolerant intermediate that the
  experimental literature places at the centre of acquired resistance. Q is
  slow-cycling rather than arrested, so it divides.
  - **R -> Q reversion is opt-in and defaults to off.** A per-cell reversion
    rate is meaningful only for an epigenetically stable R. For a genetic R,
    apparent resensitisation during a drug holiday is competitive dilution
    driven by R's fitness cost, which the birth and death rates already
    carry; modelling it as a transition double-counts it. It is also close to
    unidentifiable: in a sensitivity analysis of a 28-day design it was a
    numerically exact null direction, and even with abundant R and a
    cycling-fraction marker it reached only 0.26 of the identifiable span.
- **`dynamics.transitions` and `dynamics.transition_rates` in configuration.**
  The orchestrator derived its edge set from the *number* of states, so routes
  outside the canonical set were unreachable from configuration at all. Edges
  listed without an explicit rate default to zero rather than inheriting the
  P <-> Q defaults.
- **`RateSet.gr_value()`** reports the Hafner normalized growth-rate
  inhibition metric from the dominant eigenvalue, with an explicit warning
  that it is an asymptotic summary: with a fitter resistant state it can call
  a drug ineffective that clears the sensitive population within the assay.
- **`RateSet.low_density_rate_matrix()`** extracts the first-moment generator
  that `asymptotic_growth_rate` and `stable_state_fractions` each rebuilt.
- **`umimic.inference.identifiability`** is new: a practical-identifiability
  diagnostic reporting which parameters a given design can actually constrain.
  A model can fit well while most of its parameters are set by the prior, and
  nothing in the fit reveals it. On a single-concentration count-only design
  the diagnostic correctly reports EC50 and Hill as unconstrained.

### Limitations that were not safely documentable

Two entries on the known-limitations list turned out to defeat the feature
they qualified rather than merely bound it.

- **BLI attenuation used a single mean tumour volume.** Attenuation grows with
  tumour size -- that is the entire reason the model exists -- so applying one
  constant correction across a growing tumour erased the size-dependence and
  introduced a systematic tilt: over-correcting early and under-correcting
  late, which reads as a biological trend. Measured across a 50 -> 1600 mm^3
  trajectory the induced error ran from -63% to +63%. `BLIObservation` now
  indexes array-valued covariates per time point, and the likelihood passes
  per-point volumes instead of their mean.
- **Biomarkers were unreachable from configuration.** `BiomarkerObservation`
  existed and was topology-aware, but `Experiment._build_components` had no
  branch for it, so `modalities: [biomarker]` silently produced a model
  without it. Added, along with `observations.biomarker_type` and validation
  that rejects unknown modality names rather than dropping them.
- **Small-count warning for the Gaussian branch.** Matching the first two
  moments does not make a Gaussian a negative binomial: at mu = 5 with
  phi = 10 it places ~3.4% of its mass below zero and omits the count
  distribution's skew, and this is also where the LNA supplying the process
  variance breaks down. The moment-mode likelihood now warns once below an
  expected count of 20 and points at `mode="ode"` or a particle filter.

- **`RateSet.finite_horizon_gr()`** measures GR over the actual assay window
  rather than from the dominant eigenvalue, which is what a 72-hour screen
  reports and what published GR values should be compared against. The
  asymptotic and finite-horizon answers can disagree in *sign*: for the
  persister model at C=1 the eigenvalue gives GR = +0.81 (a drug that barely
  works, because a rare resistant clone governs the long run) while the
  72-hour assay gives GR = -0.27 (net kill). Without a slow-emerging state
  the two converge as the window lengthens.

Genuinely closed as documented limitations: `linear_chain` (rejected at
configuration) and signalling coupling (toy scaffold, ODE-only, refused
elsewhere).

### Inference caught up with the forward model

The simulation side gained resistance, persistence and plasticity before the
likelihood did. That asymmetry was the most dangerous kind: the fit ran and
converged.

- **A resistant compartment was fitted as an immortal, fully fit clone.**
  `_build_rate_set` built a sensitive P/Q model only, so with `states: [P, Q,
  A, R]` the R state inherited P's baseline division rate and received *no*
  death rate. Under drug at C=100 that gave R birth 0.04 and death 0.0 while
  P was killed at 0.06 -- R absorbed the entire trajectory. Added
  `RESISTANCE_PARAM_NAMES` and `PERSISTER_PARAM_NAMES` (with matching
  `PriorSpec.default_resistance()` / `default_persister()`), covering
  per-state division (`b0_Q`, `b0_R`), per-state death (`d0_R`), residual
  drug sensitivity (`emax_death_Q`, `emax_death_R`), the persister route
  (`u_QR`) and its drug-induced component (`induced_QR`).
- **`ModelLikelihood` now refuses to fit a state it cannot describe.** The
  check is role-based: a dividing state needs its own `b0_<X>`, a dying state
  needs `d0_<X>` unless it has a documented fallback (Q falls back to
  `d0_P * 0.5`). P/Q models are unaffected; a four-state model with the
  default parameter set is rejected with a message naming the sets that fit.
- **`generate_synthetic()` discarded every modality but the first.** It passed
  `list(models.values())[0]` to the generator, so a multimodal configuration
  emitted single-modality data and a simulate-then-fit round trip did not
  exercise the described experiment. It now hands over the full
  `MultimodalObservation`.
- **Configuration combinations that fit the wrong model are rejected.**
  `drug_mechanism: cytostatic` with a cytotoxic-only `parameter_set` would
  attribute reduced division to increased death; `parameter_set: mechanism`
  with `forward_mode: ode` is under-identified, because the mean confounds
  b and d and only the LNA process variance separates them. Both now raise.
- **`initial_fractions` reaches the high-level API.** `ModelLikelihood`
  supported it, but `Experiment.fit()` never passed it and `simulate()` and
  `generate_synthetic()` still placed every cell in P. Added
  `dynamics.initial_fractions` (`"proliferating"`, `"stable"`, or explicit
  fractions), threaded through all three.
- **`quiescent_sensitivity` is configurable.** It existed on
  `RateSet.cytotoxic_drug()` but not in `DynamicsConfig`, so the configured
  path always left Q refractory with no way to change it.
- **`RateSet.all_rates_at()` used the full state sum for the density term**,
  including apoptotic cells, so the diagnostic reported birth = 0 after a
  cytotoxic pulse even with the viable population far below K. It now uses
  `topology.density_total()`, matching the solvers.
- **`CellDynamicsODE` docstring** still wrote `dR/dt = uPR*P - dR*R`, omitting
  the division term the code applies now that R proliferates.

### Hierarchical inference

`HierarchicalModel` had never been reviewed and carried the same defect
families corrected elsewhere in this release:

- **Walker axis flattened.** `get_chain(flat=True)` collapsed the walker
  dimension, so R-hat and ESS were undefined for every parameter of a
  hierarchical fit. Draws are now `(n_walkers, n_draws)`, with log-likelihood
  and log-posterior stored separately.
- **Non-reproducible.** Walker initialisation used bare `np.random.randn` and
  an unseeded `priors.sample()`; there was no `rng` argument at all. A seed
  now determines the fit.
- **Cell-counts only.** No observation model was passed to the per-group
  likelihoods, so a multimodal dataset was fitted on one modality -- the same
  gap fixed in `Experiment.fit`.
- **Forward mode hardcoded to `"ode"`**, discarding the LNA process variance.
  Now selectable, defaulting to `"moment"`.
- **Silent fallbacks.** A parameter absent from the priors was initialised
  from a hardcoded `0.01`. Missing priors are now an error, as are
  parameters listed as both shared and group-varying.
- **Group label collisions.** Series sharing a `group_id` overwrote one
  another in the returned samples; labels are now made unique and prefer
  `replicate_id`.

### Fixed

- **Posterior predictive checks scored the anchor observation.** The check
  replicated every non-missing point, including the one consumed by the
  initial condition. That point is reproduced almost exactly by construction,
  flattering the fit and skewing the Bayesian p-value. Both the likelihood and
  the check now share `ModelLikelihood.scored_mask()`.
- **`division_states` differed by construction route.**
  `ModelTopology.persister_resistance()` gave `[P, Q, R]` while the equivalent
  configuration gave `[P, R]`, so the same conceptual model had different
  dynamics and a different Ki-67 fraction depending on how it was built.
  `dynamics.division_states` now makes this explicit.
- **Drug-dependent transitions were unreachable from configuration.**
  `transition_induction` and `transition_factor` had no configuration
  surface, leaving drug-induced plasticity library-only. Added
  `dynamics.induced_transitions` and `dynamics.transition_fold_change`.
- `HillFoldChange`, `PhenotypeRateProfile` and `TransitionRateProfile` are now
  exported from `umimic.dynamics`.
- `save_config` now dumps in JSON mode. Tuple-valued fields (such as
  `dynamics.transitions`) were emitted as `!!python/tuple`, which
  `yaml.safe_load` refuses to read, so an affected configuration saved
  cleanly and then failed to load.

### Consistency and API honesty

- **Two clearance parameters, one of them dead.** `ModelTopology` carried an
  `apoptotic_clearance_rate` that no solver read, alongside the
  `RateSet.clearance_rate` the simulators actually use. Setting the topology
  field looked effective and changed nothing. The field is removed and
  assigning it now raises, pointing at the setting that works.
- **Decreasing dose-response curves used as rate modulators.** RateSet's rate
  laws (`b0 * (1 - m)`, `d0 + m`, `u0 * (1 + m)`) require `m` to be the
  *magnitude of the drug effect*, so a modulator must increase with
  concentration. `FourParameterLogistic` decreases in its usual orientation:
  passed as `birth_modulation` it set the untreated division rate to zero and
  made higher doses *increase* growth. Modulators are now validated at
  construction, and `FourParameterLogistic.as_effect()` reorients a fitted
  viability curve into a valid modulator.
- **Synthetic generation ignored the configured modalities.**
  `generate_invitro_plate` emitted cell counts only, even when built with a
  multimodal observation model, so a multimodal configuration silently
  produced single-modality data. Worse, `generate_invivo_cohort` inlined its
  own BLI and volume formulas -- with a hardcoded 1e-3 mm^3/cell volume scale
  -- rather than using the observation models, so generated data need not
  match the likelihood later fitted to it. Both paths now sample from the
  configured models and accept a `modalities` argument.
- **`inference.mode` advertised modes the orchestrator does not implement.**
  `"smc"` and `"hierarchical"` passed configuration validation and then failed
  inside `Experiment.fit()` with "Unknown inference mode", after the data had
  been loaded. The config Literal now admits only `"mle"` and `"mcmc"`; the
  error message points at `umimic.inference.ParticleMCMC` and
  `umimic.inference.hierarchical` for the direct APIs.
- **Luciferin kinetics units documented as phenomenological.** `dose` is an
  administered amount while `km` is an intracellular concentration, with no
  volume of distribution connecting them, so `C_luc` is in arbitrary units and
  only `dose / km` is meaningful. `signal_fraction` is peak-normalized and is
  the intended interface; the docstring now says so rather than implying
  calibrated biochemistry.

### Removed

- **PyMC backend.** Its model attached the data through a constant
  `pm.Potential`, so it sampled the prior and the observations had no effect on
  the posterior. `backend="pymc"` now raises `NotImplementedError` explaining
  why. It will return with a differentiable forward model or a tested PyTensor
  likelihood wrapper.
- **`dashboard` CLI command and `[dashboard]` extra.** No dashboard application
  ships with the package, so the command could only fail.
- **`dynamics.linear_chain` (phase-type dwell times) is now rejected.** No
  solver consumed it, so configuring it silently left dwell times exponential.
  It raises a validation error until the feature is implemented.
- **Signaling coupling with a non-ODE method is now rejected.** Rate coupling
  is threaded only through the deterministic ODE; the stochastic simulators,
  the LNA and the inference likelihood ignore it, so enabling signalling and
  running Gillespie, tau-leaping or a fit returned uncoupled results that
  looked coupled.

### Added

- `umimic --version`.
- `umimic generate` now writes `dataset.csv` and `config.yaml` to `--output`,
  in a documented interchange format that `umimic fit --data` reads back.
- Exact stochastic simulation under time-varying exposure via Extrande
  thinning; `GillespieSimulator` selects it automatically and `exposure_mode`
  makes the contract explicit. The direct method is no longer described as
  exact for continuously varying PK.
- Tau-leaping critical-reaction handling and SSA fallback for small
  populations; populations can no longer go negative.
- Infusion dosing (`iv_infusion` with a duration) and per-dose route handling.
- Split rank-normalized R-hat and bulk ESS; both return `None` when undefined
  instead of reporting 1.0.
- A posterior predictive check that actually simulates replicate datasets and
  reports Bayesian p-values per modality.
- Missing observations (NaN), `replicate_id`, and `units` in `TimeSeriesData`,
  with schema validation; information criteria count only real observations.
- Reproducible sampling: `MCMCSampler(rng=...)` fully determines a run, and
  posterior draws keep their `(n_chains, n_draws)` shape with log-likelihood
  and log-posterior stored separately.
- `RateSet.mixed_drug()`, so the `mixed` CLI choice does something.
- 45 analytical-agreement acceptance tests (`tests/test_numerics/`) covering
  the logistic solution, Bateman and one/two-compartment IV analytics, PK grid
  invariance, birth-death moments, tau-leaping convergence, and configuration
  fidelity.
- `LICENSE` (MIT).

## [Unreleased]

### Added
- CHANGELOG.md and CONTRIBUTING.md for project governance.
- Expanded Sphinx documentation: quickstart, API reference, configuration guide.

### Fixed
- CLI logging now uses a package-scoped logger (`umimic`) instead of overwriting
  the root logger via `logging.basicConfig(force=True)`.
- Dashboard command no longer relies on a single hard-coded path; searches
  multiple candidate locations and reports clear errors.
- Lambda closures in likelihood and experiment modules now use default-argument
  binding to avoid late-binding capture bugs in loops.
- `load_config()` validates file existence and YAML content before parsing,
  with actionable error messages.
- Public dataset loader (`_find_data_dir`) now lists available datasets and
  suggests the correct download command on failure.
- `UMIMIC_DATA_ROOT` env var now warns and falls back gracefully when pointing
  to a non-existent directory.

### Improved
- CLI exception handling catches specific error types (`FileNotFoundError`,
  `ValueError`, `ImportError`, `KeyboardInterrupt`) with user-friendly messages
  instead of a bare `except Exception`.
- Parameter naming convention documented in `RateSet` docstring with a full
  inference-vector-to-model-field mapping table.
- Inference parameter comments in `DEFAULT_PARAM_NAMES` now reference the
  corresponding `RateSet` and observation model fields.

## [0.0.2] - 2026-03-01

### Added
- CLI entry point (`umimic`) with `simulate`, `fit`, `generate`, `dashboard` commands.
- Run logging with `--log-file` and `--log-level` options.
- Pydantic-based YAML configuration (`ExperimentConfig`).
- Experiment orchestrator class.
- ODE, Gillespie, and tau-leaping simulation engines.
- Moment equations (Linear Noise Approximation).
- One-compartment and two-compartment PK models with dosing schedules.
- Unified `ExposureProfile` abstraction for in-vitro / in-vivo exposure.
- Observation models: cell counts (Negative Binomial), BLI, tumor volume, biomarkers.
- Multimodal observation model combiner.
- MLE (multi-start), MCMC (emcee/PyMC), SMC, and Kalman inference engines.
- Hierarchical Bayesian inference.
- Prior specification and log-prior evaluation.
- Convergence diagnostics.
- Synthetic data generator (`generate_invitro_plate`, `generate_invivo_cohort`).
- Public dataset loaders: BESTDR, PhenoPop, Hafner/Niepel GR, TSHS Tumor Growth, NCI-60.
- Visualization: trajectories, dose-response curves, posteriors, diagnostics.
- Read the Docs configuration and Sphinx/MyST documentation.
- GitHub Actions workflows for trusted PyPI publishing and release smoke tests.

## [0.0.1] - 2026-02-15

### Added
- Initial project structure and packaging.
- Core cell-state dynamics module.
- Basic ODE solver.
