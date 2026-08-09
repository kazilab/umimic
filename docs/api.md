# API reference

Public Python API overview by module. For formulas and contracts see
{doc}`scientific-assumptions`.

## Pipeline

### `umimic.pipeline.experiment.Experiment`

Primary orchestrator: builds topology, rates, observations, and exposure from
config; runs simulation, synthetic data, and fit.

```python
from umimic.pipeline.config import ExperimentConfig
from umimic.pipeline.experiment import Experiment

config = ExperimentConfig(name="my_exp")
exp = Experiment(config)
```

**Methods**

- `simulate(rate_set=None, method=None, concentrations=None)` — Forward
  simulation. With `concentrations`, returns plate-style
  `dict[float, SimulationResult]` (constant C per arm). Otherwise uses
  `self.exposure` (constant or PK).
- `generate_synthetic(rate_set=None)` — `ExperimentalDataset` from config.
- `fit(data, **kwargs)` — MLE or emcee MCMC → `InferenceResult`.

### `umimic.pipeline.config`

- `load_config(path)` / `save_config(config, path)` — YAML I/O (JSON-mode dump
  so transitions round-trip without `!!python/tuple`).
- `ExperimentConfig` and nested configs — see {doc}`configuration`.

### `umimic.pipeline.results`

- `save_result(result, path)` — JSON summary (MLE stats; MCMC mean/std/CI, not
  full chains).
- `load_result(path)` — `dict` (not a live `InferenceResult`).
- `compare_results({label: InferenceResult})` — side-by-side estimates.

### `umimic.pipeline.transfer`

- `TransferLearning(invitro_result, transfer_params, shrinkage)` —
  in vitro MCMC/MLE → lognormal priors. `shrinkage` ∈ (0, 1].
- `build_priors()` → `PriorSpec`; `summarize_transfer()`.

---

## Dynamics

### `umimic.dynamics.states`

- `CellType` — `P`, `Q`, `A`, `R`.
- `ModelTopology` — active states, transitions, division/death, density mask.
  Factories: `two_state()`, `three_state()`, `four_state()`,
  `persister_resistance()`.
- `StateVector` — snapshot helper.

### `umimic.dynamics.rates`

- `RateSet` — concentration-modulated birth, death, transitions.
  - Growth: `asymptotic_growth_rate(c, topology)`, `doubling_time`,
    `gr_value`, `finite_horizon_gr`, `stable_state_fractions`.
  - `net_growth_rate(c)` — **only** \(b - d_P\); not multi-state growth.
  - Factories: `cytotoxic_drug`, `cytostatic_drug`, `mixed_drug`,
    `resistant_clone`, `persister_resistance` / `from_profiles`.
- Dose–response: `EmaxHill`, `HillFoldChange`, `FourParameterLogistic`
  (use `.as_effect()` for modulators), `ConstantRate`.
- Profiles: `PhenotypeRateProfile`, `TransitionRateProfile`.

### Solvers

| Class | Module | Role |
|-------|--------|------|
| `CellDynamicsODE` | `ode_system` | Mean-field ODE; optional `rate_multiplier_fn` |
| `GillespieSimulator` | `gillespie` | SSA; Extrande if exposure varies |
| `TauLeapingSimulator` | `tau_leaping` | Hybrid leap + critical reactions |
| `MomentODE` | `moment_equations` | LNA mean + covariance |

---

## Pharmacokinetics

### `umimic.pk.exposure`

- `ExposureProfile.constant(c)` / `.from_pk(pk, dosing)`
- `concentration(t)` / `__call__(t)`
- `precompute(t_grid)` / `clear_cache()` — optional linear interpolation

### `umimic.pk.compartment`

- `OneCompartmentPK(vd, ke, ka=None, f_oral=1.0)`
- `TwoCompartmentPK(vc, vp, cl, q, ka=None, f_oral=1.0)`
- `solve(dosing, t_eval)` — multi-dose, grid-invariant integration

### `umimic.pk.dosing`

- `Dose`, `DosingSchedule` —
  `constant_invitro`, `single_bolus`, `repeated`, `oral_repeated`

### `umimic.pk.luciferin`

- `LuciferinKinetics` — phenomenological substrate timing (minutes)
- `TissueAttenuation` — point source or volume-averaged attenuation

---

## Observations

All models accept `topology=` so operators match the state layout.

| Class | Likelihood | Notes |
|-------|------------|--------|
| `CellCountObservation` | NegBin or Gaussian+process | Keys: `overdispersion` |
| `BLIObservation` | Lognormal (median scale) | Keys: `sigma_log_bli`; luciferin + attenuation |
| `TumorVolumeObservation` | Lognormal | Keys: `sigma_v`; `beta` calibration |
| `BiomarkerObservation` | Beta fractions | `ki67` / `caspase`; key `biomarker_precision` |
| `MultimodalObservation` | Sum of available modalities | Conditional independence given \(x\) |

`sample(..., process_variance=...)` matches the log-likelihood noise model
(used by posterior predictive checks).

---

## Inference

### `umimic.inference.likelihood`

- `ModelLikelihood(topology, data, param_names, mode, observation_model, …)`
- `PARAMETER_SETS`: `default`, `mechanism`, `resistance`, `persister`
- `build_rate_set(params)`, `KNOWN_PARAM_NAMES`, `resolve_initial_fractions`

### Point estimation and sampling

| Class | Role |
|-------|------|
| `MLEstimator` | Multi-start MLE / MAP |
| `MCMCSampler` | emcee only (`backend="emcee"`) |
| `ExtendedKalmanFilter` | LNA + Gaussian observation updates |
| `ParticleFilter` / `ParticleMCMC` | Bootstrap PF + PMCMC |
| `HierarchicalModel` | Partial pooling across series |
| `PriorSpec` | `default_invitro`, `default_mechanism`, `default_resistance`, `default_persister` |

### Diagnostics

- `compute_rhat`, `effective_sample_size`, `summarize_mcmc`
- `posterior_predictive_check` (scored mask + process variance when available)
- `analyze_identifiability`, `likelihood_identifiability`

Sample layout: `(n_chains, n_draws)` per parameter.

---

## Signaling

- `SignalingNetwork` — abstract interface
- `ToyMapkAktNetwork` — two-node scaffold (not a pathway model);
  `direction="inhibitory"` default

Coupling into cell rates is built in `Experiment` for ODE only.

---

## Visualization

| Function | Notes |
|----------|--------|
| `plot_population_trajectories` | ODE / single run |
| `plot_ensemble` | Percentile bands by default (`ci_method`) |
| `plot_dose_response_trajectories` | Multi-concentration curves |
| `plot_rate_dose_response` | Rates + asymptotic growth |
| `plot_net_growth_curve` | Asymptotic \(g(C)\); g0/g50 annotations |
| `plot_mechanism_comparison` | P birth vs death fold-change |
| `plot_posterior_marginals` / `plot_pair` | Pooled posteriors |
| `plot_trace` | **One line per chain/walker** |
| `plot_residuals` / `plot_fit_quality` | Exploratory fit checks |

---

## Data

### Schemas

- `TimeSeriesData` — times, observations dict, concentration, metadata
- `ExperimentalDataset` — collection of series

### Synthetic and public data

- `SyntheticDataGenerator` — `generate_invitro_plate`, `generate_invivo_cohort`
- `load_csv`, `load_bestdr`, `load_phenopop`, `load_tshs_tumor`,
  `load_hafner_gr`, `load_nci60`, `list_available_datasets`

---

## Result types (`umimic.types`)

| Type | Contents |
|------|----------|
| `SimulationResult` | `times`, `populations`, `viable`, `metadata` |
| `EnsembleResult` | trajectories; `mean()`, `std()` |
| `FilterResult` | EKF means/covs, marginal LL, `diverged` |
| `MLEResult` | parameters, LL, AIC/BIC, SE, converged |
| `MCMCResult` | samples `(n_chains, n_draws)`, traces, diagnostics |
| `InferenceResult` | `method`, `mle` and/or `mcmc`, `context` |
