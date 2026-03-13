# API Reference

This page provides an overview of U-MIMIC's public Python API, organized by module.

## Pipeline

### `umimic.pipeline.experiment.Experiment`

The main user-facing class that ties all components together.

```python
from umimic.pipeline.config import ExperimentConfig
from umimic.pipeline.experiment import Experiment

config = ExperimentConfig(name="my_exp")
exp = Experiment(config)
```

**Methods:**

- `simulate(rate_set=None, method=None, concentrations=None)` -- Run forward simulation.
  Returns `SimulationResult` or `dict[float, SimulationResult]` for dose-response.
- `generate_synthetic(rate_set=None)` -- Generate synthetic `ExperimentalDataset`.
- `fit(data, **kwargs)` -- Run inference. Returns `InferenceResult`.

### `umimic.pipeline.config`

- `load_config(path)` -- Load and validate a YAML config file into `ExperimentConfig`.
- `save_config(config, path)` -- Save an `ExperimentConfig` to YAML.
- `ExperimentConfig` -- Top-level Pydantic model (see {doc}`configuration`).

## Dynamics

### `umimic.dynamics.states`

- `CellType` -- Enum: `P` (proliferating), `Q` (quiescent), `A` (apoptotic), `R` (resistant).
- `ModelTopology` -- Defines which states and transitions are active.
  Factory methods: `two_state()`, `three_state()`, `four_state()`.

### `umimic.dynamics.rates`

- `RateSet` -- Complete parameter set for concentration-modulated birth, death,
  and transition rates. See docstring for the inference parameter naming convention.
  Factory methods: `cytotoxic_drug()`, `cytostatic_drug()`.
- `DoseResponseFunction` -- Abstract base for dose-response curves.
- `EmaxHill` -- Emax/Hill model: `Emax * C^Hill / (EC50^Hill + C^Hill)`.
- `FourParameterLogistic` -- 4PL sigmoidal model.
- `ConstantRate` -- Constant (no drug modulation).

### `umimic.dynamics.ode_system`

- `CellDynamicsODE` -- Deterministic ODE solver using `scipy.integrate.solve_ivp`.
  Methods: `solve(y0, t_span, t_eval)`, `solve_dose_response(y0, t_span, concentrations, t_eval)`.

### `umimic.dynamics.gillespie`

- `GillespieSimulator` -- Exact stochastic simulation (Gillespie SSA).
  Method: `simulate(y0, t_max, t_eval)`.

### `umimic.dynamics.tau_leaping`

- `TauLeapingSimulator` -- Approximate stochastic simulation.
  Method: `simulate(y0, t_max, t_eval)`.

### `umimic.dynamics.moment_equations`

- `MomentODE` -- Linear Noise Approximation: solves mean + covariance dynamics.
  Method: `solve(mu0, t_span, t_eval)`.

## Pharmacokinetics

### `umimic.pk.exposure`

- `ExposureProfile` -- Unified drug concentration interface.
  - `ExposureProfile.constant(concentration)` -- In-vitro constant exposure.
  - `ExposureProfile.from_pk(pk_model, dosing)` -- In-vivo PK-driven exposure.
  - `concentration(t)` -- Get concentration at time `t`.

### `umimic.pk.compartment`

- `OneCompartmentPK(vd, ke, ka=None)` -- One-compartment PK model.
- `TwoCompartmentPK(vc, vp, cl, q, ka=None)` -- Two-compartment PK model.

### `umimic.pk.dosing`

- `DosingSchedule` -- Dosing event schedule.
  Factory methods: `constant_invitro(conc)`, `single_bolus(dose, time)`,
  `repeated(dose, interval, n_doses)`, `oral_repeated(...)`.

## Observations

### `umimic.observations.base`

- `ObservationModel` -- Abstract base class.
  Methods: `log_likelihood(obs, latent, params)`, `sample(latent, params)`, `param_names()`.

### `umimic.observations.cell_counts`

- `CellCountObservation(overdispersion)` -- Negative Binomial observation model
  for discrete cell counts. Supports LNA-informed variance.

### `umimic.observations.bli`

- `BLIObservation(alpha, sigma_log)` -- Bioluminescence imaging observation model.

### `umimic.observations.tumor_volume`

- `TumorVolumeObservation(beta, sigma_v)` -- Tumor volume observation model.

### `umimic.observations.multimodal`

- `MultimodalObservation(models)` -- Combines multiple observation models.

## Inference

### `umimic.inference.likelihood`

- `ModelLikelihood(topology, data, param_names, mode, observation_model)` --
  Central log-likelihood function. Groups replicates by concentration to share
  ODE solutions. Callable: `likelihood(theta) -> float`.

### `umimic.inference.mle`

- `MLEstimator(likelihood, bounds, priors, method)` -- Maximum likelihood estimation.
  Methods: `fit(initial_guess, n_restarts) -> MLEResult`.

### `umimic.inference.mcmc`

- `MCMCSampler(likelihood, priors, backend)` -- MCMC sampling (emcee or PyMC).
  Methods: `sample(n_samples, n_chains, n_warmup) -> MCMCResult`.

### `umimic.inference.priors`

- `PriorSpec` -- Prior distribution specification.
  Methods: `log_prior(params) -> float`, `sample(rng) -> dict`.
  Factory: `PriorSpec.default_invitro()`.

## Data

### `umimic.data.schemas`

- `TimeSeriesData` -- Single replicate: `times`, `observations` dict, `concentration`.
- `ExperimentalDataset` -- Collection of `TimeSeriesData` with metadata.

### `umimic.data.synthetic`

- `SyntheticDataGenerator(rate_set, topology, obs_model, rng)` --
  Methods: `generate_invitro_plate(...)`, `generate_invivo_cohort(...)`.

### `umimic.data.loaders`

- `load_csv(path, data_config)` -- Load experimental data from CSV.

### `umimic.data.public_datasets`

- `load_bestdr(data_root, cell_line, drug)` -- BESTDR breast cancer data.
- `load_phenopop(data_root, population)` -- PhenoPop Ba/F3 imaging data.
- `load_tshs_tumor(data_root, treatment_group)` -- TSHS xenograft tumor volumes.
- `load_hafner_gr(data_root, cell_line, drug)` -- Hafner/Niepel GR metrics.
- `load_nci60(data_path, cell_line)` -- NCI-60 growth inhibition.
- `list_available_datasets()` -- List datasets and their download status.

## Result Types

Defined in `umimic.types`:

- `SimulationResult` -- Forward simulation output (`times`, `populations` dict).
- `EnsembleResult` -- Multiple stochastic trajectories.
- `MLEResult` -- MLE output (`parameters`, `log_likelihood`, `aic`, `bic`, `se`, `converged`).
- `MCMCResult` -- MCMC output (`samples`, diagnostics).
- `InferenceResult` -- Unified container (`method`, `mle`, `mcmc`, `context`).
