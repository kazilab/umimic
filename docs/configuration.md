# Configuration Reference

U-MIMIC uses YAML configuration files validated by [Pydantic](https://docs.pydantic.dev/).
All fields have sensible defaults, so you only need to specify what you want to change.

## Top-Level: `ExperimentConfig`

| Field        | Type              | Default       | Description                          |
|------------- |-------------------|---------------|--------------------------------------|
| `name`       | `str`             | `"experiment"`| Experiment name                      |
| `context`    | `str`             | `"in_vitro"`  | `"in_vitro"` or `"in_vivo"`         |
| `seed`       | `int`             | `42`          | Random seed for reproducibility      |
| `dynamics`   | `DynamicsConfig`  | see below     | Cell-state dynamics settings         |
| `pk`         | `PKConfig`        | see below     | Pharmacokinetic model settings       |
| `dosing`     | `DosingConfig`    | see below     | Dosing schedule settings             |
| `observations`| `ObservationConfig`| see below   | Observation model settings           |
| `inference`  | `InferenceConfig` | see below     | Inference engine settings            |
| `priors`     | `PriorConfig`     | see below     | Prior distribution settings          |
| `data`       | `DataConfig`      | see below     | Data loading settings                |
| `simulation` | `SimulationConfig`| see below     | Simulation / synthetic data settings |

## `DynamicsConfig`

| Field               | Type            | Default    | Description                          |
|----------------------|----------------|------------|--------------------------------------|
| `states`             | `list[str]`    | `["P","Q"]`| Active cell states (P, Q, A, R)      |
| `density_dependent`  | `bool`         | `false`    | Enable logistic density dependence   |
| `carrying_capacity`  | `float | null` | `null`     | Carrying capacity K (cells)          |
| `clearance_rate`     | `float`        | `0.1`      | Apoptotic cell clearance rate (1/h)  |

## `PKConfig`

| Field   | Type    | Default    | Description                              |
|---------|---------|------------|------------------------------------------|
| `model` | `str`   | `"none"`   | `"none"`, `"one_compartment"`, `"two_compartment"` |
| `vd`    | `float` | `10.0`     | Volume of distribution (1-compartment)   |
| `ke`    | `float` | `0.1`      | Elimination rate constant (1/h)          |
| `ka`    | `float` | `null`     | Absorption rate constant (oral dosing)   |
| `vc`    | `float` | `null`     | Central volume (2-compartment)           |
| `vp`    | `float` | `null`     | Peripheral volume (2-compartment)        |
| `cl`    | `float` | `null`     | Clearance (2-compartment)                |
| `q`     | `float` | `null`     | Inter-compartmental clearance            |

## `DosingConfig`

| Field           | Type            | Default      | Description                         |
|-----------------|-----------------|--------------|-------------------------------------|
| `type`          | `str`           | `"constant"` | `"constant"`, `"single_bolus"`, `"repeated_bolus"`, `"oral"` |
| `concentrations`| `list[float]`   | `null`       | In-vitro concentration list         |
| `dose_amount`   | `float`         | `null`       | Dose amount per administration      |
| `interval`      | `float`         | `null`       | Dosing interval (hours)             |
| `n_doses`       | `int`           | `null`       | Number of doses                     |
| `start_time`    | `float`         | `0.0`        | Time of first dose (hours)          |

## `ObservationConfig`

| Field                      | Type          | Default          | Description                     |
|---------------------------|---------------|------------------|---------------------------------|
| `modalities`               | `list[str]`  | `["cell_counts"]`| Enabled modalities              |
| `cell_count_overdispersion`| `float`      | `10.0`           | Negative Binomial overdispersion|
| `bli_alpha`                | `float`      | `1000.0`         | BLI photons-per-cell scaling    |
| `bli_sigma_log`            | `float`      | `0.3`            | BLI log-normal noise sigma      |
| `volume_beta`              | `float`      | `1e-3`           | Volume conversion factor        |
| `volume_sigma`             | `float`      | `0.2`            | Volume measurement noise        |
| `biomarker_precision`      | `float`      | `50.0`           | Biomarker precision parameter   |

## `InferenceConfig`

| Field         | Type    | Default    | Description                             |
|---------------|---------|------------|-----------------------------------------|
| `mode`        | `str`   | `"mle"`    | `"mle"`, `"mcmc"`, `"smc"`, `"hierarchical"` |
| `backend`     | `str`   | `"scipy"`  | `"scipy"`, `"emcee"`, `"pymc"`, `"particle"` |
| `n_samples`   | `int`   | `2000`     | MCMC samples per chain                  |
| `n_chains`    | `int`   | `4`        | Number of MCMC chains                   |
| `n_warmup`    | `int`   | `1000`     | MCMC warmup / burn-in samples           |
| `n_particles` | `int`   | `500`      | SMC particle count                      |
| `n_restarts`  | `int`   | `5`        | MLE multi-start restarts                |

## `SimulationConfig`

| Field           | Type    | Default       | Description                        |
|-----------------|---------|---------------|------------------------------------|
| `method`        | `str`   | `"gillespie"` | `"ode"`, `"gillespie"`, `"tau_leaping"` |
| `initial_cells` | `int`   | `100`         | Initial cell count                 |
| `t_max`         | `float` | `72.0`        | Simulation duration (hours)        |
| `dt_obs`        | `float` | `4.0`         | Observation interval (hours)       |
| `n_replicates`  | `int`   | `4`           | Replicates per condition           |
| `seed`          | `int`   | `42`          | Simulation random seed             |

## `PriorConfig`

Prior distributions for inference parameters. Each entry is a dict of distribution
parameters (keys depend on the distribution family).

| Field        | Default                                | Description                |
|-------------|----------------------------------------|----------------------------|
| `b0`        | `{scale: 0.04, s: 0.5}`              | Birth rate prior           |
| `d0_P`      | `{scale: 0.01, s: 0.5}`              | Death rate (P) prior       |
| `emax_death`| `{scale: 0.1}`                         | Emax death prior           |
| `ec50_death`| `{scale: 1.0, s: 1.0}`               | EC50 death prior           |
| `hill_death`| `{scale: 1.5, s: 0.3}`               | Hill coefficient prior     |

## Complete Example

```yaml
name: cytotoxic_breastcancer
context: in_vitro
seed: 123

dynamics:
  states: [P, Q]
  density_dependent: false

dosing:
  type: constant
  concentrations: [0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

observations:
  modalities: [cell_counts]
  cell_count_overdispersion: 15.0

simulation:
  method: gillespie
  initial_cells: 200
  t_max: 96.0
  dt_obs: 6.0
  n_replicates: 3

inference:
  mode: mle
  backend: scipy
  n_restarts: 10
```
