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
| `signaling`  | `SignalingConfig` | see below     | Intracellular signaling dynamics settings |
| `coupling`   | `CouplingConfig`  | see below     | Signaling-to-fate coupling settings  |

## `DynamicsConfig`

| Field               | Type            | Default    | Description                          |
|----------------------|----------------|------------|--------------------------------------|
| `states`             | `list[str]`    | `["P","Q"]`| Active cell states (P, Q, A, R)      |
| `density_dependent`  | `bool`         | `false`    | Enable logistic density dependence   |
| `carrying_capacity`  | `float | null` | `null`     | Carrying capacity K (cells)          |
| `clearance_rate`     | `float`        | `0.1`      | Apoptotic cell clearance rate (1/h)  |
| `default_birth_base` | `float`        | `0.04`     | Baseline birth rate (1/h) used for default simulation `RateSet` |
| `default_death_base_p` | `float`      | `0.01`     | Baseline death rate (1/h) for P used in default simulation `RateSet` |
| `default_death_base_q` | `float`      | `0.005`    | Baseline death rate (1/h) for Q used in default simulation `RateSet` |

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

## `SignalingConfig`

| Field          | Type                  | Default          | Description |
|----------------|-----------------------|------------------|-------------|
| `enabled`      | `bool`               | `false`          | Enable signaling-aware simulation/inference paths |
| `model`        | `str`                | `"none"`         | Signaling model identifier (`"none"` or `"toy_mapk_akt"`) |
| `initial_state`| `dict[str, float]`   | `{}`             | Initial signaling node activities |
| `parameters`   | `dict[str, float]`   | `{}`             | Signaling model parameters |
| `observed_nodes` | `list[str]`        | `[]`             | Node names expected in signaling measurements |

## `CouplingConfig`

| Field       | Type                       | Default    | Description |
|-------------|----------------------------|------------|-------------|
| `enabled`   | `bool`                     | `false`    | Enable signaling-to-fate coupling |
| `function`  | `str`                      | `"hill"`   | Coupling family (`"hill"` or `"logistic"`) |
| `targets`   | `list[str]`                | `[]`       | Targets to modulate: broad (`birth`, `death`, `transition`) or specific (`death:P`, `transition:P->R`) |
| `parameters`| `dict[str, float]`         | `{}`       | Coupling hyper-parameters |
| `max_effect_by_target` | `dict[str, float]` | `{}` | Optional per-target effect scales overriding global `parameters.max_effect` |
| `ec50_by_target` | `dict[str, float]` | `{}` | Optional per-target EC50 overrides for Hill coupling |
| `hill_by_target` | `dict[str, float]` | `{}` | Optional per-target Hill-coefficient overrides for Hill coupling |
| `k_by_target` | `dict[str, float]` | `{}` | Optional per-target logistic slope overrides (`parameters.k`) |
| `center_by_target` | `dict[str, float]` | `{}` | Optional per-target logistic midpoint overrides (`parameters.center`) |

### Coupling Tuning Guidance

For practical initialization, use these approximate ranges before fitting:

- Hill coupling:
  - `max_effect`: `0.1` to `2.0`
  - `ec50`: `1e-3` to `10.0`
  - `hill`: `0.5` to `4.0`
- Logistic coupling:
  - `max_effect`: `0.1` to `2.0`
  - `k`: `0.1` to `10.0`
  - `center`: `0.0` to `1.0`
- Per-target overrides:
  - `max_effect_by_target`: `0.0` to `3.0`
  - `ec50_by_target`: `1e-3` to `20.0`
  - `hill_by_target`: `0.5` to `5.0`
  - `k_by_target`: `0.1` to `20.0`
  - `center_by_target`: `-1.0` to `2.0`

Programmatic helper:

```python
from umimic.pipeline.config import CouplingConfig

print(CouplingConfig.recommended_parameter_ranges())
```

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
  default_birth_base: 0.04
  default_death_base_p: 0.01
  default_death_base_q: 0.005

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

signaling:
  enabled: true
  model: toy_mapk_akt
  initial_state:
    mapk: 0.2
    akt: 0.1
  parameters:
    decay: 0.4

coupling:
  enabled: true
  function: hill
  targets: [birth, death:P, transition:P->R]
  parameters:
    max_effect: 0.8
    ec50: 1.0
    hill: 2.0
  max_effect_by_target:
    birth: 0.4
    death:P: 1.2
    transition:P->R: 0.6
  ec50_by_target:
    birth: 0.2
    death:P: 1.5
  hill_by_target:
    birth: 2.5
    transition:P->R: 1.2
  k_by_target:
    birth: 3.0
  center_by_target:
    birth: 0.25
```
