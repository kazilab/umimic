# Configuration reference

U-MIMIC uses YAML configuration files validated by
[Pydantic](https://docs.pydantic.dev/). All fields have defaults; specify only
what you need to change.

Cross-field checks reject combinations that would fit the wrong model (for
example cytostatic drug with a death-only parameter set, or
`parameter_set=mechanism` with `forward_mode=ode`). See
{doc}`scientific-assumptions` and the package note
`umimic/pipeline/SCIENTIFIC_ASSUMPTIONS.md`.

## Top-level: `ExperimentConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `"experiment"` | Experiment name |
| `context` | `str` | `"in_vitro"` | `"in_vitro"` or `"in_vivo"` |
| `seed` | `int` | `42` | Global RNG seed |
| `dynamics` | `DynamicsConfig` | see below | Cell-state dynamics |
| `pk` | `PKConfig` | see below | Pharmacokinetics |
| `dosing` | `DosingConfig` | see below | Dosing / concentrations |
| `observations` | `ObservationConfig` | see below | Observation models |
| `inference` | `InferenceConfig` | see below | Inference engine |
| `priors` | `PriorConfig` | see below | Prior stubs (pipeline uses set factories) |
| `data` | `DataConfig` | see below | Data loading |
| `simulation` | `SimulationConfig` | see below | Simulation / synthetic data |
| `signaling` | `SignalingConfig` | see below | Intracellular scaffold |
| `coupling` | `CouplingConfig` | see below | Signaling → rate multipliers |

## Alignment checklist

| Situation | Required settings |
|-----------|-------------------|
| States include **R** | Fit with `inference.parameter_set: resistance` or `persister` |
| Cytostatic or mixed drug | `parameter_set: mechanism` and `forward_mode: moment` |
| Signaling coupling | `simulation.method: ode`; `signaling.enabled` + `coupling.enabled` |
| Incomplete oral absorption | `pk.f_oral` in `(0, 1]`, oral dosing, `ka` set |
| Multimodal data | `observations.modalities` must match data columns |

## `DynamicsConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `states` | `list[str]` | `["P","Q"]` | Active states (`P`, `Q`, `A`, `R`) |
| `transitions` | `list[[src,tgt]]` or null | `null` | Explicit edges; default = canonical edges among active states |
| `transition_rates` | `dict[str,float]` or null | `null` | Baseline rates keyed `"P->Q"`; missing edges default to 0 |
| `induced_transitions` | `dict` or null | `null` | Additive drug-induced edges: `{emax, ec50?, hill?}` |
| `transition_fold_change` | `dict` or null | `null` | Multipliers: `{low?, high, ec50?, hill?}` |
| `division_states` | `list[str]` or null | `null` | Defaults to P and R if present |
| `density_dependent` | `bool` | `false` | Logistic crowding on birth |
| `carrying_capacity` | `float` or null | `null` | \(K\) (cells) |
| `density_counts_apoptotic` | `bool` | `false` | Whether corpses enter density (default: no) |
| `clearance_rate` | `float` | `0.1` | Apoptotic clearance (1/h) on `RateSet` |
| `drug_mechanism` | `str` or null | `null` | `"cytotoxic"`, `"cytostatic"`, `"mixed"` |
| `emax_death` / `ec50_death` / `hill_death` | `float` | `0.05` / `1` / `1.5` | Cytotoxic EmaxHill |
| `emax_birth` / `ec50_birth` / `hill_birth` | `float` | `0.8` / `1` / `1.5` | Cytostatic EmaxHill |
| `quiescent_sensitivity` | `float` | `0.0` | Q death Emax relative to P, in [0, 1] |
| `default_birth_base` | `float` | `0.04` | Baseline birth (1/h) |
| `default_death_base_p` | `float` | `0.01` | Baseline death P |
| `default_death_base_q` | `float` | `0.005` | Baseline death Q |
| `default_death_base_r` | `float` | `0.01` | Baseline death R |
| `resistant_fitness_cost` | `float` | `0.0` | Fractional cut of R birth vs P, in [0, 1) |
| `initial_fractions` | `str` / `dict` / null | `null` | `"proliferating"`, `"stable"`, or explicit fractions |
| `linear_chain` | — | — | **Rejected** if set (phase-type not implemented) |

Default rates are **illustrative** culture scales for simulation startup, not
cell-line truth.

## `PKConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"none"` | `"none"`, `"one_compartment"`, `"two_compartment"` |
| `vd` | `float` | `10.0` | Volume of distribution (1C) |
| `ke` | `float` | `0.1` | Elimination rate (1/h) |
| `ka` | `float` or null | `null` | Oral absorption rate |
| `f_oral` | `float` | `1.0` | Oral bioavailability \(F\) in (0, 1]; IV unaffected |
| `vc`, `vp`, `cl`, `q` | `float` or null | `null` | Two-compartment parameters |

Drug PK times are in **hours**.

## `DosingConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `str` | `"constant"` | `"constant"`, `"single_bolus"`, `"repeated_bolus"`, `"oral"` |
| `concentrations` | `list[float]` | `[0.0]` | In-vitro concentration list (required for `constant`) |
| `dose_amount` | `float` or null | `null` | Amount per dose (required for bolus/oral types) |
| `interval` | `float` or null | `null` | Hours between doses |
| `n_doses` | `int` or null | `null` | Number of doses |
| `start_time` | `float` | `0.0` | Time of first dose (hours) |

Each `type` validates that its required fields are present (no silent
`0 → default` substitutions).

## `ObservationConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `modalities` | `list[str]` | `["cell_counts"]` | Subset of `cell_counts`, `bli`, `volume`, `biomarker` |
| `cell_count_overdispersion` | `float` | `10.0` | NegBin \(\phi\) |
| `bli_alpha` | `float` | `1000.0` | Photons per cell (calibration) |
| `bli_sigma_log` | `float` | `0.3` | BLI log-scale SD |
| `volume_beta` | `float` | `1e-5` | mm³ per cell (~10⁵ cells/mm³) |
| `volume_sigma` | `float` | `0.2` | Volume log-scale SD |
| `biomarker_precision` | `float` | `50.0` | Beta precision \(\kappa\) |
| `biomarker_type` | `str` | `"ki67"` | `"ki67"` or `"caspase"` |

Unknown modality names are rejected at config validation.

## `InferenceConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `str` | `"mle"` | `"mle"` or `"mcmc"` (orchestrator). SMC/hierarchical via `umimic.inference` directly |
| `backend` | `str` | `"scipy"` | `"scipy"` (MLE), `"emcee"` (MCMC). PyMC withdrawn |
| `forward_mode` | `str` | `"moment"` | `"moment"` (LNA + process variance) or `"ode"` |
| `parameter_set` | `str` | `"default"` | `"default"`, `"mechanism"`, `"resistance"`, `"persister"` |
| `n_samples` | `int` | `2000` | MCMC draws per chain after warmup |
| `n_chains` | `int` | `4` | Walkers / chains (emcee raises to ≥ 2·dim+2) |
| `n_warmup` | `int` | `1000` | Burn-in |
| `n_restarts` | `int` | `5` | MLE multi-start restarts |
| `n_particles` | `int` | `500` | For direct ParticleMCMC use only |

**Guards:**

- `drug_mechanism` in `{cytostatic, mixed}` requires birth terms →
  `parameter_set=mechanism`.
- `parameter_set=mechanism` requires `forward_mode=moment`.

## `SignalingConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Build pathway trajectory |
| `model` | `str` | `"none"` | `"none"` or `"toy_mapk_akt"` |
| `direction` | `str` | `"inhibitory"` | `"inhibitory"` or `"stimulatory"` |
| `initial_state` | `dict` | `{}` | Node activities in [0, 1] |
| `parameters` | `dict` | `{}` | Toy params: baselines, drives, decay |
| `observed_nodes` | `list[str]` | `[]` | Optional measurement node names |

The toy model is a **scaffold**, not a MAPK cascade. Coupling works with
**ODE** cell dynamics only.

## `CouplingConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Apply rate multipliers |
| `function` | `str` | `"hill"` | `"hill"` or `"logistic"` |
| `targets` | `list[str]` | `[]` | `birth`, `death`, `transition`, or `death:P`, `transition:P->Q` |
| `parameters` | `dict` | `{}` | Global `max_effect`, `ec50`, `hill`, weights, … |
| `max_effect_by_target` | `dict` | `{}` | Per-target \(m_{\max}\ge 0\) |
| `ec50_by_target` / `hill_by_target` | `dict` | `{}` | Hill overrides |
| `k_by_target` / `center_by_target` | `dict` | `{}` | Logistic overrides |

**Multiplier:** \(\max(0,\,1 + m_{\max}\,e(a))\). High pathway activity
**boosts** rates above the bare RateSet; low activity → factor 1.

Recommended ranges: `CouplingConfig.recommended_parameter_ranges()` or
`umimic config suggest-coupling-ranges`.

## `SimulationConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | `str` | `"gillespie"` | `"ode"`, `"gillespie"`, `"tau_leaping"` |
| `initial_cells` | `int` | `100` | Initial total count |
| `t_max` | `float` | `72.0` | Duration (hours) |
| `dt_obs` | `float` | `4.0` | Observation spacing (hours) |
| `n_replicates` | `int` | `4` | Wells / animals for synthetic data |
| `seed` | `int` | `42` | Simulation seed (also top-level `seed`) |

## `PriorConfig`

Legacy / documentation stubs for prior scales. Live fits use
`PriorSpec.default_invitro()`, `default_mechanism()`, `default_resistance()`,
or `default_persister()` according to `parameter_set`.

## Complete examples

### In vitro cytotoxic plate

```yaml
name: cytotoxic_plate
context: in_vitro
seed: 123

dynamics:
  states: [P, Q]
  drug_mechanism: cytotoxic
  quiescent_sensitivity: 0.0
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
  method: ode
  initial_cells: 200
  t_max: 96.0
  dt_obs: 6.0
  n_replicates: 3

inference:
  mode: mle
  parameter_set: default
  forward_mode: moment
  n_restarts: 10
```

### Resistance topology

```yaml
dynamics:
  states: [P, Q, A, R]
  drug_mechanism: cytotoxic
  resistant_fitness_cost: 0.1
inference:
  parameter_set: resistance
  forward_mode: ode   # or moment
  mode: mle
```

### In vivo oral PK

```yaml
context: in_vivo
pk:
  model: one_compartment
  vd: 10.0
  ke: 0.1
  ka: 0.5
  f_oral: 0.6
dosing:
  type: oral
  dose_amount: 100.0
  interval: 24.0
  n_doses: 7
  start_time: 0.0
observations:
  modalities: [cell_counts, volume]
simulation:
  method: ode
  t_max: 168.0
  dt_obs: 24.0
```
