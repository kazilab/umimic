# Quickstart

## Installation

```bash
pip install umimic
```

From a source checkout (editable, recommended for development):

```bash
cd script/
pip install -e .
```

Optional extras:

```bash
pip install -e ".[inference]"   # emcee MCMC
pip install -e ".[all]"         # full stack
```

Requires **Python ≥ 3.11**.

## First simulation (CLI)

```bash
umimic simulate --drug-type cytotoxic --output results/sim
```

Writes trajectory and dose–response plots under `results/sim/`. Growth plots
use **multi-state asymptotic growth** \(g(C)\), not naive \(b - d_P\).

Coupling parameter ranges for signaling-to-rate links:

```bash
umimic config suggest-coupling-ranges
umimic config suggest-coupling-ranges --format yaml
```

## Programmatic simulation

```python
from umimic.pipeline.config import ExperimentConfig
from umimic.pipeline.experiment import Experiment

config = ExperimentConfig(
    name="quickstart",
    dynamics={
        "states": ["P", "Q"],
        "drug_mechanism": "cytotoxic",  # attaches death Emax on P
    },
    dosing={"concentrations": [0, 0.1, 0.3, 1, 3, 10, 30]},
    simulation={"method": "ode", "t_max": 72.0},
)

exp = Experiment(config)

# Plate-style multi-dose: constant C per arm
results = exp.simulate(concentrations=config.dosing.concentrations)

for conc, result in results.items():
    print(f"C={conc}: final viable = {result.viable[-1]:.0f}")
```

### Population growth rate (multi-state)

```python
from umimic.dynamics.rates import RateSet
from umimic.dynamics.states import ModelTopology

rates = RateSet()
topo = ModelTopology.two_state()

g = rates.asymptotic_growth_rate(0.0, topo)  # long-run rate (1/h)
T2 = rates.doubling_time(0.0, topo)
# rates.net_growth_rate(c) is only b(C) - d_P(C) — not culture doubling
```

## Generate synthetic data and fit

```python
from umimic.pipeline.config import ExperimentConfig
from umimic.pipeline.experiment import Experiment

config = ExperimentConfig(
    dynamics={"states": ["P", "Q"], "drug_mechanism": "cytotoxic"},
    dosing={"concentrations": [0, 0.3, 1, 3, 10]},
    simulation={
        "method": "ode",
        "t_max": 72.0,
        "dt_obs": 6.0,
        "n_replicates": 3,
        "initial_cells": 100,
    },
    inference={
        "mode": "mle",
        "parameter_set": "default",  # cytotoxic-only parameter vector
        "forward_mode": "moment",    # LNA process variance (recommended)
        "n_restarts": 5,
    },
)
exp = Experiment(config)

dataset = exp.generate_synthetic()
result = exp.fit(dataset)

print(f"Converged: {result.mle.converged}")
print(f"Parameters: {result.mle.parameters}")
print(f"AIC: {result.mle.aic:.1f}")
```

### Align parameter set with the model

| Model | `inference.parameter_set` | Notes |
|-------|---------------------------|--------|
| P/Q cytotoxic | `default` | Death modulation only |
| Cytostatic / mixed | `mechanism` | Requires `forward_mode: moment` |
| States include **R** | `resistance` | Must estimate `b0_R`, `d0_R`, … |
| P↔Q→R persisters | `persister` | Induced Q→R, Q rates |

## YAML configuration

```yaml
name: my_experiment
context: in_vitro

dynamics:
  states: [P, Q]
  drug_mechanism: cytotoxic
  density_dependent: false
  # initial_fractions: stable   # optional: relaxed phenotype mix

dosing:
  type: constant
  concentrations: [0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

simulation:
  method: ode
  initial_cells: 100
  t_max: 72.0
  dt_obs: 4.0
  n_replicates: 4

observations:
  modalities: [cell_counts]
  cell_count_overdispersion: 10.0

inference:
  mode: mle
  parameter_set: default
  forward_mode: moment
  n_restarts: 5
```

```bash
umimic generate --config experiment.yaml --output results/synthetic
umimic fit --config results/synthetic/config.yaml \
           --data results/synthetic/dataset.csv --output results/fit
```

## In vivo PK sketch

```yaml
context: in_vivo
pk:
  model: one_compartment
  vd: 10.0
  ke: 0.1
  ka: 0.5
  f_oral: 0.6          # oral bioavailability F ∈ (0, 1]
dosing:
  type: oral
  dose_amount: 100.0
  interval: 24.0
  n_doses: 7
  start_time: 0.0
```

Drug time is in **hours**. Luciferin imaging (BLI) times are in **minutes**.

## Resistance / persister models

```python
from umimic.dynamics.states import ModelTopology
from umimic.dynamics.rates import RateSet

topo = ModelTopology.four_state()              # R divides
# topo = ModelTopology.persister_resistance()  # P <-> Q -> R

rates = RateSet.resistant_clone(resistance=1.0, fitness_cost=0.1)
```

In config, set `dynamics.states: [P, Q, A, R]` and
`inference.parameter_set: resistance` (or `persister`) before fitting.

## Loading public datasets

```python
from umimic.data.public_datasets import load_bestdr, list_available_datasets

datasets = list_available_datasets()
for name, info in datasets.items():
    print(f"{name}: {info['description']} (downloaded: {info['downloaded']})")

data = load_bestdr(cell_line="MCF7", drug="paclitaxel")
```

## Signaling (scaffold only)

The toy MAPK/AKT model is **plumbing for rate coupling**, not a pathway model.
Coupling is supported for **ODE** simulation only (`simulation.method: ode`).
See {doc}`scientific-assumptions`.

```yaml
signaling:
  enabled: true
  model: toy_mapk_akt
  direction: inhibitory   # default: inhibitors lower pathway activity
coupling:
  enabled: true
  targets: [birth]
  parameters:
    max_effect: 0.5       # high activity boosts rates up to 1 + max_effect
```

## Next steps

- {doc}`configuration` — full YAML field reference
- {doc}`scientific-assumptions` — formulas and footguns by package
- {doc}`api` — module API overview
- Project [README](https://github.com/kazilab/umimic/blob/main/README.md)
