# Quickstart

## Installation

Install from source (editable mode recommended for development):

```bash
cd script/
pip install -e .
```

For inference backends (MCMC):

```bash
pip install -e ".[inference]"
```

For the full stack:

```bash
pip install -e ".[all]"
```

## Your First Simulation

Run a cytotoxic dose-response simulation from the CLI:

```bash
umimic simulate --drug-type cytotoxic --output results/sim
```

This generates trajectory plots and dose-response curves under `results/sim/`.

## Programmatic Usage

```python
from umimic.pipeline.config import ExperimentConfig
from umimic.pipeline.experiment import Experiment

# Use defaults (P/Q two-state model)
config = ExperimentConfig(
    name="quickstart",
    dosing={"concentrations": [0, 0.1, 0.3, 1, 3, 10, 30]},
    simulation={"method": "ode", "t_max": 72.0},
)

exp = Experiment(config)
results = exp.simulate(concentrations=config.dosing.concentrations)

# results is a dict: concentration -> SimulationResult
for conc, result in results.items():
    print(f"C={conc}: final P cells = {result.populations['P'][-1]:.0f}")
```

## Generate Synthetic Data

```python
dataset = exp.generate_synthetic()
print(f"Generated {dataset.n_series} time series")
print(f"Concentrations: {dataset.concentrations}")
```

## Fit a Model

```python
from umimic.pipeline.config import ExperimentConfig
from umimic.pipeline.experiment import Experiment

config = ExperimentConfig(
    inference={"mode": "mle", "n_restarts": 5},
)
exp = Experiment(config)

# Generate synthetic data, then recover parameters
dataset = exp.generate_synthetic()
result = exp.fit(dataset)

print(f"Converged: {result.mle.converged}")
print(f"Parameters: {result.mle.parameters}")
print(f"AIC: {result.mle.aic:.1f}")
```

## YAML Configuration

Create an `experiment.yaml` file:

```yaml
name: my_experiment
context: in_vitro

dynamics:
  states: [P, Q]
  density_dependent: false

dosing:
  type: constant
  concentrations: [0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

simulation:
  method: gillespie
  initial_cells: 100
  t_max: 72.0
  dt_obs: 4.0
  n_replicates: 4
  seed: 42

observations:
  modalities: [cell_counts]
  cell_count_overdispersion: 10.0

inference:
  mode: mle
  n_restarts: 5
```

Then run:

```bash
umimic simulate --config experiment.yaml --output results/
umimic fit --config experiment.yaml --output results/fit/
```

## Loading Public Datasets

```python
from umimic.data.public_datasets import load_bestdr, list_available_datasets

# See what's available
datasets = list_available_datasets()
for name, info in datasets.items():
    print(f"{name}: {info['description']} (downloaded: {info['downloaded']})")

# Load BESTDR data (requires download first)
data = load_bestdr(cell_line="MCF7", drug="paclitaxel")
```

## Next Steps

- {doc}`configuration` -- Full configuration reference
- {doc}`api` -- API reference for all modules
