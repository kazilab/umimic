# U-MIMIC

**Unified Mechanistic Inference from Multimodal Imaging and Counts**

Developed by: Data Analysis Team @KaziLab.se

U-MIMIC is a Python package for modeling and analyzing tumor population dynamics
under drug treatment. It combines mechanistic simulation with statistical
inference to support both in-vitro and in-vivo experimental workflows.

## Features

- **Mechanistic dynamics** -- ODE, Gillespie SSA, and tau-leaping simulation of
  P/Q/A/R cell-state models with dose-response modulation
- **Pharmacokinetics** -- One- and two-compartment PK models with flexible dosing
  schedules, unified via `ExposureProfile`
- **Observation models** -- Cell counts (Negative Binomial), bioluminescence
  imaging, tumor volume, and biomarkers; combinable via `MultimodalObservation`
- **Inference** -- MLE (multi-start), MCMC (emcee/PyMC), SMC, Kalman filtering,
  and hierarchical Bayesian estimation
- **Public datasets** -- Loaders for BESTDR, PhenoPop, Hafner/Niepel GR,
  TSHS Tumor Growth, and NCI-60
- **CLI** -- `umimic simulate`, `fit`, `generate`, `dashboard` commands with
  YAML configuration and run logging

## Installation

```bash
pip install .
```

With optional extras:

```bash
pip install ".[inference]"    # MCMC backends (emcee, PyMC)
pip install ".[dashboard]"    # Streamlit dashboard
pip install ".[all]"          # Everything
```

Requires Python >= 3.11.

## Quickstart

### CLI

```bash
# Run a dose-response simulation
umimic simulate --drug-type cytotoxic --output results/sim

# Fit model to data
umimic fit --config experiment.yaml --data data.csv --output results/fit

# Generate synthetic data
umimic generate --config experiment.yaml --output results/synthetic
```

### Python API

```python
from umimic.pipeline.config import ExperimentConfig
from umimic.pipeline.experiment import Experiment

config = ExperimentConfig(
    dosing={"concentrations": [0, 0.1, 1, 10]},
    simulation={"method": "ode", "t_max": 72.0},
)
exp = Experiment(config)

# Simulate
results = exp.simulate(concentrations=[0, 0.1, 1, 10])

# Generate synthetic data and fit
dataset = exp.generate_synthetic()
result = exp.fit(dataset)
print(result.mle.parameters)
```

## Configuration

U-MIMIC uses YAML configuration validated by Pydantic. See
[docs/configuration.md](docs/configuration.md) for the full reference.

Minimal example:

```yaml
name: my_experiment
dynamics:
  states: [P, Q]
dosing:
  concentrations: [0, 0.1, 0.3, 1, 3, 10, 30]
simulation:
  method: gillespie
  t_max: 72.0
inference:
  mode: mle
  n_restarts: 5
```

## Run Logging

CLI commands support execution logging:

```bash
umimic simulate --output results/sim --log-file results/sim/run.log --log-level DEBUG
```

If `--log-file` is not provided, a timestamped log file is written to the
command output directory. Logging uses a package-scoped logger (`umimic`) and
does not interfere with other logging configuration.

## Documentation

Full documentation is available on Read the Docs. To build locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

- [Quickstart](docs/quickstart.md)
- [Configuration Reference](docs/configuration.md)
- [API Reference](docs/api.md)
- [Read the Docs Setup](docs/readthedocs-github.md)
- [PyPI Trusted Publishing](docs/pypi-trusted-publishing.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and
how to add new simulation methods, observation models, or dataset loaders.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

MIT
