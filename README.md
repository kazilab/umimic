# U-MIMIC

**Unified Mechanistic Inference from Multimodal Imaging and Counts**

[![Tests](https://github.com/kazilab/umimic/actions/workflows/tests.yml/badge.svg)](https://github.com/kazilab/umimic/actions/workflows/tests.yml)
[![Docs](https://github.com/kazilab/umimic/actions/workflows/docs.yml/badge.svg)](https://github.com/kazilab/umimic/actions/workflows/docs.yml)
[![Release Smoke](https://github.com/kazilab/umimic/actions/workflows/release-smoke.yml/badge.svg)](https://github.com/kazilab/umimic/actions/workflows/release-smoke.yml)
[![PyPI Publish](https://github.com/kazilab/umimic/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/kazilab/umimic/actions/workflows/publish-pypi.yml)
[![Coverage](https://codecov.io/gh/kazilab/umimic/graph/badge.svg)](https://codecov.io/gh/kazilab/umimic)

Developed by: Data Analysis Team @KaziLab.se

U-MIMIC is a Python package for modeling and analyzing tumor population dynamics
under drug treatment. It combines mechanistic simulation with statistical
inference to support both in-vitro and in-vivo experimental workflows.

## Features

- **Mechanistic dynamics** -- ODE, Gillespie SSA (exact under time-varying
  exposure via Extrande thinning), and tau-leaping simulation of P/Q/A/R
  cell-state models with dose-response modulation
- **Phenotype plasticity** -- transitions carry a baseline, a fold-change, and
  an *additive induced* component, so a route absent without treatment can
  appear under it. `ModelTopology.persister_resistance()` provides the
  P <-> Q -> R persister route into resistance.
- **Pharmacokinetics** -- One- and two-compartment PK models with flexible dosing
  schedules, unified via `ExposureProfile`
- **Observation models** -- Cell counts (Negative Binomial), bioluminescence
  imaging, tumor volume, and biomarkers; combinable via `MultimodalObservation`
- **Inference** -- MLE (multi-start), MCMC (emcee), SMC/particle MCMC, Kalman
  filtering, and hierarchical Bayesian estimation. Every configured modality
  enters the likelihood under conditional independence.
- **Identifiability diagnostics** -- `umimic.inference.analyze_identifiability`
  reports which parameters a given experimental design can actually
  constrain, so a converged fit does not imply the data measured anything
- **Public datasets** -- Loaders for BESTDR, PhenoPop, Hafner/Niepel GR,
  TSHS Tumor Growth, and NCI-60
- **CLI** -- `umimic simulate`, `fit`, `generate`, `config` commands with
  YAML configuration and run logging (`umimic --version` reports the release)

## Installation

```bash
pip install umimic
```

From a source checkout:

```bash
pip install .
```

With optional extras:

```bash
pip install ".[inference]"    # MCMC backend (emcee)
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

# Generate synthetic data (writes dataset.csv + config.yaml)
umimic generate --config experiment.yaml --output results/synthetic

# ...and fit the dataset that was just generated
umimic fit --config results/synthetic/config.yaml \
           --data results/synthetic/dataset.csv --output results/fit

# Show recommended SAEM coupling tuning ranges
umimic config suggest-coupling-ranges

# Print as YAML for config authoring
umimic config suggest-coupling-ranges --format yaml
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

## Examples

Example notebooks for common workflows are available in [`examples/`](examples/):

- [`examples/quickstart_api.ipynb`](examples/quickstart_api.ipynb): minimal simulation + fit flow
- [`examples/mcmc_inference.ipynb`](examples/mcmc_inference.ipynb): Bayesian/MCMC-oriented workflow

Additional full validation notebooks are available in [`notebooks/`](notebooks/).

## Configuration

U-MIMIC uses YAML configuration validated by Pydantic. See
[docs/configuration.md](docs/configuration.md) for the full reference.

The default dynamics rates (`default_birth_base`, `default_death_base_p`,
`default_death_base_q`) are generic baseline kinetics for simulation startup.
For biological studies, tune these values to your system or estimate them from
data via inference.

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

## Testing and Coverage

Run tests locally:

```bash
pip install -e ".[dev]"
pytest
```

Generate a local coverage report:

```bash
pytest --cov=umimic --cov-report=term-missing --cov-report=xml
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and
how to add new simulation methods, observation models, or dataset loaders.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Citation

If you use U-MIMIC in academic work, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).

## License

MIT
