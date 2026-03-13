# Contributing to U-MIMIC

Thank you for your interest in contributing to U-MIMIC! This guide will help you
get started.

## Development Setup

1. Clone the repository and install in editable mode with dev extras:

```bash
git clone https://github.com/kazilab/umimic.git
cd umimic/script
pip install -e ".[dev,inference]"
```

2. Verify your setup:

```bash
pytest
```

## Project Structure

```
script/
  umimic/
    pipeline/       # CLI, config, experiment orchestration
    dynamics/       # ODE, Gillespie, tau-leaping simulation
    pk/             # Pharmacokinetics and dosing
    observations/   # Observation/likelihood models
    inference/      # MLE, MCMC, SMC, Kalman estimation
    data/           # Data loading, synthetic generation, public datasets
    visualization/  # Plotting utilities
    types.py        # Shared result dataclasses
  tests/            # Mirrors umimic/ structure
  docs/             # Sphinx/MyST documentation
```

## Code Style

- **Formatter / Linter**: We use [Ruff](https://docs.astral.sh/ruff/) (`ruff check` and `ruff format`).
- **Type hints**: All public functions should have type annotations.
- **Line length**: 100 characters.
- **Python version**: 3.11+.

Run the linter before committing:

```bash
ruff check umimic/ tests/
ruff format umimic/ tests/
```

## Running Tests

```bash
# Full suite
pytest

# Skip slow tests (integration, MCMC)
pytest -m "not slow"

# With coverage
pytest --cov=umimic --cov-report=term-missing
```

## Making Changes

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-change
   ```

2. Make your changes. Write tests for new functionality.

3. Run linting and tests:
   ```bash
   ruff check umimic/ tests/
   pytest
   ```

4. Commit with a descriptive message:
   ```bash
   git commit -m "Add support for two-compartment PK oral dosing"
   ```

5. Open a pull request against `main`.

## Adding a New Simulation Method

1. Create a new module in `umimic/dynamics/` (e.g., `hybrid.py`).
2. Implement a simulator class following the same interface as
   `GillespieSimulator` or `TauLeapingSimulator`.
3. Register the method name in `Experiment.simulate()` in `pipeline/experiment.py`.
4. Add tests in `tests/test_dynamics/`.

## Adding a New Observation Model

1. Subclass `ObservationModel` from `umimic/observations/base.py`.
2. Implement `log_likelihood()`, `sample()`, and `param_names()`.
3. Register in `Experiment._build_components()` so config can enable it.
4. Add tests in `tests/test_observations/`.

## Adding a Public Dataset Loader

1. Add a `load_<name>()` function in `umimic/data/public_datasets.py`.
2. Register it in `list_available_datasets()`.
3. Add download support in `umimic/data/public/download_datasets.py`.
4. Add a test in `tests/test_release/`.

## Parameter Naming Convention

When adding inference parameters, follow the naming convention documented in
`RateSet`'s docstring (e.g., `b0` for birth base rate, `d0_P` for death base
rate of proliferating cells). See `inference/likelihood.py:DEFAULT_PARAM_NAMES`
for the full mapping.

## Documentation

Docs are built with Sphinx and MyST-Parser (Markdown):

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## Reporting Issues

Please open an issue on GitHub with:
- A description of the problem or feature request.
- Steps to reproduce (for bugs).
- Your Python version and OS.
