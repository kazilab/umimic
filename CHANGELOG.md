# Changelog

All notable changes to U-MIMIC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CHANGELOG.md and CONTRIBUTING.md for project governance.
- Expanded Sphinx documentation: quickstart, API reference, configuration guide.

### Fixed
- CLI logging now uses a package-scoped logger (`umimic`) instead of overwriting
  the root logger via `logging.basicConfig(force=True)`.
- Dashboard command no longer relies on a single hard-coded path; searches
  multiple candidate locations and reports clear errors.
- Lambda closures in likelihood and experiment modules now use default-argument
  binding to avoid late-binding capture bugs in loops.
- `load_config()` validates file existence and YAML content before parsing,
  with actionable error messages.
- Public dataset loader (`_find_data_dir`) now lists available datasets and
  suggests the correct download command on failure.
- `UMIMIC_DATA_ROOT` env var now warns and falls back gracefully when pointing
  to a non-existent directory.

### Improved
- CLI exception handling catches specific error types (`FileNotFoundError`,
  `ValueError`, `ImportError`, `KeyboardInterrupt`) with user-friendly messages
  instead of a bare `except Exception`.
- Parameter naming convention documented in `RateSet` docstring with a full
  inference-vector-to-model-field mapping table.
- Inference parameter comments in `DEFAULT_PARAM_NAMES` now reference the
  corresponding `RateSet` and observation model fields.

## [0.0.2] - 2026-03-01

### Added
- CLI entry point (`umimic`) with `simulate`, `fit`, `generate`, `dashboard` commands.
- Run logging with `--log-file` and `--log-level` options.
- Pydantic-based YAML configuration (`ExperimentConfig`).
- Experiment orchestrator class.
- ODE, Gillespie, and tau-leaping simulation engines.
- Moment equations (Linear Noise Approximation).
- One-compartment and two-compartment PK models with dosing schedules.
- Unified `ExposureProfile` abstraction for in-vitro / in-vivo exposure.
- Observation models: cell counts (Negative Binomial), BLI, tumor volume, biomarkers.
- Multimodal observation model combiner.
- MLE (multi-start), MCMC (emcee/PyMC), SMC, and Kalman inference engines.
- Hierarchical Bayesian inference.
- Prior specification and log-prior evaluation.
- Convergence diagnostics.
- Synthetic data generator (`generate_invitro_plate`, `generate_invivo_cohort`).
- Public dataset loaders: BESTDR, PhenoPop, Hafner/Niepel GR, TSHS Tumor Growth, NCI-60.
- Visualization: trajectories, dose-response curves, posteriors, diagnostics.
- Read the Docs configuration and Sphinx/MyST documentation.
- GitHub Actions workflows for trusted PyPI publishing and release smoke tests.

## [0.0.1] - 2026-02-15

### Added
- Initial project structure and packaging.
- Core cell-state dynamics module.
- Basic ODE solver.
