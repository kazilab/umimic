# U-MIMIC Documentation

**Unified Mechanistic Inference from Multimodal Imaging and Counts**

U-MIMIC is a Python package for modeling and analyzing tumor population
dynamics under drug treatment. It provides mechanistic simulation (ODE,
Gillespie SSA with Extrande under time-varying PK, tau-leaping), multimodal
observation models (cell counts, BLI, tumor volume, biomarkers), and
inference workflows (MLE, emcee MCMC, particle MCMC, hierarchical Bayesian).

Developed by the Data Analysis Team @KaziLab.se.

> **v0.0.4** corrected several modelling and numerical defects. Results from
> earlier releases should be regenerated — see the project
> [CHANGELOG](https://github.com/kazilab/umimic/blob/main/CHANGELOG.md).

## Getting started

```bash
pip install umimic
umimic --help
```

```{toctree}
:maxdepth: 2
:caption: User guide

quickstart
configuration
scientific-assumptions
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```

```{toctree}
:maxdepth: 2
:caption: Infrastructure

readthedocs-github
pypi-trusted-publishing
```

## Package map

| Package | Role |
|---------|------|
| `umimic.dynamics` | Rates, topology, ODE / SSA / LNA |
| `umimic.observations` | Counts, BLI, volume, biomarkers |
| `umimic.pk` | PK compartments, dosing, exposure |
| `umimic.inference` | Likelihood, MLE, MCMC, EKF, SMC |
| `umimic.pipeline` | Config, `Experiment`, CLI, transfer |
| `umimic.signaling` | Toy pathway scaffold + rate coupling |
| `umimic.visualization` | Trajectories, growth curves, posteriors |
| `umimic.data` | Schemas, synthetic data, public loaders |
