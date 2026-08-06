# Examples

This directory contains compact, runnable notebook examples for common U-MIMIC
workflows:

- `quickstart_api.ipynb`: minimal configuration, simulation, synthetic data
  generation, and MLE fitting.
- `mcmc_inference.ipynb`: basic Bayesian setup using MCMC mode.

## Running locally

From the project root:

```bash
pip install -e ".[inference]"
jupyter lab
```

Open a notebook from this directory and run the cells in order.
