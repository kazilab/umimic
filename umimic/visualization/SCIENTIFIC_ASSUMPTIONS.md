# Scientific assumptions — `umimic.visualization`

What each figure actually plots, so labels match the dynamics and inference
contracts. Styling is secondary; mislabeled growth rates and flattened MCMC
traces are not.

**Scope:** `style.py`, `trajectories.py`, `dose_response.py`, `posteriors.py`,
`diagnostics.py`.

---

## 1. Role

Visualization is a thin layer over result objects (`SimulationResult`,
`EnsembleResult`, `MCMCResult`, `RateSet`). It must not invent a second
definition of “growth rate” or “95% interval” that disagrees with
`umimic.dynamics` / `umimic.inference`.

---

## 2. Trajectories

| Function | Content |
|----------|---------|
| `plot_population_trajectories` | Compartment curves; optional viable total (non-`A`); optional `cell_counts` overlay |
| `plot_dose_response_trajectories` | One state vs time, colored by concentration |
| `plot_ensemble` | Stochastic paths + mean + uncertainty band |

### Ensemble bands

Default **`ci_method="percentile"`**: 2.5th and 97.5th percentiles across
trajectories at each time (nonparametric, appropriate for count ensembles).

Optional **`ci_method="gaussian"`**: mean ± 1.96·SD (parametric; can go
negative before clipping at 0). Prefer percentiles for Gillespie/tau-leaping
outputs.

---

## 3. Dose–response and growth metrics

### Rates

`plot_rate_dose_response` can show birth, death, transitions, and:

| Name | Quantity |
|------|----------|
| `asymptotic_growth` (default) | \(g(C)=\) dominant eigenvalue of the low-density multi-state rate matrix (`RateSet.asymptotic_growth_rate`) |
| `net_growth` / `net_growth_P` | Naive \(b(C)-d_P(C)\) only — **not** culture doubling when Q/R/transitions matter |

Pass `topology=` for multi-state models (default: two-state P/Q).

### Growth curve / g0 / g50

`plot_net_growth_curve` (name kept for API stability) defaults to
**asymptotic** \(g(C)\):

- **g0**: concentration where \(g(C)=0\) (long-run arrest).
- **g50**: concentration where \(g(C)=\tfrac12 g(0)\).

Crossings use log-linear interpolation on the concentration grid.

`metric="naive_p"` draws \(b-d_P\) with an explicit “not multi-state” label —
for comparison only.

These are **not** automatically Hafner finite-horizon GR values; for assay
window GR use `RateSet.finite_horizon_gr` / `gr_value` outside this module.

### Mechanism comparison

`plot_mechanism_comparison` shows **P** birth and death (absolute and fold
change). It is a cytostatic vs cytotoxic sketch for P rates, not multi-state
\(g(C)\).

---

## 4. Posteriors

| Function | Content |
|----------|---------|
| `plot_posterior_marginals` | Pooled samples; mean + 2.5/97.5% percentile lines |
| `plot_trace` | **One series per chain/walker**; x-axis = draw index within chain |
| `plot_pair` | Lower-triangle scatter; diagonal marginals |

### MCMC layout

Samples are `(n_chains, n_draws)`. Traces must **not** flatten walkers into
one pseudo-time series (that invents a false iteration axis and hides
walker disagreement). Histograms still pool all chains for a marginal.

Emcee walkers are an interacting ensemble, not independent chains; multi-line
traces still help spot stuck walkers and gross non-stationarity. Formal
R-hat/ESS live in `umimic.inference.diagnostics`.

---

## 5. Fit diagnostics

| Function | Content |
|----------|---------|
| `plot_residuals` | Raw \(y-\hat y\) vs time and vs predicted |
| `plot_fit_quality` | Obs vs prediction; optional band from caller |

Raw residuals assume additive scale. For NegBin / lognormal modalities use
model-appropriate residuals (Pearson, deviance, log-scale) elsewhere; these
plots are exploratory.

---

## 6. Style

`apply_umimic_style()` updates global `matplotlib.rcParams` for the session.
State colors: P blue, Q orange, A red, R green.

---

## 7. Checklist

1. Multi-state growth plots: pass the same `topology` as the simulation.
2. Do not read `net_growth` curves as population doubling without checking
   the label / metric.
3. Ensemble bands: default percentiles; use gaussian only knowingly.
4. MCMC: use `plot_trace` for per-chain paths; use `summarize_mcmc` for R-hat.
5. Residual plots: remember the observation error model.

---

## 8. What this layer does *not* claim

- Publication-ready automatic model validation.
- Finite-horizon GR or IC50 from the growth curve alone.
- That Gaussian residual plots certify a NegBin fit.
- Interactive dashboards or report generation.
