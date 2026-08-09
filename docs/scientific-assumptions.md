# Scientific assumptions

Each core package documents formulas, defaults, and common misuses in a
`SCIENTIFIC_ASSUMPTIONS.md` file next to the source. This page indexes them
and summarizes the rules that most often affect interpretation.

## Package notes (source)

Full write-ups live next to the code (not duplicated here):

| Package | Path in the repository |
|---------|-------------------------|
| Dynamics | `umimic/dynamics/SCIENTIFIC_ASSUMPTIONS.md` |
| Inference | `umimic/inference/SCIENTIFIC_ASSUMPTIONS.md` |
| Observations | `umimic/observations/SCIENTIFIC_ASSUMPTIONS.md` |
| PK | `umimic/pk/SCIENTIFIC_ASSUMPTIONS.md` |
| Signaling | `umimic/signaling/SCIENTIFIC_ASSUMPTIONS.md` |
| Visualization | `umimic/visualization/SCIENTIFIC_ASSUMPTIONS.md` |
| Pipeline | `umimic/pipeline/SCIENTIFIC_ASSUMPTIONS.md` |

## Growth rates

- **Asymptotic growth** \(g(C)\): dominant eigenvalue of the low-density
  multi-state rate matrix. Use for doubling time and long-run GR-style metrics
  (`RateSet.asymptotic_growth_rate`, `doubling_time`, `gr_value`).
- **Naive net** \(b - d_P\): single-compartment diagnostic only
  (`net_growth_rate`). For the default P/Q model it overstates growth relative
  to \(g(C)\).
- **Finite-horizon GR**: use `finite_horizon_gr` when a rare resistant state
  would dominate the asymptotic eigenvalue but not a short assay.

Visualization dose–response plots default to **asymptotic** \(g(C)\).

## Inference contracts

- One shared `build_rate_set` maps parameter names into dynamics (no second
  silent mapping).
- Unknown parameter names are rejected at likelihood construction.
- The first anchor observation (default: cell counts) can condition the
  initial state and is then **excluded** from the score (`condition_on_first`).
- `parameter_set=mechanism` needs `forward_mode=moment` (LNA variance separates
  cytostatic vs cytotoxic action).
- States with R require `resistance` or `persister` parameter sets at **fit**
  time; simulation-only configs may use defaults for forward runs.

## Observations

- Counts: NegBin without process variance; Gaussian with
  \(\mathrm{Var} = V_{\mathrm{proc}} + \mu + \mu^2/\phi\) in moment mode.
- BLI / volume: lognormal (scale = **median**); process variance via
  \(\sigma_{\log}^2 + V/N^2\).
- Free noise keys: `overdispersion`, `sigma_log_bli`, `sigma_v`,
  `biomarker_precision`.
- Ki-67: fraction of **division_states** / viable (includes dividing R).

## Pharmacokinetics

- Drug PK times in **hours**; luciferin imaging in **minutes**.
- Oral bioavailability `f_oral` ∈ (0, 1] scales oral deposits only.
- Plate multi-dose `simulate(concentrations=…)` uses constant C per arm; time
  varying PK uses `ExposureProfile` from the configured dosing schedule.
- Optional `precompute(t_grid)` caches PK for linear interpolation (SSA hot
  paths); peaks between knots are under-resolved by design.

## Signaling coupling

- Toy MAPK/AKT is a **scaffold**, not a cascade model.
- Rate multiplier: \(\max(0,\,1 + m_{\max}\,e(a))\) — high pathway activity
  **boosts** rates above the bare RateSet; pathway-off returns to 1.
- Implemented for **deterministic ODE** cell dynamics only.

## MCMC and visualization

- Sample layout: `(n_chains, n_draws)` (emcee walkers kept un-collapsed).
- Trace plots draw **one series per chain**; do not flatten walkers for
  diagnostics.
- Ensemble uncertainty bands default to **percentiles** (2.5–97.5%), not
  mean ± 1.96 SD.

## Transfer learning

- In vitro MCMC (or MLE ± SE) → lognormal priors for selected PD parameters.
- `shrinkage` ∈ (0, 1]: smaller values **inflate** prior SD (weaker prior).
