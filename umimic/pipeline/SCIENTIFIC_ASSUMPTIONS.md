# Scientific assumptions — `umimic.pipeline`

How configuration becomes a runnable experiment: topology, rates, exposure,
observations, simulation, and inference. Package-specific math lives under
`dynamics`, `inference`, `observations`, `pk`, and `signaling`; this note is
the **wiring contract**.

**Scope:** `config.py`, `experiment.py`, `runner.py`, `results.py`, `transfer.py`.

---

## 1. Role of the pipeline

`Experiment(config)` is the primary integration surface:

1. Build `ModelTopology`, `RateSet`, observation models, and `ExposureProfile`.
2. **Simulate** forward (ODE / Gillespie / tau-leaping; multi-dose plate).
3. **Generate** synthetic data for testing.
4. **Fit** via MLE or emcee MCMC using the same likelihood contracts as
   `umimic.inference`.

CLI entry: `umimic.pipeline.runner`. Results are summarized to JSON; full MCMC
chains are not archived by default.

---

## 2. Topology from configuration

States are taken from `dynamics.states` by **name**, not by count (so
`[P, Q, R]` is never silently remapped to `[P, Q, A]`).

| Setting | Default behaviour |
|---------|-------------------|
| `transitions` | If omitted: edges from the four-state template that involve only active states |
| `division_states` | P and R if present (R must divide to be a clone); Q not by default |
| `death_states` | P, Q, R if present |
| density | From `density_dependent`, `carrying_capacity`, `density_counts_apoptotic` |

Clearance lives on `RateSet.clearance_rate` (`dynamics.clearance_rate`), not on
the topology.

**Initial fractions** (`dynamics.initial_fractions`):

- `proliferating` / `null` — all cells in P (structural assumption).
- `stable` — relaxed mix at the first dosing concentration.
- explicit dict — named fractions, renormalised.

---

## 3. Rates and drug mechanism

Baseline kinetics come from `dynamics.default_*` fields. If R is active:

- Own death baseline and birth `b0 * (1 - resistant_fitness_cost)`.
- Explicit `birth_modulation_by_state[R] = None` (no inherited cytostasis).

`dynamics.drug_mechanism`:

| Value | Effect |
|-------|--------|
| `cytotoxic` | Death Emax on P; Q scaled by `quiescent_sensitivity` |
| `cytostatic` | Birth Emax on shared birth modulation |
| `mixed` | Both |
| unset | No concentration modulation until fitted / factory rates |

Configured `induced_transitions` and `transition_fold_change` map to
`transition_induction` / `transition_factor` (EmaxHill / HillFoldChange).

**Simulation-only configs with R need not use `parameter_set=resistance`.**
Fitting does: `ModelLikelihood` rejects free states that would become immortal
clones. Prefer matching set to topology before `fit`.

---

## 4. Exposure and dosing

| `pk.model` | Exposure |
|------------|----------|
| `none` | Constant: first of `dosing.concentrations`, or 0 |
| `one_compartment` / `two_compartment` | PK + `DosingSchedule` |

`pk.f_oral` ∈ (0, 1] is oral bioavailability (IV routes unchanged).

`simulate(concentrations=[…])` runs a **plate-style** dose–response: constant
concentration per arm (not the time-varying PK curve). Time-varying PK is used
for single-trajectory `simulate()` when `pk.model ≠ none`.

---

## 5. Observations

`observations.modalities` builds a `MultimodalObservation`. Each model receives
the experiment topology.

`fit` keeps only modalities present in the data; configured-but-missing
modalities warn and drop. Data-only modalities with no model never score.

---

## 6. Simulation methods

| Method | Exposure | Signaling coupling |
|--------|----------|--------------------|
| `ode` | Yes | Yes (`rate_multiplier_fn`) |
| `gillespie` | Yes | **Rejected** if signaling enabled |
| `tau_leaping` | Yes | **Rejected** if signaling enabled |

Signaling is open-loop vs exposure; see `signaling/SCIENTIFIC_ASSUMPTIONS.md`.

---

## 7. Inference via `Experiment.fit`

| Config | Meaning |
|--------|---------|
| `inference.mode` | `mle` or `mcmc` (emcee only) |
| `inference.forward_mode` | `moment` (LNA + process variance) or `ode` |
| `inference.parameter_set` | `default` / `mechanism` / `resistance` / `persister` |

**Guards at config construction:**

- Cytostatic/mixed mechanism requires birth-modulation parameters → use
  `parameter_set=mechanism`.
- `mechanism` requires `forward_mode=moment` (mean alone confounds b vs d).

**Not orchestrated here:** ParticleMCMC, hierarchical models — use
`umimic.inference` directly.

Priors are the matching `PriorSpec.default_*` factory for the parameter set.
MAP uses priors when the MLE backend is not pure scipy-without-prior; see
`MLEstimator` construction in `experiment.py`.

---

## 8. Transfer learning

`TransferLearning` turns in vitro MCMC (or MLE ± SE) into lognormal priors for
selected PD parameters.

- `shrinkage ∈ (0, 1]`: `1` = full transfer strength; smaller values **inflate**
  prior SD by `1/shrinkage` (weaker prior). Values ≤ 0 or > 1 are rejected.
- Only positive samples enter lognormal fits; tiny widths are floored.

This is empirical Bayes-style transfer, not a joint hierarchical model of both
contexts.

---

## 9. Results IO

`save_result` writes method, context, point estimates, MLE stats, and MCMC
**summaries** (mean, std, 95% CI, diagnostics) — not full sample arrays.
`load_result` returns a plain dict. For full chain plots, keep the live
`MCMCResult` or extend serialization.

---

## 10. Checklist

1. Align `dynamics.states` with `inference.parameter_set` before fitting.
2. Cytostatic/mixed → mechanism set + moment mode.
3. Signaling → ODE only; disable for Gillespie/fit.
4. Oral incomplete absorption → set `pk.f_oral` and `ka` with oral dosing.
5. Plate multi-dose vs in vivo PK are different `simulate` paths.
6. Multimodal: modalities in config ⊆ modalities in data (or accept drops).
7. Transfer: `shrinkage` in (0, 1]; do not use 0.
8. Treat short-data MLE on high-dimensional resistance sets with caution
   (bounds, restarts, more doses/time).

---

## 11. What this layer does not claim

- Automatic model selection across topologies.
- That default baseline rates are cell-line-specific truth.
- Full posterior archiving or experiment database.
- Joint in vitro / in vivo hierarchical sampling (transfer is prior construction).
