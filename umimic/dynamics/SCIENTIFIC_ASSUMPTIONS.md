# Scientific assumptions — `umimic.dynamics`

This note records the biological and mathematical assumptions behind the cell-population dynamics layer. It is written for someone fitting or interpreting models, not for API reference (see module and class docstrings for that).

**Scope:** `rates.py`, `states.py`, `ode_system.py`, `gillespie.py`, `tau_leaping.py`, `moment_equations.py`.

**Status:** Core rate laws, multitype growth, ODE / SSA / LNA consistency, and Hafner-style GR metrics match standard theory. Known footguns are called out below; most are configuration risks rather than inverted formulas.

---

## 1. Biological picture

The model is a **multitype continuous-time Markov chain (CTMC)** of cell counts:

| Symbol | Meaning |
|--------|---------|
| **P** | Proliferating / drug-sensitive cycling cells |
| **Q** | Quiescent or slow-cycling (e.g. drug-tolerant persisters) |
| **R** | Resistant clone (must be able to divide to outgrow under treatment) |
| **A** | Apoptotic compartment (sink fed by death; optional clearance) |

Each viable cell of type \(i\) independently:

- **divides** (if \(i\) is a division state) at rate \(b_i(C, N)\),
- **dies** (if \(i\) is a death state) at rate \(d_i(C)\),
- **converts** \(i \to j\) at rate \(u_{ij}(C)\).

Dead cells may accumulate in **A** and leave at clearance rate \(\kappa\). Drug concentration \(C = C(t)\) comes from an exposure / PK profile; total crowding \(N\) is defined by the topology’s density mask (see §4).

**Not modelled here:** spatial structure, explicit cell cycle phases (beyond P/Q labels), stochastic gene expression inside a cell, or phase-type (Erlang) dwell times. The topology field `linear_chain_stages` is reserved and **not implemented** — all dwell times are exponential.

---

## 2. Rate laws (`RateSet`)

### 2.1 Dose–response building blocks

Modulators used as **effect magnitudes** \(m(C)\) or induction rates \(a(C)\) must be **non-decreasing** in \(C\). Construction-time validation rejects decreasing curves for birth, death, legacy transition modulation, and induction.

| Class | Formula / role |
|-------|----------------|
| `EmaxHill` | \(m(C) = E_{\max}\, C^{h}/(EC_{50}^{h}+C^{h})\) on \([0,E_{\max}]\) |
| `HillFoldChange` | \(f(C) = f_0 + (f_\infty-f_0)\, C^{h}/(EC_{50}^{h}+C^{h})\) — fold multiplier; may fall with dose (suppression) |
| `FourParameterLogistic` | Standard 4PL for *observed* response (often decreasing viability). **Not** a valid rate modulator until reoriented with `.as_effect()` |
| `ConstantRate` | \(m(C) \equiv \text{value}\) |

`FourParameterLogistic.as_effect()` returns an increasing curve from 0 to \(\mathrm{top}-\mathrm{bottom}\), algebraically equivalent to an Emax/Hill form with that amplitude.

**Parameter constraints (Emax/Hill family):** \(E_{\max}\ge 0\), \(EC_{50}>0\), \(h>0\). Birth modulators are further capped so their peak is \(\le 1\) (complete cytostasis); values \(>1\) only clamp \(b\) to zero and leave a flat likelihood ridge.

### 2.2 Birth, death, transitions

For phenotype \(i\) and edge \(i\to j\):

\[
\begin{aligned}
b_i(C,N) &= b_{0,i}\,[1-m_{b,i}(C)]\,\max\!\bigl(0,\,1-N/K\bigr), \\
d_i(C) &= d_{0,i} + m_{d,i}(C), \\
u_{ij}(C) &= u_{0,ij}\, f_{ij}(C)\, [1 + m_{ij}(C)] + a_{ij}(C).
\end{aligned}
\]

| Term | Meaning |
|------|---------|
| \(m_b\) | Fractional **cytostatic** suppression of division (\(0\) = none, \(1\) = no division) |
| \(m_d\) | **Additive** death-rate increase (units: 1/time) |
| \(f_{ij}\) | Non-negative fold-change of an existing transition (induction or suppression) |
| \(m_{ij}\) | Legacy multiplicative boost \((1+m)\); prefer \(f\) for new models |
| \(a_{ij}\) | Additive **induction** rate — required for de novo transitions when \(u_0=0\) (purely multiplicative laws cannot create \(0\cdot(\cdot)=0\)) |
| \(K\) | Carrying capacity when density dependence is on; omit for unlimited growth |

**Convention, not a universal law:** cytostasis is fractional; cytotoxicity is additive. Other papers use fractional death or shared Hill parameters; here the two mechanisms are intentionally separable (variance signatures, independent EC50/Emax).

Rates are clamped at zero after evaluation. Negative concentrations are treated as zero in Hill-type curves.

### 2.3 Per-phenotype overrides

Shared `birth_base` / `birth_modulation` apply unless overridden via `birth_base_by_state` / `birth_modulation_by_state`. An **explicit `None`** on a resistant state means “do not inherit P’s cytostatic suppression” — required for resistance models.

Preferred construction for multi-phenotype fitness:

- `RateSet.from_profiles(...)` — optional `topology=` rejects dividing states that would silently inherit P’s drug response.
- `RateSet.resistant_clone(...)` — P/Q + R with resistance fraction and fitness cost.
- `RateSet.persister_resistance(...)` — illustrative P→Q→R defaults (not fitted parameters).

### 2.4 Factory defaults worth knowing

| Factory | Notable default |
|---------|-----------------|
| `cytotoxic_drug` / `mixed_drug` | `quiescent_sensitivity=0`: Q gets **no** drug-induced death. Matches cycle-specific agents; **understates kill** for agents active on non-cycling cells. Set deliberately. |
| `cytostatic_drug` | Birth modulation only; Q baseline death is half of P’s default scale. |
| `resistant_clone` | At `resistance=1`, R has no drug death increment; birth \(b_0(1-\mathrm{fitness\_cost})\). |
| Bare `RateSet()` + `ModelTopology.four_state()` | R **divides** but uses the **same** baseline rates as P unless you override — not a resistance model by itself. |

Baseline scale (defaults): \(b_0\sim 0.04\,\mathrm{h}^{-1}\) (~17 h interdivision if no death), clearance \(\kappa\sim 0.1\,\mathrm{h}^{-1}\). These are illustrative culture-scale numbers, not universal constants.

---

## 3. Growth metrics (what “growth rate” means)

### 3.1 Naive net rate — do not use for multi-state doubling time

```text
net_growth_rate(C) = b_P(C) - d_P(C)
```

Ignores transitions out of P and death in other states. For the default P/Q topology this is **larger** than the true long-run rate (e.g. ~0.030 vs ~0.025 h⁻¹). Kept for simple diagnostics only.

### 3.2 Asymptotic growth rate (correct multi-state object)

Build the **low-density rate matrix** \(A(C)\) on viable states (A excluded): birth and death on the diagonal; transition \(i\to j\) subtracts \(u_{ij}\) from \((i,i)\) and adds it to \((j,i)\). Then

\[
g(C) = \text{dominant eigenvalue of } A(C)
\quad\text{(largest real part)}.
\]

This is the exponential rate of total viable population after the phenotype mix relaxes. Doubling time:

\[
T_2 = \ln 2 / g(C) \quad (g>0),\qquad \infty \text{ otherwise}.
\]

Stable phenotype fractions = normalized dominant right eigenvector of \(A(C)\).

### 3.3 GR (Hafner-style)

**Asymptotic** (exponential regime):

\[
\mathrm{GR}(C) = 2^{g(C)/g(C_{\mathrm{ref}})} - 1.
\]

Interpretation: 1 = no effect, 0 = cytostasis, \(<0\) = net kill. Requires \(g(C_{\mathrm{ref}})>0\).

**Finite-horizon** (what a fixed-length assay measures):

\[
\mathrm{GR} = 2^{\log_2(x_{\mathrm{treated}}/x_0)\,/\,\log_2(x_{\mathrm{control}}/x_0)} - 1,
\]

with viable counts from the deterministic ODE over the assay window.

**Footgun:** with a rare but fitter R, \(g(C)\) is eventually set by R even when R is negligible over 72 h. Asymptotic GR can report “drug ineffective” while the finite-horizon GR (and the experiment) show clearance of the sensitive bulk. Prefer `finite_horizon_gr` for assay comparisons; use asymptotic GR for long-run / evolutionary summaries.

---

## 4. Topology and density (`ModelTopology`)

| Setting | Scientific meaning |
|---------|-------------------|
| `division_states` | Who contributes birth reactions / ODE birth terms |
| `death_states` | Who can die (and feed A if present) |
| `transitions` | Allowed phenotype conversions (not death) |
| `density_dependent` + `carrying_capacity` \(K\) | Logistic factor \(\max(0,1-N/K)\) on birth |
| `density_counts_apoptotic` | If `False` (default), corpses do **not** crowd division |

**Why exclude A from density by default:** after a cytotoxic pulse, counting corpses as space occupancy suppresses regrowth until clearance finishes — an artefact unless the assay truly has volume competition from dead cells.

**Clearance rate** lives only on `RateSet.clearance_rate`. Assigning a clearance attribute on the topology is rejected (it used to be a silent no-op).

**R must divide** in four-state and persister–resistance topologies. A non-dividing “resistant” compartment is an absorbing sink filled only by conversion; it cannot form a self-sustaining clone under treatment.

**Persister route:** `ModelTopology.persister_resistance` emphasises Q→R rather than only P→R. Epigenetic R→Q reversion is off by default; genetic R should not use per-cell reversion (apparent resensitisation is competitive dilution via fitness cost).

---

## 5. Deterministic mean field (`CellDynamicsODE`)

The ODE is the expectation of the jump process in the large-population limit:

\[
\begin{aligned}
\dot X_i &= b_i(C,N)\, X_i - d_i(C)\, X_i
  - \sum_j u_{ij}(C)\, X_i + \sum_j u_{ji}(C)\, X_j, \\
\dot A &= \sum_{i\neq A} d_i(C)\, X_i - \kappa A
\quad\text{(when A is tracked)}.
\end{aligned}
\]

Optional `rate_multiplier_fn(t, key)` scales named rates in time (e.g. scheduled interventions); non-finite or negative multipliers are ignored (treated as 1).

**Assumption:** well-mixed, continuous counts; no demographic stochasticity. Appropriate for large \(N\) or as the LNA mean.

---

## 6. Stochastic simulation

### 6.1 Reaction network (`build_reactions`)

| Reaction | Stoichiometry (sketch) | Propensity |
|----------|------------------------|------------|
| Birth of type \(i\) | \(+1\) on \(i\) | \(b_i(C,N)\, X_i\) |
| Death of \(i\) | \(-1\) on \(i\); \(+1\) on A if tracked | \(d_i(C)\, X_i\) |
| Clearance | \(-1\) on A | \(\kappa\, A\) |
| Transition \(i\to j\) | \(-1\) on \(i\), \(+1\) on \(j\) | \(u_{ij}(C)\, X_i\) |

Binary fission is encoded as net \(+1\) (one cell becomes two). Propensities are mass-action in counts (independent cells).

### 6.2 Gillespie SSA (`GillespieSimulator`)

| Exposure | Method | Exact? |
|----------|--------|--------|
| Constant \(C\) | Direct method | Yes |
| Time-varying \(C(t)\) | Extrande thinning | Yes, **if** the propensity upper bound is valid |
| Forced `"direct"` + varying \(C\) | Frozen propensities per interval | **Approximate**; metadata `exact=False` |

**Extrande caveat:** the bound on total propensity over a look-ahead window is a **grid maximum × `bound_safety`**, not a proven analytic envelope. Extremely sharp PK peaks between sample points could under-bound. Increase `bound_samples` / `bound_safety` or shorten `lookahead` if exposure is spiked.

Negative counts after a jump raise an error (inconsistent propensity/stoichiometry) rather than silent clipping.

### 6.3 Tau-leaping (`TauLeapingSimulator`)

Poisson leaps with **critical-reaction** hybrid (Cao, Gillespie & Petzold, 2006): near-exhaustion reactions fire by SSA; leaps that would go negative are rejected and retried with halved \(\tau\). Below `ssa_threshold` total cells, the simulator uses exact SSA. This is an approximation that becomes exact as \(\tau\to 0\).

---

## 7. Linear noise approximation (`MomentODE`)

Propagates mean \(\mu(t)\) and covariance \(\Sigma(t)\):

\[
\dot\mu = f(t,\mu),\qquad
\dot\Sigma = J(t,\mu)\,\Sigma + \Sigma\, J(t,\mu)^\top + D(t,\mu),
\]

where \(f\) is the same nonlinear drift as the ODE, \(J=\partial f/\partial\mu\), and \(D=\sum_k a_k\,\nu_k\nu_k^\top\) with propensities \(a_k\) and stoichiometries \(\nu_k\).

**Density-dependent Jacobian:** birth \(f_i = B_i(C)\,(1-N/K)_+\,\mu_i\) with \(N=m\cdot\mu\) yields

\[
\frac{\partial f_i}{\partial\mu_j}
= b_i\,\delta_{ij} - \frac{B_i(C)\,\mu_i}{K}\, m_j
\quad\text{when } N < K
\]

(and zero birth contribution when \(N\ge K\)). Using \(J\mu\) instead of \(f(\mu)\) for the mean would be **wrong** under logistic growth (fixed point \(K/2\) instead of \(K\)).

**Default \(\Sigma_0\):** diagonal with entries \(\max(\mu_{0,i},1)\) (Poisson-like). For a known exact initial count vector use \(\Sigma_0=0\).

**Validity:** LNA is a large-population / Gaussian approximation. Near extinction it can produce non-PSD covariances; the solver symmetrizes and clamps small negative eigenvalues, and logs a warning when the defect is large.

**Scientific use:** mean trajectories alone often cannot separate cytostatic from cytotoxic mechanisms; process variance can. That is why moments are the “fast” backbone for likelihoods.

---

## 8. Cross-solver consistency (what must agree)

| Quantity | ODE | SSA / tau-leap | LNA |
|----------|-----|----------------|-----|
| Mean drift \(f\) | `rhs` | expectation of jumps | `drift` |
| Crowding total \(N\) | `topology.density_total` | same | same |
| Per-state birth (e.g. R) | `cell_type=` on `birth_rate` | same | same |
| Diffusion / jump noise | — | full process | \(D\) from same reactions |

Growth diagnostics that depend only on rates + topology (`asymptotic_growth_rate`, `gr_value`) use the low-density matrix and do not require integration, but they still assume the same rate laws as the simulators.

---

## 9. Checklist before trusting a fit or figure

1. **Multi-state growth:** use `asymptotic_growth_rate` / `doubling_time` / GR helpers — not `net_growth_rate`.
2. **Resistance:** give R its own birth (and usually death) profile; do not rely on bare default `RateSet` with a four-state topology.
3. **Quiescent kill:** set `quiescent_sensitivity` (or explicit Q death modulation) for the drug class.
4. **Induced plasticity:** use `transition_induction` (or `TransitionRateProfile.induction`) when \(u_0=0\).
5. **Suppressing a transition:** use `HillFoldChange(high < low)` in `transition_factor`, not a decreasing \(m\).
6. **Viability 4PL as modulator:** call `.as_effect()` first.
7. **Assay GR vs eigenvalue GR:** use `finite_horizon_gr` for fixed windows when R can dominate asymptotically.
8. **Time-varying PK + SSA:** prefer `exposure_mode="thinning"` (or auto); do not treat frozen-propensity `"direct"` as exact.
9. **Crowding after kill:** leave `density_counts_apoptotic=False` unless corpses truly occupy space in your assay.
10. **LNA near extinction:** treat covariance with caution; check warnings / PSD.

---

## 10. What this layer deliberately does *not* claim

- Spatial or pharmacokinetic distribution inside a tumour (exposure is a scalar \(C(t)\)).
- Mechanistic intracellular networks (those live under `umimic.signaling` and couple in separately if used).
- Identifiability of every rate from bulk data (multi-state conversion rates are often weakly identified from total viable counts alone).
- That default numerical constants are patient- or cell-line-specific parameters.

---

## References (conceptual)

- Multitype continuous-time branching / CTMC cell models; low-density rate matrix and dominant eigenvalue as asymptotic growth rate.
- Gillespie SSA; Extrande / thinning for time-inhomogeneous propensities (Voliotis et al., related literature).
- Hybrid tau-leaping with critical reactions (Cao, Gillespie & Petzold, *J. Chem. Phys.* 2006).
- Linear noise approximation / chemical Langevin covariance equation.
- Hafner et al. growth-rate inhibition (GR) metrics for dose–response under proliferation.

Implementation details and edge-case behaviour are enforced in code and in `tests/test_dynamics/` plus `tests/test_numerics/` (semantics, modelling, plasticity, acceptance).
