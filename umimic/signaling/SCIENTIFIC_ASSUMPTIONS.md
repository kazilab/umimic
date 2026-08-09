# Scientific assumptions — `umimic.signaling`

Intracellular pathway interface, the toy MAPK/AKT scaffold, and how pathway
activity multiplies cell-population rates in the experiment orchestrator.

**Scope:** `network.py`, `models.py`, plus coupling in
`umimic.pipeline.experiment` / `CouplingConfig`.

---

## 1. Role in U-MIMIC

Cell dynamics (`umimic.dynamics`) evolve phenotype counts with rates that may
depend on drug concentration \(C(t)\). Signaling is an **optional upstream
layer**:

1. Integrate a pathway state \(y(t)\) under exposure \(C(t)\).
2. Map \(y(t)\) to a scalar activity \(a(t)\).
3. Multiply selected birth / death / transition rates by \(m(t)\ge 0\).

This is open-loop: cell state does not feed back into the pathway or into PK.
Only the **deterministic cell ODE** consumes the multiplier. Gillespie,
tau-leaping, and inference refuse signaling with an error rather than ignore
it.

---

## 2. Interface (`SignalingNetwork`)

| Method | Contract |
|--------|----------|
| `node_names` | Ordered labels for the state vector |
| `initial_state()` | \(y(0)\) |
| `rhs(t, y, concentration)` | \(\dot y\) given drug concentration |
| `observable_map(y)` | Named activities (default: identity on nodes) |

Subclasses own units, bounds, and biology. The abstract layer does not solve
ODEs or define likelihoods for phospho-data.

---

## 3. Toy MAPK/AKT model (scaffold only)

### Warning

`ToyMapkAktNetwork` is **not** a Raf–MEK–ERK or PI3K–AKT–mTOR model. There is
no cascade, feedback, or data-calibrated parameter. Use it to exercise
coupling and config plumbing, not to draw pathway conclusions.

### Equations

Two independent nodes \(i\in\{\mathrm{mapk},\mathrm{akt}\}\):

\[
\begin{aligned}
T_i(C) &= \mathrm{clip}\bigl(b_i + s\, d_i\, C,\, 0,\, 1\bigr),\\
\dot y_i &= \delta\,\bigl(T_i(C) - \mathrm{clip}(y_i,0,1)\bigr).
\end{aligned}
\]

| Symbol | Field | Constraint |
|--------|-------|------------|
| \(b_i\) | `mapk_baseline`, `akt_baseline` | \([0,1]\) |
| \(d_i\) | `mapk_drive`, `akt_drive` | \(\ge 0\) |
| \(\delta\) | `decay` | \(> 0\) |
| \(s\) | `direction` | \(-1\) inhibitory (default), \(+1\) stimulatory |

Steady state at fixed \(C\): \(y_i^\star = T_i(C)\). Activity is bounded in
\([0,1]\). Nodes do **not** cross-talk.

### Drug direction

Default **`inhibitory`**: higher \(C\) lowers set points (typical targeted
inhibitors). **`stimulatory`** is explicit only (agonist / relief-of-feedback
toy). Config field: `signaling.direction`.

### Concentration scaling

Drive is linear in \(C\) then clipped. With defaults, MAPK target hits 0 by
\(C \approx 2\) in whatever unit exposure uses. Rescale `*_drive` to your
concentration unit; do not treat defaults as µM EC50s.

---

## 4. Rate coupling (pipeline)

### Formula

\[
a(t) = \max\bigl(0,\, w_{\mathrm{mapk}} y_{\mathrm{mapk}}(t)
  + w_{\mathrm{akt}} y_{\mathrm{akt}}(t)\bigr),
\]

\[
e(a) =
\begin{cases}
\dfrac{a^{h}}{EC_{50}^{h}+a^{h}} & \text{Hill},\\[6pt]
\dfrac{1}{1+e^{-k(a-c)}} & \text{logistic},
\end{cases}
\qquad
m(t,\mathrm{key}) = \max\bigl(0,\, 1 + m_{\max}(\mathrm{key})\, e(a(t))\bigr).
\]

with \(m_{\max}\ge 0\) and \(e\in[0,1]\).

### Semantics (critical)

| Pathway activity | Multiplier |
|------------------|------------|
| High | up to \(1+m_{\max}\) |
| Low / off | \(\to 1\) (bare RateSet) |

So:

- **RateSet baselines are the pathway-off floor.**
- Active oncogenic signaling *boosts* targeted rates (e.g. birth).
- An **inhibitor** that lowers \(y\) removes that boost; it does not, by
  itself, drive rates below the RateSet via this formula.

This is **not** a free-signed map “drug multiplies death.” Encoding
“pathway activity suppresses death” would need a different link (e.g. negative
\(m_{\max}\), which config rejects).

Targets: `birth`, `death`, `transition`, or fine keys `death:P`,
`transition:P->Q`, etc. (see `CouplingConfig`).

### Integration

1. Pathway ODE integrated on the cell simulation grid under \(C(t)\).
2. Failed pathway solve → **RuntimeError** (no silent drop of coupling).
3. Activity interpolated linearly in \(t\) for the multiplier callback.
4. Cell ODE multiplies named rates by \(m(t,\mathrm{key})\).

Supported for **ODE cell dynamics only**.

---

## 5. Config checklist

1. `signaling.enabled` + `model: toy_mapk_akt` only for scaffold demos.
2. Set `signaling.direction` for inhibitors vs agonists.
3. Scale `mapk_drive` / `akt_drive` to exposure units.
4. Set `coupling.targets` and \(m_{\max}\) knowing rates only go **up** from
   baseline with high activity.
5. Do not enable signaling for Gillespie / tau-leaping / `fit()` — they error.
6. Put pathway initial conditions in \([0,1]\) under known node names.

---

## 6. What this layer does *not* claim

- Mechanistic MAPK or AKT biochemistry.
- Cell-to-cell pathway heterogeneity or stochastic gene expression.
- Two-way feedback from phenotype composition to signaling.
- Identifiability of pathway parameters from bulk counts alone.
- That Hill \(EC_{50}\) on activity equals a drug EC50 in concentration units.

---

## References (conceptual)

- Phenomenological activity variables for rate modulation (not detailed kinetic
  schemes).
- Hill / logistic link functions from a latent activity to a bounded effect.
- Separation of measurement PK/exposure from intracellular and population
  layers.
