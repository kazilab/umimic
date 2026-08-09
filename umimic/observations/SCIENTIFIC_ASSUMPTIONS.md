# Scientific assumptions — `umimic.observations`

How latent cell-state vectors become data: observation operators, noise
models, process-variance fusion, and multimodal product likelihoods. For
rates and simulators see `umimic/dynamics/SCIENTIFIC_ASSUMPTIONS.md`; for
how these models enter estimation see
`umimic/inference/SCIENTIFIC_ASSUMPTIONS.md`.

**Scope:** `base.py`, `cell_counts.py`, `bli.py`, `tumor_volume.py`,
`biomarkers.py`, `multimodal.py`.

---

## 1. Latent state and observation operators

The latent state is a vector \(x\) ordered as `topology.active_states`. An
observable is a linear map

\[
N = H x, \qquad \mathrm{Var}(N) = H\,\Sigma\,H^\top
\]

with row vector \(H\) built from the topology (not from assumed index
positions). Supported kinds:

| Kind | Selector |
|------|----------|
| `viable` | all non-apoptotic states |
| `total` | every state |
| `proliferating` | P only |
| `quiescent` | Q only |
| `dead` | A only |
| `resistant` | R only |

**Always pass `topology=`** into observation models used with non-canonical
layouts. Without it, a 3-state vector is assumed `[P, Q, A]`; if the model is
actually `[P, Q, R]`, `viable` / `dead` are wrong.

Projected means use a numerical floor (\(\sim 10^{-6}\)) so log-scale
modalities never see a hard zero; this is not a biological “minimum
population.”

---

## 2. Conditional independence (multimodal)

Given the latent state \(x\), modalities are treated as independent:

\[
p(y_{\mathrm{counts}}, y_{\mathrm{BLI}}, y_{\mathrm{vol}}, \ldots \mid x)
= \prod_m p(y_m \mid x).
\]

Missing modalities (or NaNs) are skipped, not imputed. Shared animal effects
or assay batch correlation across modalities are **not** modelled; dependence
enters only through the common latent trajectory (and, for BLI, optional
paired volume for attenuation).

---

## 3. Cell counts — Negative binomial / Gaussian

### Mean

\[
\mu = H_{\mathrm{count\_type}}\, x
\]

Default `count_type="viable"`. Other choices (`total`, `proliferating`,
`dead`) change which compartment is scored.

### Noise

**ODE / no process variance** (exact discrete model):

\[
Y \sim \mathrm{NegBin}(\mu, \phi), \qquad
\mathrm{Var}(Y\mid\mu) = \mu + \frac{\mu^2}{\phi}.
\]

scipy parameterization: \(n=\phi\), \(p=\phi/(\phi+\mu)\). Larger \(\phi\) →
less overdispersion (Poisson limit \(\phi\to\infty\)).

**Moment mode with LNA process variance** \(V = H\Sigma H^\top\):

\[
Y \approx \mathcal{N}\!\bigl(\mu,\; V + \mu + \mu^2/\phi\bigr).
\]

This is the law of total variance: process variance of the latent mean plus
conditional observation variance. Matching the first two moments does **not**
make a Gaussian a NegBin. At small \(\mu\) (\(\lesssim 20\)) the Gaussian
puts mass below zero and lacks skew; a one-time warning is issued. Prefer
`mode="ode"` or a particle filter for low-count data.

### Sampling

Matches the likelihood branch: NegBin without process variance; Gaussian with
total variance (clipped at 0) when process variance is provided. Free
parameter key: **`overdispersion`**.

### EKF

Linear update with \(H = H_{\mathrm{count\_type}}\) and
\(R = \mu + \mu^2/\phi\) (measurement only; process noise lives in the
filtered \(\Sigma\)).

---

## 4. Tumor volume — lognormal

\[
\log Y_V \sim \mathcal{N}\!\bigl(\log(\beta N_{\mathrm{viable}}),\; \sigma_V\bigr)
\]

with optional process inflation

\[
\sigma_{\mathrm{eff}}
= \sqrt{\sigma_V^2 + V / N_{\mathrm{viable}}^2}
\quad\text{(delta method for signal \(\propto N\))}.
\]

**Scale convention:** \(\beta N\) is the **median** of \(Y_V\), not the mean.
The mean is \(\beta N\, e^{\sigma_V^2/2}\).

**Defaults:** \(\beta \sim 10^{-5}\,\mathrm{mm}^3/\mathrm{cell}\)
(\(\sim 10^5\) cells/mm³ of tissue including stroma) is a physical scale, not
a free growth parameter. Inference typically estimates \(\sigma_V\)
(`sigma_v`) and leaves \(\beta\) fixed so \(N\) and \(\beta\) are not
confounded.

**EKF:** update on the log scale with \(H = H_{\mathrm{viable}} / N\).

Modality key: `volume` (data may also use `tumor_volume`; the likelihood maps
paired volume into BLI attenuation).

---

## 5. BLI — lognormal with measurement physics

\[
Y_{\mathrm{BLI}}
= \alpha\, N_{\mathrm{luc}}\, g(C_{\mathrm{luc}})\, \mathrm{Att}\, \varepsilon,
\quad
\log\varepsilon \sim \mathcal{N}(0,\sigma_{\log}^2).
\]

- \(N_{\mathrm{luc}}\): viable cells (all assumed reporter-positive unless the
  topology encodes otherwise).
- \(g\): luciferin kinetic factor from `LuciferinKinetics` when imaging time is
  set; otherwise 1. Units of that kinetics model are **phenomenological**
  (see `umimic.pk.luciferin`).
- \(\mathrm{Att}\): tissue attenuation; can use paired `tumor_volume` or
  `tumor_depth` per time point (not a series-mean volume).
- \(\alpha\): photons per cell — fixed calibration, not a free growth rate.

Same median/lognormal convention and process-variance delta method as volume.
Free noise key in inference: **`sigma_log_bli`** (legacy alias `sigma_log`
still accepted).

**EKF:** log-scale; factors \(\alpha, g, \mathrm{Att}\) cancel from
\(d\log h / dx = H_{\mathrm{viable}}/N\).

---

## 6. Biomarkers — Beta fractions

Snapshot immunostaining fractions:

| Type | Fraction |
|------|----------|
| `ki67` | (sum of **division_states**) / viable |
| `caspase` | A / total (including dead) |

Ki-67 uses every dividing state, so a proliferating R compartment counts as
positive. Without a topology, only P is treated as cycling.

Noise:

\[
Y \sim \mathrm{Beta}(\kappa f,\, \kappa(1-f)), \quad \kappa > 0.
\]

Extinct population (\(x=0\)): fraction undefined → log-likelihood contribution
**0** (term dropped), samples are NaN — not a fake 0.5.

Free parameter key: **`biomarker_precision`** (legacy alias `precision`).

**Not used in EKF:** no `linearize` (no Gaussian approximation for Beta
fractions). Use batch likelihood or particle filtering. LNA process variance
on counts is **not** folded into the Beta likelihood (a naive map would be
misleading without a fraction Jacobian).

Clips \(f\) and \(y\) slightly inside \((0,1)\) for numerical support of the
Beta density.

---

## 7. Parameter keys (inference alignment)

| Modality | Construction attrs | Free / override keys in `params` |
|----------|--------------------|----------------------------------|
| Counts | `overdispersion` | `overdispersion` |
| BLI | `alpha`, `sigma_log` | `sigma_log_bli` (+ alias `sigma_log`) |
| Volume | `beta`, `sigma_v` | `sigma_v` |
| Biomarker | `precision` | `biomarker_precision` (+ alias `precision`) |

These free keys match `OBSERVATION_PARAM_NAMES` in
`umimic.inference.likelihood`. Calibrations \(\alpha\), \(\beta\) stay on the
observation object unless you rebuild it.

---

## 8. Sampling and posterior predictive checks

`sample(..., params=..., process_variance=...)` uses the **same** noise model
as `log_likelihood`:

- param overrides for free noise parameters;
- optional LNA process variance when provided.

`posterior_predictive_check` in moment mode projects \(\Sigma\) onto each
modality’s observable and passes that variance into `sample`, so predictive
replicates are not measurement-noise-only.

In ODE mode there is no \(\Sigma\); replicates use measurement noise alone
(consistent with the ODE likelihood).

---

## 9. Checklist

1. Pass `topology=` matching the dynamics topology.
2. Low counts: avoid moment-mode Gaussian NegBin; use ODE or PF.
3. Free observation noise in θ only via the keys in §7.
4. Multimodal data: register every modality you want scored in
   `MultimodalObservation` / experiment config.
5. BLI + volume: provide paired volumes if attenuation should grow with
   tumour size.
6. Ki-67: confirm `division_states` match what the antibody reports.
7. Caspase: denominator is total cells including dead — match assay
   definition.
8. Do not interpret lognormal `expected_value` as arithmetic mean.
9. Biomarkers + EKF: not supported; use likelihood or PF.
10. Independence across modalities is a modelling choice, not a biological
    law.

---

## 10. What this layer does *not* claim

- Spatial optical transport beyond simple attenuation / depth–volume maps.
- Separate luciferase-negative viable subpopulations unless encoded in the
  state vector.
- Shared animal-level random effects across modalities.
- Structural zeros / dropout models for empty wells (floor mean instead).
- That Gaussian ≈ NegBin is adequate below the small-count warning threshold.

---

## References (conceptual)

- Negative binomial overdispersion for bulk cell counts.
- Lognormal measurement error for multiplicative imaging / caliper assays.
- Delta method mapping process variance on \(N\) to log-signal variance.
- Beta likelihood for compositional / fractional biomarkers.
- Conditional independence given latent state for multimodal fusion.
