# Scientific assumptions — `umimic.pk`

Pharmacokinetics, dosing schedules, the unified exposure interface used by
dynamics, and BLI substrate/optics helpers. Drug time is in **hours**;
luciferin imaging time is in **minutes**.

**Scope:** `dosing.py`, `compartment.py`, `exposure.py`, `luciferin.py`.

---

## 1. Role in U-MIMIC

Dynamics never implement PK themselves. They call an exposure function

\[
C_{\mathrm{drug}} = C(t)
\]

supplied by `ExposureProfile` (or any callable). That concentration enters
birth/death/transition rate laws. BLI observation models may additionally use
`LuciferinKinetics` and `TissueAttenuation` for measurement physics, not for
cytotoxic exposure.

---

## 2. Dosing (`Dose`, `DosingSchedule`)

| Route | Meaning |
|-------|---------|
| `iv_bolus` | Instantaneous amount into central compartment |
| `iv_infusion` | Constant rate \(R = \mathrm{amount}/\mathrm{duration}\) over \([t, t+\Delta)\) |
| `oral` | Instantaneous amount into absorption (gut) compartment |

In vitro constant exposure uses `DosingSchedule.constant_invitro(c)` /
`fixed_concentration` (no PK ODE).

**Assumptions**

- Amounts are non-negative; infusion requires \(\Delta > 0\).
- No scheduled lag, missed doses, or adherence noise.
- Units of amount must be consistent with \(V_d\) or \(V_c\) (e.g. mg and L →
  mg/L); the code does not convert units.

---

## 3. One-compartment PK

### Parameters

| Symbol | Field | Constraint |
|--------|-------|------------|
| \(V_d\) | `vd` | \(> 0\) |
| \(k_e\) | `ke` | \(\ge 0\) |
| \(k_a\) | `ka` | \(> 0\) if oral doses are used; else `None` |
| \(F\) | `f_oral` | \((0, 1]\); default 1 |

### Equations

**IV bolus** (analytical superposition when every dose is IV bolus):

\[
C(t) = \sum_i \frac{D_i}{V_d}\, e^{-k_e (t-t_i)}\,\mathbf{1}_{t\ge t_i}.
\]

**Oral** (amount \(A\) in gut, \(F D\) deposited at dose times):

\[
\dot A = -k_a A,\qquad
\dot C = \frac{k_a A}{V_d} + \frac{R(t)}{V_d} - k_e C.
\]

**IV infusion:** \(R(t) = D/\Delta\) while active, else 0.

Half-life: \(t_{1/2} = \ln 2 / k_e\) (\(\infty\) if \(k_e=0\)).

### Bioavailability

Oral doses add \(F\cdot D\) to the gut compartment. IV bolus and infusion are
**not** scaled by \(F\). Default \(F=1\) is complete absorption — set
`f_oral` for incomplete oral bioavailability.

There is **no absorption lag** (dose is available immediately in the gut
compartment at `dose.time`).

---

## 4. Two-compartment PK

Clearance parameterization:

\[
k_e = \frac{CL}{V_c},\quad
k_{12} = \frac{Q}{V_c},\quad
k_{21} = \frac{Q}{V_p}.
\]

\[
\begin{aligned}
\dot C_1 &= -(k_e+k_{12})C_1 + k_{21} C_2 \frac{V_p}{V_c}
  + \frac{k_a A + R(t)}{V_c},\\
\dot C_2 &= k_{12} C_1 \frac{V_c}{V_p} - k_{21} C_2,\\
\dot A &= -k_a A.
\end{aligned}
\]

Readout is **central** concentration \(C_1\). Peripheral concentration is
internal state only. Oral \(F\) applies as in the one-compartment model.

**Not modelled:** effect compartment, protein binding / free fraction,
nonlinear clearance, time-varying \(V\) or \(CL\).

---

## 5. Integration contract

- The internal state is advanced through **every** dose time and infusion
  start/stop, independent of the requested `t_eval` grid.
- Within each segment, `solve_ivp` dense output samples requested times.
- Instantaneous doses at a time \(t^*\) are applied **before** \(C(t^*)\) is
  recorded (post-dose convention).
- Simultaneous doses at the same \(t\) **accumulate**.
- Concentrations are clamped at zero after integration (numerical floor only).

Sparse and dense `t_eval` therefore agree at shared times (for the analytical
IV-bolus path, exactly; for numerical paths, to integrator tolerance).

---

## 6. Exposure profile

| Factory | Behaviour |
|---------|-----------|
| `ExposureProfile.constant(c)` | \(C(t)\equiv c\) (in vitro) |
| `ExposureProfile.from_pk(pk, dosing)` | \(C(t)\) from compartment model |
| empty | \(C(t)\equiv 0\) |

`profile(t)` is the dynamics `exposure_fn`.

### Optional grid cache (`precompute`)

By default every query re-solves the PK (exact on the model). For hot paths
(SSA / Extrande propensity probes):

```python
exposure.precompute(np.linspace(0, t_max, n))
```

then uses **linear interpolation** on that grid. Callers must put dose and
infusion breakpoints on the grid if peak shape matters; inter-knot peaks are
under-resolved. Outside the grid, values are clamped to the endpoints.
`clear_cache()` restores exact solves.

There is no silent automatic cache: approximation is opt-in.

---

## 7. Luciferin kinetics (BLI substrate)

Phenomenological Bateman + Michaelis–Menten (time in **minutes**):

\[
C_{\mathrm{luc}}(t)
= D\frac{k_a}{k_a-k_e}\bigl(e^{-k_e t}-e^{-k_a t}\bigr),\quad
g=\frac{C}{K_m+C}.
\]

`signal_fraction` returns \(g(t)/g(t_{\mathrm{peak}})\), so only **relative**
timing off-peak matters. \(D\) and \(K_m\) are not true biochemical µM; do not
compare to published intracellular luciferin concentrations.

Defaults place a standard IP-like dose near saturation at peak. Parameters
must be positive rates / non-negative dose.

---

## 8. Tissue optical attenuation (BLI)

Point source at depth \(d\):

\[
\mathrm{Att} = e^{-\mu_{\mathrm{eff}} d}.
\]

Finite tumour of volume \(V\): sphere of radius \(r\), tissue depth above the
top `reference_depth`, **volume-averaged** attenuation (not centroid-only).
Centroid-only attenuation overstates loss in large tumours and can look like
progressive cell death under growth.

\(\mu_{\mathrm{eff}}\ge 0\), `reference_depth` \(\ge 0\). This is not a full
radiative-transport simulation.

---

## 9. Coupling checklist

1. Drug PK times in **hours**; luciferin imaging time in **minutes**.
2. Match amount and volume units (e.g. mg and L).
3. Oral incomplete absorption → set `f_oral` (and `ka`); IV ignores \(F\).
4. Central \(C(t)\) drives rates — if the pharmacology is free or tissue
   concentration, map externally or extend the model.
5. For Extrande / many SSA queries, `precompute` a grid that includes dose
   times; otherwise accept full re-solves.
6. In vitro: use constant exposure, not a zero-clearance PK hack.
7. BLI: set luciferin imaging time near peak or accept `signal_fraction < 1`.

---

## 10. What this layer does *not* claim

- Population PK / covariates / inter-occasion variability.
- Metabolite or multi-drug interaction models.
- Physiological PBPK organ trees.
- Absolute photon output from luciferin (only relative timing and simple
  attenuation).
- That interpolated cached exposure equals the ODE solution between knots.

---

## References (conceptual)

- Linear one- and two-compartment mammillary models; clearance form of
  micro-constants.
- Superposition for linear multi-dose PK.
- Bateman function for first-order absorption and elimination.
- Effective attenuation coefficient models for planar BLI.
