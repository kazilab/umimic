# Mathematical model

This page explains how U-MIMIC calculates, from the drug exposure to the
fitted parameters. It follows one calculation from start to finish and gives
the equations each step uses. The `SCIENTIFIC_ASSUMPTIONS.md` note beside each
package covers edge cases and pitfalls in more depth (see
[Scientific assumptions](scientific-assumptions.md)), and the table in
section 13 names the code that implements each equation.

Time is in hours throughout, except luciferin imaging time, which is in
minutes. Concentrations are in whatever unit the dosing uses; the package does
not convert units.

## 1. The calculation at a glance

Given a parameter vector $\theta$ and data $y$, U-MIMIC computes the
log-likelihood $\ell(\theta) = \log p(y \mid \theta)$ in five steps, and a
sixth explores it:

1. **Exposure.** The drug concentration over time, $C(t)$: constant in vitro,
   or from a pharmacokinetic (PK) model of the dosing schedule in vivo
   (section 3).
2. **Rates.** $\theta$ sets each cell state's division, death and switching
   rates as functions of $C$ (section 4).
3. **Population dynamics.** The rates define a stochastic model of cell
   numbers. U-MIMIC predicts its mean $\mu(t)$ and, in moment mode, its
   covariance $\Sigma(t)$ (section 5).
4. **Observation.** Each measurement is a noisy readout of a combination of
   cell numbers, with a noise law chosen for that kind of measurement
   (section 7).
5. **Likelihood.** Each scored measurement contributes a log-probability, and
   their sum is $\ell(\theta)$ (section 8).
6. **Estimation.** Maximum likelihood, MCMC and the other fitting methods
   search or sample $\ell(\theta)$, with priors where given (section 9).

Section 6 covers growth summaries computed directly from the rates, and
section 10 the diagnostics that say which parameters the data constrain.

## 2. Cell states and the stochastic model

The population is a vector of cell counts $x$ over the states in use:

| State | Meaning |
|---|---|
| P | Proliferating, drug-sensitive cells |
| Q | Quiescent or slow-cycling cells, such as drug-tolerant persisters |
| A | Apoptotic cells: filled by death, emptied by clearance |
| R | Resistant cells, which must divide to outgrow treatment |

A *topology* says which states exist, which divide, which die and which
switch into which:

| Topology | States | Divide | Die | Switching |
|---|---|---|---|---|
| `two_state` | P, Q | P | P, Q | P ⇄ Q |
| `three_state` (default) | P, Q, A | P | P, Q | P ⇄ Q |
| `four_state` | P, Q, A, R | P, R | P, Q, R | P ⇄ Q, P → R |
| `persister_resistance` | P, Q, A, R | P, Q, R | P, Q, R | P ⇄ Q, Q → R; optionally R → Q and P → R |

Each viable cell in state $i$ acts independently of the others. It

- divides at rate $b_i(C, N)$, one cell becoming two;
- dies at rate $d_i(C)$, moving to A when A is tracked;
- switches to state $j$ at rate $u_{ij}(C)$.

Apoptotic cells are cleared at rate $\kappa$. This is a multitype
continuous-time Markov chain; without crowding it is a multitype branching
process. Every waiting time is exponential, and space and cell-cycle phases
are not modelled.

Crowding uses the total $N = \sum_i m_i x_i$, where $m_i = 1$ for viable
states and, by default, $m_A = 0$, so dead cells do not take up space.

## 3. Drug exposure

**In vitro**, each series has a constant concentration, $C(t) = c$.

**In vivo**, $C(t)$ comes from a compartment model of the dosing schedule. In
the one-compartment model, with volume $V_d$, elimination rate $k_e$,
absorption rate $k_a$ and oral bioavailability $F$:

$$
\frac{dA_g}{dt} = -k_a A_g,
\qquad
\frac{dC}{dt} = \frac{k_a A_g + R(t)}{V_d} - k_e C .
$$

An oral dose $D$ adds $F D$ to the gut amount $A_g$; an intravenous bolus adds
$D / V_d$ to $C$; an infusion of $D$ over a duration $\Delta$ gives
$R(t) = D / \Delta$ while it runs. When every dose is a bolus, the solution is
a sum over doses $k$ at times $t_k$:

$$
C(t) = \sum_k \frac{D_k}{V_d}\, e^{-k_e (t - t_k)}\, \mathbf{1}[t \ge t_k].
$$

The two-compartment model has central and peripheral volumes $V_c$ and $V_p$,
clearance $CL$ and intercompartmental clearance $Q$, with
$k_e = CL / V_c$, $k_{12} = Q / V_c$ and $k_{21} = Q / V_p$:

$$
\begin{aligned}
\frac{dC_1}{dt} &= -(k_e + k_{12})\, C_1 + k_{21} \frac{V_p}{V_c}\, C_2
  + \frac{k_a A_g + R(t)}{V_c}, \\
\frac{dC_2}{dt} &= k_{12} \frac{V_c}{V_p}\, C_1 - k_{21}\, C_2 .
\end{aligned}
$$

The central concentration $C_1$ drives the cell rates. Integration restarts
at every dose and infusion boundary, a dose at time $t$ is applied before
$C(t)$ is read, and doses at the same time add. There is no absorption lag,
protein binding, effect compartment or nonlinear clearance.

## 4. Rate laws

### 4.1 Dose–response curves

Drug effects use the Emax/Hill curve, which is 0 without drug and rises to
$E_{\max}$:

$$
E(C) = E_{\max} \frac{C^h}{EC_{50}^h + C^h},
\qquad E_{\max} \ge 0,\ EC_{50} > 0,\ h > 0 .
$$

### 4.2 The rates of each state

$$
\begin{aligned}
b_i(C, N) &= \underbrace{b_{0,i}\, \bigl[1 - E_{b,i}(C)\bigr]}_{B_i(C)}
  \Bigl(1 - \frac{N}{K}\Bigr)_{+}, \\
d_i(C) &= d_{0,i} + E_{d,i}(C), \\
u_{ij}(C) &= u_{0,ij}\, f_{ij}(C)\, \bigl[1 + m_{ij}(C)\bigr] + a_{ij}(C),
\end{aligned}
$$

where $(z)_+ = \max(z, 0)$. These are the drug's three separate modes of action:

- **Cytostatic.** $E_{b,i}$ is the fraction of division blocked, capped at 1
  (complete arrest).
- **Cytotoxic.** $E_{d,i}$ adds to the death rate, in units of 1/h.
- **Plasticity.** $f_{ij}$ is a fold change on an existing switching route
  (above 1 induces it, below 1 suppresses it), and $a_{ij}$ is an additive
  rate that lets the drug open a route that is absent without it
  ($u_{0,ij} = 0$). The factor $1 + m_{ij}$ remains for older
  configurations.

The crowding factor applies only when a carrying capacity $K$ is set. Every
rate is clamped at zero. Making cytostasis a fraction and cytotoxicity an
addition is a modelling choice: it keeps the two mechanisms separate, each
with its own potency and efficacy.

### 4.3 From fitted parameters to rates

Fits name their parameters. Each name sets one part of the rate laws:

| Parameter | Sets |
|---|---|
| `b0` | $b_{0,P}$, and $b_{0,i}$ of any other dividing state without its own |
| `b0_Q`, `b0_R` | $b_{0,Q}$, $b_{0,R}$. These states then do not inherit P's cytostatic effect. |
| `d0_P`, `d0_Q`, `d0_R` | $d_{0,i}$. Without `d0_Q`, $d_{0,Q} = 0.5\, d_{0,P}$. |
| `emax_death`, `ec50_death`, `hill_death` | $E_{d,P}$ |
| `emax_death_Q`, `emax_death_R` | $E_{d,Q}$, $E_{d,R}$, with P's $EC_{50}$ and $h$. Without them, Q and R have no drug-induced death. |
| `emax_birth`, `ec50_birth`, `hill_birth` | $E_b$, shared by every dividing state without its own `b0_*` |
| `u_PQ`, `u_QP`, `u_PR`, `u_QR`, `u_RQ` | $u_{0,ij}$ |
| `induced_PQ`, `induced_QR` | $a_{ij}$, an Emax/Hill curve whose $EC_{50}$ and $h$ are `ec50_induction` and `hill_induction`, or else the death curve's |
| `clearance_rate` | $\kappa$ |
| `overdispersion`, `sigma_log_bli`, `sigma_v`, `biomarker_precision` | Measurement noise (section 7) |
| `sigma_extrinsic` | Between-well spread (section 8.4) |

The built-in parameter sets choose which of these are free:

| Set | Free parameters |
|---|---|
| `default` | `b0`, `d0_P`, the three death-curve parameters, `u_PQ`, `u_QP`, `overdispersion`. A purely cytotoxic drug. |
| `mechanism` | `default` plus the three birth-curve parameters. Admits both mechanisms; requires moment mode (section 5.4). |
| `resistance` | `default` plus `b0_R`, `d0_R`, `emax_death_R`, `u_PR` |
| `persister` | `default` plus `b0_Q`, `b0_R`, `d0_Q`, `d0_R`, `emax_death_Q`, `emax_death_R`, `u_QR`, `induced_QR` |

A fit is refused if a dividing or dying state would have no rate of its own,
because it would silently inherit P's or have none.

## 5. Population dynamics

### 5.1 The mean: deterministic equations

The expected cell numbers follow, for each viable state $i$:

$$
\frac{dx_i}{dt} = \bigl[b_i(C, N) - d_i(C)\bigr]\, x_i
  - \sum_{j} u_{ij}(C)\, x_i + \sum_{j} u_{ji}(C)\, x_j,
\qquad
\frac{dx_A}{dt} = \sum_{i} d_i(C)\, x_i - \kappa\, x_A ,
$$

with $b_i = 0$ for states that do not divide and $d_i = 0$ for states that do
not die. This is the large-population limit of the stochastic model. With
constant exposure and no crowding the system is linear,
$\dot x = M(C)\, x$, and is solved exactly as $x(t) = e^{M(C)\, t}\, x(0)$.
Otherwise it is integrated numerically, restarting at exposure breakpoints so
the solver cannot step over a dose.

### 5.2 Stochastic simulation

The same model as discrete events, each with a stoichiometry vector $\nu_k$
(the change in $x$) and a propensity $a_k$ (its rate):

| Event | Change | Propensity |
|---|---|---|
| Division of $i$ | $x_i + 1$ | $b_i(C, N)\, x_i$ |
| Death of $i$ | $x_i - 1$, and $x_A + 1$ when A is tracked | $d_i(C)\, x_i$ |
| Switch $i \to j$ | $x_i - 1$, $x_j + 1$ | $u_{ij}(C)\, x_i$ |
| Clearance | $x_A - 1$ | $\kappa\, x_A$ |

**Gillespie's direct method** (constant exposure; exact). With total
propensity $a_0 = \sum_k a_k$, the next event comes after a waiting time
$\tau \sim \mathrm{Exp}(a_0)$ and is event $k$ with probability $a_k / a_0$.

**Extrande thinning** (time-varying exposure; exact while its bound holds).
Over a look-ahead window $[t, t + L]$, take a bound $B \ge a_0$ and draw
$\tau \sim \mathrm{Exp}(B)$. If $\tau > L$, move to $t + L$ with no event.
Otherwise, at $t + \tau$, fire event $k$ with probability $a_k(t + \tau) / B$,
or nothing with probability $1 - a_0(t + \tau) / B$. The bound is 1.05 times
the largest $a_0$ found on a 16-point grid over the window, and the window is
halved whenever the bound is exceeded.

**Tau-leaping** (approximate; exact as $\tau \to 0$). Each event fires
$\mathrm{Poisson}(a_k \tau)$ times per leap, with $\tau$ chosen so that
propensities change by at most 3%. Events within 10 firings of exhausting a
state are simulated exactly, a leap that would make a count negative is
retried with half the step, and small populations are simulated exactly.

### 5.3 Mean and covariance: the linear noise approximation

Moment mode propagates the mean $\mu(t)$ and the covariance $\Sigma(t)$ of the
cell numbers together:

$$
\frac{d\mu}{dt} = f(t, \mu),
\qquad
\frac{d\Sigma}{dt} = J\,\Sigma + \Sigma\, J^{\top} + D,
\qquad
D = \sum_k a_k(t, \mu)\, \nu_k \nu_k^{\top},
$$

where $f$ is the right-hand side of the equations in 5.1 and
$J = \partial f / \partial \mu$ its Jacobian. Under crowding, the division
terms contribute

$$
\frac{\partial f_i}{\partial \mu_j} \ni b_i\, \delta_{ij} - \frac{B_i(C)\, \mu_i}{K}\, m_j
\qquad (N < K).
$$

The likelihood starts from $\Sigma(0) = \mathrm{diag}\bigl(\max(\mu_i(0), 1)\bigr)$,
a Poisson-like uncertainty in seeding. This is a large-population, Gaussian
approximation. Near extinction the covariance can stop being positive
semidefinite; the solver then symmetrises it, clips small negative
eigenvalues and warns when the defect is large.

### 5.4 Why the variance matters, and why it is rarely enough

For a linear birth–death process started from $n_0$ cells, with $r = b - d$:

$$
\mathbb{E}[X_t] = n_0\, e^{r t},
\qquad
\mathrm{Var}[X_t] = n_0\, \frac{b + d}{b - d}\, e^{r t} \bigl(e^{r t} - 1\bigr).
$$

The mean depends only on $b - d$, but the variance also depends on $b + d$.
In principle, then, the variance tells a drug that blocks division
(lower $b$) from one that kills (higher $d$) when both slow growth equally.
The linear noise approximation reproduces both moments exactly for this
process. For example, with $b = 0.04$, $d = 0.01$, $n_0 = 500$ and
$t = 72$ h, the closed form and the package both give a mean of 4,335.6 and a
variance of 55,431.

In practice, measurement noise swamps it. Counting noise adds
$\mu + \mu^2 / \phi$ (section 7.2), so the share of the observed variance that
carries the information is

$$
\text{share} = \frac{V_{\text{proc}}}{V_{\text{proc}} + \mu + \mu^2 / \phi} .
$$

In the example above with $\phi = 10$ that share is about 3%; at the
package's default P/Q settings it is about 1.5%. In the package's own tests,
cytostatic and cytotoxic drugs with matched means were told apart at chance.
Treat a division/death split from counts alone as determined by the prior
unless you have shown otherwise for your design. Information about death
comes from a second observable: a dead-cell count (section 7.2) or clone
sizes (section 9.5).

## 6. Growth summaries

Growth summaries come straight from the rates, without simulation. The
low-density rate matrix $M(C)$ over the viable states (A excluded) is

$$
M_{ii} = b_i(C) - d_i(C) - \sum_j u_{ij}(C),
\qquad
M_{ji} = u_{ij}(C) \quad (i \ne j),
$$

with $b_i$ at zero density. From it:

- **Asymptotic growth rate** $g(C)$: the eigenvalue of $M(C)$ with the largest
  real part, the rate the population approaches once its state mix settles.
- **Doubling time** $T_2 = \ln 2 / g(C)$ when $g > 0$, otherwise infinite.
- **Stable state mix**: the dominant right eigenvector of $M(C)$, normalised to
  sum to 1.
- **Asymptotic GR**: $\mathrm{GR}(C) = 2^{\,g(C) / g(C_{\text{ref}})} - 1$,
  where 1 is no effect, 0 complete cytostasis and negative values net loss.
- **Finite-horizon GR**: what a fixed-length assay measures, from viable
  counts simulated with the deterministic equations over the assay window:

$$
\mathrm{GR} = 2^{\,\log_2(x_T / x_0) \,/\, \log_2(x^{\text{ctrl}}_T / x_0)} - 1 .
$$

For example, at the default rates without drug ($b_0 = 0.04$, $d_{0,P} = 0.01$,
$d_{0,Q} = 0.005$, $u_{PQ} = 0.005$, $u_{QP} = 0.003$, in 1/h):

$$
M = \begin{pmatrix} 0.025 & 0.003 \\ 0.005 & -0.008 \end{pmatrix},
$$

which gives $g = 0.0254$/h, a doubling time of 27.2 h and a stable mix of 87%
P and 13% Q. The single-state rate $b_0 - d_{0,P} = 0.030$/h (23.1 h) overstates
growth because it ignores cells lost to Q. When a rare but fitter resistant
state is present, it eventually sets $g$ even if it is negligible during the
assay, so compare assays with the finite-horizon GR.

## 7. Observation models

### 7.1 What each measurement reads

Each measurement type reads a sum of states, $N = h^{\top} x$, where $h$
selects the states: viable (all except A, the default), total, or P, Q, A or R
alone. In moment mode it also carries the process variance of that sum,
$V_{\text{proc}} = h^{\top} \Sigma\, h$.

### 7.2 Cell counts

With the deterministic forward model, counts follow a negative binomial with
mean $\mu = h^{\top} x$ and overdispersion $\phi$:

$$
P(Y = y) = \binom{y + \phi - 1}{y}
  \Bigl(\frac{\phi}{\phi + \mu}\Bigr)^{\phi}
  \Bigl(\frac{\mu}{\phi + \mu}\Bigr)^{y},
\qquad
\mathrm{Var}(Y) = \mu + \frac{\mu^2}{\phi} .
$$

Larger $\phi$ means less noise, with the Poisson distribution as the limit. In
moment mode the total variance is $V_{\text{proc}} + \mu + \mu^2 / \phi$, and
the count stays negative binomial with its size parameter lowered so that its
variance equals that total:

$$
\phi_{\text{eff}} = \frac{\mu^2}{V_{\text{proc}} + \mu^2 / \phi} .
$$

This keeps the law discrete and skewed, as counts are. A Gaussian with the
same mean and variance is used only when the total variance is at or below
$\mu$, or when it is requested. The package warns when the expected count
falls below 20 or the skew of the total distribution exceeds 0.2, where moment
mode becomes unreliable. A dead-cell count is a second count channel that
reads A, with its own overdispersion parameter.

### 7.3 Tumour volume

Volume is lognormal around $\beta N$, where $\beta$ is a fixed volume per cell
(default $10^{-5}$ mm³) and $N$ the viable count:

$$
\log Y_V \sim \mathcal{N}\bigl(\log(\beta N),\ \sigma_V^2 + V_{\text{proc}} / N^2\bigr).
$$

The term $V_{\text{proc}} / N^2$ is the process variance carried to the log
scale (the delta method). $\beta N$ is the median, not the mean, and
$\sigma_V$ (`sigma_v`) is the free noise parameter.

### 7.4 Bioluminescence (BLI)

$$
\log Y_{\text{BLI}} \sim \mathcal{N}\bigl(\log(\alpha\, N\, g\, \mathrm{Att}),\
  \sigma_{\text{BLI}}^2 + V_{\text{proc}} / N^2\bigr),
$$

where $\alpha$ is a fixed photon output per cell and $\sigma_{\text{BLI}}$
(`sigma_log_bli`) the free noise parameter. $g$ corrects for imaging away from
the luciferin peak. With imaging time $t$ in minutes,

$$
C_{\text{luc}}(t) = D\, \frac{k_a}{k_a - k_e} \bigl(e^{-k_e t} - e^{-k_a t}\bigr),
\qquad
g(t) = \frac{s(t)}{s(t_{\text{peak}})},
\quad s = \frac{C_{\text{luc}}}{K_m + C_{\text{luc}}},
$$

and $g = 1$ when no imaging time is set. These luciferin units are
phenomenological, not intracellular concentrations. $\mathrm{Att}$ is tissue
attenuation: $e^{-\mu_{\text{eff}} d}$ for a source at depth $d$. For a tumour
of volume $v$, treated as a sphere of radius $r = (3v / 4\pi)^{1/3}$ whose top
lies at depth $d_0$, attenuation is averaged over the volume:

$$
\langle \mathrm{Att} \rangle = e^{-\mu_{\text{eff}} (d_0 + r)}\,
  \frac{3\, (\xi \cosh \xi - \sinh \xi)}{\xi^3},
\qquad \xi = \mu_{\text{eff}}\, r .
$$

The volume comes from a paired volume reading at the same time point. Using
the centre's depth instead overstates the loss, more so as the tumour grows,
which can look like cell death. For $v = 500$ mm³,
$\mu_{\text{eff}} = 0.5$/mm and $d_0 = 2$ mm, the average is 0.055 (a Monte
Carlo check agrees), against 0.031 at the centre.

### 7.5 Marker fractions

Ki-67 reads the fraction of viable cells in dividing states; caspase reads
$x_A$ over all cells. A fraction $f$ is observed with Beta noise of precision
$\kappa_b$ (`biomarker_precision`):

$$
Y \sim \mathrm{Beta}\bigl(\kappa_b f,\ \kappa_b (1 - f)\bigr).
$$

Fractions are clipped to $[10^{-4}, 1 - 10^{-4}]$, a population with no cells
contributes nothing, and no process variance is added.

### 7.6 Readings below the detection limit

For BLI and volume with a detection limit $L > 0$, a reading at or below $L$
contributes the probability of falling below it rather than a density:
$\log \Phi\bigl((\log L - \log \tilde y) / \sigma\bigr)$, where $\tilde y$ is
the predicted median and $\sigma$ the total log-scale standard deviation. Without a
detection limit, a reading of zero or less is an error.

## 8. The likelihood

### 8.1 The starting state

For each series, the first non-missing reading of the anchor measurement
(cell counts by default), $y_{t_0}$, sets the starting state
$x(t_0) = y_{t_0}\, \pi$. The starting mix $\pi$ is one of:

- all cells in P (the default);
- the stable mix at the series' concentration (section 6);
- fractions given explicitly.

That reading is then not scored, so it is not used twice. All in P is a
modelling claim, and it affects early growth and the switching rates.

### 8.2 Series and replicates

Series are grouped by concentration. In vivo, one PK exposure is evaluated on
absolute experiment time. Replicates with the same anchor time and starting
state share one forward solve on the union of their time points, and each is
scored at its own times.

### 8.3 Scoring each time point

Given the latent state, measurement types are independent, so their
log-probabilities add. One exception: in moment mode, when two or more
measurements of the same population (counts, BLI, volume) are taken at one
time point, they are scored together as one multivariate normal:

$$
z \sim \mathcal{N}\bigl(\hat z,\ \mathbf{J} \Sigma \mathbf{J}^{\top} + \mathrm{diag}(R)\bigr).
$$

Here $z$ holds the readings (counts on their own scale, BLI and volume as
logarithms), $\hat z$ the predictions and $\mathbf{J}$ one row per reading:
$h^{\top}$ for counts and $h^{\top} / N$ for log-scale readings. $R$ is the
measurement variance: $\mu + \mu^2 / \phi$ for counts, $\sigma^2$ for
log-scale readings. The off-diagonal terms count the shared population
fluctuation once instead of once per measurement. The score is the normal's
log-density minus the log of each log-scale reading, which converts it back
to the readings' original scale so it is comparable with the separate laws.
If a reading is censored or the joint covariance is degenerate, each reading
falls back to its own law (section 7). Marker fractions are always scored
separately.

### 8.4 Between-well spread

Wells differ in seeding and handling, so one well can sit high or low at
every time point. The `overdispersion` parameter cannot absorb this because it
is independent between time points. With `sigma_extrinsic` $= \sigma_e > 0$,
each replicate's trajectory is scaled by a factor $s$ with
$\log s \sim \mathcal{N}(-\sigma_e^2 / 2,\ \sigma_e^2)$, so that
$\mathbb{E}[s] = 1$. The mean scales by $s$, and so does the covariance,
because scaling the starting population of a linear process scales its
variance by $s$, not $s^2$. This is exact without crowding and an
approximation with it. The factor is integrated out with $K = 9$
Gauss–Hermite nodes $x_k$ and weights $w_k$:

$$
L_r = \int L_r(s)\, p(\log s)\, d\log s
\approx \frac{1}{\sqrt{\pi}} \sum_{k=1}^{K} w_k\,
  L_r\Bigl(e^{\sqrt{2}\, \sigma_e x_k - \sigma_e^2 / 2}\Bigr).
$$

Without it, between-well spread is attributed to division and death and
inflates their sum.

### 8.5 The total

$$
\ell(\theta) = \sum_{\text{replicates } r} \log L_r(\theta),
$$

where $L_r$ is the product of that replicate's per-time-point terms
(section 8.3), integrated over its well factor when one is set (section 8.4).
A parameter vector outside the model's support (a negative rate,
$EC_{50} \le 0$, a birth $E_{\max} > 1$) or a failed solve gives
$\ell = -\infty$. The number of data points $n$ used by AIC and BIC counts
only scored readings, so it excludes anchors.

## 9. Estimation

### 9.1 Maximum likelihood (MLE and MAP)

The estimate maximises $\ell(\theta)$, or $\ell(\theta) + \log \pi(\theta)$
when priors are given (MAP). The search runs on $\log \theta$ for parameters
with a positive lower bound, and on the natural scale for the $E_{\max}$
terms, whose lower bound is 0 ("no effect"). L-BFGS-B is the default
optimiser; Nelder–Mead and differential evolution are available. The first
start is at the prior medians (or the middle of the bounds on the search
scale), and
further starts are drawn uniformly within the bounds on the search scale; the
best result is kept. Standard errors come from a finite-difference Hessian
$\mathbf{H}$ of the objective at $\hat\theta$, on the natural scale:

$$
\mathrm{SE}_j = \sqrt{[\mathbf{H}^{-1}]_{jj}},
\qquad
\mathrm{AIC} = 2k - 2\ell(\hat\theta),
\qquad
\mathrm{BIC} = k \ln n - 2\ell(\hat\theta),
$$

with $k$ free parameters. These are local: unreliable on flat ridges or at a
bound. Compare AIC only between fits of the same data with the same forward
mode, because the moment-mode, deterministic, Kalman and particle likelihoods
are different approximations.

### 9.2 MCMC

The target is $\log \pi(\theta) + \ell(\theta)$, sampled with emcee's
affine-invariant ensemble sampler. At least $2k + 2$ walkers start from prior
draws, or from a small ball around a given point. A parameter needs a prior to
be sampled. The defaults are weakly informative: lognormal for rates and
potencies, half-normal for $E_{\max}$ terms that may be zero, and uniform on
$[0, 1]$ for the birth $E_{\max}$. Convergence is checked with rank-normalised
split $\hat R$ and effective sample size (Vehtari et al., 2021). Ensemble
walkers are not independent chains.

### 9.3 Hierarchical model

Series are pooled by group $i$. Shared parameters $\theta_s$ are common to all
groups, and each random-effect parameter $p$ varies by group around a
population median $\mu_p$:

$$
\theta_{i,p} = \mu_p\, e^{\tau_p z_{i,p}},
\qquad z_{i,p} \sim \mathcal{N}(0, 1),
\qquad \mu_p \sim \pi_p,
\qquad \tau_p \sim \text{Half-Cauchy}(0, 1),
$$

which is $\theta_{i,p} \sim \mathrm{LogNormal}(\log \mu_p, \tau_p)$ written in
non-centred form. The posterior is

$$
p(\theta_s, \mu, \tau, z \mid y) \propto
\pi(\theta_s) \prod_p \pi_p(\mu_p)\, \mathrm{HC}(\tau_p)
\prod_{i,p} \varphi(z_{i,p})
\prod_i L_i(\theta_s, \theta_i),
$$

sampled with emcee on $\log \tau_p$ (with its Jacobian), and truncated at
$\lvert z \rvert \le 8$ and $\lvert \tau_p z \rvert \le 5$. This is a
partial-pooling model, not a mixed-effects model with covariates.

### 9.4 Filtering

**Extended Kalman filter.** Between readings, the mean and covariance are
propagated with the linear noise approximation (section 5.3). Each reading
$z$ with prediction $\hat z$, row $h$ and measurement variance $R$ then
updates them:

$$
S = h^{\top} \Sigma\, h + R,
\qquad
K = \Sigma\, h / S,
\qquad
\mu \leftarrow \max\bigl(\mu + K (z - \hat z),\, 0\bigr),
\qquad
\Sigma \leftarrow (I - K h^{\top})\, \Sigma\, (I - K h^{\top})^{\top} + K R K^{\top},
$$

and adds $-\tfrac{1}{2}\bigl[\log(2\pi S) + (z - \hat z)^2 / S\bigr]$ to the
log-likelihood. This is related to the batch likelihood of section 8 but is a
different objective. Marker fractions and readings below a detection limit
cannot enter it.

**Particle filter.** $n$ particles (500 by default; 200 inside particle MCMC)
are simulated with the
Gillespie method between readings. At reading $k$ each particle's weight is
multiplied by $p(y_k \mid x_k^{(i)})$, and the log-likelihood estimate is

$$
\log \hat p(y_{1:T} \mid \theta) = \sum_{k} \log \sum_{i} \bar w_{k-1}^{(i)}\,
  p\bigl(y_k \mid x_k^{(i)}\bigr),
$$

with $\bar w$ the normalised weights. Particles are resampled systematically
when the effective sample size $1 / \sum_i (\bar w^{(i)})^2$ falls below half
of $n$.

**Particle MCMC.** Metropolis–Hastings with the particle filter's estimate in
place of the likelihood. Because that estimate is unbiased, the chain still
targets the exact posterior. Strictly positive parameters move by
$\log \theta'_j = \log \theta_j + 0.01\, \varepsilon$, and parameters whose
prior allows 0 move on the natural scale, reflected at 0. A proposal is
accepted with probability

$$
\min\Bigl\lbrace 1,\
  \frac{\hat p(y \mid \theta')\, \pi(\theta')}{\hat p(y \mid \theta)\, \pi(\theta)}
  \prod_{j \in \text{log}} \frac{\theta'_j}{\theta_j} \Bigr\rbrace,
$$

where the product corrects for the log-scale proposal. The proposal and the
filter draw from independent random streams.

### 9.5 Clone sizes

Clone sizes identify death without relying on variance. For one founding cell
under a linear birth–death process with constant rates and $r = b - d$
(Kendall, 1948):

$$
P(Z_t = 0) = \alpha,
\qquad
P(Z_t = n) = (1 - \alpha)(1 - \beta)\, \beta^{\,n-1} \quad (n \ge 1),
$$

$$
\alpha = \frac{d\,(e^{r t} - 1)}{b\, e^{r t} - d},
\qquad
\beta = \frac{b\,(e^{r t} - 1)}{b\, e^{r t} - d} .
$$

The extinction fraction $\alpha$ tends to $d / b$ as $t$ grows (when $b > d$),
and the two observables invert in closed form:

$$
r = \frac{1}{t} \log \frac{1 - \alpha}{1 - \beta},
\qquad
b = \frac{r\, \beta}{(1 - \beta)(e^{r t} - 1)},
\qquad
d = b - r .
$$

For example, $b = 0.04$, $d = 0.01$ and $t = 72$ h give $\alpha = 0.228$ and
$\beta = 0.911$, which invert back to the same rates. When a surviving clone is
detected only with probability $q$, a zero means "extinct or missed":
$P(\text{observe } 0) = \alpha + (1 - \alpha)(1 - q)$ and
$P(\text{observe } n) = (1 - \alpha)\, q\, (1 - \beta)\, \beta^{\,n-1}$.
The rates are fitted by maximum likelihood over the clone-size histogram. Read
counts from barcode sequencing are not cell counts; supply $q$ from a spike-in
or rarefaction analysis, or treat the result as a bound.

### 9.6 Transfer from in vitro to in vivo

An in vitro fit becomes in vivo priors. For each transferred parameter, a
lognormal is fitted to the in vitro estimate: from MCMC draws,
$\hat\sigma = \mathrm{sd}(\log \theta)$ and median
$m = \exp(\mathrm{mean}(\log \theta))$; from an MLE,
$\hat\sigma \approx \mathrm{SE} / \hat\theta$ (10% when there is no SE) and
$m = \hat\theta$. The prior is

$$
\theta^{\text{vivo}} \sim \mathrm{LogNormal}\bigl(\log(c\, m),\ \max(\hat\sigma / \lambda,\ 0.05)\bigr),
$$

where $\lambda \in (0, 1]$ is the shrinkage (smaller is wider) and $c$ a scale
factor. Only the dimensionless `emax_death` and `hill_death` transfer with
$c = 1$. Concentrations such as $EC_{50}$ and rates such as $b_0$ need an
explicit $c$, such as a free-fraction ratio or a measured growth-rate ratio,
because in vitro and in vivo values differ. Estimates with $\hat\sigma > 2$ are
not transferred.

## 10. Identifiability, design and model checks

**Identifiability.** At a parameter value, the local information on the log
scale is

$$
\mathbf{F} = -\frac{\partial^2 \ell}{\partial \log\theta\, \partial \log\theta^{\top}},
$$

with negative eigenvalues set to 0. Writing
$\mathbf{F} = \mathbf{V} \Lambda \mathbf{V}^{\top}$ and keeping the directions
with $\sqrt{\lambda_k} > 10^{-6} \sqrt{\lambda_1}$, parameter $j$'s score is

$$
s_j = \Bigl(\sum_{k \text{ kept}} V_{jk}^2\Bigr)^{1/2} \in [0, 1],
$$

the fraction of its direction that the data can see. A score near 0 means
the data do not constrain the parameter, and its posterior will follow its
prior. This is local (practical) identifiability, not structural.

**Experimental design.** The same calculation on the noise-free data a
planned design would produce on average, with the noise parameters held fixed,
gives each parameter a status:

| Score | Status |
|---|---|
| $s_j < 0.1$ | Not constrained |
| $0.1 \le s_j < 0.99$ | Confounded: constrained only together with others, as $b$ and $d$ are under the deterministic likelihood |
| $s_j \ge 0.99$ | Constrained, with relative standard error $\approx \sqrt{[\mathbf{F}^{+}]_{jj}}$ |

**Posterior predictive checks.** Parameters drawn from the posterior simulate
new data through the same observation laws, including process variance in
moment mode, at the same scored points. A Bayesian p-value near 0 or 1 for a
summary means the model cannot reproduce that feature of the data.

## 11. Optional signalling coupling

A toy two-node pathway (labelled MAPK and AKT) can scale chosen rates. It is a
scaffold for testing coupling, not a calibrated pathway model, and works with
the deterministic equations only. Each node relaxes towards a set point:

$$
\frac{dy_i}{dt} = \delta\, \bigl(T_i(C) - y_i\bigr),
\qquad
T_i(C) = \mathrm{clip}\bigl(b_i + s\, k_i C,\ 0,\ 1\bigr),
$$

with $s = -1$ for an inhibitor. Pathway activity
$a = \max(0,\ w_1 y_1 + w_2 y_2)$ multiplies the targeted rates by
$\max\bigl(0,\ 1 + m_{\max}\, e(a)\bigr)$, where $e$ is a Hill or logistic
curve in $[0, 1]$. High activity boosts the rates, and an inhibitor removes
that boost; it cannot push a rate below its baseline.

## 12. Assumptions behind every result

- Cells act independently in a well-mixed population, with exponential
  waiting times, a single scalar exposure and no spatial structure.
- Measurement types are independent given the cell numbers, except where
  they are scored jointly (section 8.3).
- The starting mix, the calibrations ($\alpha$, $\beta$, PK parameters) and
  fixed noise settings are choices that shape the result. State them when you
  report it.
- Moment mode is a Gaussian, large-population approximation (section 5.3) and
  is biased for very noisy counts ($\phi$ below about 100).
- Standard errors and identifiability scores are local. A good mean
  trajectory does not show that MCMC has mixed.
- Recovering parameters from data simulated by the same model is a best case,
  not evidence about real data.

The [Scientific assumptions](scientific-assumptions.md) page and the
workbench handbook's "Scientific limits" section expand on these.

## 13. Where each equation lives

| Section | Code |
|---|---|
| 2, topologies | `umimic/dynamics/states.py` (`ModelTopology`) |
| 3, exposure | `umimic/pk/compartment.py`, `umimic/pk/dosing.py`, `umimic/pk/exposure.py` |
| 4, rate laws | `umimic/dynamics/rates.py` (`EmaxHill`, `RateSet`) |
| 4.3, parameter names | `umimic/inference/likelihood.py` (`build_rate_set`, `PARAMETER_SETS`) |
| 5.1, deterministic equations | `umimic/dynamics/ode_system.py` (`CellDynamicsODE`) |
| 5.2, stochastic simulation | `umimic/dynamics/gillespie.py`, `umimic/dynamics/tau_leaping.py` |
| 5.3, moments | `umimic/dynamics/moment_equations.py` (`MomentODE`) |
| 6, growth summaries | `RateSet.asymptotic_growth_rate`, `gr_value`, `finite_horizon_gr` |
| 7, observations | `umimic/observations/` (`cell_counts.py`, `tumor_volume.py`, `bli.py`, `biomarkers.py`); luciferin and attenuation in `umimic/pk/luciferin.py` |
| 8, likelihood | `umimic/inference/likelihood.py` (`ModelLikelihood`); joint scoring in `umimic/observations/multimodal.py` |
| 9.1, MLE | `umimic/inference/mle.py` |
| 9.2, MCMC and priors | `umimic/inference/mcmc.py`, `umimic/inference/priors.py`, `umimic/inference/diagnostics.py` |
| 9.3, hierarchical | `umimic/inference/hierarchical.py` |
| 9.4, filtering | `umimic/inference/kalman.py`, `umimic/inference/smc.py` |
| 9.5, clone sizes | `umimic/inference/clonal.py` |
| 9.6, transfer | `umimic/pipeline/transfer.py` |
| 10, identifiability and design | `umimic/inference/identifiability.py`, `umimic/inference/design.py` |
| 11, signalling | `umimic/signaling/models.py`, coupling in `umimic/pipeline/experiment.py` |

## References

- Andrieu, Doucet and Holenstein (2010). Particle Markov chain Monte Carlo
  methods. *J. R. Stat. Soc. B* 72, 269–342.
- Cao, Gillespie and Petzold (2006). Efficient step size selection for the
  tau-leaping simulation method. *J. Chem. Phys.* 124, 044109.
- Foreman-Mackey, Hogg, Lang and Goodman (2013). emcee: the MCMC hammer.
  *PASP* 125, 306–312.
- Gillespie (1977). Exact stochastic simulation of coupled chemical reactions.
  *J. Phys. Chem.* 81, 2340–2361.
- Hafner, Niepel, Chung and Sorger (2016). Growth rate inhibition metrics
  correct for confounders in measuring sensitivity to cancer drugs.
  *Nat. Methods* 13, 521–527.
- Kendall (1948). On the generalized "birth-and-death" process.
  *Ann. Math. Stat.* 19, 1–15.
- Van Kampen (2007). *Stochastic Processes in Physics and Chemistry*, 3rd ed.
  Elsevier. (Linear noise approximation.)
- Vehtari, Gelman, Simpson, Carpenter and Bürkner (2021). Rank-normalization,
  folding, and localization: an improved $\hat R$ for assessing convergence of
  MCMC. *Bayesian Anal.* 16, 667–718.
- Voliotis, Thomas, Grima and Bowsher (2016). Stochastic simulation of
  biomolecular networks in dynamic environments. *PLoS Comput. Biol.* 12,
  e1004923. (Extrande.)
