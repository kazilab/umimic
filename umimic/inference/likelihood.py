"""Central log-likelihood computation for U-MIMIC.

This module is the critical bridge between the dynamics model and inference.
It takes a parameter vector, builds the forward model, runs it, and evaluates
the observation log-likelihood at each data point.

Multimodal contract
-------------------
Every modality present in the data *and* configured in the observation model
contributes to the likelihood. Modalities are combined under conditional
independence given the latent state, so the total log-likelihood is the sum
over modalities and time points of the individual terms. Missing observations
(NaN) are skipped rather than imputed.

Replicate contract
------------------
Replicates sharing a concentration share one forward solve, performed on the
*union* of their observation times. Each replicate is then evaluated at its own
exact times via the solver's dense interpolant. Times are never snapped to the
nearest available grid point.

Initial-condition contract
--------------------------
The initial count is distributed across states by `initial_fractions`. The
default places every cell in P. That is a *structural assumption*, not a
neutral one: a real culture carries a P/Q mix, and starting from pure P biases
the estimated transition rates and the early growth curve. Pass
``initial_fractions="stable"`` to start from the relaxed state distribution,
or give explicit fractions.

By default the first observation of the anchoring modality is used to set the
initial condition and is therefore *excluded* from the likelihood, since using
a value both to condition the model and to score it double-counts information.
Set ``condition_on_first=False`` to instead treat the initial state as a fixed
parameter and score every observation.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from scipy.special import logsumexp

from umimic.data.schemas import TimeSeriesData
from umimic.dynamics.moment_equations import MomentODE
from umimic.dynamics.ode_system import CellDynamicsODE
from umimic.dynamics.rates import EmaxHill, RateSet
from umimic.dynamics.states import CellType, ModelTopology
from umimic.observations.base import ObservationModel, TopologyAwareObservation
from umimic.observations.cell_counts import CellCountObservation
from umimic.observations.multimodal import MultimodalObservation

logger = logging.getLogger(__name__)


# Default parameter names and their positions in the theta vector.
# These names map to RateSet fields as documented in RateSet's docstring.
#
# NOTE: this set contains only *death* modulation, so it describes a purely
# cytotoxic drug. Fitting it to cytostatic data will attribute the growth
# reduction to death, and it cannot distinguish the two mechanisms no matter
# how good the data are. Use MECHANISM_PARAM_NAMES for that.
DEFAULT_PARAM_NAMES = [
    "b0",          # RateSet.birth_base
    "d0_P",        # RateSet.death_base[CellType.P]
    "emax_death",  # RateSet.death_modulation EmaxHill.emax
    "ec50_death",  # RateSet.death_modulation EmaxHill.ec50
    "hill_death",  # RateSet.death_modulation EmaxHill.hill
    "u_PQ",        # RateSet.transition_base[(P, Q)]
    "u_QP",        # RateSet.transition_base[(Q, P)]
    "overdispersion",  # CellCountObservation.overdispersion
]

# Parameter set that admits both mechanisms. The mean alone confounds them
# (both reduce net growth); the LNA process variance scales with b + d while
# the mean scales with b - d, so in principle the variance breaks the
# degeneracy and this set requires mode="moment".
#
# In practice that route is usually underpowered, and this set should not be
# read as "cytostatic and cytotoxic are separately identifiable". At the
# package defaults (phi=10, mu~4000) the process variance is ~1.5% of the
# total observation variance, a doubling of turnover is worth ~1 nat across a
# whole dataset, and measured mechanism discrimination on mean-matched
# synthetic data is at chance (5/11 across seeds and restarts). Check the
# variance share for your own design before trusting a split.
# See umimic/inference/SCIENTIFIC_ASSUMPTIONS.md section 2a.
MECHANISM_PARAM_NAMES = [
    "b0",
    "d0_P",
    "emax_death",
    "ec50_death",
    "hill_death",
    "emax_birth",  # RateSet.birth_modulation EmaxHill.emax (cytostatic)
    "ec50_birth",  # RateSet.birth_modulation EmaxHill.ec50
    "hill_birth",  # RateSet.birth_modulation EmaxHill.hill
    "u_PQ",
    "u_QP",
    "overdispersion",
]

# Parameter set for a model carrying a resistant compartment. Without the R
# entries, R inherits the shared baseline birth rate and has no death rate at
# all -- an immortal, fully fit, drug-insensitive clone that will absorb the
# whole trajectory. The simulation side has expressed resistance properly
# since 0.0.4; this is the matching inference side.
RESISTANCE_PARAM_NAMES = [
    "b0",
    "b0_R",         # RateSet.birth_base_by_state[R]
    "d0_P",
    "d0_R",         # RateSet.death_base[R]
    "emax_death",
    "ec50_death",
    "hill_death",
    "emax_death_R",  # residual drug sensitivity of R (0 = fully resistant)
    "u_PQ",
    "u_QP",
    "u_PR",
    "overdispersion",
]

# Adds the persister route Q -> R, including its drug-induced component, and
# quiescent drug sensitivity.
#
# `induced_QR` here carries efficacy only: its potency defaults to the
# cytotoxic ec50_death/hill_death, so this set constrains the drug to drive
# plasticity and killing at the same concentration. Add "ec50_induction" and
# "hill_induction" to free that, at the cost of two more parameters that
# multi-concentration data must be rich enough to identify.
PERSISTER_PARAM_NAMES = [
    "b0",
    "b0_Q",          # slow-cycling persisters divide at their own rate
    "b0_R",
    "d0_P",
    "d0_Q",
    "d0_R",
    "emax_death",
    "ec50_death",
    "hill_death",
    "emax_death_Q",  # quiescent drug sensitivity (0 = refractory)
    "emax_death_R",
    "u_PQ",
    "u_QP",
    "u_QR",          # baseline persister -> resistant
    "induced_QR",    # drug-induced persister -> resistant
    "overdispersion",
]

PARAMETER_SETS = {
    "default": DEFAULT_PARAM_NAMES,
    "mechanism": MECHANISM_PARAM_NAMES,
    "resistance": RESISTANCE_PARAM_NAMES,
    "persister": PERSISTER_PARAM_NAMES,
}

#: States whose baseline death rate has a documented fallback, so an absent
#: parameter is a deliberate simplification rather than a silent gap.
#: Q falls back to ``d0_P * 0.5``; every other state would get zero.
_DEATH_FALLBACK_STATES = (CellType.P, CellType.Q)

#: Parameter names consumed by the observation models rather than the rates.
#: Kept here so every backend forwards the same set; a name missing from it is
#: silently ignored by the observation model, which makes it unidentifiable.
OBSERVATION_PARAM_NAMES = frozenset(
    {
        "overdispersion",
        "sigma_log_bli",
        "sigma_v",
        "biomarker_precision",
        # Between-replicate (extrinsic) spread: see ModelLikelihood's
        # `sigma_extrinsic` handling. Distinct from `overdispersion`, which is
        # independent across time points within a replicate.
        "sigma_extrinsic",
    }
)

#: Parameter names :func:`build_rate_set` reads. Anything outside this set and
#: OBSERVATION_PARAM_NAMES does not reach the forward model at all.
RATE_PARAM_NAMES = frozenset(
    {
        "b0",
        "b0_Q",
        "b0_R",
        "d0_P",
        "d0_Q",
        "d0_R",
        "emax_death",
        "ec50_death",
        "hill_death",
        "emax_death_Q",
        "emax_death_R",
        "emax_birth",
        "ec50_birth",
        "hill_birth",
        "u_PQ",
        "u_QP",
        "u_PR",
        "u_QR",
        "u_RQ",
        "induced_PQ",
        "induced_QR",
        "ec50_induction",
        "hill_induction",
        "clearance_rate",
    }
)

#: Every name any backend knows how to use.
KNOWN_PARAM_NAMES = RATE_PARAM_NAMES | OBSERVATION_PARAM_NAMES

def _scalarize_params(obs_params: dict, position: int) -> dict:
    """Index array-valued observation parameters down to one time point.

    ``_modality_params`` may attach per-time-point arrays (paired tumour
    volumes for BLI attenuation) aligned to a modality's masked points. The
    point-wise fused path needs the scalar at one position.
    """
    if position < 0:
        return dict(obs_params)
    out = {}
    for key, value in obs_params.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and position < arr.shape[0]:
            out[key] = float(arr[position])
        else:
            out[key] = value
    return out


# Which observable each modality measures, for process-variance projection.
MODALITY_OBSERVABLE = {
    "cell_counts": "viable",
    "bli": "viable",
    "volume": "viable",
    "tumor_volume": "viable",
}


def resolve_initial_fractions(
    topology: ModelTopology,
    data: TimeSeriesData,
    spec,
    rate_set: RateSet | None = None,
) -> np.ndarray:
    """Fractions of the initial count assigned to each state.

    Module-level so every backend distributes the initial count the same way.
    `spec` is None/"proliferating" (all cells in P), "stable" (the relaxed
    state distribution at this series' concentration), or explicit fractions
    as a dict or array. A per-series ``metadata["initial_fractions"]``
    overrides it.
    """
    spec = data.metadata.get("initial_fractions", spec)

    if spec is None or (isinstance(spec, str) and spec == "proliferating"):
        fractions = np.zeros(topology.n_states)
        fractions[topology.state_index(CellType.P)] = 1.0
        return fractions

    if isinstance(spec, str):
        if spec != "stable":
            raise ValueError(
                f"Unknown initial_fractions {spec!r}; expected "
                "'proliferating', 'stable', or an array/dict of fractions."
            )
        if rate_set is None:
            # No rates available yet (e.g. a structural query); fall back.
            fractions = np.zeros(topology.n_states)
            fractions[topology.state_index(CellType.P)] = 1.0
            return fractions
        conc = data.concentration if data.concentration is not None else 0.0
        return rate_set.stable_state_fractions(conc, topology)

    if isinstance(spec, dict):
        fractions = np.zeros(topology.n_states)
        for ct, value in spec.items():
            cell_type = CellType[ct] if isinstance(ct, str) else ct
            fractions[topology.state_index(cell_type)] = float(value)
    else:
        fractions = np.asarray(spec, dtype=float)
        if fractions.shape != (topology.n_states,):
            raise ValueError(
                f"initial_fractions must have one entry per active state "
                f"({topology.n_states}), got {fractions.shape}."
            )

    if np.any(fractions < 0):
        raise ValueError("initial_fractions must be non-negative.")
    total = fractions.sum()
    if total <= 0:
        raise ValueError("initial_fractions must sum to a positive value.")
    return fractions / total


def missing_state_params(topology: ModelTopology, names: set[str]) -> list[str]:
    """Active states whose rates the parameter vector does not describe.

    Checked by the role a state plays in the topology: a dividing state needs
    its own ``b0_<X>`` or it inherits the shared baseline, and a dying state
    needs ``d0_<X>`` unless it has a documented fallback. Module-level so every
    backend applies the same test.
    """
    problems = []
    for cell_type in topology.active_states:
        if cell_type is CellType.P or cell_type is CellType.A:
            continue
        missing = []
        if (
            cell_type in topology.division_states
            and f"b0_{cell_type.name}" not in names
        ):
            missing.append(f"b0_{cell_type.name}")
        if (
            cell_type in topology.death_states
            and cell_type not in _DEATH_FALLBACK_STATES
            and f"d0_{cell_type.name}" not in names
        ):
            missing.append(f"d0_{cell_type.name}")
        if missing:
            problems.append(f"{cell_type.name} needs {missing}")
    return problems


def build_rate_set(params: dict[str, float]) -> RateSet:
    """Construct a RateSet from parameter values.

    Module-level and shared by every backend on purpose. A second, partial
    copy of this mapping is the worst kind of inference bug: the sampler wires
    up parameter names the dynamics never read, so those parameters return
    their prior while the ones that *are* wired absorb the misfit. The fit
    runs and converges, which is what makes it dangerous.
    """
    death_mod = None
    if "emax_death" in params and params.get("emax_death", 0) > 0:
        death_mod = EmaxHill(
            emax=params.get("emax_death", 0.05),
            ec50=params.get("ec50_death", 1.0),
            hill=params.get("hill_death", 1.5),
        )

    birth_mod = None
    if "emax_birth" in params and params.get("emax_birth", 0) > 0:
        birth_mod = EmaxHill(
            emax=params.get("emax_birth", 0.5),
            ec50=params.get("ec50_birth", 1.0),
            hill=params.get("hill_birth", 1.5),
        )

    death_base = {CellType.P: params.get("d0_P", 0.01)}
    if "d0_Q" in params:
        death_base[CellType.Q] = params["d0_Q"]
    else:
        death_base[CellType.Q] = params.get("d0_P", 0.01) * 0.5
    if "d0_R" in params:
        death_base[CellType.R] = params["d0_R"]

    death_modulation = {}
    if death_mod is not None:
        death_modulation[CellType.P] = death_mod

    # Per-state drug sensitivity. These share the P potency (ec50, hill)
    # and differ in efficacy, which is the usual way partial resistance
    # and quiescent refractoriness are parameterised.
    ec50 = params.get("ec50_death", 1.0)
    hill = params.get("hill_death", 1.5)
    for name, cell_type in (("emax_death_Q", CellType.Q),
                            ("emax_death_R", CellType.R)):
        if params.get(name, 0.0) > 0:
            death_modulation[cell_type] = EmaxHill(
                emax=params[name], ec50=ec50, hill=hill
            )

    # Per-state division. Without an explicit entry a state inherits the
    # shared birth_base, which would hand a resistant clone the sensitive
    # population's growth rate.
    birth_base_by_state = {}
    birth_modulation_by_state = {}
    for name, cell_type in (("b0_Q", CellType.Q), ("b0_R", CellType.R)):
        if name in params:
            birth_base_by_state[cell_type] = params[name]
            # Explicit None: these states do not inherit P's cytostatic
            # response; their drug effect is their own emax_death_* term.
            birth_modulation_by_state[cell_type] = None

    trans_base = {}
    for name, edge in (
        ("u_PQ", (CellType.P, CellType.Q)),
        ("u_QP", (CellType.Q, CellType.P)),
        ("u_PR", (CellType.P, CellType.R)),
        ("u_QR", (CellType.Q, CellType.R)),
        ("u_RQ", (CellType.R, CellType.Q)),
    ):
        if name in params:
            trans_base[edge] = params[name]

    # Drug-induced transitions: additive, so a route absent without
    # treatment can be estimated as appearing under it.
    #
    # Potency defaults to the cytotoxic potency, which ties the concentration
    # at which the drug drives plasticity to the one at which it kills. That
    # is a parsimony choice, not biology: the two are separate pharmacology,
    # and a fitted induced_* term read as independent potency would be wrong.
    # Supply ec50_induction / hill_induction to estimate them separately.
    ec50_ind = params.get("ec50_induction", ec50)
    hill_ind = params.get("hill_induction", hill)
    trans_induction = {}
    for name, edge in (
        ("induced_PQ", (CellType.P, CellType.Q)),
        ("induced_QR", (CellType.Q, CellType.R)),
    ):
        if params.get(name, 0.0) > 0:
            trans_induction[edge] = EmaxHill(
                emax=params[name], ec50=ec50_ind, hill=hill_ind
            )

    rate_set = RateSet(
        birth_base=params.get("b0", 0.04),
        birth_modulation=birth_mod,
        birth_base_by_state=birth_base_by_state,
        birth_modulation_by_state=birth_modulation_by_state,
        death_base=death_base,
        death_modulation=death_modulation,
        transition_base=trans_base,
        transition_induction=trans_induction,
    )
    if "clearance_rate" in params:
        rate_set.clearance_rate = params["clearance_rate"]
    return rate_set


class ModelLikelihood:
    """Central likelihood function for parameter estimation.

    Given a parameter vector theta, this class:
    1. Constructs a RateSet from the parameters
    2. Runs the forward model (moment ODE or deterministic ODE)
    3. Evaluates the observation log-likelihood for every available modality
    4. Returns the total log-likelihood

    This serves as the objective function for MLE and the target for MCMC.
    """

    def __init__(
        self,
        topology: ModelTopology,
        data: TimeSeriesData | list[TimeSeriesData],
        param_names: list[str] | None = None,
        mode: str = "moment",
        observation_model: ObservationModel | MultimodalObservation | None = None,
        condition_on_first: bool = True,
        anchor_modality: str = "cell_counts",
        initial_fractions=None,
        n_quadrature: int = 9,
    ):
        """
        Args:
            topology: Model topology (which states/transitions).
            data: Observed data (single series or list for multiple replicates).
            param_names: Names of parameters in the theta vector.
            mode: Forward model mode - 'moment' (LNA) or 'ode' (deterministic).
            observation_model: Observation model. May be a single
                ObservationModel or a MultimodalObservation combining several.
                Default: CellCountObservation on the given topology.
            condition_on_first: Use the first anchor observation as the initial
                condition and exclude it from the likelihood.
            anchor_modality: Modality used to set the initial condition.
            initial_fractions: How the initial count is split across states.
                None or "proliferating" puts every cell in P (the historical
                behaviour, and a structural assumption -- see the class
                docstring); "stable" uses the relaxed state distribution;
                a dict or array gives explicit fractions. A per-series
                override may be supplied as metadata["initial_fractions"].
        """
        self.topology = topology
        self.data_list = data if isinstance(data, list) else [data]
        self.param_names = param_names or DEFAULT_PARAM_NAMES
        self.mode = mode
        self.condition_on_first = condition_on_first
        self.anchor_modality = anchor_modality
        self.initial_fractions = initial_fractions
        # Gauss-Hermite nodes for integrating out `sigma_extrinsic`. Only
        # used when that parameter is present and positive.
        if n_quadrature < 3 or n_quadrature % 2 == 0:
            raise ValueError(
                f"n_quadrature must be an odd integer >= 3, got {n_quadrature}."
            )
        self.n_quadrature = n_quadrature
        self._n_evals = 0

        if observation_model is None:
            observation_model = CellCountObservation(
                overdispersion=10.0, topology=topology
            )
        self.obs_model = observation_model

        # Resolve modality -> model, so every configured modality is used.
        if isinstance(observation_model, MultimodalObservation):
            self._modality_models = dict(observation_model.models)
        else:
            name = getattr(observation_model, "modality_name", None) or "cell_counts"
            self._modality_models = {name: observation_model}

        # Make sure every observation model maps states via this topology.
        for model in self._modality_models.values():
            if isinstance(model, TopologyAwareObservation):
                model.set_topology(topology)

        # Pre-group data by concentration to avoid redundant ODE solves.
        # Replicates at the same concentration share identical dynamics.
        self._conc_groups: dict[float, list[TimeSeriesData]] = defaultdict(list)
        for d in self.data_list:
            conc = d.concentration if d.concentration is not None else 0.0
            self._conc_groups[conc].append(d)

        # Union of observation times per concentration group, so a shared solve
        # covers every replicate's schedule exactly.
        self._group_times: dict[float, np.ndarray] = {}
        for conc, series in self._conc_groups.items():
            union = np.unique(np.concatenate([s.times for s in series]))
            self._group_times[conc] = union

        self._validate_param_names()
        self._check_states_are_parameterised()

        self._active_modalities = sorted(
            {
                m
                for s in self.data_list
                for m in s.modalities
                if m in self._modality_models
            }
        )
        if not self._active_modalities:
            raise ValueError(
                "No configured modality is present in the data. Data has "
                f"{sorted({m for s in self.data_list for m in s.modalities})}, "
                f"observation model provides {sorted(self._modality_models)}."
            )
        logger.debug("Likelihood active modalities: %s", self._active_modalities)

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------

    def _validate_param_names(self) -> None:
        """Reject parameters the forward model would never read.

        A name outside the known set is optimised or sampled, priced by any
        prior that happens to cover it, and then dropped on the way into
        :func:`build_rate_set` and the observation models -- so its posterior
        is its prior and the wired parameters absorb the misfit. That failure
        is invisible in a fit curve, so it is an error rather than a warning.
        ParticleMCMC applies the same check; keep them aligned.
        """
        # Observation models may declare their own parameter keys, so that two
        # channels of the same kind (viable counts and dead counts) can carry
        # separate noise parameters instead of competing for one. Collect them
        # from the models actually in use rather than hard-coding the names.
        known = set(KNOWN_PARAM_NAMES)
        for model in self._modality_models.values():
            key = getattr(model, "overdispersion_key", None)
            if key:
                known.add(key)

        unknown = [n for n in self.param_names if n not in known]
        if unknown:
            raise ValueError(
                f"Parameter(s) {unknown} are not read by the forward model or "
                "any observation model, so estimating them would return the "
                f"prior. Known parameters: {sorted(known)}."
            )

    def _missing_state_params(self, names: set[str]) -> list[str]:
        """Active states whose rates the parameter vector does not describe."""
        return missing_state_params(self.topology, names)

    def _check_states_are_parameterised(self) -> None:
        """Every active state must have its own rate parameters.

        A state present in the topology but absent from the parameter vector
        does not simply sit still: it inherits the shared baseline birth rate
        and receives no death rate, so a resistant compartment becomes an
        immortal, fully fit, drug-insensitive clone that absorbs the whole
        trajectory. That fit runs and converges, which is what makes it
        dangerous.
        """
        names = set(self.param_names)
        problems = self._missing_state_params(names)

        if (
            CellType.Q in self.topology.death_states
            and "d0_Q" not in names
            and "d0_P" in names
        ):
            # The fallback is a modelling choice, not a neutral default: it
            # asserts quiescent cells die at exactly half the proliferating
            # rate, and that ratio is then fixed rather than estimated. Say so
            # once, because nothing downstream distinguishes an estimated
            # d0_Q from an imputed one.
            logger.warning(
                "Q is an active death state but d0_Q is not being estimated; "
                "it is fixed at 0.5 * d0_P. That ratio is an assumption, not "
                "a fitted quantity -- add 'd0_Q' to param_names to estimate "
                "quiescent death separately."
            )

        if problems:
            candidates = [
                label
                for label, params in PARAMETER_SETS.items()
                if not self._missing_state_params(set(params))
            ]
            raise ValueError(
                "The topology contains states the parameter vector does not "
                f"describe: {'; '.join(problems)}. Such a state inherits the "
                "shared birth rate and has no death rate, making it an "
                "immortal fully-fit clone during inference. Use one of the "
                f"parameter sets {candidates or list(PARAMETER_SETS)} or pass "
                "an explicit param_names covering every active state."
            )

    def theta_to_params(self, theta: np.ndarray) -> dict[str, float]:
        """Convert parameter vector to named dict."""
        return {name: theta[i] for i, name in enumerate(self.param_names)}

    def params_to_theta(self, params: dict[str, float]) -> np.ndarray:
        """Convert named params to vector."""
        return np.array([params[name] for name in self.param_names])

    def _build_rate_set(self, params: dict[str, float]) -> RateSet:
        """Construct a RateSet from parameter values."""
        return build_rate_set(params)

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------

    def _initial_state(
        self, data: TimeSeriesData, rate_set: RateSet | None = None
    ) -> np.ndarray:
        """Initial latent state for a replicate.

        Replicate-specific: each series may start from its own measured count.
        The count is distributed across states by `initial_fractions`; see the
        class docstring for why the default matters.
        """
        n0 = None
        if data.has_modality(self.anchor_modality):
            values = data.observations[self.anchor_modality]
            present = np.flatnonzero(~np.isnan(values))
            if present.size:
                n0 = float(values[present[0]])
        if n0 is None:
            n0 = float(data.metadata.get("initial_cells", 100.0))
        n0 = max(n0, 1.0)

        fractions = self._resolve_initial_fractions(data, rate_set)
        return n0 * fractions

    def _resolve_initial_fractions(
        self, data: TimeSeriesData, rate_set: RateSet | None
    ) -> np.ndarray:
        """Fractions of the initial count assigned to each state."""
        return resolve_initial_fractions(
            self.topology, data, self.initial_fractions, rate_set
        )

    def _solve_forward(
        self,
        rate_set: RateSet,
        conc: float,
        times: np.ndarray,
        mu0: np.ndarray,
    ):
        """Solve the forward model for one concentration on the union grid.

        Returns:
            (means_fn, covs_fn) callables mapping a time array to the mean
            trajectory and (for moment mode) the projected covariances.
        """
        def exposure_fn(t, _c=conc):
            return _c

        t_span = (float(times[0]), float(times[-1]))

        if self.mode == "moment":
            moment_ode = MomentODE(rate_set, self.topology, exposure_fn)
            t_sol, means, covs = moment_ode.solve(mu0, t_span=t_span, t_eval=times)
            return t_sol, means, covs

        ode = CellDynamicsODE(rate_set, self.topology, exposure_fn)
        result = ode.solve(mu0, t_span, times)
        means = np.column_stack(
            [result.populations[ct.name] for ct in self.topology.active_states]
        )
        return result.times, means, None

    # ------------------------------------------------------------------
    # Likelihood evaluation
    # ------------------------------------------------------------------

    def scored_mask(self, data: TimeSeriesData, modality: str) -> np.ndarray:
        """Time points of `modality` that actually enter the likelihood.

        Non-missing observations, minus the anchor point when it is consumed
        by the initial condition. Posterior predictive checks must use the
        same mask: replicating a point the model was *conditioned* on
        reproduces it almost exactly by construction, which flatters the fit
        and skews the Bayesian p-value.
        """
        mask = data.observed_mask(modality)
        if self.condition_on_first and modality == self.anchor_modality:
            present = np.flatnonzero(mask)
            if present.size:
                mask = mask.copy()
                mask[present[0]] = False
        return mask

    def _evaluate_replicate(
        self,
        data: TimeSeriesData,
        params: dict[str, float],
        t_sol: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray | None,
    ) -> float:
        """Log-likelihood for one replicate, marginal over its random effect.

        `sigma_extrinsic` is the between-replicate spread: wells differ in
        seeding density and handling, so one well sits systematically high or
        low across *all* its time points. `overdispersion` cannot represent
        that -- it is independent between time points, so with enough of them
        it averages away, while a well effect never does. Left unmodelled the
        spread has nowhere to go but the mechanistic rates: on BESTDR, where
        seeding differs ~20x between wells, moment-mode fits inflate b0 and
        d0_P several-fold trying to explain it as demographic noise.

        The effect enters as a multiplicative factor on the trajectory. That
        is exact rather than convenient: for a density-independent model the
        dynamics are linear, so scaling the initial population scales the mean
        *and* the LNA covariance by the same factor. It is centred
        (`- sigma^2/2`) so switching it on does not shift the mean, and
        integrated out by Gauss-Hermite quadrature -- the marginal likelihood
        of the replicate, not a plug-in estimate of its effect.
        """
        sigma = float(params.get("sigma_extrinsic", 0.0) or 0.0)
        if not np.isfinite(sigma) or sigma <= 0.0:
            return self._evaluate_replicate_core(data, params, t_sol, means, covs)

        nodes, weights = np.polynomial.hermite.hermgauss(self.n_quadrature)
        # u ~ N(0, sigma^2)  ==>  int f(u) p(u) du = (1/sqrt(pi)) sum w_k f(sqrt(2) sigma x_k)
        log_terms = []
        for x_k, w_k in zip(nodes, weights):
            log_scale = np.sqrt(2.0) * sigma * x_k - 0.5 * sigma**2
            ll_k = self._evaluate_replicate_core(
                data, params, t_sol, means, covs, log_scale=log_scale
            )
            if np.isfinite(ll_k):
                log_terms.append(np.log(w_k) + ll_k)
        if not log_terms:
            return -np.inf
        return float(logsumexp(np.asarray(log_terms)) - 0.5 * np.log(np.pi))

    def _evaluate_replicate_core(
        self,
        data: TimeSeriesData,
        params: dict[str, float],
        t_sol: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray | None,
        log_scale: float = 0.0,
    ) -> float:
        """Log-likelihood for one replicate at a fixed random-effect value."""
        # Map this replicate's times onto the shared union grid. Because the
        # grid is the union of all replicate times, every time is present; we
        # locate it exactly rather than snapping to a neighbour.
        idx = np.searchsorted(t_sol, data.times)
        idx = np.clip(idx, 0, len(t_sol) - 1)
        if not np.allclose(t_sol[idx], data.times, rtol=0, atol=1e-9):
            missing = data.times[~np.isclose(t_sol[idx], data.times, rtol=0, atol=1e-9)]
            raise ValueError(
                "Replicate observation times are not present in the shared "
                f"solution grid (first missing: {missing[0]}). This indicates "
                "the union grid was built from a different series set."
            )

        latent = np.maximum(means[idx], 0.0)
        replicate_covs = covs
        if log_scale != 0.0:
            scale = float(np.exp(log_scale))
            latent = latent * scale
            # Linear dynamics: both the mean and the LNA covariance are
            # proportional to the initial population, so the covariance scales
            # by `scale`, not `scale**2`.
            if replicate_covs is not None:
                replicate_covs = replicate_covs * scale
        covs = replicate_covs

        total_ll = 0.0

        # Modalities that share the latent population must be fused, not
        # multiplied: BLI and volume are both deterministic functions of the
        # same N_viable, so folding the LNA process variance into each marginal
        # separately counts one fluctuation twice and narrows the posterior.
        fused_modalities = self._fusable_modalities(data)
        if covs is not None and len(fused_modalities) >= 2:
            fused_ll, fused_handled = self._fused_replicate_ll(
                data, params, idx, latent, covs, fused_modalities
            )
            if not np.isfinite(fused_ll):
                return -np.inf
            total_ll += fused_ll
        else:
            fused_handled = frozenset()

        for modality in self._active_modalities:
            if modality in fused_handled:
                continue
            if not data.has_modality(modality):
                continue

            model = self._modality_models[modality]
            values = data.observations[modality]
            mask = self.scored_mask(data, modality)

            if not np.any(mask):
                continue

            obs_params = self._modality_params(modality, params, data, idx, mask)

            # Project process covariance onto this modality's observable:
            # Var(H x) = H Sigma H^T, not the sum of all covariance entries.
            process_vars = None
            if covs is not None and isinstance(model, TopologyAwareObservation):
                observable = MODALITY_OBSERVABLE.get(modality)
                if observable is not None:
                    process_vars = np.asarray(
                        model.project_variance(observable, covs[idx][mask])
                    )

            ll = model.log_likelihood_batch(
                values[mask], latent[mask], obs_params, process_vars
            )
            if not np.isfinite(ll):
                return -np.inf
            total_ll += ll

        return total_ll

    def _fusable_modalities(self, data: TimeSeriesData) -> list[str]:
        """Active modalities on this replicate that can share one joint normal.

        A modality qualifies when it measures a projection of the latent state
        (so the LNA covariance reaches it) and supplies a Gaussian
        linearization. Biomarker fractions have no linearization and are scored
        exactly, on their own.
        """
        out = []
        for modality in self._active_modalities:
            if not data.has_modality(modality):
                continue
            if MODALITY_OBSERVABLE.get(modality) is None:
                continue
            model = self._modality_models[modality]
            if type(model).linearize is ObservationModel.linearize:
                continue
            out.append(modality)
        return out

    def _fused_replicate_ll(
        self,
        data: TimeSeriesData,
        params: dict[str, float],
        idx: np.ndarray,
        latent: np.ndarray,
        covs: np.ndarray,
        modalities: list[str],
    ) -> tuple[float, frozenset[str]]:
        """Score latent-sharing modalities jointly, point by point.

        At a time point where two or more of them are observed, one
        multivariate normal carries the correlation induced by the shared
        population. Where only one is observed there is nothing to correlate,
        so the exact per-modality likelihood is used instead -- it keeps the
        true observation law (negative binomial, lognormal) rather than the
        Gaussian approximation the fused path necessarily uses.

        Returns the summed log-likelihood and the set of modalities it covers,
        which the caller must then skip.
        """
        masks = {m: self.scored_mask(data, m) for m in modalities}
        obs_params = {
            m: self._modality_params(m, params, data, idx, masks[m])
            for m in modalities
        }
        # Position of each scored point within that modality's masked arrays,
        # so array-valued parameters (per-time-point tumour volumes) can be
        # indexed back to a scalar for the point-wise calls below.
        positions = {m: np.cumsum(masks[m]) - 1 for m in modalities}

        fusion = MultimodalObservation(
            {m: self._modality_models[m] for m in modalities}
        )

        total = 0.0
        for j in range(len(data.times)):
            present = {}
            point_params = {}
            for m in modalities:
                if not masks[m][j]:
                    continue
                value = data.observations[m][j]
                if not np.isfinite(value):
                    continue
                present[m] = float(value)
                point_params[m] = _scalarize_params(
                    obs_params[m], int(positions[m][j])
                )

            if not present:
                continue

            state = latent[j]
            if len(present) >= 2:
                ll = fusion.joint_log_likelihood(
                    present, state, covs[idx[j]], point_params
                )
                if ll is not None:
                    if not np.isfinite(ll):
                        return -np.inf, frozenset(modalities)
                    total += ll
                    continue
                # Fusion declined (degenerate covariance, or fewer than two
                # modalities could linearize); fall through and score each
                # exactly.

            for m, value in present.items():
                model = self._modality_models[m]
                observable = MODALITY_OBSERVABLE[m]
                pv = float(model.project_variance(observable, covs[idx[j]]))
                ll = model.log_likelihood(value, state, point_params[m], pv)
                if not np.isfinite(ll):
                    return -np.inf, frozenset(modalities)
                total += ll

        return total, frozenset(modalities)

    def _modality_params(
        self,
        modality: str,
        params: dict[str, float],
        data: TimeSeriesData,
        idx: np.ndarray,
        mask: np.ndarray,
    ) -> dict:
        """Assemble the parameter dict handed to one modality's model.

        Modality-specific names are passed through consistently, and paired
        volume measurements are forwarded so the BLI model can apply
        volume-dependent optical attenuation.
        """
        obs_params = {
            k: v for k, v in params.items() if k in OBSERVATION_PARAM_NAMES
        }

        if modality == "bli":
            for volume_key in ("volume", "tumor_volume"):
                if not data.has_modality(volume_key):
                    continue
                paired = data.observations[volume_key][mask]
                if paired.size and not np.all(np.isnan(paired)):
                    # Per-time-point volumes: attenuation grows with tumour
                    # size, so a single mean volume would apply one constant
                    # correction across the whole trajectory and erase the
                    # size-dependence being modelled. Gaps are filled with the
                    # series mean so a missing volume does not drop the BLI
                    # point entirely.
                    filled = np.where(
                        np.isnan(paired), np.nanmean(paired), paired
                    )
                    obs_params["tumor_volume"] = filled
                break

        return obs_params

    def __call__(self, theta: np.ndarray) -> float:
        """Evaluate total log-likelihood at parameter vector theta.

        Args:
            theta: Parameter vector.

        Returns:
            Total log-likelihood (sum across all data series and modalities).
        """
        self._n_evals += 1
        params = self.theta_to_params(theta)

        # Check parameter bounds (all rates must be non-negative)
        for name, val in params.items():
            if not np.isfinite(val) or val < 0:
                return -np.inf

        try:
            rate_set = self._build_rate_set(params)
        except ValueError as exc:
            # A parameter vector the rate laws reject (ec50 <= 0, hill <= 0,
            # a birth Emax above 1) has no valid forward model, so its
            # likelihood is zero. Raising here instead would abort an
            # optimiser or sampler that merely stepped outside the support.
            logger.debug("Rejected parameter vector: %s", exc)
            return -np.inf

        total_ll = 0.0

        for conc, series in self._conc_groups.items():
            times = self._group_times[conc]
            if len(times) < 2:
                continue

            # Replicates may start from different measured initial counts. When
            # they do, each needs its own solve; otherwise one solve is shared.
            initials = [self._initial_state(d, rate_set) for d in series]
            shared = all(np.allclose(x, initials[0]) for x in initials)

            try:
                if shared:
                    t_sol, means, covs = self._solve_forward(
                        rate_set, conc, times, initials[0]
                    )
                    solutions = [(t_sol, means, covs)] * len(series)
                else:
                    solutions = [
                        self._solve_forward(rate_set, conc, times, x)
                        for x in initials
                    ]
            except (RuntimeError, ValueError) as exc:
                logger.debug("Forward solve failed at conc=%s: %s", conc, exc)
                return -np.inf

            for data, (t_sol, means, covs) in zip(series, solutions):
                ll = self._evaluate_replicate(data, params, t_sol, means, covs)
                if not np.isfinite(ll):
                    return -np.inf
                total_ll += ll

        return total_ll

    def neg_log_likelihood(self, theta: np.ndarray) -> float:
        """Negative log-likelihood (for minimization)."""
        return -self.__call__(theta)

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    @property
    def n_evaluations(self) -> int:
        return self._n_evals

    @property
    def n_observations(self) -> int:
        """Number of observations actually entering the likelihood.

        Counts only non-missing values in modalities the model uses, minus any
        anchor points consumed by the initial condition. This is the sample
        size for AIC/BIC.
        """
        total = 0
        for data in self.data_list:
            for modality in self._active_modalities:
                if not data.has_modality(modality):
                    continue
                n = int(np.count_nonzero(data.observed_mask(modality)))
                if (
                    self.condition_on_first
                    and modality == self.anchor_modality
                    and n > 0
                ):
                    n -= 1
                total += n
        return total

    def bic(self, theta: np.ndarray) -> float:
        """Bayesian information criterion at theta."""
        n = self.n_observations
        if n <= 0:
            return np.inf
        return float(self.n_params * np.log(n) - 2.0 * self.__call__(theta))

    def aic(self, theta: np.ndarray) -> float:
        """Akaike information criterion at theta."""
        return float(2.0 * self.n_params - 2.0 * self.__call__(theta))
