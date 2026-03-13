"""Central log-likelihood computation for U-MIMIC.

This module is the critical bridge between the dynamics model and inference.
It takes a parameter vector, builds the forward model, runs it, and evaluates
the observation log-likelihood at each data point.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np

from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.rates import RateSet, EmaxHill
from umimic.dynamics.moment_equations import MomentODE
from umimic.dynamics.ode_system import CellDynamicsODE
from umimic.observations.base import ObservationModel
from umimic.observations.cell_counts import CellCountObservation
from umimic.data.schemas import TimeSeriesData


# Default parameter names and their positions in the theta vector.
# These names map to RateSet fields as documented in RateSet's docstring.
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


class ModelLikelihood:
    """Central likelihood function for parameter estimation.

    Given a parameter vector theta, this class:
    1. Constructs a RateSet from the parameters
    2. Runs the forward model (moment ODE or deterministic ODE)
    3. Evaluates the observation log-likelihood at each data point
    4. Returns the total log-likelihood

    This serves as the objective function for MLE and the target for MCMC.
    """

    def __init__(
        self,
        topology: ModelTopology,
        data: TimeSeriesData | list[TimeSeriesData],
        param_names: list[str] | None = None,
        mode: str = "moment",
        observation_model: ObservationModel | None = None,
    ):
        """
        Args:
            topology: Model topology (which states/transitions).
            data: Observed data (single series or list for multiple replicates).
            param_names: Names of parameters in the theta vector.
            mode: Forward model mode - 'moment' (LNA) or 'ode' (deterministic).
            observation_model: Observation model (default: CellCountObservation).
        """
        self.topology = topology
        self.data_list = data if isinstance(data, list) else [data]
        self.param_names = param_names or DEFAULT_PARAM_NAMES
        self.mode = mode
        self.obs_model = observation_model or CellCountObservation(overdispersion=10.0)
        self._n_evals = 0

        # Pre-group data by concentration to avoid redundant ODE solves.
        # Replicates at the same concentration share identical dynamics.
        self._conc_groups: dict[float, list[TimeSeriesData]] = defaultdict(list)
        for d in self.data_list:
            conc = d.concentration if d.concentration is not None else 0.0
            self._conc_groups[conc].append(d)

    def theta_to_params(self, theta: np.ndarray) -> dict[str, float]:
        """Convert parameter vector to named dict."""
        return {name: theta[i] for i, name in enumerate(self.param_names)}

    def params_to_theta(self, params: dict[str, float]) -> np.ndarray:
        """Convert named params to vector."""
        return np.array([params[name] for name in self.param_names])

    def _build_rate_set(self, params: dict[str, float]) -> RateSet:
        """Construct a RateSet from parameter values."""
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

        death_modulation = {}
        if death_mod is not None:
            death_modulation[CellType.P] = death_mod

        trans_base = {}
        if "u_PQ" in params:
            trans_base[(CellType.P, CellType.Q)] = params["u_PQ"]
        if "u_QP" in params:
            trans_base[(CellType.Q, CellType.P)] = params["u_QP"]
        if "u_PR" in params:
            trans_base[(CellType.P, CellType.R)] = params["u_PR"]

        return RateSet(
            birth_base=params.get("b0", 0.04),
            birth_modulation=birth_mod,
            death_base=death_base,
            death_modulation=death_modulation,
            transition_base=trans_base,
        )

    def _solve_forward(
        self,
        rate_set: RateSet,
        conc: float,
        times: np.ndarray,
        n0: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Solve the forward model for a single concentration.

        Returns:
            (t_sol, means, covs) where covs is None for ODE mode.
        """
        exposure_fn = lambda t, _c=conc: _c

        mu0 = np.zeros(self.topology.n_states)
        mu0[0] = max(n0, 1.0)

        covs = None
        if self.mode == "moment":
            moment_ode = MomentODE(rate_set, self.topology, exposure_fn)
            t_sol, means, covs = moment_ode.solve(
                mu0, t_span=(times[0], times[-1]), t_eval=times
            )
        else:
            ode = CellDynamicsODE(rate_set, self.topology, exposure_fn)
            result = ode.solve(mu0, (times[0], times[-1]), times)
            means = np.column_stack(
                [result.populations[ct.name] for ct in self.topology.active_states]
            )
            t_sol = result.times

        return t_sol, means, covs

    def _evaluate_replicate(
        self,
        data: TimeSeriesData,
        params: dict[str, float],
        t_sol: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray | None,
    ) -> float:
        """Evaluate log-likelihood for a single replicate using cached ODE solution.

        Uses batch evaluation for all time points at once.
        """
        times = data.times
        if len(times) < 2:
            return 0.0

        if "cell_counts" not in data.observations:
            return 0.0

        obs_params = {"overdispersion": params.get("overdispersion", 10.0)}
        all_obs = data.observations["cell_counts"]

        # Match solution times to data times. When t_eval was passed to solve,
        # t_sol should already align with times. Use direct indexing when possible.
        if len(t_sol) == len(times) and np.allclose(t_sol, times, atol=1e-10):
            latent_states = np.maximum(means, 0)
            process_vars = np.sum(covs, axis=(1, 2)) if covs is not None else None
        else:
            # Fallback: find closest time indices
            t_idx = np.searchsorted(t_sol, times, side="left")
            t_idx = np.clip(t_idx, 0, len(t_sol) - 1)
            latent_states = np.maximum(means[t_idx], 0)
            process_vars = (
                np.sum(covs[t_idx], axis=(1, 2)) if covs is not None else None
            )

        return self.obs_model.log_likelihood_batch(
            all_obs, latent_states, obs_params, process_vars
        )

    def __call__(self, theta: np.ndarray) -> float:
        """Evaluate total log-likelihood at parameter vector theta.

        Args:
            theta: Parameter vector.

        Returns:
            Total log-likelihood (sum across all data series).
        """
        self._n_evals += 1
        params = self.theta_to_params(theta)

        # Check parameter bounds (all rates must be non-negative)
        for name, val in params.items():
            if val < 0:
                return -np.inf

        # Build rate set once for this parameter vector
        rate_set = self._build_rate_set(params)

        total_ll = 0.0

        # Evaluate by concentration group: solve ODE once per unique concentration,
        # then evaluate log-likelihood for each replicate against the shared solution.
        for conc, data_list in self._conc_groups.items():
            # Use the union of all time points for the ODE solve.
            # For typical experiments, all replicates share the same schedule.
            representative = data_list[0]
            times = representative.times
            if len(times) < 2:
                continue

            n0 = representative.observations.get("cell_counts", np.array([100]))[0]

            try:
                t_sol, means, covs = self._solve_forward(
                    rate_set, conc, times, n0
                )
            except Exception:
                return -np.inf

            for data in data_list:
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
