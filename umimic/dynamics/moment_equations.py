"""Linear Noise Approximation: moment ODE system.

Propagates mean mu(t) and covariance Sigma(t) for the cell population
branching process. This is the 'fast mode' foundation for inference.

The mean follows the deterministic ODE, and the covariance evolves via:
    d(Sigma)/dt = A(t)*Sigma + Sigma*A(t)^T + D(t)
where A is the Jacobian of the drift and D is the diffusion matrix.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.rates import RateSet

logger = logging.getLogger(__name__)


class MomentODE:
    """Moment ODE system for the linear noise approximation.

    Simultaneously propagates:
    - mu(t): mean population vector (n_states)
    - Sigma(t): covariance matrix (n_states x n_states)

    Variance carries mechanistic information the mean does not: cytostatic
    and cytotoxic action can produce the same mean trajectory while differing
    in b + d. This class propagates that variance correctly (ODE/SSA/LNA agree
    to <0.3% in the mean, 0.92-1.01 in the variance ratio).

    Whether it is *detectable* is a separate question and usually the answer
    is no: what matters is the process variance as a share of total
    observation variance, which at the package defaults is ~1.5%. Propagating
    the signature is not the same as being able to fit on it.
    See umimic/inference/SCIENTIFIC_ASSUMPTIONS.md section 2a.
    """

    def __init__(
        self,
        rate_set: RateSet,
        topology: ModelTopology,
        exposure_fn: Callable[[float], float],
        psd_tolerance: float = 1e-6,
    ):
        self.rate_set = rate_set
        self.topology = topology
        self.exposure_fn = exposure_fn
        self.psd_tolerance = psd_tolerance
        self.n = topology.n_states
        self._idx = {ct: i for i, ct in enumerate(topology.active_states)}

        # Pre-compute topology-dependent indices and stoichiometry matrices
        self._precompute_structure()

    def _precompute_structure(self) -> None:
        """Pre-compute stoichiometry outer products and index arrays.

        This eliminates per-call np.zeros/np.outer allocation and Python
        loops in rate_matrix() and diffusion_matrix(), which are called
        hundreds of times per ODE solve.
        """
        n = self.n
        idx = self._idx
        topo = self.topology

        # Division state indices, paired with their cell types so per-state
        # birth rates (e.g. a resistant clone's) can be resolved at runtime.
        self._division_states = list(topo.division_states)
        self._division_idx = np.array(
            [idx[ct] for ct in self._division_states], dtype=np.intp
        )

        # Death state indices (excluding A)
        self._death_states = []
        self._death_idx = []
        for ct in topo.death_states:
            if ct != CellType.A:
                self._death_states.append(ct)
                self._death_idx.append(idx[ct])
        self._death_idx = np.array(self._death_idx, dtype=np.intp)

        self._has_A = topo.has_state(CellType.A)
        self._a_idx = idx[CellType.A] if self._has_A else -1

        # Transition (src_idx, tgt_idx, src_CellType, tgt_CellType)
        self._trans_src_idx = np.array(
            [idx[s] for s, t in topo.transitions], dtype=np.intp
        )
        self._trans_tgt_idx = np.array(
            [idx[t] for s, t in topo.transitions], dtype=np.intp
        )
        self._trans_pairs = list(topo.transitions)  # keep CellType pairs for rate lookup

        # Pre-compute stoichiometry outer products for diffusion matrix.
        # Each reaction has a fixed stoichiometry vector; its outer product
        # (stoich @ stoich.T) is topology-dependent and never changes.
        # At runtime we only need to multiply by the propensity scalar.

        self._diff_outers = []  # list of (outer_matrix, state_idx, reaction_type)

        # Birth reactions
        for ct in topo.division_states:
            i = idx[ct]
            outer = np.zeros((n, n))
            outer[i, i] = 1.0  # stoich=[0,...,+1,...,0] → outer is e_i @ e_i^T
            self._diff_outers.append(("birth", i, outer))

        # Death reactions
        for ct in self._death_states:
            i = idx[ct]
            stoich = np.zeros(n)
            stoich[i] = -1.0
            if self._has_A:
                stoich[self._a_idx] = 1.0
            self._diff_outers.append(("death", i, np.outer(stoich, stoich)))

        # Clearance
        if self._has_A:
            outer = np.zeros((n, n))
            outer[self._a_idx, self._a_idx] = 1.0  # stoich=e_a, outer=e_a@e_a^T
            self._diff_outers.append(("clearance", self._a_idx, outer))

        # Transitions
        for k, (src, tgt) in enumerate(topo.transitions):
            i_src = idx[src]
            i_tgt = idx[tgt]
            stoich = np.zeros(n)
            stoich[i_src] = -1.0
            stoich[i_tgt] = 1.0
            self._diff_outers.append(("transition", k, np.outer(stoich, stoich)))

        # Stack all outer products into a single 3D array for einsum
        self._n_reactions = len(self._diff_outers)
        self._outer_stack = np.array([o[2] for o in self._diff_outers])  # (R, n, n)

    def drift(self, t: float, mu: np.ndarray) -> np.ndarray:
        """Nonlinear drift f(t, mu) = d(mu)/dt.

        This is the mean-field vector field and matches
        :meth:`umimic.dynamics.ode_system.CellDynamicsODE.rhs` exactly.

        It must be evaluated directly rather than as ``J @ mu``: with density
        dependence the drift is quadratic in ``mu``, so ``J @ mu`` differs from
        ``f(mu)`` (for a one-state logistic it yields ``r*mu*(1 - 2*mu/K)``,
        whose fixed point is K/2 instead of K).
        """
        n = self.n
        f = np.zeros(n)
        c = self.exposure_fn(t)
        mu_pos = np.maximum(mu, 0.0)
        total = self.topology.density_total(mu_pos)
        K = self.topology.carrying_capacity if self.topology.density_dependent else None

        # Birth. Rates are resolved per dividing state so that a resistant
        # clone can carry its own division rate and drug sensitivity.
        for i, ct in zip(self._division_idx, self._division_states):
            b = self.rate_set.birth_rate(c, total, K, cell_type=ct)
            f[i] += b * mu_pos[i]

        # Death (dead cells flow into A when that state is tracked)
        for j, ct in enumerate(self._death_states):
            i = self._death_idx[j]
            d = self.rate_set.death_rate(ct, c)
            f[i] -= d * mu_pos[i]
            if self._has_A:
                f[self._a_idx] += d * mu_pos[i]

        # Apoptotic clearance
        if self._has_A:
            f[self._a_idx] -= self.rate_set.clearance_rate * mu_pos[self._a_idx]

        # Transitions
        for k, (src, tgt) in enumerate(self._trans_pairs):
            rate = self.rate_set.transition_rate(src, tgt, c)
            i_src = self._trans_src_idx[k]
            i_tgt = self._trans_tgt_idx[k]
            flux = rate * mu_pos[i_src]
            f[i_src] -= flux
            f[i_tgt] += flux

        return f

    def jacobian(self, t: float, mu: np.ndarray) -> np.ndarray:
        """Jacobian J(t, mu) with J[i, j] = d(f_i)/d(mu_j), evaluated at mu.

        Used only for the covariance equation. All terms are linear in the
        state except birth under density dependence, where
        ``f_i = B(C) * (1 - N/K) * mu_i`` with ``N = m . mu`` the crowding
        total and ``m`` the density mask, giving

            d(f_i)/d(mu_j) = b(C, N) * delta_ij - B(C) * mu_i * m_j / K

        The mask matters: when apoptotic cells do not occupy space, the
        density gradient is zero in the apoptotic column and a non-zero entry
        there would misstate the covariance.
        """
        n = self.n
        J = np.zeros((n, n))
        c = self.exposure_fn(t)
        mu_pos = np.maximum(mu, 0.0)
        mask = self.topology.density_mask
        total = float(mask @ mu_pos)
        K = self.topology.carrying_capacity if self.topology.density_dependent else None

        # Birth
        # The density factor is clamped at zero; past carrying capacity the
        # birth term is identically zero and so is its derivative.
        density_active = K is not None and K > 0 and total < K
        for i, ct in zip(self._division_idx, self._division_states):
            b = self.rate_set.birth_rate(c, total, K, cell_type=ct)
            B = self.rate_set.modulated_birth_base(c, cell_type=ct)
            J[i, i] += b
            if density_active and B > 0 and mu_pos[i] > 0:
                J[i, :] -= (B * mu_pos[i] / K) * mask

        # Death
        for j, ct in enumerate(self._death_states):
            i = self._death_idx[j]
            d = self.rate_set.death_rate(ct, c)
            J[i, i] -= d
            if self._has_A:
                J[self._a_idx, i] += d

        # Apoptotic clearance
        if self._has_A:
            J[self._a_idx, self._a_idx] -= self.rate_set.clearance_rate

        # Transitions
        for k, (src, tgt) in enumerate(self._trans_pairs):
            rate = self.rate_set.transition_rate(src, tgt, c)
            i_src = self._trans_src_idx[k]
            i_tgt = self._trans_tgt_idx[k]
            J[i_src, i_src] -= rate
            J[i_tgt, i_src] += rate

        return J

    def rate_matrix(self, t: float, mu: np.ndarray) -> np.ndarray:
        """Deprecated alias for :meth:`jacobian`.

        Retained for backwards compatibility. Note that this is the Jacobian,
        not the mean drift; use :meth:`drift` for d(mu)/dt.
        """
        return self.jacobian(t, mu)

    def diffusion_matrix(self, t: float, mu: np.ndarray) -> np.ndarray:
        """Construct the diffusion matrix D(t).

        Uses pre-computed stoichiometry outer products (from __init__) and
        computes D = sum_k propensity_k * outer_k via np.einsum.
        """
        c = self.exposure_fn(t)
        mu_pos = np.maximum(mu, 0)
        total = self.topology.density_total(mu_pos)
        K = self.topology.carrying_capacity if self.topology.density_dependent else None

        # Build propensity vector aligned with self._diff_outers / self._outer_stack
        propensities = np.empty(self._n_reactions)
        idx = 0

        # Birth reactions
        if len(self._division_idx) > 0:
            for i, ct in zip(self._division_idx, self._division_states):
                b = self.rate_set.birth_rate(c, total, K, cell_type=ct)
                propensities[idx] = b * mu_pos[i]
                idx += 1
        # Death reactions
        for j, ct in enumerate(self._death_states):
            i = self._death_idx[j]
            d = self.rate_set.death_rate(ct, c)
            propensities[idx] = d * mu_pos[i]
            idx += 1
        # Clearance
        if self._has_A:
            propensities[idx] = self.rate_set.clearance_rate * mu_pos[self._a_idx]
            idx += 1
        # Transitions
        for k, (src, tgt) in enumerate(self._trans_pairs):
            i_src = self._trans_src_idx[k]
            rate = self.rate_set.transition_rate(src, tgt, c)
            propensities[idx] = rate * mu_pos[i_src]
            idx += 1

        # D = sum_k propensity_k * outer_k  via einsum
        return np.einsum("k,kij->ij", propensities, self._outer_stack)

    def rhs(self, t: float, state_flat: np.ndarray) -> np.ndarray:
        """Combined RHS for [mu_flat, Sigma_flat].

        State vector layout:
        - [0:n] = mu (mean vector)
        - [n:n+n*n] = Sigma (covariance, flattened row-major)
        """
        n = self.n
        mu = state_flat[:n]
        Sigma = state_flat[n:].reshape(n, n)

        # Mean dynamics: the nonlinear drift, evaluated directly.
        dmu = self.drift(t, mu)

        # Covariance dynamics: dSigma/dt = J*Sigma + Sigma*J^T + D
        J = self.jacobian(t, mu)
        D = self.diffusion_matrix(t, mu)
        dSigma = J @ Sigma + Sigma @ J.T + D

        return np.concatenate([dmu, dSigma.flatten()])

    def solve(
        self,
        mu0: np.ndarray,
        Sigma0: np.ndarray | None = None,
        t_span: tuple[float, float] = (0, 72),
        t_eval: np.ndarray | None = None,
        method: str = "RK45",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integrate the moment equations.

        Args:
            mu0: Initial mean vector (n_states,).
            Sigma0: Initial covariance matrix (n_states, n_states).
                    Default: diagonal with variance = mu0 (Poisson-like).
            t_span: Time interval.
            t_eval: Times to record.
            method: ODE solver method.

        Returns:
            (times, means, covariances) where:
            - times: (n_times,)
            - means: (n_times, n_states)
            - covariances: (n_times, n_states, n_states)
        """
        n = self.n
        if Sigma0 is None:
            Sigma0 = np.diag(np.maximum(mu0, 1.0))

        y0 = np.concatenate([mu0, Sigma0.flatten()])

        sol = solve_ivp(
            self.rhs,
            t_span,
            y0,
            t_eval=t_eval,
            method=method,
            rtol=1e-6,
            atol=1e-8,
        )

        if not sol.success:
            logger.warning(
                "Moment ODE solve with %s failed, retrying with BDF: %s",
                method,
                sol.message,
            )
            sol = solve_ivp(
                self.rhs, t_span, y0, t_eval=t_eval,
                method="BDF", rtol=1e-6, atol=1e-8,
            )
            if not sol.success:
                raise RuntimeError(
                    f"Moment-equation integration failed after BDF retry: {sol.message}"
                )

        times = sol.t
        n_times = len(times)
        means = sol.y[:n, :].T  # (n_times, n_states)

        # Reshape covariances: vectorized symmetrization + PSD enforcement
        cov_flat = sol.y[n:, :].T  # (n_times, n*n)
        covs = cov_flat.reshape(n_times, n, n).copy()
        # Symmetrize all at once (asymmetry here is pure integrator round-off)
        covs = (covs + np.swapaxes(covs, 1, 2)) / 2

        # PSD enforcement. Small negative eigenvalues are round-off and are
        # clamped silently; a large one means the LNA has genuinely broken down
        # (typically near-extinction) and must not be papered over quietly.
        for k in range(n_times):
            min_eig = np.linalg.eigvalsh(covs[k])[0]
            if min_eig < 0:
                scale = max(float(np.max(np.abs(np.diag(covs[k])))), 1.0)
                if -min_eig > self.psd_tolerance * scale:
                    logger.warning(
                        "Moment covariance at t=%.4g has eigenvalue %.3e "
                        "(%.1f%% of the covariance scale); the linear noise "
                        "approximation is unreliable here.",
                        times[k],
                        min_eig,
                        100.0 * (-min_eig) / scale,
                    )
                covs[k] += (-min_eig + 1e-8) * np.eye(n)

        return times, means, covs
