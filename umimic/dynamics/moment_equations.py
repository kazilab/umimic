"""Linear Noise Approximation: moment ODE system.

Propagates mean mu(t) and covariance Sigma(t) for the cell population
branching process. This is the 'fast mode' foundation for inference.

The mean follows the deterministic ODE, and the covariance evolves via:
    d(Sigma)/dt = A(t)*Sigma + Sigma*A(t)^T + D(t)
where A is the Jacobian of the drift and D is the diffusion matrix.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.rates import RateSet


class MomentODE:
    """Moment ODE system for the linear noise approximation.

    Simultaneously propagates:
    - mu(t): mean population vector (n_states)
    - Sigma(t): covariance matrix (n_states x n_states)

    The key insight from BESTDR: variance carries mechanistic information.
    Different mechanisms (cytostatic vs cytotoxic) produce the same mean
    but different variance signatures, enabling statistical separation.
    """

    def __init__(
        self,
        rate_set: RateSet,
        topology: ModelTopology,
        exposure_fn: Callable[[float], float],
    ):
        self.rate_set = rate_set
        self.topology = topology
        self.exposure_fn = exposure_fn
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

        # Division state indices
        self._division_idx = np.array(
            [idx[ct] for ct in topo.division_states], dtype=np.intp
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

    def rate_matrix(self, t: float, mu: np.ndarray) -> np.ndarray:
        """Construct the Jacobian matrix A(t) of the drift.

        A[i,j] = d(drift_i)/d(x_j) evaluated at mu.
        """
        n = self.n
        A = np.zeros((n, n))
        c = self.exposure_fn(t)
        total = float(np.sum(np.maximum(mu, 0)))
        K = self.topology.carrying_capacity if self.topology.density_dependent else None

        # Birth contributions (vectorized over division states)
        if len(self._division_idx) > 0:
            b = self.rate_set.birth_rate(c, total, K)
            for i in self._division_idx:
                A[i, i] += b
                if K is not None and K > 0 and self.rate_set.birth_base > 0 and mu[i] > 0:
                    mod = (1.0 - float(self.rate_set.birth_modulation(c))
                           if self.rate_set.birth_modulation else 1.0)
                    density_term = self.rate_set.birth_base * mod * mu[i] / K
                    A[i, :] -= density_term

        # Death contributions
        for j, ct in enumerate(self._death_states):
            i = self._death_idx[j]
            d = self.rate_set.death_rate(ct, c)
            A[i, i] -= d
            if self._has_A:
                A[self._a_idx, i] += d

        # Apoptotic clearance
        if self._has_A:
            A[self._a_idx, self._a_idx] -= self.rate_set.clearance_rate

        # Transitions (vectorized index assignments)
        for k, (src, tgt) in enumerate(self._trans_pairs):
            rate = self.rate_set.transition_rate(src, tgt, c)
            i_src = self._trans_src_idx[k]
            i_tgt = self._trans_tgt_idx[k]
            A[i_src, i_src] -= rate
            A[i_tgt, i_src] += rate

        return A

    def diffusion_matrix(self, t: float, mu: np.ndarray) -> np.ndarray:
        """Construct the diffusion matrix D(t).

        Uses pre-computed stoichiometry outer products (from __init__) and
        computes D = sum_k propensity_k * outer_k via np.einsum.
        """
        c = self.exposure_fn(t)
        total = float(np.sum(np.maximum(mu, 0)))
        K = self.topology.carrying_capacity if self.topology.density_dependent else None
        mu_pos = np.maximum(mu, 0)

        # Build propensity vector aligned with self._diff_outers / self._outer_stack
        propensities = np.empty(self._n_reactions)
        idx = 0

        # Birth reactions
        if len(self._division_idx) > 0:
            b = self.rate_set.birth_rate(c, total, K)
            for i in self._division_idx:
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

        # Mean dynamics (same as deterministic ODE)
        A = self.rate_matrix(t, mu)
        dmu = A @ mu

        # Covariance dynamics: dSigma/dt = A*Sigma + Sigma*A^T + D
        D = self.diffusion_matrix(t, mu)
        dSigma = A @ Sigma + Sigma @ A.T + D

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
            sol = solve_ivp(
                self.rhs, t_span, y0, t_eval=t_eval,
                method="BDF", rtol=1e-6, atol=1e-8,
            )

        times = sol.t
        n_times = len(times)
        means = sol.y[:n, :].T  # (n_times, n_states)

        # Reshape covariances: vectorized symmetrization + PSD enforcement
        cov_flat = sol.y[n:, :].T  # (n_times, n*n)
        covs = cov_flat.reshape(n_times, n, n).copy()
        # Symmetrize all at once
        covs = (covs + np.swapaxes(covs, 1, 2)) / 2
        # Cheap PSD enforcement: clamp negative eigenvalues only where needed
        for k in range(n_times):
            min_eig = np.linalg.eigvalsh(covs[k])[0]
            if min_eig < 0:
                covs[k] += (-min_eig + 1e-8) * np.eye(n)

        return times, means, covs
