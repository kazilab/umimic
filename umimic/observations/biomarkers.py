"""Snapshot biomarker observation models (Ki-67, cleaved caspase, etc.).

These are optional observations from immunostaining at terminal or
biopsy time points that directly measure cell state fractions, dramatically
improving identifiability of the branching process parameters.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from umimic.dynamics.states import ModelTopology
from umimic.observations.base import ObservationModel, TopologyAwareObservation


class BiomarkerObservation(TopologyAwareObservation, ObservationModel):
    """Biomarker observation for cell state fractions.

    Models snapshot immunostaining observations:
    - Ki-67: proliferative fraction f_P = P / (P + Q + R)
    - Cleaved caspase / TUNEL: apoptotic fraction f_A = A / (P + Q + A + R)

    Observations are modeled as Beta-distributed around the true fraction:
        Y ~ Beta(alpha, beta) where alpha = kappa * f, beta = kappa * (1 - f)
        and kappa is the precision parameter.
    """

    #: Canonical modality key used by data schemas and configuration.
    modality_name = "biomarker"

    #: Biomarker types with a defined mapping from latent state to fraction.
    SUPPORTED_TYPES = ("ki67", "caspase")

    def __init__(
        self,
        biomarker_type: str = "ki67",
        precision: float = 50.0,
        topology: ModelTopology | None = None,
    ):
        """
        Args:
            biomarker_type: 'ki67' (cycling fraction) or 'caspase'
                (apoptotic fraction).
            precision: Beta distribution precision (higher = less noise).
            topology: Model topology, used to identify the relevant states.
                For Ki-67 this determines which states count as cycling, so
                omitting it in a model where R proliferates undercounts the
                proliferative fraction.
        """
        TopologyAwareObservation.__init__(self, topology)
        if not np.isfinite(precision) or precision <= 0:
            raise ValueError(f"precision must be positive, got {precision}.")
        if biomarker_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unknown biomarker_type {biomarker_type!r}; expected one of "
                f"{self.SUPPORTED_TYPES}."
            )
        self.biomarker_type = biomarker_type
        self.precision = precision

    def _cycling_operator(self, n_states: int) -> np.ndarray:
        """Indicator of the states that stain Ki-67 positive.

        Ki-67 marks cycling cells, so it is *every dividing state*, not P
        alone. With a proliferating resistant compartment, using P alone
        undercounts the proliferative fraction by exactly the R population.
        """
        if self.topology is None or self.topology.n_states != n_states:
            # No topology: fall back to the canonical assumption that only P
            # cycles, which is what the operator helper would give anyway.
            return self.operator("proliferating", n_states)
        H = np.zeros(n_states)
        for ct in self.topology.division_states:
            H[self.topology.state_index(ct)] = 1.0
        return H

    def _get_fraction(self, latent_state: np.ndarray) -> float:
        """Compute the target fraction from latent state.

        Returns NaN when the population is extinct: no cells means no stain,
        and reporting 0.5 would inject a spurious "50% positive" observation
        exactly where the data carry no information.
        """
        x = np.maximum(np.asarray(latent_state, dtype=float), 0.0)
        total = float(np.sum(x))
        if total <= 0:
            return float("nan")

        if self.biomarker_type == "ki67":
            # Cycling fraction: dividing states / viable
            cycling = float(self._cycling_operator(x.shape[-1]) @ x)
            viable = self._project("viable", latent_state, floor=1e-12)
            return cycling / viable
        elif self.biomarker_type == "caspase":
            # Apoptotic fraction: A / total
            A = self._project("dead", latent_state, floor=0.0)
            return A / total

        raise ValueError(
            f"Unknown biomarker_type {self.biomarker_type!r}; expected 'ki67' "
            "or 'caspase'. A 'custom' type is not supported: it previously "
            "returned a constant 0.5 regardless of the latent state, which is "
            "a fixed fake observation rather than a customisable one."
        )

    def log_likelihood(
        self,
        observed: float | np.ndarray,
        latent_state: np.ndarray,
        params: dict | None = None,
        process_variance: float | None = None,
    ) -> float:
        """Log-likelihood under the Beta model.

        An extinct population contributes nothing: there are no cells to
        stain, so the fraction is undefined and the term is dropped rather
        than evaluated against a fabricated 0.5.
        """
        f = self._get_fraction(latent_state)
        if not np.isfinite(f):
            return 0.0

        kappa = self.precision
        if params and "biomarker_precision" in params:
            kappa = float(params["biomarker_precision"])
        if not np.isfinite(kappa) or kappa <= 0:
            raise ValueError(
                f"biomarker_precision must be positive, got {kappa}."
            )

        obs_val = float(observed)
        if not np.isfinite(obs_val) or not 0.0 <= obs_val <= 1.0:
            raise ValueError(
                f"Biomarker observations are fractions and must lie in [0, 1], "
                f"got {observed!r}."
            )

        f = np.clip(f, 1e-4, 1 - 1e-4)
        obs_val = np.clip(obs_val, 1e-4, 1 - 1e-4)
        return float(stats.beta.logpdf(obs_val, kappa * f, kappa * (1 - f)))

    def sample(
        self,
        latent_state: np.ndarray,
        rng: np.random.Generator,
        params: dict | None = None,
    ) -> float:
        """Sample a biomarker observation. Extinct populations give NaN."""
        f = self._get_fraction(latent_state)
        if not np.isfinite(f):
            return float("nan")
        f = np.clip(f, 1e-4, 1 - 1e-4)
        kappa = self.precision
        if params and "biomarker_precision" in params:
            kappa = float(params["biomarker_precision"])
        return float(rng.beta(kappa * f, kappa * (1 - f)))

    def expected_value(self, latent_state: np.ndarray) -> float:
        return self._get_fraction(latent_state)

    def param_names(self) -> list[str]:
        return ["precision"]
