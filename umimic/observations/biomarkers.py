"""Snapshot biomarker observation models (Ki-67, cleaved caspase, etc.).

These are optional observations from immunostaining at terminal or
biopsy time points that directly measure cell state fractions, dramatically
improving identifiability of the branching process parameters.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from umimic.observations.base import ObservationModel


class BiomarkerObservation(ObservationModel):
    """Biomarker observation for cell state fractions.

    Models snapshot immunostaining observations:
    - Ki-67: proliferative fraction f_P = P / (P + Q + R)
    - Cleaved caspase / TUNEL: apoptotic fraction f_A = A / (P + Q + A + R)

    Observations are modeled as Beta-distributed around the true fraction:
        Y ~ Beta(alpha, beta) where alpha = kappa * f, beta = kappa * (1 - f)
        and kappa is the precision parameter.
    """

    def __init__(
        self,
        biomarker_type: str = "ki67",
        precision: float = 50.0,
    ):
        """
        Args:
            biomarker_type: 'ki67' (proliferative), 'caspase' (apoptotic),
                           or 'custom' with custom fraction function.
            precision: Beta distribution precision (higher = less noise).
        """
        self.biomarker_type = biomarker_type
        self.precision = precision

    def _get_fraction(self, latent_state: np.ndarray) -> float:
        """Compute the target fraction from latent state."""
        total = float(np.sum(np.maximum(latent_state, 0)))
        if total <= 0:
            return 0.5

        if self.biomarker_type == "ki67":
            # Proliferative fraction: P / viable
            P = max(latent_state[0], 0)
            viable = total - (latent_state[2] if len(latent_state) > 2 else 0)
            return P / max(viable, 1e-6)
        elif self.biomarker_type == "caspase":
            # Apoptotic fraction: A / total
            A = latent_state[2] if len(latent_state) > 2 else 0
            return max(A, 0) / total
        else:
            return 0.5

    def log_likelihood(
        self,
        observed: float | np.ndarray,
        latent_state: np.ndarray,
        params: dict | None = None,
        process_variance: float | None = None,
    ) -> float:
        """Log-likelihood under Beta model."""
        f = self._get_fraction(latent_state)
        f = np.clip(f, 1e-4, 1 - 1e-4)
        kappa = self.precision
        if params and "biomarker_precision" in params:
            kappa = params["biomarker_precision"]

        a = kappa * f
        b = kappa * (1 - f)
        observed = np.clip(float(observed), 1e-4, 1 - 1e-4)
        return float(stats.beta.logpdf(observed, a, b))

    def sample(
        self,
        latent_state: np.ndarray,
        rng: np.random.Generator,
        params: dict | None = None,
    ) -> float:
        """Sample a biomarker observation."""
        f = self._get_fraction(latent_state)
        f = np.clip(f, 1e-4, 1 - 1e-4)
        kappa = self.precision
        a = kappa * f
        b = kappa * (1 - f)
        return float(rng.beta(a, b))

    def expected_value(self, latent_state: np.ndarray) -> float:
        return self._get_fraction(latent_state)

    def param_names(self) -> list[str]:
        return ["precision"]
