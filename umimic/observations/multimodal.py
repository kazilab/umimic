"""Composite multimodal observation model.

Combines multiple observation modalities under the assumption of
conditional independence given the latent state:

    p(Y_counts, Y_BLI, Y_volume | X) = p(Y_counts|X) * p(Y_BLI|X) * p(Y_volume|X)

This is the key multimodal fusion capability of U-MIMIC.
"""

from __future__ import annotations

import numpy as np

from umimic.observations.base import ObservationModel


class MultimodalObservation:
    """Composite observation model combining multiple modalities.

    Joint log-likelihood is the sum of individual log-likelihoods
    (conditional independence given latent state).

    Gracefully handles missing modalities: if a modality has no
    observation at a time point, it is simply excluded from the sum.
    """

    def __init__(self, models: dict[str, ObservationModel]):
        """
        Args:
            models: Dict mapping modality name to ObservationModel.
                   Example: {"cell_counts": CellCountObservation(...),
                             "bli": BLIObservation(...),
                             "volume": TumorVolumeObservation(...)}
        """
        self.models = models

    def log_likelihood(
        self,
        observations: dict[str, float | None],
        latent_state: np.ndarray,
        params: dict | None = None,
    ) -> float:
        """Joint log-likelihood across all available modalities.

        Args:
            observations: Dict mapping modality name to observed value.
                         None or missing entries are skipped.
            latent_state: Latent cell population state vector.
            params: Additional parameters.
        """
        total = 0.0
        for name, model in self.models.items():
            if name in observations and observations[name] is not None:
                ll = model.log_likelihood(observations[name], latent_state, params)
                if np.isfinite(ll):
                    total += ll
                else:
                    return -np.inf
        return total

    def sample(
        self,
        latent_state: np.ndarray,
        rng: np.random.Generator,
        params: dict | None = None,
    ) -> dict[str, float]:
        """Generate synthetic observations from all modalities."""
        return {
            name: model.sample(latent_state, rng, params)
            for name, model in self.models.items()
        }

    def expected_values(self, latent_state: np.ndarray) -> dict[str, float]:
        """Expected observation from each modality."""
        result = {}
        for name, model in self.models.items():
            try:
                result[name] = model.expected_value(latent_state)
            except NotImplementedError:
                pass
        return result

    @property
    def modality_names(self) -> list[str]:
        return list(self.models.keys())

    def param_names(self) -> list[str]:
        all_params = []
        for name, model in self.models.items():
            for p in model.param_names():
                all_params.append(f"{name}.{p}")
        return all_params
