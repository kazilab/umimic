"""Interfaces for intracellular signaling network models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SignalingNetwork(ABC):
    """Abstract signaling network dynamics model."""

    @property
    @abstractmethod
    def node_names(self) -> list[str]:
        """Ordered names for state vector entries."""

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        """Return initial node activity vector."""

    @abstractmethod
    def rhs(self, t: float, y: np.ndarray, concentration: float) -> np.ndarray:
        """Time derivative for signaling node activities."""

    def observable_map(self, y: np.ndarray) -> dict[str, float]:
        """Map raw signaling state into named observable activities."""
        return {
            name: float(y[i])
            for i, name in enumerate(self.node_names)
        }
