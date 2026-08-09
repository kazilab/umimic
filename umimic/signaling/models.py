"""Concrete signaling network models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from umimic.signaling.network import SignalingNetwork


@dataclass
class ToyMapkAktNetwork(SignalingNetwork):
    """Minimal 2-node MAPK/AKT signaling toy model.

    .. warning::
       This is a **scaffold for coupling and inference plumbing, not a pathway
       model**. Two nodes with linear drive and decay cannot represent MAPK or
       AKT regulation, and no parameter here is calibrated against data. Do not
       read biological conclusions out of it.

    Drug direction is explicit. Most targeted oncology agents (MEK, RAF, EGFR,
    PI3K, AKT inhibitors) *reduce* pathway activity, so `direction` defaults to
    ``"inhibitory"``. The previous behaviour was stimulatory: drug raised
    signalling, and because rate coupling multiplies by ``1 + effect``, higher
    concentration then *increased* proliferation -- backwards for an inhibitor.
    Set ``direction="stimulatory"`` deliberately if modelling an agonist or a
    relief-of-feedback effect.

    Nodes are held in [0, 1]; activity cannot go negative or unbounded.
    """

    mapk_baseline: float = 0.3
    akt_baseline: float = 0.25
    mapk_drive: float = 0.15
    akt_drive: float = 0.10
    decay: float = 0.5
    direction: str = "inhibitory"

    #: Parameter names the pipeline may forward from config.
    CONFIG_PARAM_NAMES = frozenset(
        {
            "mapk_baseline",
            "akt_baseline",
            "mapk_drive",
            "akt_drive",
            "decay",
            "direction",
        }
    )

    def __post_init__(self) -> None:
        if self.direction not in ("inhibitory", "stimulatory"):
            raise ValueError(
                f"direction must be 'inhibitory' or 'stimulatory', got "
                f"{self.direction!r}."
            )
        for name in ("mapk_baseline", "akt_baseline"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"{name} must lie in [0, 1], got {value}."
                )
            setattr(self, name, value)
        for name in ("mapk_drive", "akt_drive"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative, got {value}."
                )
            setattr(self, name, value)
        if not np.isfinite(self.decay) or self.decay <= 0.0:
            raise ValueError(
                f"decay must be a positive rate constant, got {self.decay}."
            )
        self.decay = float(self.decay)

    @property
    def _sign(self) -> float:
        return -1.0 if self.direction == "inhibitory" else 1.0

    @property
    def node_names(self) -> list[str]:
        return ["mapk", "akt"]

    def initial_state(self) -> np.ndarray:
        return np.array([self.mapk_baseline, self.akt_baseline], dtype=float)

    def rhs(self, t: float, y: np.ndarray, concentration: float) -> np.ndarray:
        """Relaxation toward a drug-shifted set point.

        Each node decays toward its baseline and is pushed away from it by the
        drug, in the direction given by `direction`. Writing it as relaxation
        to a set point (rather than an unbounded drive term) keeps activity in
        [0, 1] instead of growing without limit at high concentration.
        """
        mapk, akt = np.clip(np.asarray(y, dtype=float), 0.0, 1.0)
        c = max(float(concentration), 0.0)
        sign = self._sign

        mapk_target = np.clip(self.mapk_baseline + sign * self.mapk_drive * c, 0.0, 1.0)
        akt_target = np.clip(self.akt_baseline + sign * self.akt_drive * c, 0.0, 1.0)

        d_mapk = self.decay * (mapk_target - mapk)
        d_akt = self.decay * (akt_target - akt)
        return np.array([d_mapk, d_akt], dtype=float)
