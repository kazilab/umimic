"""YAML configuration loading and validation using pydantic."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from umimic.inference.likelihood import PARAMETER_SETS

logger = logging.getLogger(__name__)

_COUPLING_TARGET_PATTERN = re.compile(
    r"^(birth|death|transition|death:[PQAR]|transition:[PQAR]->[PQAR])$"
)


def _is_valid_coupling_target(token: str) -> bool:
    return _COUPLING_TARGET_PATTERN.fullmatch(token) is not None


class DynamicsConfig(BaseModel):
    """Configuration for cell-state dynamics."""

    states: list[str] = ["P", "Q"]

    # Explicit phenotype transitions as [source, target] pairs. When omitted,
    # the canonical edges among the configured states are used (P<->Q, P->R).
    # Setting this is the only way to express routes the canonical set lacks,
    # notably the persister route Q->R; previously the edge set was derived
    # from the state count and could not be changed at all.
    transitions: list[tuple[str, str]] | None = None

    # Baseline transition rates (1/hour), keyed "P->Q". Edges present in
    # `transitions` but absent here default to 0, which is deliberate for the
    # persister route: Q->R should be driven by drug induction rather than a
    # silent non-zero baseline.
    transition_rates: dict[str, float] | None = None

    # Drug-dependent transition components, keyed "P->Q" like transition_rates.
    #   induced_transitions: additive rate created by drug, so a route absent
    #     without treatment (baseline 0) can appear under it. This is the only
    #     way to express drug-induced plasticity such as P->Q persistence.
    #   transition_fold_change: multiplier on the baseline; values below 1 at
    #     saturation mean the drug suppresses the route (e.g. blocking Q->P
    #     resensitisation).
    # Each maps an edge to Emax/EC50/Hill (induction) or low/high/EC50/Hill
    # (fold change).
    induced_transitions: dict[str, dict[str, float]] | None = None
    transition_fold_change: dict[str, dict[str, float]] | None = None

    # States that divide. Defaults to P plus R when present. Persister models
    # need Q here too: a slow-cycling drug-tolerant state is not arrested, and
    # leaving it out makes the same conceptual model behave differently
    # depending on whether it was built from config or from
    # ModelTopology.persister_resistance().
    division_states: list[str] | None = None

    density_dependent: bool = False
    carrying_capacity: float | None = None
    # Phase-type (Erlang) dwell-time stages per state. NOT IMPLEMENTED: no
    # solver consumes this, so setting it would silently have no dynamical
    # effect. It is rejected rather than ignored -- see the validator below.
    linear_chain: dict[str, int] | None = None
    clearance_rate: float = 0.1

    # Drug mechanism. "cytotoxic" raises death, "cytostatic" lowers birth,
    # "mixed" does both. None leaves rates unmodulated.
    drug_mechanism: Literal["cytotoxic", "cytostatic", "mixed"] | None = None
    emax_death: float = 0.05
    ec50_death: float = 1.0
    hill_death: float = 1.5
    emax_birth: float = 0.8
    ec50_birth: float = 1.0
    hill_birth: float = 1.5
    default_birth_base: float = Field(
        0.04,
        description=(
            "Default baseline proliferation rate (1/hour) used for simulation when no "
            "fitted parameters are supplied."
        ),
    )
    default_death_base_p: float = Field(
        0.01,
        description=(
            "Default baseline death rate (1/hour) for proliferative state P, used in "
            "forward simulation defaults."
        ),
    )
    default_death_base_q: float = Field(
        0.005,
        description=(
            "Default baseline death rate (1/hour) for quiescent state Q, used in "
            "forward simulation defaults."
        ),
    )
    default_death_base_r: float = Field(
        0.01,
        description=(
            "Default baseline death rate (1/hour) for resistant state R. R is "
            "drug-insensitive but not immortal; leaving it at zero would make "
            "the resistant compartment an inert sink."
        ),
    )
    resistant_fitness_cost: float = Field(
        0.0,
        ge=0.0,
        lt=1.0,
        description=(
            "Fractional reduction of R's division rate relative to P, i.e. the "
            "fitness cost of carrying resistance. 0 means R grows as fast as "
            "an untreated sensitive cell."
        ),
    )
    quiescent_sensitivity: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Q's drug-induced death relative to P's, in [0, 1]. An explicit "
            "modelling choice: 0 treats quiescent cells as fully refractory, "
            "which suits cell-cycle-specific agents but understates kill for "
            "agents active against non-cycling cells."
        ),
    )
    initial_fractions: str | dict[str, float] | None = Field(
        None,
        description=(
            "How the initial cell count is split across states. None or "
            "'proliferating' puts every cell in P -- a structural assumption "
            "that biases early growth and transition-rate estimates. 'stable' "
            "uses the relaxed state distribution; a mapping gives explicit "
            "fractions."
        ),
    )
    density_counts_apoptotic: bool = Field(
        False,
        description=(
            "Whether apoptotic cells occupy space in the density-dependent "
            "growth term. Counting corpses suppresses division until they are "
            "cleared, producing artefactual growth inhibition after a "
            "cytotoxic pulse."
        ),
    )

    @field_validator(
        "clearance_rate",
        "default_birth_base",
        "default_death_base_p",
        "default_death_base_q",
        "default_death_base_r",
        "emax_death",
        "emax_birth",
    )
    @classmethod
    def _non_negative_rates(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Rate parameters must be non-negative")
        return v

    @field_validator("ec50_death", "ec50_birth", "hill_death", "hill_birth")
    @classmethod
    def _positive_dose_response(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("EC50 and Hill parameters must be > 0")
        return v

    @field_validator("carrying_capacity")
    @classmethod
    def _non_negative_optional_capacity(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("carrying_capacity must be non-negative")
        return v

    @field_validator("states")
    @classmethod
    def _recognized_unique_states(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one cell state must be configured")
        known = {"P", "Q", "A", "R"}
        unknown = [s for s in v if s not in known]
        if unknown:
            raise ValueError(
                f"Unknown cell state(s) {unknown}; expected a subset of {sorted(known)}"
            )
        if len(set(v)) != len(v):
            raise ValueError(f"Duplicate cell states in {v}")
        if "P" not in v:
            raise ValueError(
                "State 'P' (proliferating) is required: it carries the "
                "initial condition and the division reaction"
            )
        return v

    @model_validator(mode="after")
    def _check_transitions(self) -> DynamicsConfig:
        """Transitions must connect configured, non-apoptotic states."""
        if self.transitions is None:
            return self

        seen: set[tuple[str, str]] = set()
        for edge in self.transitions:
            if len(edge) != 2:
                raise ValueError(
                    f"Each transition must be a [source, target] pair, got {edge!r}"
                )
            src, tgt = edge
            for state in (src, tgt):
                if state not in self.states:
                    raise ValueError(
                        f"Transition {src}->{tgt} refers to state {state!r}, "
                        f"which is not in states={self.states}"
                    )
            if src == tgt:
                raise ValueError(f"Transition {src}->{tgt} is a self-loop")
            if "A" in (src, tgt):
                raise ValueError(
                    f"Transition {src}->{tgt} involves the apoptotic state. A "
                    "is a sink fed by death and cleared by clearance_rate; it "
                    "is not reached by phenotype transitions."
                )
            if (src, tgt) in seen:
                raise ValueError(f"Duplicate transition {src}->{tgt}")
            seen.add((src, tgt))

        for field_name, required in (
            ("transition_rates", ()),
            ("induced_transitions", ("emax",)),
            ("transition_fold_change", ("high",)),
        ):
            entries = getattr(self, field_name)
            if not entries:
                continue
            for key, value in entries.items():
                if "->" not in key:
                    raise ValueError(
                        f"{field_name} key {key!r} must be of the form 'P->Q'"
                    )
                src, tgt = (part.strip() for part in key.split("->", 1))
                if (src, tgt) not in seen:
                    raise ValueError(
                        f"{field_name} specifies {key!r}, which is not in "
                        f"transitions={self.transitions}"
                    )
                if field_name == "transition_rates":
                    if value < 0:
                        raise ValueError(
                            f"transition_rates[{key!r}] must be non-negative"
                        )
                    continue
                missing = [p for p in required if p not in value]
                if missing:
                    raise ValueError(
                        f"{field_name}[{key!r}] is missing {missing}"
                    )
                for pname, pval in value.items():
                    if pval < 0:
                        raise ValueError(
                            f"{field_name}[{key!r}][{pname!r}] must be "
                            "non-negative"
                        )

        if ("R", "Q") in seen or ("R", "P") in seen:
            logger.warning(
                "Transitions out of R are configured. A per-cell reversion "
                "rate is meaningful only if R is an epigenetically stable "
                "state; for a genetic R, resensitisation is competitive "
                "dilution already carried by R's birth and death rates, and "
                "modelling it as a transition double-counts it. Reversion is "
                "also close to unidentifiable from aggregate observables."
            )
        return self

    @model_validator(mode="after")
    def _check_initial_fractions(self) -> DynamicsConfig:
        spec = self.initial_fractions
        if spec is None:
            return self
        if isinstance(spec, str):
            if spec not in ("proliferating", "stable"):
                raise ValueError(
                    f"initial_fractions must be 'proliferating', 'stable', or "
                    f"a mapping of state to fraction; got {spec!r}"
                )
            return self
        unknown = [s for s in spec if s not in self.states]
        if unknown:
            raise ValueError(
                f"initial_fractions refers to state(s) {unknown} not in "
                f"states={self.states}"
            )
        if any(v < 0 for v in spec.values()):
            raise ValueError("initial_fractions must be non-negative")
        if sum(spec.values()) <= 0:
            raise ValueError("initial_fractions must sum to a positive value")
        return self

    @model_validator(mode="after")
    def _check_division_states(self) -> DynamicsConfig:
        if self.division_states is None:
            return self
        unknown = [s for s in self.division_states if s not in self.states]
        if unknown:
            raise ValueError(
                f"division_states {unknown} are not in states={self.states}"
            )
        if "A" in self.division_states:
            raise ValueError("The apoptotic state cannot divide")
        if not self.division_states:
            raise ValueError("At least one state must divide")
        return self

    @model_validator(mode="after")
    def _check_density_dependence(self) -> DynamicsConfig:
        if self.density_dependent and not self.carrying_capacity:
            raise ValueError(
                "density_dependent=True requires a positive carrying_capacity"
            )
        return self

    @field_validator("linear_chain")
    @classmethod
    def _reject_unimplemented_linear_chain(
        cls, v: dict[str, int] | None
    ) -> dict[str, int] | None:
        """Refuse a setting no solver reads.

        Phase-type dwell times are documented on ModelTopology but are not
        consumed by the ODE, LNA, Gillespie or tau-leaping engines. Accepting
        the option would let a user believe they had configured Erlang dwell
        times while the model still used exponential ones.
        """
        if v:
            raise ValueError(
                "linear_chain (phase-type dwell times) is not implemented in "
                "this release: no solver consumes it, so setting it would have "
                "no effect on the dynamics. Remove it from the configuration; "
                "all states currently have exponential dwell times."
            )
        return v


class PKConfig(BaseModel):
    """Configuration for pharmacokinetic model."""

    model: Literal["none", "one_compartment", "two_compartment"] = "none"
    vd: float = 10.0
    ke: float = 0.1
    ka: float | None = None
    vc: float | None = None
    vp: float | None = None
    cl: float | None = None
    q: float | None = None

    @field_validator("vd", "ke")
    @classmethod
    def _positive_pk_required(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("PK parameters vd and ke must be > 0")
        return v

    @field_validator("ka", "vc", "vp", "cl", "q")
    @classmethod
    def _positive_pk_optional(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("Optional PK parameters must be > 0 when provided")
        return v


class DosingConfig(BaseModel):
    """Configuration for dosing schedule."""

    type: Literal["constant", "single_bolus", "repeated_bolus", "oral"] = "constant"
    # Default is an explicit drug-free control rather than an implicit fallback
    # applied later, so the effective value is visible in the configuration.
    concentrations: list[float] | None = Field(default_factory=lambda: [0.0])
    dose_amount: float | None = None
    interval: float | None = None
    n_doses: int | None = None
    start_time: float = 0.0

    @field_validator("concentrations")
    @classmethod
    def check_concentrations(cls, v: list[float] | None) -> list[float] | None:
        if v and any(c < 0 for c in v):
            raise ValueError("Concentrations must be non-negative")
        return v

    @field_validator("start_time")
    @classmethod
    def _non_negative_start_time(cls, v: float) -> float:
        if v < 0:
            raise ValueError("start_time must be non-negative")
        return v

    @field_validator("dose_amount")
    @classmethod
    def _positive_optional_dose_amount(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("dose_amount must be > 0 when provided")
        return v

    @field_validator("interval")
    @classmethod
    def _positive_optional_interval(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("interval must be > 0 when provided")
        return v

    @field_validator("n_doses")
    @classmethod
    def _positive_optional_n_doses(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("n_doses must be > 0 when provided")
        return v

    @model_validator(mode="after")
    def _require_fields_for_type(self) -> DosingConfig:
        """Each dosing type must carry the fields it actually needs.

        Filling these in downstream with `value or default` would turn an
        explicit 0 into a silent 100 and hide an incomplete configuration.
        """
        required = {
            "constant": ("concentrations",),
            "single_bolus": ("dose_amount",),
            "repeated_bolus": ("dose_amount", "interval", "n_doses"),
            "oral": ("dose_amount", "interval", "n_doses"),
        }[self.type]

        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"dosing type {self.type!r} requires {missing} to be set"
            )
        if self.type == "constant" and not self.concentrations:
            raise ValueError("dosing type 'constant' requires a non-empty "
                             "concentrations list")
        return self


class ObservationConfig(BaseModel):
    """Configuration for observation models."""

    modalities: list[str] = ["cell_counts"]
    cell_count_overdispersion: float = 10.0
    bli_alpha: float = 1000.0
    bli_sigma_log: float = 0.3
    # Volume per cell (mm^3/cell). Tumour cellularity is roughly 1e5-1e6
    # cells/mm^3 (a ~15-20 um cell is ~2e-6 mm^3 before packing and stroma),
    # so the physically plausible range is about 1e-6 to 1e-5 mm^3/cell.
    # The previous default of 1e-3 implied only 1000 cells/mm^3 -- about two
    # orders of magnitude too few -- which broke the absolute calibration
    # between volume and cell number and any BLI-volume coupling relying on it.
    volume_beta: float = 1e-5
    volume_sigma: float = 0.2
    biomarker_precision: float = 50.0
    biomarker_type: Literal["ki67", "caspase"] = "ki67"

    @field_validator("modalities")
    @classmethod
    def _known_modalities(cls, v: list[str]) -> list[str]:
        """Reject modalities the orchestrator cannot build.

        An unrecognised name previously fell through the builder loop
        silently, so a typo -- or `biomarker`, which had no branch at all --
        produced a model with that modality quietly missing.
        """
        known = {"cell_counts", "bli", "volume", "biomarker"}
        unknown = [m for m in v if m not in known]
        if unknown:
            raise ValueError(
                f"Unknown observation modality/modalities {unknown}; expected "
                f"a subset of {sorted(known)}"
            )
        if not v:
            raise ValueError("At least one observation modality is required")
        if len(set(v)) != len(v):
            raise ValueError(f"Duplicate modalities in {v}")
        return v


class SignalingConfig(BaseModel):
    """Configuration for intracellular signaling dynamics."""

    enabled: bool = False
    model: Literal["none", "toy_mapk_akt"] = "none"
    initial_state: dict[str, float] = Field(default_factory=dict)
    parameters: dict[str, float] = Field(default_factory=dict)
    observed_nodes: list[str] = Field(default_factory=list)

    @field_validator("initial_state", "parameters")
    @classmethod
    def _finite_values(cls, v: dict[str, float]) -> dict[str, float]:
        if not all(np.isfinite(val) for val in v.values()):
            raise ValueError("Signaling state/parameter values must be finite")
        return v


class CouplingConfig(BaseModel):
    """Configuration for signaling-to-fate coupling functions."""

    enabled: bool = False
    function: Literal["hill", "logistic"] = "hill"
    targets: list[str] = Field(default_factory=list)
    parameters: dict[str, float] = Field(default_factory=dict)
    max_effect_by_target: dict[str, float] = Field(default_factory=dict)
    ec50_by_target: dict[str, float] = Field(default_factory=dict)
    hill_by_target: dict[str, float] = Field(default_factory=dict)
    k_by_target: dict[str, float] = Field(default_factory=dict)
    center_by_target: dict[str, float] = Field(default_factory=dict)

    @staticmethod
    def recommended_parameter_ranges() -> dict[str, dict[str, tuple[float, float]]]:
        """Practical starter ranges for coupling parameter tuning.

        These ranges are heuristics to initialize searches; they are not strict
        biological constraints.
        """
        return {
            "hill": {
                "max_effect": (0.1, 2.0),
                "ec50": (1e-3, 10.0),
                "hill": (0.5, 4.0),
            },
            "logistic": {
                "max_effect": (0.1, 2.0),
                "k": (0.1, 10.0),
                "center": (0.0, 1.0),
            },
            "target_overrides": {
                "max_effect_by_target": (0.0, 3.0),
                "ec50_by_target": (1e-3, 20.0),
                "hill_by_target": (0.5, 5.0),
                "k_by_target": (0.1, 20.0),
                "center_by_target": (-1.0, 2.0),
            },
        }

    @field_validator("parameters")
    @classmethod
    def _finite_parameters(cls, v: dict[str, float]) -> dict[str, float]:
        if not all(np.isfinite(val) for val in v.values()):
            raise ValueError("Coupling parameter values must be finite")
        return v

    @field_validator("targets")
    @classmethod
    def _validate_targets(cls, v: list[str]) -> list[str]:
        allowed_prefix = ("birth", "death", "transition")
        for token in v:
            if not token or not token.startswith(allowed_prefix):
                raise ValueError(f"Invalid coupling target: {token}")
            if not _is_valid_coupling_target(token):
                raise ValueError(
                    f"Invalid coupling target format: {token}. "
                    "Use birth, death, transition, death:<STATE>, "
                    "or transition:<SRC->TGT>."
                )
        return v

    @field_validator("max_effect_by_target")
    @classmethod
    def _validate_max_effect_by_target(
        cls, v: dict[str, float]
    ) -> dict[str, float]:
        for token, value in v.items():
            if not _is_valid_coupling_target(token):
                raise ValueError(
                    f"Invalid max_effect_by_target key: {token}. "
                    "Use birth, death, transition, death:<STATE>, "
                    "or transition:<SRC->TGT>."
                )
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"max_effect_by_target values must be finite and non-negative ({token})"
                )
        return v

    @field_validator("ec50_by_target")
    @classmethod
    def _validate_ec50_by_target(
        cls, v: dict[str, float]
    ) -> dict[str, float]:
        for token, value in v.items():
            if not _is_valid_coupling_target(token):
                raise ValueError(
                    f"Invalid ec50_by_target key: {token}. "
                    "Use birth, death, transition, death:<STATE>, "
                    "or transition:<SRC->TGT>."
                )
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"ec50_by_target values must be finite and > 0 ({token})"
                )
        return v

    @field_validator("hill_by_target")
    @classmethod
    def _validate_hill_by_target(
        cls, v: dict[str, float]
    ) -> dict[str, float]:
        for token, value in v.items():
            if not _is_valid_coupling_target(token):
                raise ValueError(
                    f"Invalid hill_by_target key: {token}. "
                    "Use birth, death, transition, death:<STATE>, "
                    "or transition:<SRC->TGT>."
                )
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"hill_by_target values must be finite and > 0 ({token})"
                )
        return v

    @field_validator("k_by_target")
    @classmethod
    def _validate_k_by_target(
        cls, v: dict[str, float]
    ) -> dict[str, float]:
        for token, value in v.items():
            if not _is_valid_coupling_target(token):
                raise ValueError(
                    f"Invalid k_by_target key: {token}. "
                    "Use birth, death, transition, death:<STATE>, "
                    "or transition:<SRC->TGT>."
                )
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"k_by_target values must be finite and > 0 ({token})"
                )
        return v

    @field_validator("center_by_target")
    @classmethod
    def _validate_center_by_target(
        cls, v: dict[str, float]
    ) -> dict[str, float]:
        for token, value in v.items():
            if not _is_valid_coupling_target(token):
                raise ValueError(
                    f"Invalid center_by_target key: {token}. "
                    "Use birth, death, transition, death:<STATE>, "
                    "or transition:<SRC->TGT>."
                )
            if not np.isfinite(value):
                raise ValueError(
                    f"center_by_target values must be finite ({token})"
                )
        return v


class InferenceConfig(BaseModel):
    """Configuration for inference engine."""

    # Modes Experiment.fit() actually dispatches. "smc" and "hierarchical"
    # were previously accepted here but unimplemented in the orchestrator, so
    # they failed at run time with "Unknown inference mode" after the data had
    # been loaded. The ParticleMCMC and hierarchical estimators remain
    # available directly from umimic.inference.
    mode: Literal["mle", "mcmc"] = "mle"
    # "pymc" is deliberately absent: the backend was withdrawn in 0.0.4
    # because its model ignored the data.
    backend: Literal["scipy", "emcee", "particle"] = "scipy"

    # Forward model used to evaluate the likelihood. "moment" (LNA) also
    # supplies the process variance, which is what carries the mechanistic
    # birth-versus-death signature; "ode" is faster but discards it. This is
    # independent of `mode`: tying it to MLE-vs-MCMC silently disabled the
    # variance signature on the default path.
    forward_mode: Literal["moment", "ode"] = "moment"

    # Parameters to estimate. None selects the cytotoxic-only default set,
    # which cannot distinguish cytostatic from cytotoxic action because it
    # contains no birth-modulation terms; use "mechanism" for a set that can.
    parameter_set: Literal["default", "mechanism", "resistance", "persister"] = (
        "default"
    )

    n_samples: int = 2000
    n_chains: int = 4
    n_warmup: int = 1000
    #: Only read by :class:`umimic.inference.ParticleMCMC`, which the pipeline
    #: does not dispatch; see the backend validator below.
    n_particles: int = 500
    n_restarts: int = 5

    @field_validator("n_samples", "n_chains", "n_warmup", "n_particles", "n_restarts")
    @classmethod
    def _positive_counts(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Inference counts must be > 0")
        return v

    @model_validator(mode="after")
    def _backend_is_dispatched(self) -> InferenceConfig:
        """Reject a backend the orchestrator does not actually run.

        Experiment.fit() silently substituted emcee for "particle", so a
        config asking for particle inference got gradient-free MAP or an
        ensemble sampler instead and reported success. A backend that is not
        dispatched has to fail here, before any data is loaded.
        """
        if self.backend == "particle":
            raise ValueError(
                "backend='particle' is not dispatched by Experiment.fit(): it "
                "would silently run emcee (mode='mcmc') or Nelder-Mead MAP "
                "(mode='mle') instead. Particle inference is available "
                "directly as umimic.inference.ParticleMCMC, which takes a "
                "single TimeSeriesData."
            )
        return self


class PriorConfig(BaseModel):
    """Configuration for prior distributions."""

    b0: dict[str, float] = {"dist_param_scale": 0.04, "dist_param_s": 0.5}
    d0_P: dict[str, float] = {"dist_param_scale": 0.01, "dist_param_s": 0.5}
    emax_death: dict[str, float] = {"dist_param_scale": 0.1}
    ec50_death: dict[str, float] = {"dist_param_scale": 1.0, "dist_param_s": 1.0}
    hill_death: dict[str, float] = {"dist_param_scale": 1.5, "dist_param_s": 0.3}


class DataConfig(BaseModel):
    """Configuration for data loading."""

    format: str = "csv"
    path: str | None = None
    time_column: str = "hours"
    count_column: str = "viable_count"
    concentration_column: str = "concentration"
    replicate_column: str = "replicate_id"


class SimulationConfig(BaseModel):
    """Configuration for synthetic data generation."""

    method: Literal["ode", "gillespie", "tau_leaping"] = "gillespie"
    initial_cells: int = 100
    t_max: float = 72.0
    dt_obs: float = 4.0
    n_replicates: int = 4
    seed: int = 42

    @field_validator("initial_cells", "n_replicates")
    @classmethod
    def _positive_ints(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("initial_cells and n_replicates must be > 0")
        return v

    @field_validator("t_max", "dt_obs")
    @classmethod
    def _positive_times(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("t_max and dt_obs must be > 0")
        return v

    @field_validator("seed")
    @classmethod
    def _non_negative_seed(cls, v: int) -> int:
        if v < 0:
            raise ValueError("seed must be non-negative")
        return v


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    name: str = "experiment"
    context: Literal["in_vitro", "in_vivo"] = "in_vitro"
    dynamics: DynamicsConfig = DynamicsConfig()
    pk: PKConfig = PKConfig()
    dosing: DosingConfig = DosingConfig()
    observations: ObservationConfig = ObservationConfig()
    inference: InferenceConfig = InferenceConfig()
    priors: PriorConfig = PriorConfig()
    data: DataConfig = DataConfig()
    simulation: SimulationConfig = SimulationConfig()
    signaling: SignalingConfig = SignalingConfig()
    coupling: CouplingConfig = CouplingConfig()
    seed: int = 42

    @model_validator(mode="after")
    def _check_inference_matches_the_model(self) -> ExperimentConfig:
        """Cross-section checks that stop quietly wrong fits.

        Each of these parses fine in isolation and produces a converged fit
        that answers the wrong question.
        """
        inference = self.inference
        params = PARAMETER_SETS[inference.parameter_set]

        # A cytostatic or mixed mechanism cannot be recovered by a parameter
        # set with no birth-modulation terms; the fit will attribute the
        # growth reduction to death instead.
        if self.dynamics.drug_mechanism in ("cytostatic", "mixed") and not any(
            "birth" in name for name in params
        ):
            raise ValueError(
                f"drug_mechanism={self.dynamics.drug_mechanism!r} acts on "
                f"division, but inference.parameter_set="
                f"{inference.parameter_set!r} contains no birth-modulation "
                "parameters, so the fit would attribute the effect to death. "
                "Use parameter_set='mechanism'."
            )

        # Separating cytostatic from cytotoxic action relies on the LNA
        # process variance: the mean confounds them, since both reduce net
        # growth. Fitting the mechanism set without it is under-identified.
        if inference.parameter_set == "mechanism" and inference.forward_mode == "ode":
            raise ValueError(
                "parameter_set='mechanism' needs forward_mode='moment'. The "
                "mean alone cannot separate reduced division from increased "
                "death; it is the LNA process variance (scaling with b + d "
                "while the mean scales with b - d) that breaks the degeneracy."
            )

        # NOTE: whether the parameter set covers every active state is checked
        # by ModelLikelihood at fit time, not here. A configuration that only
        # ever simulates is legitimate -- the simulation side parameterises R
        # correctly from `dynamics.*` -- so rejecting it at construction would
        # block valid forward-only use.

        return self


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to YAML config file.

    Returns:
        Validated ExperimentConfig.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML is empty or cannot be parsed into a valid config.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Provide a valid YAML config file. Example minimal config:\n\n"
            "  name: my_experiment\n"
            "  dynamics:\n"
            "    states: [P, Q]\n"
            "  dosing:\n"
            "    concentrations: [0, 0.1, 1, 10]\n"
        )

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(
            f"Config file is empty: {path}\n"
            "The file must contain valid YAML. See README.md for examples."
        )

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping (got {type(raw).__name__}): {path}"
        )

    return ExperimentConfig(**raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Save configuration to a YAML file.

    Dumped in JSON mode so that tuple-valued fields (such as
    `dynamics.transitions`) become plain lists. `yaml.dump` would otherwise
    emit `!!python/tuple` tags, which `yaml.safe_load` refuses to read -- the
    config would save cleanly and then fail to load.
    """
    path = Path(path)
    with open(path, "w") as f:
        yaml.dump(
            config.model_dump(mode="json"),
            f,
            default_flow_style=False,
            sort_keys=False,
        )
