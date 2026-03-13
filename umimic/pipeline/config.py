"""YAML configuration loading and validation using pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class DynamicsConfig(BaseModel):
    """Configuration for cell-state dynamics."""

    states: list[str] = ["P", "Q"]
    density_dependent: bool = False
    carrying_capacity: float | None = None
    linear_chain: dict[str, int] | None = None
    clearance_rate: float = 0.1


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


class DosingConfig(BaseModel):
    """Configuration for dosing schedule."""

    type: Literal["constant", "single_bolus", "repeated_bolus", "oral"] = "constant"
    concentrations: list[float] | None = None
    dose_amount: float | None = None
    interval: float | None = None
    n_doses: int | None = None
    start_time: float = 0.0


class ObservationConfig(BaseModel):
    """Configuration for observation models."""

    modalities: list[str] = ["cell_counts"]
    cell_count_overdispersion: float = 10.0
    bli_alpha: float = 1000.0
    bli_sigma_log: float = 0.3
    volume_beta: float = 1e-3
    volume_sigma: float = 0.2
    biomarker_precision: float = 50.0


class InferenceConfig(BaseModel):
    """Configuration for inference engine."""

    mode: Literal["mle", "mcmc", "smc", "hierarchical"] = "mle"
    backend: Literal["scipy", "emcee", "pymc", "particle"] = "scipy"
    n_samples: int = 2000
    n_chains: int = 4
    n_warmup: int = 1000
    n_particles: int = 500
    n_restarts: int = 5


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
    seed: int = 42


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
    """Save configuration to YAML file."""
    path = Path(path)
    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
