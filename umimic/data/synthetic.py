"""Synthetic data generation from the U-MIMIC model."""

from __future__ import annotations

from typing import Callable

import numpy as np

from umimic.dynamics.states import ModelTopology
from umimic.dynamics.rates import RateSet
from umimic.dynamics.gillespie import GillespieSimulator
from umimic.dynamics.ode_system import CellDynamicsODE
from umimic.observations.base import ObservationModel, TopologyAwareObservation
from umimic.observations.cell_counts import CellCountObservation
from umimic.data.schemas import TimeSeriesData, ExperimentalDataset


class SyntheticDataGenerator:
    """Generate synthetic datasets from the U-MIMIC model.

    Useful for:
    - Validating inference pipelines (parameter recovery)
    - Sensitivity analysis
    - Power analysis (experimental design)
    """

    def __init__(
        self,
        rate_set: RateSet,
        topology: ModelTopology,
        observation_model: ObservationModel | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.rate_set = rate_set
        self.topology = topology
        self.observation_model = observation_model or CellCountObservation(
            overdispersion=10.0, topology=topology
        )
        self.rng = rng or np.random.default_rng(42)

        # Normalize to modality -> model so generation and inference draw on
        # the same observation models. Hand-rolled sampling formulas here
        # would be a second definition of each modality, free to drift from
        # the one the likelihood uses.
        from umimic.observations.multimodal import MultimodalObservation

        if isinstance(self.observation_model, MultimodalObservation):
            self._modality_models = dict(self.observation_model.models)
        else:
            name = getattr(self.observation_model, "modality_name", "cell_counts")
            self._modality_models = {name: self.observation_model}

        for model in self._modality_models.values():
            if isinstance(model, TopologyAwareObservation):
                model.set_topology(topology)

    @property
    def modalities(self) -> list[str]:
        """Modalities this generator can emit."""
        return list(self._modality_models)

    def _resolve_modalities(self, requested: list[str] | None) -> list[str]:
        """Validate a requested modality list against the configured models."""
        if requested is None:
            return self.modalities
        unknown = [m for m in requested if m not in self._modality_models]
        if unknown:
            raise ValueError(
                f"No observation model configured for modality/modalities "
                f"{unknown}. Available: {self.modalities}. Pass a "
                "MultimodalObservation to the generator to emit more."
            )
        return list(requested)

    def _state_at(self, result, index: int) -> np.ndarray:
        """Latent state vector at a recorded time index."""
        return np.array(
            [result.populations[ct.name][index] for ct in self.topology.active_states]
        )

    def _sample_observations(
        self, result, n_times: int, modalities: list[str]
    ) -> dict[str, np.ndarray]:
        """Draw synthetic observations from the configured models."""
        observations = {m: np.zeros(n_times) for m in modalities}
        for i in range(n_times):
            state = self._state_at(result, i)
            for modality in modalities:
                observations[modality][i] = self._modality_models[modality].sample(
                    state, self.rng
                )
        return observations

    def generate_invitro_plate(
        self,
        initial_cells: np.ndarray | None = None,
        concentrations: list[float] | None = None,
        n_wells_per_dose: int = 4,
        t_max: float = 72.0,
        dt_obs: float = 4.0,
        method: str = "gillespie",
        modalities: list[str] | None = None,
    ) -> ExperimentalDataset:
        """Simulate a typical in vitro dose-response plate experiment.

        Args:
            initial_cells: Initial state vector (default: [100, 0] for P, Q).
            concentrations: Drug concentrations to simulate.
            n_wells_per_dose: Number of replicate wells per concentration.
            t_max: Total experiment duration (hours).
            dt_obs: Observation interval (hours).
            method: Simulation method ('gillespie', 'tau_leaping', 'ode').
            modalities: Which observations to emit. Defaults to every modality
                the generator's observation model provides -- previously this
                path emitted cell counts only, even when the generator was
                built with a multimodal observation model, so a multimodal
                configuration silently produced single-modality data.

        Returns:
            ExperimentalDataset with one TimeSeriesData per well.
        """
        modalities = self._resolve_modalities(modalities)
        if initial_cells is None:
            initial_cells = np.zeros(self.topology.n_states)
            initial_cells[0] = 100  # 100 proliferating cells

        if concentrations is None:
            concentrations = [0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

        t_record = np.arange(0, t_max + dt_obs, dt_obs)
        series = []

        for conc in concentrations:
            def exposure_fn(t, c=conc):
                return c

            for well_idx in range(n_wells_per_dose):
                if method == "gillespie":
                    sim = GillespieSimulator(
                        self.rate_set, self.topology, exposure_fn, self.rng
                    )
                    result = sim.simulate(initial_cells, t_max, t_record)
                elif method == "ode":
                    ode = CellDynamicsODE(self.rate_set, self.topology, exposure_fn)
                    result = ode.solve(initial_cells, (0, t_max), t_record)
                else:
                    from umimic.dynamics.tau_leaping import TauLeapingSimulator

                    sim = TauLeapingSimulator(
                        self.rate_set, self.topology, exposure_fn, rng=self.rng
                    )
                    result = sim.simulate(initial_cells, t_max, t_record)

                # Generate noisy observations from the latent trajectory,
                # using the configured observation models.
                observations = self._sample_observations(
                    result, len(t_record), modalities
                )

                ts = TimeSeriesData(
                    times=t_record,
                    observations=observations,
                    concentration=conc,
                    group_id=f"C{conc}_W{well_idx}",
                    replicate_id=f"C{conc}_W{well_idx}",
                    metadata={
                        "true_trajectory": {
                            k: v.copy() for k, v in result.populations.items()
                        },
                        "method": method,
                    },
                )
                series.append(ts)

        return ExperimentalDataset(
            series=series,
            name="synthetic_invitro_plate",
            context="in_vitro",
        )

    def generate_invivo_cohort(
        self,
        initial_cells: np.ndarray | None = None,
        exposure_fn: Callable[[float], float] | None = None,
        n_animals: int = 8,
        t_max: float = 28.0 * 24,  # 28 days in hours
        obs_times: np.ndarray | None = None,
        modalities: list[str] | None = None,
        method: str = "gillespie",
    ) -> ExperimentalDataset:
        """Simulate an in vivo study cohort.

        Args:
            initial_cells: Initial tumor cell state vector.
            exposure_fn: Time-varying concentration function C(t).
            n_animals: Number of animals.
            t_max: Total study duration (hours).
            obs_times: Measurement time points (hours).
            modalities: Which observations to generate ('cell_counts', 'bli', 'volume').
            method: Simulation method.

        Returns:
            ExperimentalDataset with one TimeSeriesData per animal.
        """
        if initial_cells is None:
            initial_cells = np.zeros(self.topology.n_states)
            initial_cells[0] = 1000  # larger initial tumor

        if exposure_fn is None:
            def exposure_fn(t):  # vehicle control
                return 0.0

        if obs_times is None:
            # Typical in vivo: measure twice per week for 4 weeks
            obs_times = np.array([0, 72, 168, 240, 336, 408, 504, 576, 672])

        modalities = self._resolve_modalities(modalities)

        series = []
        for animal_idx in range(n_animals):
            if method == "gillespie":
                sim = GillespieSimulator(
                    self.rate_set, self.topology, exposure_fn, self.rng
                )
                result = sim.simulate(initial_cells, t_max, obs_times)
            else:
                ode = CellDynamicsODE(self.rate_set, self.topology, exposure_fn)
                result = ode.solve(initial_cells, (0, t_max), obs_times)

            # Draw from the configured observation models rather than
            # re-deriving each modality here. The inlined formulas this
            # replaces hardcoded their own alpha and beta (including the old
            # 1e-3 mm^3/cell volume scale), so generated data could not be
            # consistent with the likelihood that was later fitted to it.
            observations = self._sample_observations(
                result, len(obs_times), modalities
            )

            ts = TimeSeriesData(
                times=obs_times,
                observations=observations,
                group_id=f"animal_{animal_idx}",
                replicate_id=f"animal_{animal_idx}",
                metadata={
                    "true_trajectory": {
                        k: v.copy() for k, v in result.populations.items()
                    },
                    "method": method,
                },
            )
            series.append(ts)

        return ExperimentalDataset(
            series=series,
            name="synthetic_invivo_cohort",
            context="in_vivo",
        )
