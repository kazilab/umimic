"""Experiment orchestrator: ties all components together."""

from __future__ import annotations

from typing import Any

import numpy as np

from umimic.pipeline.config import ExperimentConfig
from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.rates import RateSet, EmaxHill
from umimic.dynamics.ode_system import CellDynamicsODE
from umimic.dynamics.gillespie import GillespieSimulator
from umimic.dynamics.tau_leaping import TauLeapingSimulator
from umimic.pk.dosing import DosingSchedule
from umimic.pk.exposure import ExposureProfile
from umimic.observations.cell_counts import CellCountObservation
from umimic.observations.bli import BLIObservation
from umimic.observations.tumor_volume import TumorVolumeObservation
from umimic.observations.multimodal import MultimodalObservation
from umimic.inference.likelihood import ModelLikelihood
from umimic.inference.mle import MLEstimator
from umimic.inference.mcmc import MCMCSampler
from umimic.inference.priors import PriorSpec
from umimic.data.schemas import TimeSeriesData, ExperimentalDataset
from umimic.data.synthetic import SyntheticDataGenerator
from umimic.types import SimulationResult, InferenceResult


class Experiment:
    """Main experiment orchestrator.

    This is the primary user-facing class that:
    1. Builds all components from a configuration
    2. Runs simulations
    3. Fits models to data
    4. Produces predictions and diagnostics
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._build_components()

    def _build_components(self):
        """Instantiate all components from configuration."""
        # Topology
        state_map = {"P": CellType.P, "Q": CellType.Q, "A": CellType.A, "R": CellType.R}
        active = [state_map[s] for s in self.config.dynamics.states if s in state_map]

        if len(active) <= 2:
            self.topology = ModelTopology.two_state()
        elif len(active) == 3:
            self.topology = ModelTopology.three_state()
        else:
            self.topology = ModelTopology.four_state()

        self.topology.density_dependent = self.config.dynamics.density_dependent
        self.topology.carrying_capacity = self.config.dynamics.carrying_capacity
        self.topology.apoptotic_clearance_rate = self.config.dynamics.clearance_rate

        # Default rate set (used for simulation; inference will estimate these)
        self.rate_set = RateSet(
            birth_base=0.04,
            death_base={CellType.P: 0.01, CellType.Q: 0.005},
        )

        # Observation models
        obs_models = {}
        for mod in self.config.observations.modalities:
            if mod == "cell_counts":
                obs_models[mod] = CellCountObservation(
                    overdispersion=self.config.observations.cell_count_overdispersion
                )
            elif mod == "bli":
                obs_models[mod] = BLIObservation(
                    alpha=self.config.observations.bli_alpha,
                    sigma_log=self.config.observations.bli_sigma_log,
                )
            elif mod == "volume":
                obs_models[mod] = TumorVolumeObservation(
                    beta=self.config.observations.volume_beta,
                    sigma_v=self.config.observations.volume_sigma,
                )
        self.observation_model = MultimodalObservation(obs_models)

        # PK and exposure
        self._build_exposure()

    def _build_exposure(self):
        """Build exposure profile from config."""
        if self.config.pk.model == "none":
            conc = 0.0
            if self.config.dosing.concentrations:
                conc = self.config.dosing.concentrations[0]
            self.exposure = ExposureProfile.constant(conc)
        else:
            from umimic.pk.compartment import OneCompartmentPK, TwoCompartmentPK

            if self.config.pk.model == "one_compartment":
                pk = OneCompartmentPK(
                    vd=self.config.pk.vd,
                    ke=self.config.pk.ke,
                    ka=self.config.pk.ka,
                )
            else:
                pk = TwoCompartmentPK(
                    vc=self.config.pk.vc or 10.0,
                    vp=self.config.pk.vp or 20.0,
                    cl=self.config.pk.cl or 1.0,
                    q=self.config.pk.q or 0.5,
                    ka=self.config.pk.ka,
                )

            dosing = self._build_dosing()
            self.exposure = ExposureProfile.from_pk(pk, dosing)

    def _build_dosing(self) -> DosingSchedule:
        """Build dosing schedule from config."""
        dc = self.config.dosing
        if dc.type == "constant":
            conc = dc.concentrations[0] if dc.concentrations else 0.0
            return DosingSchedule.constant_invitro(conc)
        elif dc.type == "single_bolus":
            return DosingSchedule.single_bolus(dc.dose_amount or 100.0, dc.start_time)
        elif dc.type == "repeated_bolus":
            return DosingSchedule.repeated(
                dc.dose_amount or 100.0,
                dc.interval or 24.0,
                dc.n_doses or 7,
                start_time=dc.start_time,
            )
        elif dc.type == "oral":
            return DosingSchedule.oral_repeated(
                dc.dose_amount or 100.0,
                dc.interval or 24.0,
                dc.n_doses or 7,
                dc.start_time,
            )
        return DosingSchedule()

    def simulate(
        self,
        rate_set: RateSet | None = None,
        method: str | None = None,
        concentrations: list[float] | None = None,
    ) -> dict[float, SimulationResult] | SimulationResult:
        """Run forward simulation.

        Args:
            rate_set: Rate parameters (default: use experiment defaults).
            method: Simulation method ('ode', 'gillespie', 'tau_leaping').
            concentrations: If provided, run dose-response simulation.

        Returns:
            SimulationResult or dict of concentration -> SimulationResult.
        """
        rs = rate_set or self.rate_set
        method = method or self.config.simulation.method

        sc = self.config.simulation
        t_eval = np.arange(0, sc.t_max + sc.dt_obs, sc.dt_obs)
        y0 = np.zeros(self.topology.n_states)
        y0[0] = sc.initial_cells

        if concentrations is not None:
            # Dose-response simulation
            ode = CellDynamicsODE(rs, self.topology, lambda t, _c=0.0: _c)
            return ode.solve_dose_response(y0, (0, sc.t_max), concentrations, t_eval)

        if method == "ode":
            ode = CellDynamicsODE(rs, self.topology, self.exposure)
            return ode.solve(y0, (0, sc.t_max), t_eval)
        elif method == "gillespie":
            sim = GillespieSimulator(rs, self.topology, self.exposure, self.rng)
            return sim.simulate(y0, sc.t_max, t_eval)
        elif method == "tau_leaping":
            sim = TauLeapingSimulator(
                rs, self.topology, self.exposure, rng=self.rng
            )
            return sim.simulate(y0, sc.t_max, t_eval)
        else:
            raise ValueError(f"Unknown method: {method}")

    def generate_synthetic(
        self,
        rate_set: RateSet | None = None,
    ) -> ExperimentalDataset:
        """Generate a synthetic dataset using the experiment's configuration."""
        rs = rate_set or self.rate_set
        obs_model = list(self.observation_model.models.values())[0] if self.observation_model.models else CellCountObservation()

        gen = SyntheticDataGenerator(rs, self.topology, obs_model, self.rng)

        sc = self.config.simulation
        concentrations = self.config.dosing.concentrations

        if self.config.context == "in_vitro":
            y0 = np.zeros(self.topology.n_states)
            y0[0] = sc.initial_cells
            return gen.generate_invitro_plate(
                initial_cells=y0,
                concentrations=concentrations,
                n_wells_per_dose=sc.n_replicates,
                t_max=sc.t_max,
                dt_obs=sc.dt_obs,
                method=sc.method,
            )
        else:
            y0 = np.zeros(self.topology.n_states)
            y0[0] = sc.initial_cells
            return gen.generate_invivo_cohort(
                initial_cells=y0,
                exposure_fn=self.exposure,
                n_animals=sc.n_replicates,
                t_max=sc.t_max,
                modalities=self.config.observations.modalities,
                method=sc.method,
            )

    def fit(
        self,
        data: TimeSeriesData | ExperimentalDataset,
        **kwargs,
    ) -> InferenceResult:
        """Run inference on observed data.

        Args:
            data: Observed data (single series or dataset).
            **kwargs: Additional arguments passed to the inference engine.

        Returns:
            InferenceResult with parameter estimates.
        """
        if isinstance(data, ExperimentalDataset):
            data_list = data.series
        elif isinstance(data, list):
            data_list = data
        else:
            data_list = [data]

        ic = self.config.inference
        priors = PriorSpec.default_invitro()

        # Build likelihood
        likelihood = ModelLikelihood(
            topology=self.topology,
            data=data_list,
            mode="ode" if ic.mode == "mle" else "moment",
        )

        if ic.mode == "mle":
            estimator = MLEstimator(
                likelihood=likelihood,
                priors=priors if ic.backend != "scipy" else None,
                method="L-BFGS-B" if ic.backend == "scipy" else "Nelder-Mead",
            )
            mle_result = estimator.fit(n_restarts=ic.n_restarts)
            return InferenceResult(
                method="mle",
                mle=mle_result,
                context=self.config.context,
            )

        elif ic.mode == "mcmc":
            sampler = MCMCSampler(
                likelihood=likelihood,
                priors=priors,
                backend=ic.backend if ic.backend in ("emcee", "pymc") else "emcee",
            )
            mcmc_result = sampler.sample(
                n_samples=ic.n_samples,
                n_chains=ic.n_chains,
                n_warmup=ic.n_warmup,
            )
            return InferenceResult(
                method="mcmc",
                mcmc=mcmc_result,
                context=self.config.context,
            )

        else:
            raise ValueError(f"Unknown inference mode: {ic.mode}")
