"""Experiment orchestrator: ties all components together."""

from __future__ import annotations

import logging

from typing import Any, Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from umimic.pipeline.config import ExperimentConfig
from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.rates import EmaxHill, HillFoldChange, RateSet
from umimic.dynamics.ode_system import CellDynamicsODE
from umimic.dynamics.gillespie import GillespieSimulator
from umimic.dynamics.tau_leaping import TauLeapingSimulator
from umimic.pk.dosing import DosingSchedule
from umimic.pk.exposure import ExposureProfile
from umimic.observations.cell_counts import CellCountObservation
from umimic.observations.bli import BLIObservation
from umimic.observations.tumor_volume import TumorVolumeObservation
from umimic.observations.biomarkers import BiomarkerObservation
from umimic.observations.multimodal import MultimodalObservation
from umimic.inference.likelihood import PARAMETER_SETS, ModelLikelihood
from umimic.inference.mle import MLEstimator
from umimic.inference.mcmc import MCMCSampler
from umimic.inference.priors import PriorSpec
from umimic.data.schemas import TimeSeriesData, ExperimentalDataset
from umimic.data.synthetic import SyntheticDataGenerator
from umimic.signaling.models import ToyMapkAktNetwork
from umimic.signaling.network import SignalingNetwork
from umimic.types import SimulationResult, InferenceResult

LOGGER = logging.getLogger(__name__)


def _edge(state_map: dict, key: str) -> tuple:
    """Parse a 'P->Q' configuration key into a (source, target) pair."""
    src, tgt = (part.strip() for part in key.split("->", 1))
    return state_map[src], state_map[tgt]


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
        # Topology: build exactly the states that were requested. Dispatching
        # on the *number* of states would silently turn [P, Q, R] into
        # [P, Q, A], changing what the model means.
        state_map = {"P": CellType.P, "Q": CellType.Q, "A": CellType.A, "R": CellType.R}
        active = [state_map[s] for s in self.config.dynamics.states]

        # Explicit transitions when configured; otherwise the canonical edges
        # among the active states. Deriving them from the state *count* made
        # routes like the persister edge Q->R unreachable from configuration.
        if self.config.dynamics.transitions is not None:
            transitions = [
                (state_map[src], state_map[tgt])
                for src, tgt in self.config.dynamics.transitions
            ]
        else:
            transitions = [
                (src, tgt)
                for src, tgt in ModelTopology.four_state().transitions
                if src in active and tgt in active
            ]

        if self.config.dynamics.division_states is not None:
            division_states = [
                state_map[s] for s in self.config.dynamics.division_states
            ]
        else:
            # R divides: a resistant state that cannot proliferate is an
            # absorbing sink, not a clone that can outgrow the sensitive
            # population under treatment. Q is arrested unless configured
            # otherwise, since the default P/Q model treats it as quiescent
            # rather than as a slow-cycling persister.
            division_states = [ct for ct in (CellType.P, CellType.R) if ct in active]

        self.topology = ModelTopology(
            active_states=active,
            transitions=transitions,
            division_states=division_states,
            death_states=[
                ct for ct in (CellType.P, CellType.Q, CellType.R) if ct in active
            ],
        )

        self.topology.density_dependent = self.config.dynamics.density_dependent
        self.topology.carrying_capacity = self.config.dynamics.carrying_capacity
        self.topology.density_counts_apoptotic = (
            self.config.dynamics.density_counts_apoptotic
        )
        # Clearance is set on the RateSet below; the topology does not carry it.

        # Default simulation rates are config-driven and represent generic
        # plausible baseline kinetics until user- or data-specific fitting.
        # clearance_rate must reach the RateSet: the simulators read it from
        # there, not from the topology.
        dyn = self.config.dynamics
        death_base = {
            CellType.P: dyn.default_death_base_p,
            CellType.Q: dyn.default_death_base_q,
        }
        birth_base_by_state = {}
        birth_modulation_by_state = {}
        if CellType.R in active:
            # The resistant clone needs its own division and death rates, and
            # must not inherit the drug's birth suppression.
            death_base[CellType.R] = dyn.default_death_base_r
            birth_base_by_state[CellType.R] = dyn.default_birth_base * (
                1.0 - dyn.resistant_fitness_cost
            )
            birth_modulation_by_state[CellType.R] = None

        rate_set_kwargs = {}
        if dyn.transitions is not None:
            # Edges without an explicit rate default to 0 rather than
            # inheriting RateSet's P<->Q defaults, which would attach an
            # unintended baseline to a configured route such as Q->R.
            configured = dyn.transition_rates or {}
            rate_set_kwargs["transition_base"] = {
                (state_map[src], state_map[tgt]): float(
                    configured.get(f"{src}->{tgt}", 0.0)
                )
                for src, tgt in dyn.transitions
            }

        # Drug-dependent transition components. Without these the additive
        # induction term is library-only and no configured model can express
        # drug-induced plasticity.
        if dyn.induced_transitions:
            rate_set_kwargs["transition_induction"] = {
                _edge(state_map, key): EmaxHill(
                    emax=spec["emax"],
                    ec50=spec.get("ec50", 1.0),
                    hill=spec.get("hill", 1.0),
                )
                for key, spec in dyn.induced_transitions.items()
            }
        if dyn.transition_fold_change:
            rate_set_kwargs["transition_factor"] = {
                _edge(state_map, key): HillFoldChange(
                    low=spec.get("low", 1.0),
                    high=spec["high"],
                    ec50=spec.get("ec50", 1.0),
                    hill=spec.get("hill", 1.0),
                )
                for key, spec in dyn.transition_fold_change.items()
            }

        self.rate_set = RateSet(
            birth_base=dyn.default_birth_base,
            birth_base_by_state=birth_base_by_state,
            birth_modulation_by_state=birth_modulation_by_state,
            death_base=death_base,
            clearance_rate=dyn.clearance_rate,
            **rate_set_kwargs,
        )
        self._apply_drug_mechanism()

        # Observation models. Each is given the topology so that "viable" and
        # "dead" resolve to the right state indices for this model.
        obs_models = {}
        for mod in self.config.observations.modalities:
            if mod == "cell_counts":
                obs_models[mod] = CellCountObservation(
                    overdispersion=self.config.observations.cell_count_overdispersion,
                    topology=self.topology,
                )
            elif mod == "bli":
                obs_models[mod] = BLIObservation(
                    alpha=self.config.observations.bli_alpha,
                    sigma_log=self.config.observations.bli_sigma_log,
                    topology=self.topology,
                )
            elif mod == "volume":
                obs_models[mod] = TumorVolumeObservation(
                    beta=self.config.observations.volume_beta,
                    sigma_v=self.config.observations.volume_sigma,
                    topology=self.topology,
                )
            elif mod == "biomarker":
                obs_models[mod] = BiomarkerObservation(
                    biomarker_type=self.config.observations.biomarker_type,
                    precision=self.config.observations.biomarker_precision,
                    topology=self.topology,
                )
        self.observation_model = MultimodalObservation(obs_models)

        # PK and exposure
        self._build_exposure()

    def _apply_drug_mechanism(self) -> None:
        """Attach dose-response modulation for the configured mechanism.

        Without this the configured mechanism never reaches the RateSet and
        every simulation is drug-free regardless of concentration.
        """
        mechanism = getattr(self.config.dynamics, "drug_mechanism", None)
        if not mechanism:
            return

        dyn = self.config.dynamics
        if mechanism in ("cytotoxic", "mixed"):
            self.rate_set.death_modulation[CellType.P] = EmaxHill(
                emax=dyn.emax_death,
                ec50=dyn.ec50_death,
                hill=dyn.hill_death,
            )
            # Quiescent cells are refractory unless configured otherwise.
            # Previously this was fixed at fully refractory with no way to
            # change it from configuration.
            if dyn.quiescent_sensitivity > 0 and CellType.Q in self.topology.active_states:
                self.rate_set.death_modulation[CellType.Q] = EmaxHill(
                    emax=dyn.emax_death * dyn.quiescent_sensitivity,
                    ec50=dyn.ec50_death,
                    hill=dyn.hill_death,
                )
        if mechanism in ("cytostatic", "mixed"):
            self.rate_set.birth_modulation = EmaxHill(
                emax=dyn.emax_birth,
                ec50=dyn.ec50_birth,
                hill=dyn.hill_birth,
            )

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
                # Explicit None checks: `x or default` would replace a
                # deliberate 0 with the default.
                pkc = self.config.pk
                pk = TwoCompartmentPK(
                    vc=10.0 if pkc.vc is None else pkc.vc,
                    vp=20.0 if pkc.vp is None else pkc.vp,
                    cl=1.0 if pkc.cl is None else pkc.cl,
                    q=0.5 if pkc.q is None else pkc.q,
                    ka=pkc.ka,
                )

            dosing = self._build_dosing()
            self.exposure = ExposureProfile.from_pk(pk, dosing)

    def _build_dosing(self) -> DosingSchedule:
        """Build dosing schedule from config."""
        # DosingConfig validation guarantees the fields each type needs, so no
        # `or default` fallbacks are required here; an explicit 0 stays 0.
        dc = self.config.dosing
        if dc.type == "constant":
            return DosingSchedule.constant_invitro(dc.concentrations[0])
        elif dc.type == "single_bolus":
            return DosingSchedule.single_bolus(dc.dose_amount, dc.start_time)
        elif dc.type == "repeated_bolus":
            return DosingSchedule.repeated(
                dc.dose_amount,
                dc.interval,
                dc.n_doses,
                start_time=dc.start_time,
            )
        elif dc.type == "oral":
            return DosingSchedule.oral_repeated(
                dc.dose_amount,
                dc.interval,
                dc.n_doses,
                dc.start_time,
            )
        raise ValueError(f"Unhandled dosing type {dc.type!r}")

    def simulate(
        self,
        rate_set: RateSet | None = None,
        method: str | None = None,
        concentrations: Sequence[float] | None = None,
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
        y0 = self._initial_state_vector(sc.initial_cells, rs)

        if concentrations is not None:
            # Dose-response simulation
            ode = CellDynamicsODE(rs, self.topology, lambda t, _c=0.0: _c)
            return ode.solve_dose_response(y0, (0, sc.t_max), concentrations, t_eval)

        if method == "ode":
            rate_multiplier_fn: Callable[[float, str], float] | None = None
            signaling_meta: dict[str, Any] | None = None
            if self.config.signaling.enabled:
                signaling_bundle = self._build_signaling_rate_multiplier(t_eval)
                if signaling_bundle is not None:
                    rate_multiplier_fn, signaling_meta = signaling_bundle

            ode = CellDynamicsODE(
                rs,
                self.topology,
                self.exposure,
                rate_multiplier_fn=rate_multiplier_fn,
            )
            result = ode.solve(y0, (0, sc.t_max), t_eval)
            if signaling_meta is not None:
                result.metadata["signaling"] = signaling_meta
            return result
        elif method == "gillespie":
            self._reject_unsupported_signaling(method)
            sim = GillespieSimulator(rs, self.topology, self.exposure, self.rng)
            return sim.simulate(y0, sc.t_max, t_eval)
        elif method == "tau_leaping":
            self._reject_unsupported_signaling(method)
            sim = TauLeapingSimulator(
                rs, self.topology, self.exposure, rng=self.rng
            )
            return sim.simulate(y0, sc.t_max, t_eval)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _build_signaling_network(self) -> tuple[SignalingNetwork, np.ndarray] | None:
        """Construct signaling network from config."""
        scfg = self.config.signaling
        if not scfg.enabled or scfg.model == "none":
            return None
        if scfg.model == "toy_mapk_akt":
            model_kwargs = {
                k: v
                for k, v in scfg.parameters.items()
                if k in {"mapk_baseline", "akt_baseline", "mapk_drive", "akt_drive", "decay"}
            }
            network = ToyMapkAktNetwork(**model_kwargs)
            y0 = network.initial_state().copy()
            if scfg.initial_state:
                idx_map = {name: i for i, name in enumerate(network.node_names)}
                for name, value in scfg.initial_state.items():
                    if name in idx_map:
                        y0[idx_map[name]] = float(value)
            return network, y0
        return None

    def _build_signaling_rate_multiplier(
        self, t_eval: np.ndarray
    ) -> tuple[Callable[[float, str], float], dict[str, Any]] | None:
        """Create signaling-to-rate coupling callback for ODE simulation."""
        signaling = self._build_signaling_network()
        if signaling is None:
            return None

        network, y0 = signaling
        t_span = (float(t_eval[0]), float(t_eval[-1]))
        sol = solve_ivp(
            lambda t, y: network.rhs(t, y, float(self.exposure(t))),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )
        if not sol.success:
            return None

        idx_map = {name: i for i, name in enumerate(network.node_names)}
        mapk = sol.y[idx_map["mapk"]] if "mapk" in idx_map else np.zeros_like(sol.t)
        akt = sol.y[idx_map["akt"]] if "akt" in idx_map else np.zeros_like(sol.t)

        cparams = self.config.coupling.parameters
        max_effect_by_target = self.config.coupling.max_effect_by_target
        ec50_by_target = self.config.coupling.ec50_by_target
        hill_by_target = self.config.coupling.hill_by_target
        k_by_target = self.config.coupling.k_by_target
        center_by_target = self.config.coupling.center_by_target
        w_mapk = float(cparams.get("mapk_weight", 0.5))
        w_akt = float(cparams.get("akt_weight", 0.5))
        max_effect = float(cparams.get("max_effect", 1.0))
        ec50 = max(float(cparams.get("ec50", 1.0)), 1e-9)
        hill = max(float(cparams.get("hill", 1.0)), 1e-6)
        logistic_k = float(cparams.get("k", 1.0))
        logistic_center = float(cparams.get("center", 0.5))
        activity = np.maximum(w_mapk * mapk + w_akt * akt, 0.0)
        targets = set(self.config.coupling.targets)

        def applies_to_key(key: str) -> bool:
            if key == "birth":
                return "birth" in targets
            if key.startswith("death:"):
                return key in targets or "death" in targets
            if key.startswith("transition:"):
                return key in targets or "transition" in targets
            return False

        def resolve_max_effect(key: str) -> float:
            if key in max_effect_by_target:
                return float(max_effect_by_target[key])
            if key.startswith("death:") and "death" in max_effect_by_target:
                return float(max_effect_by_target["death"])
            if key.startswith("transition:") and "transition" in max_effect_by_target:
                return float(max_effect_by_target["transition"])
            if key == "birth" and "birth" in max_effect_by_target:
                return float(max_effect_by_target["birth"])
            return max_effect

        def resolve_target_param(
            key: str, mapping: dict[str, float], default: float
        ) -> float:
            if key in mapping:
                return float(mapping[key])
            if key.startswith("death:") and "death" in mapping:
                return float(mapping["death"])
            if key.startswith("transition:") and "transition" in mapping:
                return float(mapping["transition"])
            if key == "birth" and "birth" in mapping:
                return float(mapping["birth"])
            return default

        def normalized_effect(sig: float, key: str) -> float:
            if self.config.coupling.function == "logistic":
                key_k = max(resolve_target_param(key, k_by_target, logistic_k), 1e-9)
                key_center = resolve_target_param(key, center_by_target, logistic_center)
                return float(1.0 / (1.0 + np.exp(-key_k * (sig - key_center))))
            key_ec50 = max(resolve_target_param(key, ec50_by_target, ec50), 1e-9)
            key_hill = max(resolve_target_param(key, hill_by_target, hill), 1e-6)
            return float((sig**key_hill) / (key_ec50**key_hill + sig**key_hill))

        def signaling_rate_multiplier(t: float, key: str) -> float:
            if not self.config.coupling.enabled or not applies_to_key(key):
                return 1.0
            sig = float(np.interp(t, sol.t, activity))
            effect = normalized_effect(max(sig, 0.0), key)
            return max(0.0, 1.0 + resolve_max_effect(key) * effect)

        meta = {
            "enabled": True,
            "model": self.config.signaling.model,
            "coupling_enabled": self.config.coupling.enabled,
            "coupling_mode": "direct_rate_multiplier",
            "coupling_targets": sorted(targets),
            "max_effect_by_target": dict(max_effect_by_target),
            "ec50_by_target": dict(ec50_by_target),
            "hill_by_target": dict(hill_by_target),
            "k_by_target": dict(k_by_target),
            "center_by_target": dict(center_by_target),
            "nodes": network.node_names,
        }
        return signaling_rate_multiplier, meta

    def generate_synthetic(
        self,
        rate_set: RateSet | None = None,
    ) -> ExperimentalDataset:
        """Generate a synthetic dataset using the experiment's configuration."""
        rs = rate_set or self.rate_set

        # Hand over every configured modality. Taking only the first one made
        # a multimodal configuration emit single-modality data, so a
        # simulate-then-fit round trip did not exercise the experiment that
        # was actually described.
        obs_model = (
            self.observation_model
            if self.observation_model.models
            else CellCountObservation(topology=self.topology)
        )

        gen = SyntheticDataGenerator(rs, self.topology, obs_model, self.rng)

        sc = self.config.simulation
        concentrations = self.config.dosing.concentrations

        if self.config.context == "in_vitro":
            y0 = self._initial_state_vector(sc.initial_cells, rs)
            return gen.generate_invitro_plate(
                initial_cells=y0,
                concentrations=concentrations,
                n_wells_per_dose=sc.n_replicates,
                t_max=sc.t_max,
                dt_obs=sc.dt_obs,
                method=sc.method,
            )
        else:
            y0 = self._initial_state_vector(sc.initial_cells, rs)
            return gen.generate_invivo_cohort(
                initial_cells=y0,
                exposure_fn=self.exposure,
                n_animals=sc.n_replicates,
                t_max=sc.t_max,
                modalities=self.config.observations.modalities,
                method=sc.method,
            )

    def _initial_state_vector(
        self, initial_cells: float, rate_set: RateSet | None = None
    ) -> np.ndarray:
        """Distribute the initial count across states.

        Placing every cell in P is a structural assumption, not a neutral
        one: a passaged culture carries a phenotype mix, and starting pure
        biases early growth and the estimated transition rates.
        `dynamics.initial_fractions` selects "proliferating" (the historical
        default), "stable" (the relaxed state distribution), or explicit
        fractions.
        """
        spec = self.config.dynamics.initial_fractions
        n_states = self.topology.n_states

        if spec is None or spec == "proliferating":
            y0 = np.zeros(n_states)
            y0[self.topology.state_index(CellType.P)] = initial_cells
            return y0

        if spec == "stable":
            rates = rate_set or self.rate_set
            conc = 0.0
            if self.config.dosing.concentrations:
                conc = float(self.config.dosing.concentrations[0])
            return initial_cells * rates.stable_state_fractions(conc, self.topology)

        fractions = np.zeros(n_states)
        state_map = {"P": CellType.P, "Q": CellType.Q, "A": CellType.A, "R": CellType.R}
        for name, value in spec.items():
            fractions[self.topology.state_index(state_map[name])] = float(value)
        total = fractions.sum()
        if total <= 0:
            raise ValueError("initial_fractions must sum to a positive value.")
        return initial_cells * fractions / total

    def _reject_unsupported_signaling(self, method: str) -> None:
        """Refuse to run a method that silently ignores signaling coupling.

        Rate coupling is threaded through `CellDynamicsODE.rate_multiplier_fn`
        only. The stochastic simulators, the LNA and the inference likelihood
        do not consume it, so running them with coupling enabled would quietly
        produce uncoupled results that look like coupled ones.
        """
        if self.config.signaling.enabled:
            raise NotImplementedError(
                f"Signaling coupling is implemented for the deterministic ODE "
                f"only; method={method!r} would ignore it and return uncoupled "
                "dynamics. Use method='ode', or disable signaling.enabled."
            )

    def _likelihood_observation_model(self, data_list):
        """Select the configured observation models present in the data.

        Modalities that are configured but absent from the data are dropped
        (they would contribute nothing); a modality present in the data but not
        configured is ignored, since no observation model defines its
        likelihood. If nothing overlaps, that is a configuration error worth
        surfacing rather than silently fitting counts alone.
        """
        available = {m for series in data_list for m in series.modalities}
        usable = {
            name: model
            for name, model in self.observation_model.models.items()
            if name in available
        }
        if not usable:
            raise ValueError(
                f"None of the configured modalities "
                f"{sorted(self.observation_model.models)} are present in the "
                f"data, which provides {sorted(available)}. Set "
                "observations.modalities to match the data."
            )
        if len(usable) < len(self.observation_model.models):
            missing = sorted(set(self.observation_model.models) - set(usable))
            LOGGER.warning(
                "Configured modalities %s are absent from the data and will "
                "not contribute to the fit.", missing,
            )
        return MultimodalObservation(usable)

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
        priors = {
            "default": PriorSpec.default_invitro,
            "mechanism": PriorSpec.default_mechanism,
            "resistance": PriorSpec.default_resistance,
            "persister": PriorSpec.default_persister,
        }[ic.parameter_set]()

        # Hand the configured observation models to the likelihood so that
        # every modality the experiment declares actually enters the fit.
        # Without this the orchestrator silently falls back to cell counts and
        # the configured BLI/volume models are never used.
        observation_model = self._likelihood_observation_model(data_list)
        self._reject_unsupported_signaling("inference")

        # The forward mode determines whether LNA process variance is
        # available. It is configured independently of the inference backend:
        # tying it to MLE-vs-MCMC silently disabled the variance signature that
        # separates birth from death on the default path.
        likelihood = ModelLikelihood(
            topology=self.topology,
            data=data_list,
            mode=ic.forward_mode,
            observation_model=observation_model,
            param_names=PARAMETER_SETS[ic.parameter_set],
            initial_fractions=self.config.dynamics.initial_fractions,
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
                # emcee is the only sampler left: pymc was withdrawn in 0.0.4
                # and scipy, the default backend, has no sampler at all.
                backend="emcee",
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
            # Unreachable via ExperimentConfig, whose `mode` Literal admits
            # only the modes dispatched above. Kept for direct construction.
            raise ValueError(
                f"Inference mode {ic.mode!r} is not implemented by "
                "Experiment.fit(). Supported: 'mle', 'mcmc'. For SMC/PMCMC or "
                "hierarchical estimation, use umimic.inference.ParticleMCMC or "
                "umimic.inference.hierarchical directly."
            )
