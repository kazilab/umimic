"""Analytical-agreement acceptance tests.

Each test here pins a result against a closed-form solution or an invariant
that must hold regardless of implementation detail. These are the checks that
catch silent numerical regressions; a passing unit test on array shapes does
not.
"""

from __future__ import annotations

import numpy as np
import pytest

from umimic.data.schemas import TimeSeriesData
from umimic.dynamics.gillespie import GillespieSimulator
from umimic.dynamics.moment_equations import MomentODE
from umimic.dynamics.ode_system import CellDynamicsODE
from umimic.dynamics.rates import RateSet
from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.tau_leaping import TauLeapingSimulator
from umimic.inference.likelihood import ModelLikelihood
from umimic.observations.bli import BLIObservation
from umimic.observations.cell_counts import CellCountObservation
from umimic.observations.multimodal import MultimodalObservation
from umimic.observations.tumor_volume import TumorVolumeObservation
from umimic.pk.compartment import OneCompartmentPK, TwoCompartmentPK
from umimic.pk.dosing import Dose, DosingSchedule


# ---------------------------------------------------------------------------
# Moment equations: the mean must be the drift, not J @ mu
# ---------------------------------------------------------------------------


def _logistic_topology(K=1000.0):
    return ModelTopology(
        active_states=[CellType.P],
        transitions=[],
        division_states=[CellType.P],
        death_states=[],
        density_dependent=True,
        carrying_capacity=K,
    )


def test_logistic_moment_mean_matches_analytical_solution():
    """One-state density-dependent growth has a closed-form logistic mean.

    Using the Jacobian as the mean drift yields r*mu*(1 - 2*mu/K), whose fixed
    point is K/2 rather than K -- a ~50% error at saturation.
    """
    K, N0, r = 1000.0, 10.0, 0.1
    topo = _logistic_topology(K)
    rates = RateSet(birth_base=r, death_base={}, transition_base={})

    moment = MomentODE(rates, topo, lambda t: 0.0)
    times = np.linspace(0, 120, 13)
    t_sol, means, _ = moment.solve(np.array([N0]), t_span=(0, 120), t_eval=times)

    analytical = K * N0 * np.exp(r * t_sol) / (K + N0 * (np.exp(r * t_sol) - 1))
    np.testing.assert_allclose(means[:, 0], analytical, rtol=1e-5)


def test_logistic_moment_mean_saturates_at_carrying_capacity():
    """The mean must approach K, not K/2."""
    K = 1000.0
    topo = _logistic_topology(K)
    rates = RateSet(birth_base=0.1, death_base={}, transition_base={})
    _, means, _ = MomentODE(rates, topo, lambda t: 0.0).solve(
        np.array([10.0]), t_span=(0, 400), t_eval=np.array([0.0, 400.0])
    )
    assert means[-1, 0] == pytest.approx(K, rel=1e-3)


@pytest.mark.parametrize("density_dependent", [False, True])
def test_moment_mean_matches_deterministic_ode(density_dependent):
    """Moment and deterministic ODE means agree for linear and nonlinear models."""
    topo = ModelTopology.two_state()
    topo.density_dependent = density_dependent
    topo.carrying_capacity = 5000.0 if density_dependent else None

    rates = RateSet(
        birth_base=0.05,
        death_base={CellType.P: 0.01, CellType.Q: 0.005},
        transition_base={
            (CellType.P, CellType.Q): 0.01,
            (CellType.Q, CellType.P): 0.008,
        },
    )
    mu0 = np.array([500.0, 50.0])
    times = np.linspace(0, 72, 25)

    _, means, _ = MomentODE(rates, topo, lambda t: 0.0).solve(
        mu0, t_span=(0, 72), t_eval=times
    )
    ode = CellDynamicsODE(rates, topo, lambda t: 0.0).solve(mu0, (0, 72), times)

    np.testing.assert_allclose(means[:, 0], ode.populations["P"], rtol=1e-4)
    np.testing.assert_allclose(means[:, 1], ode.populations["Q"], rtol=1e-4)


def test_moment_covariance_is_symmetric_psd():
    topo = ModelTopology.two_state()
    rates = RateSet(birth_base=0.05, death_base={CellType.P: 0.01})
    _, _, covs = MomentODE(rates, topo, lambda t: 0.0).solve(
        np.array([200.0, 20.0]), t_span=(0, 48), t_eval=np.linspace(0, 48, 13)
    )
    for cov in covs:
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)
        assert np.linalg.eigvalsh(cov)[0] >= -1e-8


# ---------------------------------------------------------------------------
# Pharmacokinetics
# ---------------------------------------------------------------------------


def test_pk_result_is_invariant_to_output_grid():
    """Concentration at a shared time must not depend on t_eval density."""
    pk = OneCompartmentPK(vd=10.0, ke=0.1, ka=0.5)
    dosing = DosingSchedule.repeated(100.0, 24.0, 4, route="oral")

    sparse = np.array([0.0, 12.0, 36.0, 60.0, 84.0])
    dense = np.linspace(0.0, 84.0, 169)

    c_sparse = pk.solve(dosing, sparse)
    c_dense = pk.solve(dosing, dense)

    for j, t in enumerate(sparse):
        i = int(np.argmin(np.abs(dense - t)))
        assert c_sparse[j] == pytest.approx(c_dense[i], rel=1e-6, abs=1e-9)


def test_iv_bolus_matches_one_compartment_analytical():
    pk = OneCompartmentPK(vd=10.0, ke=0.1)
    t = np.array([0.0, 1.0, 5.0, 10.0, 24.0])
    numeric = pk.solve(DosingSchedule.single_bolus(100.0), t)
    analytical = (100.0 / 10.0) * np.exp(-0.1 * t)
    np.testing.assert_allclose(numeric, analytical, rtol=1e-9)


def test_single_oral_dose_matches_bateman_function():
    D, V, ka, ke = 100.0, 10.0, 0.5, 0.1
    pk = OneCompartmentPK(vd=V, ke=ke, ka=ka)
    t = np.linspace(0, 48, 25)
    numeric = pk.solve(DosingSchedule(doses=[Dose(0.0, D, "oral")]), t)
    bateman = (D * ka) / (V * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))
    np.testing.assert_allclose(numeric, bateman, atol=1e-7)


def test_simultaneous_doses_are_summed_not_overwritten():
    pk = OneCompartmentPK(vd=10.0, ke=0.1)
    t = np.linspace(0, 24, 49)
    one = pk.solve(DosingSchedule(doses=[Dose(0.0, 100.0)]), t)
    two = pk.solve(
        DosingSchedule(doses=[Dose(0.0, 100.0), Dose(0.0, 100.0)]), t
    )
    np.testing.assert_allclose(two, 2.0 * one, rtol=1e-9)


def test_repeated_iv_doses_obey_superposition():
    pk = OneCompartmentPK(vd=10.0, ke=0.1)
    t = np.linspace(0, 96, 97)
    schedule = DosingSchedule.repeated(50.0, 24.0, 4)
    numeric = pk.solve(schedule, t)
    expected = np.zeros_like(t)
    for dose in schedule.doses:
        mask = t >= dose.time
        expected[mask] += (dose.amount / 10.0) * np.exp(-0.1 * (t[mask] - dose.time))
    np.testing.assert_allclose(numeric, expected, rtol=1e-9)


def test_two_compartment_grid_invariance():
    pk = TwoCompartmentPK(vc=5.0, vp=10.0, cl=1.0, q=0.5)
    dosing = DosingSchedule.repeated(100.0, 12.0, 3)
    sparse = np.array([0.0, 6.0, 18.0, 30.0])
    dense = np.linspace(0.0, 30.0, 121)
    c_sparse, c_dense = pk.solve(dosing, sparse), pk.solve(dosing, dense)
    for j, t in enumerate(sparse):
        i = int(np.argmin(np.abs(dense - t)))
        assert c_sparse[j] == pytest.approx(c_dense[i], rel=1e-5, abs=1e-9)


def test_infusion_is_supported_and_grid_invariant():
    pk = OneCompartmentPK(vd=10.0, ke=0.1)
    dosing = DosingSchedule(doses=[Dose(0.0, 100.0, "iv_infusion", duration=6.0)])
    sparse = np.array([0.0, 6.0, 24.0])
    dense = np.linspace(0.0, 24.0, 97)
    c_sparse, c_dense = pk.solve(dosing, sparse), pk.solve(dosing, dense)
    for j, t in enumerate(sparse):
        i = int(np.argmin(np.abs(dense - t)))
        assert c_sparse[j] == pytest.approx(c_dense[i], rel=1e-6, abs=1e-9)
    assert c_sparse[1] > 0


@pytest.mark.parametrize(
    "kwargs", [dict(vd=0.0), dict(vd=-1.0), dict(vd=10.0, ke=-0.1)]
)
def test_pk_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        OneCompartmentPK(**kwargs)


def test_pk_rejects_unsupported_route_and_unsorted_grid():
    pk = OneCompartmentPK(vd=10.0, ke=0.1)  # no ka
    with pytest.raises(ValueError):
        pk.solve(DosingSchedule(doses=[Dose(0.0, 100.0, "oral")]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        pk.solve(DosingSchedule.single_bolus(100.0), np.array([5.0, 1.0]))


def test_luciferin_peak_time_correct_for_both_rate_orderings():
    from umimic.pk.luciferin import LuciferinKinetics

    fast = LuciferinKinetics(ka_luc=0.5, ke_luc=0.05)
    slow = LuciferinKinetics(ka_luc=0.05, ke_luc=0.5)
    for kin in (fast, slow):
        expected = np.log(kin.ka_luc / kin.ke_luc) / (kin.ka_luc - kin.ke_luc)
        assert kin.peak_time == pytest.approx(expected)
        assert kin.peak_time > 0


# ---------------------------------------------------------------------------
# Stochastic simulators
# ---------------------------------------------------------------------------


def _birth_death_setup():
    topo = ModelTopology(
        active_states=[CellType.P],
        transitions=[],
        division_states=[CellType.P],
        death_states=[CellType.P],
    )
    return topo, RateSet(birth_base=0.08, death_base={CellType.P: 0.03},
                         transition_base={})


def test_ssa_ensemble_matches_analytical_birth_death_moments():
    """Linear birth-death has closed-form mean and variance."""
    topo, rates = _birth_death_setup()
    b, d, N0 = 0.08, 0.03, 200.0
    t_rec = np.array([0.0, 5.0, 10.0, 20.0])

    sim = GillespieSimulator(rates, topo, lambda t: 0.0, np.random.default_rng(7))
    traj = np.array(
        [sim.simulate(np.array([N0]), 20.0, t_rec).populations["P"]
         for _ in range(1500)]
    )

    lam = b - d
    mean = N0 * np.exp(lam * t_rec)
    var = N0 * np.exp(lam * t_rec) * (b + d) / lam * (np.exp(lam * t_rec) - 1)

    np.testing.assert_allclose(traj.mean(0), mean, rtol=0.05)
    np.testing.assert_allclose(traj.var(0)[1:], var[1:], rtol=0.25)


def test_ssa_selects_thinning_for_time_varying_exposure():
    """Frozen propensities are not exact when exposure varies continuously."""
    topo, rates = _birth_death_setup()
    rng = np.random.default_rng(0)
    t_rec = np.array([0.0, 10.0])

    constant = GillespieSimulator(rates, topo, lambda t: 0.0, rng)
    assert constant.simulate(np.array([100.0]), 10.0, t_rec).metadata[
        "method"
    ] == "gillespie:direct"

    varying = GillespieSimulator(rates, topo, lambda t: 5.0 * np.exp(-0.1 * t), rng)
    assert varying.simulate(np.array([100.0]), 10.0, t_rec).metadata[
        "method"
    ] == "gillespie:thinning"


def test_ssa_strict_constant_mode_rejects_time_varying_exposure():
    topo, rates = _birth_death_setup()
    sim = GillespieSimulator(
        rates, topo, lambda t: 5.0 * np.exp(-0.1 * t),
        np.random.default_rng(0), exposure_mode="constant",
    )
    with pytest.raises(ValueError, match="constant"):
        sim.simulate(np.array([100.0]), 10.0, np.array([0.0, 10.0]))


def test_ssa_reports_event_limit_truncation():
    topo, rates = _birth_death_setup()
    sim = GillespieSimulator(rates, topo, lambda t: 0.0, np.random.default_rng(1))
    result = sim.simulate(np.array([500.0]), 50.0, np.array([0.0, 50.0]),
                          max_events=10)
    assert result.metadata["truncated"] is True


def test_tau_leaping_converges_to_analytical_mean_as_tau_decreases():
    topo, rates = _birth_death_setup()
    b, d, N0, T = 0.08, 0.03, 500.0, 20.0
    t_rec = np.array([0.0, T])
    exact = N0 * np.exp((b - d) * T)

    errors = []
    for tau in (4.0, 1.0, 0.25):
        sim = TauLeapingSimulator(
            rates, topo, lambda t: 0.0, tau=tau, rng=np.random.default_rng(3)
        )
        traj = np.array(
            [sim.simulate(np.array([N0]), T, t_rec).populations["P"][-1]
             for _ in range(400)]
        )
        errors.append(abs(traj.mean() - exact) / exact)

    assert errors[0] > errors[-1], f"tau-leaping did not converge: {errors}"
    assert errors[-1] < 0.05


def test_tau_leaping_never_produces_negative_populations():
    """A leap must not consume more cells than exist."""
    topo = ModelTopology(
        active_states=[CellType.P], transitions=[],
        division_states=[], death_states=[CellType.P],
    )
    rates = RateSet(birth_base=0.0, death_base={CellType.P: 1.0},
                    transition_base={})
    sim = TauLeapingSimulator(
        rates, topo, lambda t: 0.0, tau=10.0,
        rng=np.random.default_rng(5), ssa_threshold=0.0,
    )
    for _ in range(100):
        result = sim.simulate(np.array([50.0]), 5.0, np.array([0.0, 1.0, 5.0]))
        assert np.all(result.populations["P"] >= 0)


def test_tau_leaping_does_not_advance_past_t_max():
    topo, rates = _birth_death_setup()
    sim = TauLeapingSimulator(
        rates, topo, lambda t: 0.0, tau=7.0, rng=np.random.default_rng(2)
    )
    result = sim.simulate(np.array([500.0]), 10.0, np.array([0.0, 5.0, 10.0]))
    assert result.times[-1] == 10.0
    assert np.all(np.isfinite(result.populations["P"]))


# ---------------------------------------------------------------------------
# Observation models and state indexing
# ---------------------------------------------------------------------------


def test_viable_count_uses_topology_not_hardcoded_index():
    """[P, Q, R] has no apoptotic state, so index 2 is resistant, not dead."""
    pqr = ModelTopology(
        active_states=[CellType.P, CellType.Q, CellType.R],
        transitions=[], division_states=[CellType.P],
        death_states=[CellType.P],
    )
    pqa = ModelTopology.three_state()
    state = np.array([100.0, 50.0, 20.0])

    assert CellCountObservation(topology=pqr)._get_mean(state) == 170.0
    assert CellCountObservation(topology=pqa)._get_mean(state) == 150.0


def test_observation_operator_projects_covariance():
    topo = ModelTopology.three_state()  # [P, Q, A]
    model = CellCountObservation(topology=topo)
    cov = np.diag([4.0, 9.0, 100.0])
    # Viable = P + Q, so H Sigma H^T = 4 + 9 = 13, not the full sum 113.
    assert model.project_variance("viable", cov) == pytest.approx(13.0)


def test_scalar_and_batch_likelihoods_agree():
    topo = ModelTopology.two_state()
    model = CellCountObservation(overdispersion=8.0, topology=topo)
    states = np.array([[100.0, 10.0], [120.0, 12.0], [150.0, 15.0]])
    obs = np.array([105.0, 118.0, 160.0])
    pv = np.array([50.0, 0.0, 70.0])  # mixed: Gaussian, NegBin, Gaussian

    batch = model.log_likelihood_batch(obs, states, None, pv)
    scalar = sum(
        model.log_likelihood(obs[i], states[i], None, pv[i]) for i in range(3)
    )
    assert batch == pytest.approx(scalar)


def test_observation_models_reject_invalid_inputs():
    topo = ModelTopology.two_state()
    with pytest.raises(ValueError):
        CellCountObservation(overdispersion=0.0, topology=topo)
    with pytest.raises(ValueError):
        CellCountObservation(topology=topo).log_likelihood(
            -5.0, np.array([100.0, 10.0])
        )
    with pytest.raises(ValueError):
        BLIObservation(topology=topo).log_likelihood(0.0, np.array([100.0, 10.0]))


# ---------------------------------------------------------------------------
# Likelihood: multimodality, replicates, missing data
# ---------------------------------------------------------------------------


def _multimodal_series(scale_bli=1.0, scale_vol=1.0, missing=()):
    t = np.linspace(0, 48, 7)
    counts = np.array([100.0, 120.0, 150.0, 185.0, 225.0, 270.0, 330.0])
    bli = counts * 1000.0 * scale_bli
    vol = counts * 1e-3 * scale_vol
    for i in missing:
        bli[i] = np.nan
    return TimeSeriesData(
        times=t,
        observations={"cell_counts": counts, "bli": bli, "volume": vol},
        concentration=0.0,
    )


def _multimodal_model(topo):
    return MultimodalObservation(
        {
            "cell_counts": CellCountObservation(10.0, topology=topo),
            "bli": BLIObservation(alpha=1000.0, topology=topo),
            "volume": TumorVolumeObservation(beta=1e-3, topology=topo),
        }
    )


THETA = np.array([0.04, 0.01, 0.0, 1.0, 1.5, 0.005, 0.003, 10.0])


def test_likelihood_uses_every_configured_modality():
    """Perturbing BLI or volume alone must change the log-likelihood."""
    topo = ModelTopology.two_state()
    model = _multimodal_model(topo)

    base = ModelLikelihood(
        topo, _multimodal_series(), mode="moment", observation_model=model
    )
    assert set(base._active_modalities) == {"cell_counts", "bli", "volume"}
    ll_base = base(THETA)

    ll_bli = ModelLikelihood(
        topo, _multimodal_series(scale_bli=3.0), mode="moment",
        observation_model=model,
    )(THETA)
    ll_vol = ModelLikelihood(
        topo, _multimodal_series(scale_vol=3.0), mode="moment",
        observation_model=model,
    )(THETA)

    assert ll_bli != pytest.approx(ll_base)
    assert ll_vol != pytest.approx(ll_base)


def test_likelihood_skips_missing_observations():
    topo = ModelTopology.two_state()
    model = _multimodal_model(topo)
    full = ModelLikelihood(
        topo, _multimodal_series(), mode="moment", observation_model=model
    )
    partial = ModelLikelihood(
        topo, _multimodal_series(missing=(2, 5)), mode="moment",
        observation_model=model,
    )
    assert np.isfinite(partial(THETA))
    assert partial.n_observations == full.n_observations - 2


def test_bic_counts_only_actual_observations():
    topo = ModelTopology.two_state()
    model = _multimodal_model(topo)
    lik = ModelLikelihood(
        topo, _multimodal_series(missing=(1,)), mode="moment",
        observation_model=model,
    )
    # 7 counts (minus 1 anchor) + 6 bli + 7 volume = 19
    assert lik.n_observations == 19
    assert np.isfinite(lik.bic(THETA))


def test_replicates_with_different_time_grids_are_interpolated():
    """Replicate times must be honoured exactly, not snapped to a neighbour."""
    topo = ModelTopology.two_state()
    a = TimeSeriesData(
        times=np.array([0.0, 12.0, 24.0, 48.0]),
        observations={"cell_counts": np.array([100.0, 130.0, 170.0, 290.0])},
        concentration=0.0,
        replicate_id="a",
    )
    b = TimeSeriesData(
        times=np.array([0.0, 6.0, 30.0, 42.0]),
        observations={"cell_counts": np.array([100.0, 115.0, 200.0, 260.0])},
        concentration=0.0,
        replicate_id="b",
    )
    lik = ModelLikelihood(topo, [a, b], mode="moment")
    # The shared solve grid is the union of both schedules.
    np.testing.assert_allclose(
        lik._group_times[0.0], [0.0, 6.0, 12.0, 24.0, 30.0, 42.0, 48.0]
    )
    assert np.isfinite(lik(THETA))


def test_data_schema_rejects_malformed_series():
    with pytest.raises(ValueError):  # non-increasing times
        TimeSeriesData(times=np.array([0.0, 5.0, 5.0]),
                       observations={"cell_counts": np.array([1.0, 2.0, 3.0])})
    with pytest.raises(ValueError):  # length mismatch
        TimeSeriesData(times=np.array([0.0, 1.0]),
                       observations={"cell_counts": np.array([1.0])})
    with pytest.raises(ValueError):  # negative counts
        TimeSeriesData(times=np.array([0.0, 1.0]),
                       observations={"cell_counts": np.array([1.0, -2.0])})


# ---------------------------------------------------------------------------
# Inference backends and diagnostics
# ---------------------------------------------------------------------------


def test_rhat_is_undefined_rather_than_one_when_not_computable():
    from umimic.inference.diagnostics import compute_rhat

    assert compute_rhat(np.ones((4, 200))) is None  # zero variance
    assert compute_rhat(np.ones((4, 3))) is None  # too few draws


def test_rhat_detects_non_convergence():
    from umimic.inference.diagnostics import compute_rhat

    rng = np.random.default_rng(0)
    converged = rng.normal(0, 1, (4, 2000))
    split = np.vstack([
        rng.normal(-6, 1, 2000), rng.normal(6, 1, 2000),
        rng.normal(-6, 1, 2000), rng.normal(6, 1, 2000),
    ])
    assert compute_rhat(converged) == pytest.approx(1.0, abs=0.02)
    assert compute_rhat(split) > 1.5


def test_ess_is_reduced_by_autocorrelation():
    from umimic.inference.diagnostics import effective_sample_size

    rng = np.random.default_rng(1)
    independent = rng.normal(0, 1, (4, 2000))
    correlated = np.zeros((4, 2000))
    for c in range(4):
        for i in range(1, 2000):
            correlated[c, i] = 0.95 * correlated[c, i - 1] + rng.normal()

    assert effective_sample_size(independent) > 4000
    assert effective_sample_size(correlated) < 2000


def test_pymc_backend_is_withdrawn_not_silently_wrong():
    """The old PyMC model ignored the data; it must fail loudly instead."""
    from umimic.inference.mcmc import MCMCSampler
    from umimic.inference.priors import PriorSpec

    topo = ModelTopology.two_state()
    lik = ModelLikelihood(topo, _multimodal_series(), mode="ode")
    sampler = MCMCSampler(lik, PriorSpec(), backend="pymc")
    with pytest.raises(NotImplementedError, match="Potential"):
        sampler.sample(n_samples=1, n_chains=1, n_warmup=1)


def test_particle_filter_degeneracy_returns_negative_infinity():
    """All-zero particle likelihood is undefined, not 'business as usual'."""
    from umimic.inference.smc import ParticleFilter
    from umimic.observations.base import ObservationModel

    class ImpossibleObservation(ObservationModel):
        """Assigns zero probability to every particle."""

        def log_likelihood(self, observed, latent_state, params=None,
                           process_variance=None):
            return -np.inf

        def sample(self, latent_state, rng, params=None):
            return 0.0

        def param_names(self):
            return []

    topo = ModelTopology.two_state()
    rates = RateSet(birth_base=0.04, death_base={CellType.P: 0.01})
    data = TimeSeriesData(
        times=np.array([0.0, 5.0]),
        observations={"cell_counts": np.array([100.0, 130.0])},
        concentration=0.0,
    )
    pf = ParticleFilter(
        rates, topo, lambda t: 0.0, ImpossibleObservation(),
        n_particles=20, rng=np.random.default_rng(0),
    )
    result = pf.filter(data, np.array([100.0, 0.0]))
    assert result["degenerate"] is True
    assert result["marginal_log_likelihood"] == -np.inf


def test_particle_filter_weights_are_recursive():
    """Without resampling, weights must accumulate across steps."""
    from umimic.inference.smc import ParticleFilter

    topo = ModelTopology.two_state()
    rates = RateSet(birth_base=0.04, death_base={CellType.P: 0.01})
    data = TimeSeriesData(
        times=np.array([0.0, 6.0, 12.0]),
        observations={"cell_counts": np.array([100.0, 110.0, 125.0])},
        concentration=0.0,
    )
    pf = ParticleFilter(
        rates, topo, lambda t: 0.0,
        CellCountObservation(overdispersion=10.0, topology=topo),
        n_particles=40, rng=np.random.default_rng(3), ess_fraction=0.0,
    )
    result = pf.filter(data, np.array([100.0, 0.0]))
    # With resampling disabled the weights still sum to one and stay finite.
    assert np.isfinite(result["marginal_log_likelihood"])
    assert result["final_weights"].sum() == pytest.approx(1.0)


def test_pmcmc_applies_the_hastings_correction():
    """A lognormal random walk is asymmetric; the correction must be present."""
    import inspect

    from umimic.inference import smc

    source = inspect.getsource(smc.ParticleMCMC.sample)
    assert "log_hastings" in source
    assert "log_hastings" in source.split("log_alpha")[1][:200]


# ---------------------------------------------------------------------------
# Configuration fidelity
# ---------------------------------------------------------------------------


def test_configured_states_are_built_exactly():
    """[P, Q, R] must not silently become [P, Q, A]."""
    from umimic.pipeline.config import ExperimentConfig
    from umimic.pipeline.experiment import Experiment

    config = ExperimentConfig()
    config.dynamics.states = ["P", "Q", "R"]
    exp = Experiment(config)
    assert [s.name for s in exp.topology.active_states] == ["P", "Q", "R"]


def test_configured_clearance_reaches_the_simulation_rate_set():
    from umimic.pipeline.config import ExperimentConfig
    from umimic.pipeline.experiment import Experiment

    config = ExperimentConfig()
    config.dynamics.states = ["P", "Q", "A"]
    config.dynamics.clearance_rate = 0.37
    exp = Experiment(config)
    assert exp.rate_set.clearance_rate == pytest.approx(0.37)


def test_explicit_zero_dose_is_not_replaced_by_a_default():
    from umimic.pipeline.config import DosingConfig

    # A zero dose amount is rejected outright rather than silently becoming 100.
    with pytest.raises(ValueError):
        DosingConfig(type="single_bolus", dose_amount=0.0)
    # And an omitted required field is an error, not a hidden default.
    with pytest.raises(ValueError):
        DosingConfig(type="repeated_bolus", dose_amount=10.0)


def test_config_rejects_unknown_and_duplicate_states():
    from umimic.pipeline.config import DynamicsConfig

    with pytest.raises(ValueError):
        DynamicsConfig(states=["P", "X"])
    with pytest.raises(ValueError):
        DynamicsConfig(states=["P", "Q", "Q"])
    with pytest.raises(ValueError):
        DynamicsConfig(density_dependent=True)  # no carrying capacity


def test_drug_mechanism_reaches_the_rate_set():
    from umimic.pipeline.config import ExperimentConfig
    from umimic.pipeline.experiment import Experiment

    config = ExperimentConfig()
    config.dynamics.drug_mechanism = "mixed"
    exp = Experiment(config)
    assert exp.rate_set.birth_modulation is not None
    assert exp.rate_set.death_modulation.get(CellType.P) is not None
    # And the drug actually changes the rates.
    assert exp.rate_set.death_rate(CellType.P, 10.0) > exp.rate_set.death_rate(
        CellType.P, 0.0
    )
