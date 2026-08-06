"""End-to-end exercises for the inference backends.

The MCMC, SMC and diagnostics modules were substantially rewritten -- chain
shape, emcee blobs, RNG stream separation, recursive logsumexp weights, the
Hastings term, split rank-normalised R-hat, and the posterior predictive
check. Property tests elsewhere pin individual behaviours; these run the
samplers for real, because a sampler that returns correctly shaped arrays can
still be sampling the wrong distribution.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from umimic.data.schemas import TimeSeriesData
from umimic.dynamics.ode_system import CellDynamicsODE
from umimic.dynamics.rates import RateSet
from umimic.dynamics.states import CellType, ModelTopology
from umimic.inference.diagnostics import posterior_predictive_check, summarize_mcmc
from umimic.inference.likelihood import ModelLikelihood
from umimic.inference.mcmc import MCMCSampler
from umimic.inference.priors import PriorSpec
from umimic.inference.smc import ParticleFilter, ParticleMCMC
from umimic.observations.cell_counts import CellCountObservation

TRUE_B0 = 0.05
TRUE_D0 = 0.01


@pytest.fixture(scope="module")
def topology():
    return ModelTopology.two_state()


@pytest.fixture(scope="module")
def growth_data(topology):
    """Noiseless counts from a known parameter set."""
    truth = RateSet(
        birth_base=TRUE_B0,
        death_base={CellType.P: TRUE_D0, CellType.Q: 0.005},
        transition_base={
            (CellType.P, CellType.Q): 0.005,
            (CellType.Q, CellType.P): 0.003,
        },
    )
    times = np.linspace(0, 72, 10)
    result = CellDynamicsODE(truth, topology, lambda _t: 0.0).solve(
        np.array([300.0, 0.0]), (0, 72), times
    )
    counts = np.round(result.populations["P"] + result.populations["Q"])
    return TimeSeriesData(
        times=times, observations={"cell_counts": counts}, concentration=0.0
    )


def _two_param_setup(topology, data):
    likelihood = ModelLikelihood(
        topology, data, param_names=["b0", "d0_P"], mode="ode"
    )
    priors = PriorSpec()
    priors.add("b0", stats.lognorm(s=0.5, scale=0.05))
    priors.add("d0_P", stats.lognorm(s=0.5, scale=0.01))
    return likelihood, priors


# ---------------------------------------------------------------------------
# emcee backend
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mcmc_run(topology, growth_data):
    likelihood, priors = _two_param_setup(topology, growth_data)
    sampler = MCMCSampler(likelihood, priors, rng=11)
    return sampler.sample(n_samples=250, n_chains=6, n_warmup=250)


def test_mcmc_recovers_the_generating_parameter(mcmc_run):
    """The whole point: the posterior must concentrate near the truth."""
    b0 = mcmc_run.samples["b0"].ravel()
    assert np.median(b0) == pytest.approx(TRUE_B0, rel=0.35)
    lo, hi = np.percentile(b0, [2.5, 97.5])
    assert lo < TRUE_B0 < hi, f"true b0 outside the 95% interval [{lo:.4f}, {hi:.4f}]"


def test_mcmc_preserves_the_walker_axis(mcmc_run):
    for values in mcmc_run.samples.values():
        assert values.ndim == 2
        assert values.shape == (mcmc_run.n_chains, mcmc_run.n_samples)


def test_log_likelihood_and_posterior_differ_by_the_log_prior(mcmc_run, topology,
                                                              growth_data):
    """They are stored separately; check the stored values are consistent."""
    likelihood, priors = _two_param_setup(topology, growth_data)
    assert mcmc_run.log_likelihood_trace is not None
    assert mcmc_run.log_posterior_trace is not None
    assert mcmc_run.log_likelihood_trace.shape == mcmc_run.log_posterior_trace.shape

    implied_prior = mcmc_run.log_posterior_trace - mcmc_run.log_likelihood_trace
    finite = np.isfinite(implied_prior)
    assert finite.any()

    # Recompute the log-prior at a sampled point and compare.
    chain, draw = np.unravel_index(np.argmax(finite), finite.shape)
    theta = np.array(
        [mcmc_run.samples[name][chain, draw] for name in likelihood.param_names]
    )
    expected = priors.log_prior(likelihood.theta_to_params(theta))
    assert implied_prior[chain, draw] == pytest.approx(expected, rel=1e-6)


def test_mcmc_is_reproducible_and_seed_dependent(topology, growth_data):
    def run(seed):
        likelihood, priors = _two_param_setup(topology, growth_data)
        return MCMCSampler(likelihood, priors, rng=seed).sample(
            n_samples=40, n_chains=6, n_warmup=40
        )

    a, b, c = run(99), run(99), run(1234)
    np.testing.assert_allclose(a.samples["b0"], b.samples["b0"])
    assert not np.allclose(a.samples["b0"], c.samples["b0"])


def test_mcmc_initial_guess_path_runs_and_is_validated(topology, growth_data):
    likelihood, priors = _two_param_setup(topology, growth_data)
    sampler = MCMCSampler(likelihood, priors, rng=5)

    result = sampler.sample(
        n_samples=30, n_chains=6, n_warmup=30, initial_guess=np.array([0.05, 0.01])
    )
    assert result.samples["b0"].shape[1] == 30

    with pytest.raises(ValueError, match="initial_guess must have shape"):
        sampler.sample(n_samples=5, n_chains=6, n_warmup=5,
                       initial_guess=np.array([0.05]))


def test_walkers_are_initialised_in_param_names_order(topology, growth_data):
    """A prior dict in a different order must not permute the parameters."""
    likelihood = ModelLikelihood(
        topology, growth_data, param_names=["b0", "d0_P"], mode="ode"
    )
    reversed_priors = PriorSpec()
    reversed_priors.add("d0_P", stats.lognorm(s=0.01, scale=0.01))
    reversed_priors.add("b0", stats.lognorm(s=0.01, scale=0.05))

    result = MCMCSampler(likelihood, reversed_priors, rng=3).sample(
        n_samples=20, n_chains=6, n_warmup=20
    )
    # The tight priors pin each parameter near its own scale; a permutation
    # would swap them.
    assert np.median(result.samples["b0"]) == pytest.approx(0.05, rel=0.3)
    assert np.median(result.samples["d0_P"]) == pytest.approx(0.01, rel=0.3)


def test_missing_prior_is_reported(topology, growth_data):
    likelihood = ModelLikelihood(
        topology, growth_data, param_names=["b0", "d0_P"], mode="ode"
    )
    priors = PriorSpec()
    priors.add("b0", stats.lognorm(s=0.5, scale=0.05))
    with pytest.raises(ValueError, match="Priors do not cover"):
        MCMCSampler(likelihood, priors, rng=1).sample(
            n_samples=5, n_chains=6, n_warmup=5
        )


# ---------------------------------------------------------------------------
# Diagnostics on a real run
# ---------------------------------------------------------------------------


def test_summarize_reports_finite_diagnostics_for_a_real_run(mcmc_run):
    summary = summarize_mcmc(mcmc_run)
    for name, entry in summary.items():
        assert entry["rhat"] is not None, f"{name} R-hat undefined for a real chain"
        assert 0.8 < entry["rhat"] < 2.0
        assert entry["ess"] is not None and entry["ess"] > 1
        assert entry["ci_2.5"] < entry["median"] < entry["ci_97.5"]


def test_posterior_predictive_check_actually_simulates(mcmc_run, topology,
                                                       growth_data):
    likelihood, _ = _two_param_setup(topology, growth_data)
    ppc = posterior_predictive_check(
        mcmc_run, likelihood, n_sim=25, rng=np.random.default_rng(0)
    )

    assert ppc["n_draws"] > 0
    counts = ppc["modalities"]["cell_counts"]
    replicates = counts["replicates"]

    # Real replicate datasets, not a reused log-likelihood scalar.
    assert replicates.shape[0] > 1
    assert replicates.shape[1] == growth_data.n_timepoints - 1  # anchor excluded
    assert not np.allclose(replicates[0], replicates[1])

    # A well-fitting model should not be at an extreme Bayesian p-value.
    assert 0.001 < counts["p_value_mean"] < 0.999
    assert counts["replicated_mean"] == pytest.approx(
        counts["observed_mean"], rel=0.5
    )


def test_posterior_predictive_check_uses_stable_initial_fractions(
    topology, growth_data
):
    """PPC must pass the rate set into _initial_state.

    Without it, initial_fractions='stable' falls back to pure-P while the
    likelihood uses the relaxed mix, so predictive replicates are drawn from
    a different initial condition than the fitted model.
    """
    from umimic.types import MCMCResult

    # Transitions are required for a non-trivial stable phenotype mix.
    names = ["b0", "d0_P", "u_PQ", "u_QP"]
    theta = np.array([TRUE_B0, TRUE_D0, 0.005, 0.003])
    likelihood = ModelLikelihood(
        topology,
        growth_data,
        param_names=names,
        mode="ode",
        initial_fractions="stable",
    )
    params = likelihood.theta_to_params(theta)
    rate_set = likelihood._build_rate_set(params)
    expected_ic = likelihood._initial_state(growth_data, rate_set)
    pure_p_ic = likelihood._initial_state(growth_data, None)
    assert not np.allclose(expected_ic, pure_p_ic), (
        "fixture rates must produce a non-trivial stable mix"
    )

    fake = MCMCResult(
        samples={
            name: np.full((1, 4), value)
            for name, value in zip(names, theta)
        },
        n_chains=1,
        n_samples=4,
    )

    recorded: list[np.ndarray] = []
    real_initial = likelihood._initial_state

    def tracking_initial(data, rate_set=None):
        state = real_initial(data, rate_set)
        recorded.append(np.asarray(state, dtype=float).copy())
        return state

    likelihood._initial_state = tracking_initial  # type: ignore[method-assign]
    try:
        ppc = posterior_predictive_check(
            fake, likelihood, n_sim=2, rng=np.random.default_rng(0)
        )
    finally:
        likelihood._initial_state = real_initial  # type: ignore[method-assign]

    assert ppc["n_draws"] > 0
    assert recorded, "PPC never called _initial_state"
    for state in recorded:
        np.testing.assert_allclose(state, expected_ic, rtol=1e-6)
        assert not np.allclose(state, pure_p_ic)


# ---------------------------------------------------------------------------
# Particle filter
# ---------------------------------------------------------------------------


def test_particle_filter_tracks_the_data(topology, growth_data):
    truth = RateSet(
        birth_base=TRUE_B0, death_base={CellType.P: TRUE_D0, CellType.Q: 0.005}
    )
    pf = ParticleFilter(
        truth, topology, lambda _t: 0.0,
        CellCountObservation(20.0, topology=topology),
        n_particles=40, rng=np.random.default_rng(1),
    )
    result = pf.filter(growth_data, np.array([300.0, 0.0]))

    assert np.isfinite(result["marginal_log_likelihood"])
    assert not result["degenerate"] and not result["truncated"]
    observed = growth_data.observations["cell_counts"]
    tracked = result["filtered_means"].sum(axis=1)
    assert tracked[-1] == pytest.approx(observed[-1], rel=0.5)


def test_marginal_likelihood_is_higher_at_the_generating_parameters(topology,
                                                                    growth_data):
    """The estimator must prefer the truth to a clearly wrong rate."""
    def run(b0):
        rates = RateSet(
            birth_base=b0, death_base={CellType.P: TRUE_D0, CellType.Q: 0.005}
        )
        pf = ParticleFilter(
            rates, topology, lambda _t: 0.0,
            CellCountObservation(20.0, topology=topology),
            n_particles=40, rng=np.random.default_rng(2),
        )
        return pf.filter(growth_data, np.array([300.0, 0.0]))[
            "marginal_log_likelihood"
        ]

    assert run(TRUE_B0) > run(TRUE_B0 * 3)
    assert run(TRUE_B0) > run(TRUE_B0 * 0.2)


def test_filter_weights_stay_normalised_without_resampling(topology, growth_data):
    pf = ParticleFilter(
        RateSet(birth_base=TRUE_B0), topology, lambda _t: 0.0,
        CellCountObservation(20.0, topology=topology),
        n_particles=30, rng=np.random.default_rng(3), ess_fraction=0.0,
    )
    result = pf.filter(growth_data, np.array([300.0, 0.0]))
    assert result["final_weights"].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Particle MCMC
# ---------------------------------------------------------------------------


def test_pmcmc_reproduces_the_prior_under_a_flat_likelihood(topology):
    """The sharpest available check on the Metropolis-Hastings machinery.

    With a constant likelihood the target *is* the prior, so the chain must
    reproduce it. The lognormal random walk is asymmetric; without the
    Hastings term ``sum(log theta' - log theta)`` the chain drifts low --
    measured at roughly 14% bias in the mean when the term is removed.
    """
    data = TimeSeriesData.from_counts(np.linspace(0, 24, 4), [100.0] * 4)
    priors = PriorSpec()
    priors.add("b0", stats.lognorm(s=0.4, scale=0.05))

    pmcmc = ParticleMCMC(
        topology, data, CellCountObservation(10.0, topology=topology), priors,
        n_particles=2, rng=np.random.default_rng(7), param_names=["b0"],
    )
    pmcmc._particle_filter_ll = lambda params: 0.0  # flat target

    result = pmcmc.sample(
        n_samples=20000, n_warmup=2000, proposal_scale=0.4,
        initial_params={"b0": 0.05},
    )
    samples = result.samples["b0"].ravel()
    target = priors.distributions["b0"]

    assert np.mean(samples) == pytest.approx(target.mean(), rel=0.05)
    assert np.median(samples) == pytest.approx(target.median(), rel=0.05)
    assert np.std(samples) == pytest.approx(target.std(), rel=0.10)


def test_pmcmc_returns_a_chain_axis_and_traces(topology):
    data = TimeSeriesData.from_counts(np.linspace(0, 24, 4), [100.0] * 4)
    priors = PriorSpec()
    priors.add("b0", stats.lognorm(s=0.4, scale=0.05))

    pmcmc = ParticleMCMC(
        topology, data, CellCountObservation(10.0, topology=topology), priors,
        n_particles=2, rng=np.random.default_rng(1), param_names=["b0"],
    )
    pmcmc._particle_filter_ll = lambda params: 0.0
    result = pmcmc.sample(n_samples=50, n_warmup=10, initial_params={"b0": 0.05})

    assert result.samples["b0"].shape == (1, 50)
    assert result.log_likelihood_trace.shape == (1, 50)
    assert result.log_posterior_trace.shape == (1, 50)
    assert 0.0 <= result.diagnostics["acceptance_rate"] <= 1.0
    assert result.diagnostics["param_names"] == ["b0"]


def test_pmcmc_proposal_and_filter_rng_streams_are_independent(topology):
    """Pseudo-marginal validity needs the likelihood noise decoupled from
    the proposal draws."""
    data = TimeSeriesData.from_counts(np.linspace(0, 24, 4), [100.0] * 4)
    priors = PriorSpec()
    priors.add("b0", stats.lognorm(s=0.4, scale=0.05))
    pmcmc = ParticleMCMC(
        topology, data, CellCountObservation(10.0, topology=topology), priors,
        n_particles=4, rng=np.random.default_rng(0), param_names=["b0"],
    )
    assert pmcmc._proposal_rng is not pmcmc._filter_rng
    a = pmcmc._proposal_rng.standard_normal(50)
    b = pmcmc._filter_rng.standard_normal(50)
    assert not np.allclose(a, b)


def test_pmcmc_rejects_a_starting_point_outside_the_prior(topology):
    data = TimeSeriesData.from_counts(np.linspace(0, 24, 4), [100.0] * 4)
    priors = PriorSpec()
    priors.add("b0", stats.uniform(loc=0.01, scale=0.01))  # support [0.01, 0.02]

    pmcmc = ParticleMCMC(
        topology, data, CellCountObservation(10.0, topology=topology), priors,
        n_particles=2, rng=np.random.default_rng(1), param_names=["b0"],
    )
    pmcmc._particle_filter_ll = lambda params: 0.0
    with pytest.raises(ValueError, match="zero posterior density"):
        pmcmc.sample(n_samples=10, n_warmup=1, initial_params={"b0": 5.0})


# ---------------------------------------------------------------------------
# PMCMC must drive the dynamics with the *whole* parameter vector
# ---------------------------------------------------------------------------


def _treated_data():
    return TimeSeriesData.from_counts(
        np.linspace(0, 48, 5), [200.0, 190.0, 175.0, 160.0, 150.0],
        concentration=5.0,
    )


def _pmcmc_ll(topology, params, n_particles=8, **kwargs):
    """Marginal log-likelihood at `params` from a freshly seeded sampler.

    Re-seeding per call is what makes the comparisons below exact: with the
    same seed and the same effective rate set, the particle draws are
    identical, so an ignored parameter shows up as *bitwise equal*
    likelihoods rather than as a small numerical difference.
    """
    priors = PriorSpec()
    pmcmc = ParticleMCMC(
        topology, _treated_data(), CellCountObservation(10.0, topology=topology),
        priors, n_particles=n_particles, rng=np.random.default_rng(11), **kwargs,
    )
    return pmcmc._particle_filter_ll(params)


@pytest.mark.parametrize(
    "name, other",
    [
        ("emax_birth", 0.9),   # cytostatic mechanism
        ("d0_Q", 0.05),        # was hardcoded to d0_P * 0.5
        ("overdispersion", 200.0),  # observation model, not the rates
    ],
)
def test_pmcmc_dynamics_see_every_sampled_parameter(topology, name, other):
    """A parameter the filter ignores would return its prior, silently.

    The sampler would still run, still converge and still report a posterior
    for such a name; the parameters that *are* wired would absorb the misfit.
    Equality here is the failure signature.
    """
    base = {
        "b0": 0.05, "d0_P": 0.01, "emax_death": 0.02, "ec50_death": 1.0,
        "hill_death": 1.5, "emax_birth": 0.1, "ec50_birth": 1.0,
        "hill_birth": 1.5, "d0_Q": 0.005, "u_PQ": 0.005, "u_QP": 0.003,
        "u_PR": 0.001, "overdispersion": 10.0,
    }
    names = sorted(base)
    changed = {**base, name: other}

    ll_base = _pmcmc_ll(topology, base, param_names=names)
    ll_changed = _pmcmc_ll(topology, changed, param_names=names)

    assert ll_base != ll_changed, f"{name} never reaches the likelihood"


@pytest.mark.parametrize("name, other", [("u_PR", 0.03), ("emax_death_R", 0.01)])
def test_pmcmc_dynamics_see_the_resistance_parameters(name, other):
    """Resistance needs a topology carrying R, so it is checked separately."""
    four_state = ModelTopology.four_state()
    base = {
        "b0": 0.05, "b0_R": 0.03, "d0_P": 0.01, "d0_R": 0.005,
        "emax_death": 0.03, "ec50_death": 1.0, "hill_death": 1.5,
        "emax_death_R": 0.002, "u_PQ": 0.005, "u_QP": 0.003, "u_PR": 0.002,
        "overdispersion": 10.0,
    }
    names = sorted(base)
    ll_base = _pmcmc_ll(four_state, base, param_names=names)
    ll_changed = _pmcmc_ll(four_state, {**base, name: other}, param_names=names)

    assert ll_base != ll_changed, f"{name} never reaches the likelihood"


def test_pmcmc_rejects_parameters_the_model_cannot_read(topology):
    priors = PriorSpec()
    with pytest.raises(ValueError, match="not read by the forward model"):
        ParticleMCMC(
            topology, _treated_data(),
            CellCountObservation(10.0, topology=topology), priors,
            param_names=["b0", "khghg_rate"],
        )


def test_pmcmc_rejects_an_unparameterised_state(topology):
    """Same guard ModelLikelihood applies: an undescribed state is immortal."""
    four_state = ModelTopology.four_state()
    priors = PriorSpec()
    with pytest.raises(ValueError, match="immortal"):
        ParticleMCMC(
            four_state, _treated_data(),
            CellCountObservation(10.0, topology=four_state), priors,
            param_names=["b0", "d0_P", "u_PQ", "u_QP"],
        )


# ---------------------------------------------------------------------------
# The anchor observation must not be scored after seeding the particles
# ---------------------------------------------------------------------------


def _filter(topology, data, **kwargs):
    pf = ParticleFilter(
        RateSet(birth_base=TRUE_B0,
                death_base={CellType.P: TRUE_D0, CellType.Q: 0.005}),
        topology, lambda _t: 0.0,
        CellCountObservation(20.0, topology=topology),
        n_particles=25, rng=np.random.default_rng(5), ess_fraction=0.0,
    )
    return pf.filter(data, np.array([300.0, 0.0]), **kwargs)


def test_anchor_observation_is_excluded_from_the_weights(topology, growth_data):
    """Seeding the cloud from y0 and then scoring y0 uses it twice.

    Excluding it must be exactly equivalent to the observation being absent,
    which is the only statement that pins the semantics rather than just
    asserting that two numbers differ.
    """
    counts = growth_data.observations["cell_counts"].copy()
    counts[0] = np.nan
    blanked = TimeSeriesData(
        times=growth_data.times,
        observations={"cell_counts": counts},
        concentration=0.0,
    )

    scored = _filter(topology, growth_data)
    skipped = _filter(
        topology, growth_data, anchor_modality="cell_counts", anchor_index=0
    )
    absent = _filter(topology, blanked)

    assert skipped["marginal_log_likelihood"] == pytest.approx(
        absent["marginal_log_likelihood"]
    )
    assert scored["marginal_log_likelihood"] != pytest.approx(
        skipped["marginal_log_likelihood"]
    )


def test_pmcmc_conditions_on_the_first_observation_by_default(topology):
    """PMCMC's contract must match ModelLikelihood's, or their marginal
    likelihoods are on different scales and cannot be compared."""
    params = {"b0": 0.05, "d0_P": 0.01}
    conditioned = _pmcmc_ll(topology, params, param_names=["b0", "d0_P"])
    scored_all = _pmcmc_ll(
        topology, params, param_names=["b0", "d0_P"], condition_on_first=False
    )
    assert conditioned != scored_all


# ---------------------------------------------------------------------------
# Hierarchical model
# ---------------------------------------------------------------------------


def _replicate_dataset(topology, birth_rates=(0.045, 0.050, 0.055)):
    from umimic.data.schemas import ExperimentalDataset

    times = np.linspace(0, 48, 6)
    series = []
    for i, b0 in enumerate(birth_rates):
        rates = RateSet(
            birth_base=b0, death_base={CellType.P: 0.01, CellType.Q: 0.005}
        )
        result = CellDynamicsODE(rates, topology, lambda _t: 0.0).solve(
            np.array([200.0, 0.0]), (0, 48), times
        )
        counts = np.round(result.populations["P"] + result.populations["Q"])
        series.append(
            TimeSeriesData(
                times=times,
                observations={"cell_counts": counts},
                concentration=0.0,
                replicate_id=f"well_{i}",
            )
        )
    return ExperimentalDataset(series=series)


@pytest.fixture(scope="module")
def hierarchical_fit(topology):
    from umimic.inference.hierarchical import HierarchicalModel

    priors = PriorSpec()
    priors.add("b0", stats.lognorm(s=0.4, scale=0.05))
    model = HierarchicalModel(
        _replicate_dataset(topology), topology,
        shared_params=[], random_effect_params=["b0"],
        priors=priors, mode="ode", rng=3,
    )
    return model, model.fit(n_samples=150, n_warmup=150)


def test_hierarchical_preserves_the_walker_axis(hierarchical_fit):
    _, result = hierarchical_fit
    for values in result.samples.values():
        assert values.ndim == 2
        assert values.shape == (result.n_chains, result.n_samples)
    assert result.log_likelihood_trace.shape == result.log_posterior_trace.shape


def test_hierarchical_diagnostics_are_computable(hierarchical_fit):
    """Flattened samples made R-hat and ESS undefined for every parameter."""
    _, result = hierarchical_fit
    summary = summarize_mcmc(result)
    assert summary["pop_mean_b0"]["rhat"] is not None
    assert summary["pop_mean_b0"]["ess"] is not None


def test_hierarchical_recovers_group_ordering_with_shrinkage(hierarchical_fit):
    _, result = hierarchical_fit
    medians = [
        float(np.median(result.samples[f"b0_well_{i}"])) for i in range(3)
    ]
    assert medians[0] < medians[1] < medians[2], "group ordering not recovered"
    population = float(np.median(result.samples["pop_mean_b0"]))
    assert population == pytest.approx(0.050, rel=0.35)
    # Partial pooling: group estimates sit inside the spread of the truths.
    assert min(medians) > 0.040 and max(medians) < 0.060


def test_hierarchical_is_reproducible(topology):
    from umimic.inference.hierarchical import HierarchicalModel

    def run(seed):
        priors = PriorSpec()
        priors.add("b0", stats.lognorm(s=0.4, scale=0.05))
        return HierarchicalModel(
            _replicate_dataset(topology), topology,
            shared_params=[], random_effect_params=["b0"],
            priors=priors, mode="ode", rng=seed,
        ).fit(n_samples=20, n_warmup=20)

    np.testing.assert_allclose(run(7).samples["pop_mean_b0"],
                               run(7).samples["pop_mean_b0"])
    assert not np.allclose(run(7).samples["pop_mean_b0"],
                           run(8).samples["pop_mean_b0"])


def test_hierarchical_group_labels_stay_unique(topology):
    """Duplicate ids used to overwrite one another in the output dict."""
    from umimic.data.schemas import ExperimentalDataset
    from umimic.inference.hierarchical import HierarchicalModel

    dataset = _replicate_dataset(topology)
    for series in dataset.series:
        series.replicate_id = "same"
    dataset = ExperimentalDataset(series=dataset.series)

    priors = PriorSpec()
    priors.add("b0", stats.lognorm(s=0.4, scale=0.05))
    model = HierarchicalModel(
        dataset, topology, shared_params=[], random_effect_params=["b0"],
        priors=priors, mode="ode", rng=1,
    )
    assert len(set(model.group_labels)) == dataset.n_series
    assert len(set(model._parameter_labels())) == model.n_dim


def test_hierarchical_validates_its_configuration(topology):
    from umimic.inference.hierarchical import HierarchicalModel

    dataset = _replicate_dataset(topology)
    priors = PriorSpec()
    priors.add("b0", stats.lognorm(s=0.4, scale=0.05))

    with pytest.raises(ValueError, match="No prior supplied"):
        HierarchicalModel(dataset, topology, ["ec50_death"], ["b0"], priors=priors)
    with pytest.raises(ValueError, match="both shared and group-varying"):
        HierarchicalModel(dataset, topology, ["b0"], ["b0"], priors=priors)
    with pytest.raises(ValueError, match="At least one parameter"):
        HierarchicalModel(dataset, topology, [], [], priors=priors)


def test_hierarchical_log_prior_rejects_non_positive_values(topology):
    from umimic.inference.hierarchical import HierarchicalModel

    priors = PriorSpec()
    priors.add("b0", stats.lognorm(s=0.4, scale=0.05))
    model = HierarchicalModel(
        _replicate_dataset(topology), topology, [], ["b0"],
        priors=priors, mode="ode", rng=1,
    )
    bad = np.full(model.n_dim, 0.05)
    bad[1] = -0.1  # negative between-group scale
    assert model.log_prior(bad) == -np.inf


def test_pmcmc_recovers_a_parameter_from_data(topology):
    """A short but genuine PMCMC run, at a size the SSA can sustain."""
    truth = RateSet(birth_base=0.05, death_base={CellType.P: 0.01, CellType.Q: 0.005})
    times = np.linspace(0, 48, 5)
    result = CellDynamicsODE(truth, topology, lambda _t: 0.0).solve(
        np.array([60.0, 0.0]), (0, 48), times
    )
    counts = np.round(result.populations["P"] + result.populations["Q"])
    data = TimeSeriesData(
        times=times, observations={"cell_counts": counts}, concentration=0.0
    )

    priors = PriorSpec()
    priors.add("b0", stats.lognorm(s=0.6, scale=0.05))
    pmcmc = ParticleMCMC(
        topology, data, CellCountObservation(15.0, topology=topology), priors,
        n_particles=20, rng=np.random.default_rng(4), param_names=["b0"],
    )
    chain = pmcmc.sample(
        n_samples=120, n_warmup=60, proposal_scale=0.25,
        initial_params={"b0": 0.05},
    )
    samples = chain.samples["b0"].ravel()
    assert np.all(samples > 0)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    assert lo < 0.05 < hi, f"true b0 outside the interval [{lo:.4f}, {hi:.4f}]"


def test_hierarchical_admits_a_zero_shared_parameter(topology):
    """A shared Emax of zero means "no drug effect" and is a legal value.

    Its prior is half-normal, whose support starts at zero, and the rate
    builder treats zero as "no modulation". Only the random-effect components
    must be strictly positive, because the LogNormal takes their log.
    """
    from umimic.inference.hierarchical import HierarchicalModel

    priors = PriorSpec()
    priors.add("b0", stats.lognorm(s=0.4, scale=0.05))
    priors.add("emax_death", stats.halfnorm(scale=0.1))
    model = HierarchicalModel(
        _replicate_dataset(topology), topology,
        shared_params=["emax_death"], random_effect_params=["b0"],
        priors=priors, mode="ode", rng=3,
    )

    n_groups = model.n_groups
    theta = np.array([0.0, 0.05, 0.1] + [0.05] * n_groups)
    assert np.isfinite(model.log_prior(theta)), "zero Emax must be admissible"

    # Negative rates remain impossible, and the random effects stay positive.
    assert not np.isfinite(model.log_prior(np.array([-0.01, 0.05, 0.1] + [0.05] * n_groups)))
    assert not np.isfinite(model.log_prior(np.array([0.02, 0.05, 0.1] + [0.0] * n_groups)))
    assert not np.isfinite(model.log_prior(np.array([0.02, 0.0, 0.1] + [0.05] * n_groups)))
    assert not np.isfinite(model.log_prior(np.array([0.02, 0.05, 0.0] + [0.05] * n_groups)))


def test_pmcmc_honours_initial_fractions(topology):
    """PMCMC always seeded pure P, ignoring the split ModelLikelihood applies.

    Starting from pure P is a structural assumption that biases the estimated
    transition rates, so the two paths disagreeing on it made their results
    incomparable.
    """
    params = {"b0": 0.05, "d0_P": 0.01, "u_PQ": 0.005, "u_QP": 0.003}
    names = sorted(params)

    default = _pmcmc_ll(topology, params, param_names=names)
    explicit_p = _pmcmc_ll(
        topology, params, param_names=names, initial_fractions={"P": 1.0}
    )
    mostly_q = _pmcmc_ll(
        topology, params, param_names=names,
        initial_fractions={"P": 0.2, "Q": 0.8},
    )

    # Spelling out the default must not change anything.
    assert default == explicit_p
    # A genuinely different split must reach the dynamics.
    assert default != mostly_q


def test_particle_filter_scores_a_bare_non_count_model(topology):
    """The single-model branch read cell_counts whatever the model measured.

    A lone BLIObservation was therefore handed counts and scored them on a
    lognormal signal scale, while its own modality went unused.
    """
    from umimic.observations.bli import BLIObservation

    times = np.linspace(0, 24, 4)
    counts = np.array([300.0, 320.0, 340.0, 360.0])

    def run(bli_values, count_values):
        data = TimeSeriesData(
            times=times,
            observations={
                "bli": np.asarray(bli_values, dtype=float),
                "cell_counts": np.asarray(count_values, dtype=float),
            },
            concentration=0.0,
        )
        pf = ParticleFilter(
            RateSet(birth_base=TRUE_B0,
                    death_base={CellType.P: TRUE_D0, CellType.Q: 0.005}),
            topology, lambda _t: 0.0,
            BLIObservation(alpha=1.0, sigma_log=0.3, topology=topology),
            n_particles=20, rng=np.random.default_rng(6), ess_fraction=0.0,
        )
        return pf.filter(data, np.array([300.0, 0.0]))["marginal_log_likelihood"]

    base = run([300.0, 320.0, 340.0, 360.0], counts)

    # The BLI values are what this model measures, so they must matter...
    assert run([30.0, 32.0, 34.0, 36.0], counts) != base
    # ...and the counts, which it does not measure, must not.
    assert run([300.0, 320.0, 340.0, 360.0], counts * 10) == base


def test_pmcmc_boundary_parameters_can_reach_zero(topology):
    """A lognormal walk lives on log theta and can never reach the boundary.

    For a parameter whose true value is "no effect" the chain drifts toward
    -inf instead of settling, so the reported posterior depends on run length.
    Parameters whose prior admits zero get a reflected linear-scale walk,
    which is symmetric and therefore needs no Hastings term.
    """
    data = TimeSeriesData.from_counts(np.linspace(0, 24, 4), [100.0] * 4)
    priors = PriorSpec()
    priors.add("emax_death", stats.halfnorm(scale=0.1))
    priors.add("b0", stats.lognorm(s=0.4, scale=0.05))

    pmcmc = ParticleMCMC(
        topology, data, CellCountObservation(10.0, topology=topology), priors,
        n_particles=2, rng=np.random.default_rng(7),
        param_names=["b0", "emax_death"],
    )
    pmcmc._particle_filter_ll = lambda params: 0.0  # flat target: prior is truth

    result = pmcmc.sample(
        n_samples=20000, n_warmup=2000, proposal_scale=1.0,
        initial_params={"b0": 0.05, "emax_death": 0.05},
    )

    # Detection is by the prior: half-normal admits zero, lognormal does not.
    assert result.diagnostics["boundary_params"] == ["emax_death"]

    emax = result.samples["emax_death"].ravel()
    target = priors.distributions["emax_death"]
    assert np.all(emax >= 0)
    assert np.mean(emax) == pytest.approx(target.mean(), rel=0.06)
    assert np.std(emax) == pytest.approx(target.std(), rel=0.10)
    # The boundary itself is reachable, which the log walk could not manage.
    assert emax.min() < 0.01 * target.std()

    # The strictly positive parameter keeps its log walk and its own prior.
    b0 = result.samples["b0"].ravel()
    assert np.median(b0) == pytest.approx(
        priors.distributions["b0"].median(), rel=0.08
    )
