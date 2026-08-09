"""Tests for phenotype plasticity: induced transitions and the persister route.

The multiplicative transition law ``u = u0 * (1 + m)`` cannot express a
transition that is absent without drug, because ``0 * anything = 0``. The
additive induction term exists for exactly that case, and the persister route
P -> Q -> R is the model that needs it.
"""

from __future__ import annotations

import numpy as np
import pytest

from umimic.dynamics.ode_system import CellDynamicsODE
from umimic.dynamics.rates import (
    EmaxHill,
    FourParameterLogistic,
    HillFoldChange,
    PhenotypeRateProfile,
    RateSet,
    TransitionRateProfile,
)
from umimic.dynamics.states import CellType, ModelTopology


# ---------------------------------------------------------------------------
# Additive induction: the capability a multiplicative law cannot reach
# ---------------------------------------------------------------------------


def test_multiplicative_modulation_cannot_create_a_transition():
    """Pin the limitation that motivates the additive term."""
    rates = RateSet(
        transition_base={(CellType.P, CellType.Q): 0.0},
        transition_modulation={
            (CellType.P, CellType.Q): EmaxHill(emax=50.0, ec50=1.0)
        },
    )
    for c in (0.0, 1.0, 100.0):
        assert rates.transition_rate(CellType.P, CellType.Q, c) == 0.0


def test_induction_creates_a_drug_induced_transition_from_zero_baseline():
    rates = RateSet(
        transition_base={(CellType.P, CellType.Q): 0.0},
        transition_induction={
            (CellType.P, CellType.Q): EmaxHill(emax=8e-3, ec50=0.5, hill=2.0)
        },
    )
    assert rates.transition_rate(CellType.P, CellType.Q, 0.0) == 0.0
    assert rates.transition_rate(CellType.P, CellType.Q, 0.5) == pytest.approx(4e-3)
    assert rates.transition_rate(CellType.P, CellType.Q, 100.0) > 7.9e-3


def test_induction_adds_to_an_existing_baseline():
    rates = RateSet(
        transition_base={(CellType.P, CellType.Q): 1e-3},
        transition_induction={
            (CellType.P, CellType.Q): EmaxHill(emax=4e-3, ec50=1.0, hill=1.0)
        },
    )
    assert rates.transition_rate(CellType.P, CellType.Q, 0.0) == pytest.approx(1e-3)
    assert rates.transition_rate(CellType.P, CellType.Q, 1.0) == pytest.approx(3e-3)


# ---------------------------------------------------------------------------
# Fold-change: suppression without a sign hack
# ---------------------------------------------------------------------------


def test_fold_change_suppresses_a_transition():
    """Drug blocking resensitisation, expressed as a multiplier."""
    rates = RateSet(
        transition_base={(CellType.Q, CellType.P): 3e-3},
        transition_factor={
            (CellType.Q, CellType.P): HillFoldChange(
                low=1.0, high=0.1, ec50=0.5, hill=2.0
            )
        },
    )
    values = [rates.transition_rate(CellType.Q, CellType.P, c) for c in (0.0, 0.5, 50.0)]
    assert values[0] == pytest.approx(3e-3)
    assert values[0] > values[1] > values[2]
    assert values[2] == pytest.approx(3e-4, rel=0.05)


def test_fold_change_can_also_induce():
    rates = RateSet(
        transition_base={(CellType.P, CellType.Q): 1e-3},
        transition_factor={
            (CellType.P, CellType.Q): HillFoldChange(low=1.0, high=10.0, ec50=1.0)
        },
    )
    assert rates.transition_rate(CellType.P, CellType.Q, 1e6) == pytest.approx(
        1e-2, rel=1e-3
    )


def test_fold_change_is_exempt_from_the_increasing_rule():
    """A falling multiplier is legitimate; a falling *modulation* is not."""
    suppressor = HillFoldChange(low=1.0, high=0.1, ec50=0.5)
    assert not suppressor.is_increasing()
    RateSet(transition_factor={(CellType.Q, CellType.P): suppressor})  # allowed

    with pytest.raises(ValueError, match="HillFoldChange"):
        RateSet(
            transition_modulation={
                (CellType.Q, CellType.P): FourParameterLogistic(top=1.0, bottom=0.0)
            }
        )


def test_negative_fold_change_is_rejected():
    with pytest.raises(ValueError):
        HillFoldChange(low=1.0, high=-0.5)


def test_induction_must_still_increase_with_concentration():
    with pytest.raises(ValueError, match="transition_induction"):
        RateSet(
            transition_induction={
                (CellType.P, CellType.Q): FourParameterLogistic(top=1.0, bottom=0.0)
            }
        )


def test_all_three_transition_components_compose():
    rates = RateSet(
        transition_base={(CellType.P, CellType.Q): 2e-3},
        transition_factor={
            (CellType.P, CellType.Q): HillFoldChange(low=1.0, high=2.0, ec50=1.0)
        },
        transition_induction={
            (CellType.P, CellType.Q): EmaxHill(emax=1e-3, ec50=1.0, hill=1.0)
        },
    )
    # At C = EC50 both curves sit at half: 2e-3 * 1.5 + 5e-4
    assert rates.transition_rate(CellType.P, CellType.Q, 1.0) == pytest.approx(3.5e-3)


# ---------------------------------------------------------------------------
# Phenotype profiles
# ---------------------------------------------------------------------------


def test_from_profiles_builds_independent_phenotype_responses():
    topo = ModelTopology.persister_resistance()
    rates = RateSet.from_profiles(
        {
            CellType.P: PhenotypeRateProfile(
                0.04, 0.01, EmaxHill(emax=0.9, ec50=1.0)
            ),
            CellType.Q: PhenotypeRateProfile(0.002, 0.003),
            CellType.R: PhenotypeRateProfile(0.036, 0.01),
        },
        topology=topo,
    )
    # P is suppressed by drug; Q and R carry their own, unmodulated rates.
    assert rates.birth_rate(100.0, cell_type=CellType.P) < 0.005
    assert rates.birth_rate(100.0, cell_type=CellType.Q) == pytest.approx(0.002)
    assert rates.birth_rate(100.0, cell_type=CellType.R) == pytest.approx(0.036)


def test_from_profiles_refuses_to_let_a_state_inherit_p_response():
    """A dividing state with no profile used to silently become drug-sensitive."""
    topo = ModelTopology.persister_resistance()
    with pytest.raises(ValueError, match="No PhenotypeRateProfile"):
        RateSet.from_profiles(
            {CellType.P: PhenotypeRateProfile(0.04, 0.01)}, topology=topo
        )


def test_profiles_validate_their_inputs():
    with pytest.raises(ValueError):
        PhenotypeRateProfile(birth_base=-1.0, death_base=0.01)
    with pytest.raises(ValueError):
        TransitionRateProfile(base_rate=-1.0)
    with pytest.raises(ValueError, match="differ"):
        RateSet.from_profiles(
            {CellType.P: PhenotypeRateProfile(0.04, 0.01)},
            {(CellType.P, CellType.P): TransitionRateProfile(1e-3)},
        )


# ---------------------------------------------------------------------------
# Persister-resistance topology
# ---------------------------------------------------------------------------


def test_persister_topology_routes_resistance_through_Q():
    topo = ModelTopology.persister_resistance()
    edges = set(topo.transitions)
    assert (CellType.Q, CellType.R) in edges
    assert (CellType.P, CellType.R) not in edges  # no direct shortcut
    assert (CellType.R, CellType.Q) not in edges  # reversion off by default
    # Persisters are slow-cycling, not arrested.
    assert CellType.Q in topo.division_states


def test_reversion_and_direct_route_are_opt_in():
    assert (CellType.R, CellType.Q) in set(
        ModelTopology.persister_resistance(include_reversion=True).transitions
    )
    assert (CellType.P, CellType.R) in set(
        ModelTopology.persister_resistance(include_direct_pr=True).transitions
    )


def test_persister_route_is_absent_without_drug():
    topo = ModelTopology.persister_resistance()
    rates = RateSet.persister_resistance(topology=topo)
    assert rates.transition_rate(CellType.Q, CellType.R, 0.0) == 0.0
    assert rates.transition_rate(CellType.Q, CellType.R, 10.0) > 0.0


def test_reversion_is_off_by_default():
    topo = ModelTopology.persister_resistance(include_reversion=True)
    off = RateSet.persister_resistance(topology=topo)
    on = RateSet.persister_resistance(topology=topo, epigenetic_reversion=1e-4)
    assert off.transition_rate(CellType.R, CellType.Q, 0.0) == 0.0
    assert on.transition_rate(CellType.R, CellType.Q, 0.0) == pytest.approx(1e-4)


def test_persister_model_shows_nadir_then_relapse():
    """The behaviour a resistance model exists to reproduce."""
    topo = ModelTopology.persister_resistance()
    rates = RateSet.persister_resistance(topology=topo)
    ode = CellDynamicsODE(rates, topo, lambda t: 3.0)
    times = np.array([0.0, 300.0, 1500.0])
    result = ode.solve(np.array([1000.0, 0.0, 0.0, 0.0]), (0, 1500), times)

    total = sum(result.populations[k] for k in ("P", "Q", "R"))
    assert total[1] < total[0], "no nadir: treatment did not reduce the population"
    assert total[2] > total[0], "no relapse: resistance never took over"
    # Relapse is carried by R, and the sensitive compartment is gone.
    assert result.populations["R"][2] > result.populations["P"][2]


def test_persister_model_drug_works_at_high_dose():
    """GR must eventually reach cytostasis; illustrative defaults must not
    describe a drug that fails at every concentration."""
    topo = ModelTopology.persister_resistance()
    rates = RateSet.persister_resistance(topology=topo)
    assert rates.gr_value(0.0, topo) == pytest.approx(1.0)
    assert rates.gr_value(1000.0, topo) < 0.0
    # Monotone in dose.
    values = [rates.gr_value(c, topo) for c in (0.0, 1.0, 10.0, 100.0, 1000.0)]
    assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# GR metric and the shared rate matrix
# ---------------------------------------------------------------------------


def test_gr_value_matches_its_definition():
    topo = ModelTopology.two_state()
    rates = RateSet.cytostatic_drug(b0=0.05, emax_birth=1.0, ec50_birth=1.0)
    g0 = rates.asymptotic_growth_rate(0.0, topo)
    for c in (0.5, 5.0):
        g = rates.asymptotic_growth_rate(c, topo)
        assert rates.gr_value(c, topo) == pytest.approx(2 ** (g / g0) - 1)
    assert rates.gr_value(0.0, topo) == pytest.approx(1.0)


def test_gr_requires_a_growing_reference():
    topo = ModelTopology.two_state()
    dying = RateSet(birth_base=0.0, death_base={CellType.P: 0.1, CellType.Q: 0.1})
    with pytest.raises(ValueError, match="positive reference"):
        dying.gr_value(1.0, topo)


def test_low_density_matrix_underpins_growth_and_fractions():
    topo = ModelTopology.two_state()
    rates = RateSet()
    matrix = rates.low_density_rate_matrix(0.0, topo)
    assert matrix.shape == (2, 2)  # A excluded
    assert rates.asymptotic_growth_rate(0.0, topo) == pytest.approx(
        float(np.max(np.linalg.eigvals(matrix).real))
    )
    fractions = rates.stable_state_fractions(0.0, topo)
    assert fractions.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Configuration reaches the topology
# ---------------------------------------------------------------------------


def test_config_can_express_the_persister_route():
    from umimic.pipeline.config import ExperimentConfig
    from umimic.pipeline.experiment import Experiment

    config = ExperimentConfig()
    config.dynamics.states = ["P", "Q", "A", "R"]
    config.dynamics.transitions = [("P", "Q"), ("Q", "P"), ("Q", "R")]
    config.dynamics.transition_rates = {"P->Q": 0.005, "Q->P": 0.003}

    exp = Experiment(config)
    edges = set(exp.topology.transitions)
    assert (CellType.Q, CellType.R) in edges
    assert (CellType.P, CellType.R) not in edges
    # Unspecified edges default to zero rather than inheriting P<->Q defaults.
    assert exp.rate_set.transition_rate(CellType.Q, CellType.R, 0.0) == 0.0
    assert exp.rate_set.transition_rate(CellType.P, CellType.Q, 0.0) == pytest.approx(
        0.005
    )


def test_config_can_express_induced_and_suppressed_transitions():
    """The additive induction term was library-only and unreachable from YAML."""
    from umimic.pipeline.config import ExperimentConfig
    from umimic.pipeline.experiment import Experiment

    config = ExperimentConfig()
    config.dynamics.states = ["P", "Q", "A", "R"]
    config.dynamics.transitions = [("P", "Q"), ("Q", "P"), ("Q", "R")]
    config.dynamics.transition_rates = {"P->Q": 1e-4, "Q->P": 3e-3}
    config.dynamics.induced_transitions = {
        "Q->R": {"emax": 2e-4, "ec50": 1.0, "hill": 2.0}
    }
    config.dynamics.transition_fold_change = {
        "Q->P": {"low": 1.0, "high": 0.1, "ec50": 0.5, "hill": 2.0}
    }
    rates = Experiment(config).rate_set

    # Induced: absent without drug, present under it.
    assert rates.transition_rate(CellType.Q, CellType.R, 0.0) == 0.0
    assert rates.transition_rate(CellType.Q, CellType.R, 10.0) > 0.0
    # Suppressed: falls with dose.
    assert rates.transition_rate(CellType.Q, CellType.P, 0.0) > rates.transition_rate(
        CellType.Q, CellType.P, 10.0
    )


def test_config_division_states_match_the_named_constructor():
    """The same model must not differ by construction route."""
    from umimic.pipeline.config import ExperimentConfig
    from umimic.pipeline.experiment import Experiment

    config = ExperimentConfig()
    config.dynamics.states = ["P", "Q", "A", "R"]
    config.dynamics.transitions = [("P", "Q"), ("Q", "P"), ("Q", "R")]
    config.dynamics.division_states = ["P", "Q", "R"]

    built = Experiment(config).topology
    named = ModelTopology.persister_resistance()
    assert set(built.division_states) == set(named.division_states)
    assert set(built.transitions) == set(named.transitions)


def test_config_validates_division_states_and_drug_specs():
    from umimic.pipeline.config import DynamicsConfig

    with pytest.raises(ValueError, match="not in states"):
        DynamicsConfig(states=["P", "Q"], division_states=["R"])
    with pytest.raises(ValueError, match="cannot divide"):
        DynamicsConfig(states=["P", "Q", "A"], division_states=["P", "A"])
    with pytest.raises(ValueError, match="missing"):
        DynamicsConfig(
            states=["P", "Q"],
            transitions=[("P", "Q")],
            induced_transitions={"P->Q": {"ec50": 1.0}},  # no emax
        )
    with pytest.raises(ValueError, match="not in transitions"):
        DynamicsConfig(
            states=["P", "Q"],
            transitions=[("P", "Q")],
            induced_transitions={"Q->P": {"emax": 1e-3}},
        )


def test_config_rejects_malformed_transitions():
    from umimic.pipeline.config import DynamicsConfig

    with pytest.raises(ValueError, match="not in states"):
        DynamicsConfig(states=["P", "Q"], transitions=[("P", "R")])
    with pytest.raises(ValueError, match="self-loop"):
        DynamicsConfig(states=["P", "Q"], transitions=[("P", "P")])
    with pytest.raises(ValueError, match="apoptotic"):
        DynamicsConfig(states=["P", "Q", "A"], transitions=[("P", "A")])
    with pytest.raises(ValueError, match="Duplicate"):
        DynamicsConfig(states=["P", "Q"], transitions=[("P", "Q"), ("P", "Q")])
    with pytest.raises(ValueError, match="not in transitions"):
        DynamicsConfig(
            states=["P", "Q"],
            transitions=[("P", "Q")],
            transition_rates={"Q->P": 0.1},
        )


def test_transitions_survive_a_yaml_round_trip(tmp_path):
    """Tuples dumped as !!python/tuple cannot be safe_load-ed back."""
    from umimic.pipeline.config import ExperimentConfig, load_config, save_config
    from umimic.pipeline.experiment import Experiment

    config = ExperimentConfig()
    config.dynamics.states = ["P", "Q", "A", "R"]
    config.dynamics.transitions = [("P", "Q"), ("Q", "P"), ("Q", "R")]
    config.dynamics.transition_rates = {"P->Q": 0.005}

    path = tmp_path / "config.yaml"
    save_config(config, path)
    restored = load_config(path)

    assert restored.dynamics.transitions == [("P", "Q"), ("Q", "P"), ("Q", "R")]
    assert (CellType.Q, CellType.R) in set(Experiment(restored).topology.transitions)


def test_default_config_keeps_the_canonical_edges():
    from umimic.pipeline.config import ExperimentConfig
    from umimic.pipeline.experiment import Experiment

    config = ExperimentConfig()
    config.dynamics.states = ["P", "Q"]
    exp = Experiment(config)
    assert set(exp.topology.transitions) == {
        (CellType.P, CellType.Q),
        (CellType.Q, CellType.P),
    }


# ---------------------------------------------------------------------------
# Identifiability diagnostic
# ---------------------------------------------------------------------------


def test_identifiability_flags_a_parameter_the_data_cannot_see():
    from umimic.inference.identifiability import analyze_identifiability

    # y depends on a and b only through their product; c is absent entirely.
    def predict(p):
        return np.log(p["a"] * p["b"]) * np.linspace(1.0, 2.0, 20)

    report = analyze_identifiability(predict, {"a": 2.0, "b": 3.0, "c": 5.0})
    assert "c" in report.unidentifiable()
    assert report.n_identifiable == 1  # only the product is constrained
    assert "unidentifiable" in report.summary()


def test_identifiability_recognises_a_well_determined_design():
    from umimic.inference.identifiability import analyze_identifiability

    def predict(p):
        t = np.linspace(0, 10, 30)
        return p["a"] * t + p["b"] * t**2

    report = analyze_identifiability(predict, {"a": 1.0, "b": 0.5})
    assert report.n_identifiable == 2
    assert not report.unidentifiable()


def test_ec50_needs_multiple_concentrations_to_be_identifiable():
    """A single dose cannot locate a half-maximal concentration."""
    from umimic.data.schemas import TimeSeriesData
    from umimic.inference.identifiability import likelihood_identifiability
    from umimic.inference.likelihood import ModelLikelihood

    topo = ModelTopology.two_state()
    times = np.linspace(0, 72, 10)
    theta = np.array([0.05, 0.01, 0.04, 1.0, 1.5, 0.005, 0.003, 10.0])

    def series(conc):
        counts = 500 * np.exp(0.02 * times)
        return TimeSeriesData(
            times=times, observations={"cell_counts": counts}, concentration=conc
        )

    one = likelihood_identifiability(
        ModelLikelihood(topo, series(5.0), mode="moment"), theta
    )
    many = likelihood_identifiability(
        ModelLikelihood(topo, [series(c) for c in (0.0, 0.3, 1.0, 3.0, 10.0)],
                        mode="moment"),
        theta,
    )
    idx = one.param_names.index("ec50_death")
    assert one.scores[idx] < many.scores[idx]

    # `n_identifiable` is deliberately not asserted here. The analysis is now
    # based on the observed Fisher information, which is only PSD at a
    # stationary point; `theta` above is a hand-picked operating point, not a
    # fitted one, so directions with upward curvature are clipped to zero and
    # the rank undercounts for both designs alike. The per-parameter score is
    # the claim this test is about.

    # A noise parameter must be assessable at all: under the old mean-based
    # sensitivity it had an identically zero column and was always reported
    # unidentifiable, which is the opposite of the truth.
    od = one.param_names.index("overdispersion")
    assert one.scores[od] > 0.1


# ---------------------------------------------------------------------------
# Finite-horizon GR
# ---------------------------------------------------------------------------


def test_finite_horizon_gr_converges_to_the_asymptotic_value():
    """With no slow-emerging state, a long assay must recover the eigenvalue."""
    topology = ModelTopology.two_state()
    rates = RateSet.cytotoxic_drug(b0=0.05, d0=0.01, emax_death=0.04, ec50_death=1.0)
    x0 = np.array([1000.0, 0.0])

    asymptotic = rates.gr_value(1.0, topology)
    windows = [
        rates.finite_horizon_gr(1.0, topology, x0, T) for T in (24.0, 240.0, 2000.0)
    ]
    # Monotonically approaching the asymptote.
    errors = [abs(v - asymptotic) for v in windows]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.01


def test_finite_horizon_gr_disagrees_with_asymptotic_when_R_dominates():
    """The trap gr_value's docstring warns about, now with an alternative.

    A rare resistant clone sets the dominant eigenvalue at every dose, so the
    asymptotic metric reports a drug that barely works while a 72-hour assay
    measures substantial kill.
    """
    topology = ModelTopology.persister_resistance()
    rates = RateSet.persister_resistance(topology=topology)
    x0 = np.array([1000.0, 0.0, 0.0, 0.0])

    asymptotic = rates.gr_value(1.0, topology)
    assay = rates.finite_horizon_gr(1.0, topology, x0, 72.0)

    assert asymptotic > 0.5, "expected the eigenvalue to be governed by R"
    assert assay < 0.0, "expected net kill over the assay window"


def test_finite_horizon_gr_is_monotone_in_dose():
    topology = ModelTopology.two_state()
    rates = RateSet.cytotoxic_drug(b0=0.05, d0=0.01, emax_death=0.05, ec50_death=1.0)
    x0 = np.array([1000.0, 0.0])
    values = [
        rates.finite_horizon_gr(c, topology, x0, 72.0)
        for c in (0.0, 0.5, 2.0, 20.0)
    ]
    assert values[0] == pytest.approx(1.0, abs=1e-6)  # no drug, no effect
    assert values == sorted(values, reverse=True)


def test_finite_horizon_gr_validates_its_inputs():
    topology = ModelTopology.two_state()
    rates = RateSet.cytotoxic_drug()
    x0 = np.array([1000.0, 0.0])

    with pytest.raises(ValueError, match="duration must be positive"):
        rates.finite_horizon_gr(1.0, topology, x0, 0.0)
    with pytest.raises(ValueError, match="no viable cells"):
        rates.finite_horizon_gr(1.0, topology, np.zeros(2), 72.0)

    # A control that does not grow leaves GR undefined rather than silently
    # dividing by a non-positive denominator.
    dying = RateSet(birth_base=0.0, death_base={CellType.P: 0.1, CellType.Q: 0.1})
    with pytest.raises(ValueError, match="does not grow"):
        dying.finite_horizon_gr(1.0, topology, x0, 72.0)
