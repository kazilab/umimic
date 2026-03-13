"""Integration test: generate synthetic data with known parameters, fit, and verify recovery."""

import numpy as np
import pytest

from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.rates import RateSet, EmaxHill
from umimic.observations.cell_counts import CellCountObservation
from umimic.data.synthetic import SyntheticDataGenerator
from umimic.inference.likelihood import ModelLikelihood
from umimic.inference.mle import MLEstimator
from umimic.inference.priors import PriorSpec


class TestSimulateAndInfer:
    """End-to-end integration test: simulate -> infer -> verify parameter recovery."""

    @pytest.mark.slow
    def test_parameter_recovery_cytotoxic(self):
        """Generate data from a cytotoxic drug model and recover parameters via MLE."""
        # True parameters
        true_b0 = 0.04
        true_d0 = 0.01
        true_emax = 0.06
        true_ec50 = 2.0
        true_hill = 1.5

        topology = ModelTopology.two_state()
        true_rates = RateSet(
            birth_base=true_b0,
            death_base={CellType.P: true_d0, CellType.Q: true_d0 * 0.5},
            death_modulation={
                CellType.P: EmaxHill(emax=true_emax, ec50=true_ec50, hill=true_hill),
            },
            transition_base={
                (CellType.P, CellType.Q): 0.003,
                (CellType.Q, CellType.P): 0.002,
            },
        )

        # Generate synthetic data
        gen = SyntheticDataGenerator(
            true_rates, topology,
            CellCountObservation(overdispersion=15.0),
            np.random.default_rng(42),
        )
        y0 = np.array([200.0, 0.0])
        dataset = gen.generate_invitro_plate(
            initial_cells=y0,
            concentrations=[0, 0.3, 1.0, 3.0, 10.0],
            n_wells_per_dose=3,
            t_max=72.0,
            dt_obs=6.0,
            method="ode",
        )

        # Fit via MLE
        param_names = [
            "b0", "d0_P", "emax_death", "ec50_death",
            "hill_death", "u_PQ", "u_QP", "overdispersion",
        ]
        likelihood = ModelLikelihood(
            topology=topology,
            data=dataset.series,
            param_names=param_names,
            mode="ode",
        )
        priors = PriorSpec.default_invitro()
        estimator = MLEstimator(likelihood, priors=priors, method="L-BFGS-B")
        result = estimator.fit(n_restarts=3)

        # Verify: estimates should be in the right ballpark
        # (exact recovery unlikely with noise, but order of magnitude correct)
        assert result.parameters["b0"] > 0.01
        assert result.parameters["b0"] < 0.1
        assert result.parameters["d0_P"] > 0.001
        assert result.parameters["d0_P"] < 0.05

    def test_likelihood_at_true_params(self):
        """Log-likelihood at true parameters should be higher than at random params."""
        topology = ModelTopology.two_state()
        true_rates = RateSet(
            birth_base=0.04,
            death_base={CellType.P: 0.01, CellType.Q: 0.005},
            transition_base={
                (CellType.P, CellType.Q): 0.005,
                (CellType.Q, CellType.P): 0.003,
            },
        )

        gen = SyntheticDataGenerator(
            true_rates, topology,
            CellCountObservation(overdispersion=10.0),
            np.random.default_rng(123),
        )
        y0 = np.array([150.0, 0.0])
        dataset = gen.generate_invitro_plate(
            initial_cells=y0,
            concentrations=[0, 1.0, 10.0],
            n_wells_per_dose=2,
            t_max=48.0,
            dt_obs=8.0,
            method="ode",
        )

        likelihood = ModelLikelihood(
            topology=topology,
            data=dataset.series,
            mode="ode",
        )

        # True-ish params
        theta_true = np.array([0.04, 0.01, 0.0, 1.0, 1.5, 0.005, 0.003, 10.0])
        # Clearly wrong params
        theta_bad = np.array([0.01, 0.1, 0.0, 1.0, 1.5, 0.005, 0.003, 10.0])

        ll_true = likelihood(theta_true)
        ll_bad = likelihood(theta_bad)

        assert ll_true > ll_bad
