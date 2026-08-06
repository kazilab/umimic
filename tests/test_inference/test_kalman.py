"""Tests for Extended Kalman Filter."""

import numpy as np
import pytest

from umimic.inference.kalman import ExtendedKalmanFilter
from umimic.dynamics.moment_equations import MomentODE
from umimic.observations.cell_counts import CellCountObservation


class TestExtendedKalmanFilter:
    def test_filter_runs(self, two_state_topology, simple_rates, sample_data):
        """EKF should run without errors."""
        moment_ode = MomentODE(simple_rates, two_state_topology, lambda t: 0.0)
        obs_model = CellCountObservation(overdispersion=10.0)
        ekf = ExtendedKalmanFilter(moment_ode, obs_model)

        mu0 = np.array([100.0, 0.0])
        result = ekf.filter(sample_data, mu0)

        assert result.filtered_means.shape == (len(sample_data.times), 2)
        assert result.filtered_covs.shape == (len(sample_data.times), 2, 2)

    def test_marginal_ll_finite(self, two_state_topology, simple_rates, sample_data):
        """Marginal log-likelihood should be finite."""
        moment_ode = MomentODE(simple_rates, two_state_topology, lambda t: 0.0)
        obs_model = CellCountObservation(overdispersion=10.0)
        ekf = ExtendedKalmanFilter(moment_ode, obs_model)

        mu0 = np.array([100.0, 0.0])
        ll = ekf.marginal_log_likelihood(sample_data, mu0)
        assert np.isfinite(ll)

    def test_filtered_means_positive(self, two_state_topology, simple_rates, sample_data):
        """Filtered means should be non-negative."""
        moment_ode = MomentODE(simple_rates, two_state_topology, lambda t: 0.0)
        obs_model = CellCountObservation(overdispersion=10.0)
        ekf = ExtendedKalmanFilter(moment_ode, obs_model)

        mu0 = np.array([100.0, 0.0])
        result = ekf.filter(sample_data, mu0)
        assert np.all(result.filtered_means >= 0)


class TestEKFHonesty:
    """The EKF must not report a likelihood for data it did not read."""

    def _ode(self, topology, rates):
        return MomentODE(rates, topology, lambda _t: 0.0)

    def test_solver_failure_invalidates_the_filter(
        self, two_state_topology, simple_rates, sample_data
    ):
        """Carrying the previous state forward fabricated a finite likelihood.

        An optimiser walks straight toward parameters whose forward solve
        fails if failure is cheaper than a bad fit.
        """
        moment_ode = self._ode(two_state_topology, simple_rates)
        ekf = ExtendedKalmanFilter(
            moment_ode, CellCountObservation(10.0, topology=two_state_topology)
        )

        def explode(*_args, **_kwargs):
            raise RuntimeError("stiff system")

        moment_ode.solve = explode
        result = ekf.filter(sample_data, np.array([100.0, 0.0]))

        assert result.diverged is True
        assert result.marginal_log_likelihood == -np.inf

    def test_missing_observations_do_not_poison_the_likelihood(
        self, two_state_topology, simple_rates
    ):
        """A NaN produced a NaN innovation that spread to every later term."""
        from umimic.data.schemas import TimeSeriesData

        times = np.array([0.0, 6.0, 12.0, 24.0])
        counts = np.array([100.0, np.nan, 140.0, 190.0])
        data = TimeSeriesData(
            times=times, observations={"cell_counts": counts}, concentration=0.0
        )
        ekf = ExtendedKalmanFilter(
            self._ode(two_state_topology, simple_rates),
            CellCountObservation(10.0, topology=two_state_topology),
        )
        ll = ekf.marginal_log_likelihood(data, np.array([100.0, 0.0]))
        assert np.isfinite(ll)

    def test_every_configured_modality_contributes(
        self, two_state_topology, simple_rates
    ):
        """Hardcoding cell counts meant BLI and volume were silently unused."""
        from umimic.data.schemas import TimeSeriesData
        from umimic.observations.multimodal import MultimodalObservation
        from umimic.observations.tumor_volume import TumorVolumeObservation

        times = np.array([0.0, 6.0, 12.0, 24.0])
        data = TimeSeriesData(
            times=times,
            observations={
                "cell_counts": np.array([100.0, 115.0, 135.0, 190.0]),
                "volume": np.array([0.10, 0.12, 0.14, 0.20]),
            },
            concentration=0.0,
        )
        counts_only = ExtendedKalmanFilter(
            self._ode(two_state_topology, simple_rates),
            CellCountObservation(10.0, topology=two_state_topology),
        ).marginal_log_likelihood(data, np.array([100.0, 0.0]))

        both = ExtendedKalmanFilter(
            self._ode(two_state_topology, simple_rates),
            MultimodalObservation(
                {
                    "cell_counts": CellCountObservation(
                        10.0, topology=two_state_topology
                    ),
                    "volume": TumorVolumeObservation(
                        beta=1e-3, sigma_v=0.2, topology=two_state_topology
                    ),
                }
            ),
        ).marginal_log_likelihood(data, np.array([100.0, 0.0]))

        assert np.isfinite(both)
        assert both != counts_only, "volume observations never entered the filter"

    def test_a_modality_without_a_gaussian_form_is_refused(
        self, two_state_topology, simple_rates
    ):
        """Skipping it would report a likelihood that ignores that data."""
        from umimic.observations.base import ObservationModel

        class Opaque(ObservationModel):
            modality_name = "biomarker"

            def log_likelihood(self, observed, latent_state, params=None,
                               process_variance=None):
                return 0.0

            def sample(self, latent_state, rng, params=None):
                return 0.0

            def param_names(self):
                return []

        with pytest.raises(ValueError, match="no Gaussian linearization"):
            ExtendedKalmanFilter(
                self._ode(two_state_topology, simple_rates), Opaque()
            )

    def test_observation_operator_follows_the_topology(self, simple_rates):
        """A [P, Q, R] model has no apoptotic state at index 2.

        The hand-written Jacobian assumed the canonical ordering, so a custom
        topology would have scored the wrong compartment.
        """
        from umimic.dynamics.states import CellType, ModelTopology

        pqr = ModelTopology(
            active_states=[CellType.P, CellType.Q, CellType.R],
            transitions=[(CellType.P, CellType.Q), (CellType.P, CellType.R)],
            division_states=[CellType.P, CellType.R],
            death_states=[CellType.P, CellType.Q, CellType.R],
        )
        model = CellCountObservation(10.0)  # deliberately topology-free
        ExtendedKalmanFilter(MomentODE(simple_rates, pqr, lambda _t: 0.0), model)

        # The filter attached its topology, so R counts as viable.
        np.testing.assert_allclose(model.operator("viable", 3), [1.0, 1.0, 1.0])

    def test_anchor_observation_can_be_excluded(
        self, two_state_topology, simple_rates
    ):
        """Seeding initial_mu from y0 and then scoring y0 uses it twice.

        Excluding it must be exactly equivalent to that observation being
        absent -- the same contract ModelLikelihood and ParticleFilter apply.
        """
        from umimic.data.schemas import TimeSeriesData

        times = np.array([0.0, 6.0, 12.0, 24.0])
        counts = np.array([100.0, 115.0, 135.0, 190.0])
        data = TimeSeriesData(
            times=times, observations={"cell_counts": counts}, concentration=0.0
        )
        blanked = counts.copy()
        blanked[0] = np.nan
        without_anchor = TimeSeriesData(
            times=times,
            observations={"cell_counts": blanked},
            concentration=0.0,
        )

        def run(series, **kwargs):
            ekf = ExtendedKalmanFilter(
                self._ode(two_state_topology, simple_rates),
                CellCountObservation(10.0, topology=two_state_topology),
            )
            return ekf.marginal_log_likelihood(
                series, np.array([100.0, 0.0]), **kwargs
            )

        scored = run(data)
        skipped = run(data, anchor_modality="cell_counts", anchor_index=0)
        absent = run(without_anchor)

        assert skipped == pytest.approx(absent)
        assert scored != pytest.approx(skipped)
