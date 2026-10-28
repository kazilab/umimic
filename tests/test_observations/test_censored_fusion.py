"""Censored observations must contribute in batch and filter workflows."""

import numpy as np
import pytest

from umimic.data.schemas import TimeSeriesData
from umimic.dynamics.moment_equations import MomentODE
from umimic.dynamics.rates import RateSet
from umimic.dynamics.states import CellType, ModelTopology
from umimic.inference.kalman import ExtendedKalmanFilter
from umimic.inference.likelihood import ModelLikelihood
from umimic.observations.bli import BLIObservation
from umimic.observations.cell_counts import CellCountObservation
from umimic.observations.multimodal import MultimodalObservation
from umimic.observations.tumor_volume import TumorVolumeObservation


@pytest.mark.parametrize("reading", [0., 1., 50., 100.])
def test_censored_bli_is_scored_with_its_cdf_even_with_two_other_modalities(reading):
    topology = ModelTopology(
        active_states=[CellType.P], transitions=[],
        division_states=[CellType.P], death_states=[CellType.P],
    )
    models = {
        "cell_counts": CellCountObservation(topology=topology, moment_family="gaussian"),
        "bli": BLIObservation(alpha=1., lod=100., topology=topology),
        "volume": TumorVolumeObservation(beta=1., topology=topology),
    }
    data = TimeSeriesData(
        times=[0.], observations={"cell_counts": [100.], "bli": [reading], "volume": [100.]},
    )
    likelihood = ModelLikelihood(
        topology, data, param_names=["b0"], mode="moment", condition_on_first=False,
        observation_model=MultimodalObservation(models),
    )
    expected = sum(
        model.log_likelihood(data.observations[name][0], np.array([100.]), process_variance=100.)
        for name, model in models.items()
    )
    assert likelihood(np.array([.04])) == pytest.approx(expected)


@pytest.mark.parametrize("model", [BLIObservation(lod=100.), TumorVolumeObservation(lod=100.)])
@pytest.mark.parametrize("reading", [0., 50., 100.])
def test_censored_positive_and_zero_readings_have_no_exact_gaussian_update(model, reading):
    assert model.linearize(reading, np.array([100., 0.])) is None


@pytest.mark.parametrize("model", [BLIObservation(lod=100.), TumorVolumeObservation(lod=100.)])
def test_ekf_explicitly_refuses_censored_observations(model):
    topology = ModelTopology.two_state()
    data = TimeSeriesData(times=[0.], observations={model.modality_name: [50.]})
    ode = MomentODE(RateSet(), topology, lambda _: 0.)
    with pytest.raises(ValueError, match="Censored observations require"):
        ExtendedKalmanFilter(ode, model).filter(data, np.array([100., 0.]))


@pytest.mark.parametrize("model", [BLIObservation(lod=0.), TumorVolumeObservation(lod=0.)])
def test_ekf_reports_nonpositive_reading_without_lod_as_invalid_not_censored(model):
    topology = ModelTopology.two_state()
    data = TimeSeriesData(times=[0.], observations={model.modality_name: [0.]})
    ode = MomentODE(RateSet(), topology, lambda _: 0.)
    with pytest.raises(ValueError, match="no limit of detection is set"):
        ExtendedKalmanFilter(ode, model).filter(data, np.array([100., 0.]))


def test_ekf_diverges_rather_than_blaming_the_reading_on_a_nonfinite_state(monkeypatch):
    topology = ModelTopology.two_state()
    model = BLIObservation(lod=100.)
    data = TimeSeriesData(times=[0., 1.], observations={"bli": [5000., 5000.]})
    ode = MomentODE(RateSet(), topology, lambda _: 0.)
    monkeypatch.setattr(ode, "solve", lambda mu, Sigma, t_span, t_eval: (
        t_eval, np.full((1, 2), np.nan), np.full((1, 2, 2), np.nan)))
    result = ExtendedKalmanFilter(ode, model).filter(data, np.array([100., 0.]))
    assert result.diverged
    assert result.marginal_log_likelihood == -np.inf
    assert np.all(np.isnan(result.filtered_means[1]))
