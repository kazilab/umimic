"""What an experimental design can tell you, before running it.

:func:`evaluate_design` takes an :class:`~umimic.pipeline.experiment.Experiment`
whose configuration describes a design -- concentrations or dosing, the
observation schedule, replicates, readouts -- and asks how well that design
would constrain the model's parameters at given values. It builds the data the
design would produce on average (the noise-free expected observations), scores
it with the same likelihood a fit would use, and reads the curvature of the
log-likelihood there: the expected Fisher information on log parameters.

This is local and approximate. It describes precision near the assumed
parameter values, not accuracy, and it says nothing about a design whose
model is wrong. Its use is to compare candidate designs on the same footing:
which one pins down which parameters, and roughly how tightly.

Observation-noise parameters (overdispersion, BLI and volume SDs, marker
precision) are held at their configured or given values: noise-free expected
data cannot measure noise, and its curvature in them would be meaningless.
A parameter identified only in combination with others -- birth and death
rates under the ODE likelihood, which sees only their difference -- is
reported as confounded, with no standard error, rather than given the
misleading finite one a pseudo-inverse would produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from umimic.data.schemas import TimeSeriesData
from umimic.inference.identifiability import likelihood_identifiability
from umimic.inference.likelihood import OBSERVATION_PARAM_NAMES, ModelLikelihood

#: Score below which a parameter counts as not constrained by the design.
SCORE_CUTOFF = 0.1
#: Score above which a parameter is constrained on its own; between the two,
#: only in combination with others.
SEPARATE_CUTOFF = 0.99

#: Observation-noise parameters and the configuration fields that hold them.
_NOISE_FIELDS = {
    "overdispersion": "cell_count_overdispersion",
    "sigma_log_bli": "bli_sigma_log",
    "sigma_v": "volume_sigma",
    "biomarker_precision": "biomarker_precision",
}


@dataclass
class DesignReport:
    """How well one design constrains each parameter, at given values."""

    param_names: list[str]
    values: dict[str, float]
    #: Fraction of each parameter's direction the design can see, in [0, 1].
    scores: dict[str, float]
    #: Approximate standard error of log(theta): about the relative SE.
    #: None unless the design constrains the parameter on its own.
    relative_se: dict[str, float | None]
    #: "constrained", "confounded" (only in combination) or "not constrained".
    status: dict[str, str]
    n_identifiable: int
    condition_number: float
    n_observations: int
    #: Noise parameters held fixed at these values.
    fixed: dict[str, float] = field(default_factory=dict)
    #: Parameters at exactly zero, whose log-scale precision is undefined.
    unassessable: list[str] = field(default_factory=list)


def evaluate_design(experiment, parameters: dict[str, float] | None = None) -> DesignReport:
    """Expected precision of the parameters under ``experiment``'s design.

    Args:
        experiment: The experiment; its configuration is the design.
        parameters: Parameter values to evaluate at, by name; omitted names
            take the configured model's values (:meth:`Experiment.parameter_values`).
    """
    values = experiment.parameter_values()
    unknown = sorted(set(parameters or {}) - set(values))
    if unknown:
        raise ValueError(f"Parameter(s) {unknown} are not in the configured parameter set.")
    values.update({name: float(value) for name, value in (parameters or {}).items()})

    # Noise parameters are fixed, at given values when there are any.
    noise = {name: value for name, value in values.items() if name in OBSERVATION_PARAM_NAMES}
    given_noise = {
        _NOISE_FIELDS[name]: float(value)
        for name, value in (parameters or {}).items()
        if name in _NOISE_FIELDS
    }
    if given_noise:
        config = experiment.config.model_copy(
            update={"observations": experiment.config.observations.model_copy(update=given_noise)}
        )
        experiment = type(experiment)(config)
    names = [name for name in values if name not in OBSERVATION_PARAM_NAMES]

    # The expected data under the configured rates places the likelihood;
    # the rates at `values` then give the data the design would produce.
    likelihood = _likelihood(experiment, _expected_series(experiment, experiment.rate_set), names)
    rate_set = likelihood._build_rate_set({name: values[name] for name in names})
    likelihood = _likelihood(experiment, _expected_series(experiment, rate_set), names)
    theta = likelihood.params_to_theta(values)
    report = likelihood_identifiability(likelihood, theta)
    covariance = np.linalg.pinv(report.information, rcond=1e-10, hermitian=True)
    scores = {name: float(score) for name, score in zip(names, report.scores)}
    status, relative_se = {}, {}
    for index, name in enumerate(names):
        if theta[index] == 0 or scores[name] < SCORE_CUTOFF:
            status[name] = "not constrained"
        elif scores[name] < SEPARATE_CUTOFF:
            status[name] = "confounded"
        else:
            status[name] = "constrained"
        variance = covariance[index, index]
        relative_se[name] = float(np.sqrt(variance)) if status[name] == "constrained" and variance > 0 else None
    return DesignReport(
        param_names=names,
        values={name: float(values[name]) for name in names},
        scores=scores,
        relative_se=relative_se,
        status=status,
        n_identifiable=sum(state == "constrained" for state in status.values()),
        condition_number=float(report.condition_number),
        n_observations=likelihood.n_observations,
        fixed=noise,
        unassessable=[name for name, value in zip(names, theta) if value == 0],
    )


def _expected_series(experiment, rate_set) -> list[TimeSeriesData]:
    """Noise-free observations of the design: every concentration and replicate."""
    config = experiment.config
    topology = experiment.topology
    models = experiment.observation_model.models
    if config.pk.model == "none":
        results = experiment.simulate(
            rate_set=rate_set, method="ode", concentrations=config.dosing.concentrations or [0.0]
        )
    else:
        results = {None: experiment.simulate(rate_set=rate_set, method="ode")}
    series = []
    for concentration, result in results.items():
        states = np.column_stack([result.populations[state.name] for state in topology.active_states])
        observations = {
            name: np.array([model.expected_value(state) for state in states], dtype=float)
            for name, model in models.items()
        }
        for replicate in range(config.simulation.n_replicates):
            series.append(
                TimeSeriesData(
                    times=np.asarray(result.times, dtype=float),
                    observations={name: values.copy() for name, values in observations.items()},
                    concentration=concentration,
                    group_id="design" if concentration is None else f"c{concentration:g}",
                    replicate_id=str(replicate + 1),
                    metadata={"initial_cells": config.simulation.initial_cells},
                )
            )
    return series


def _likelihood(experiment, series: list[TimeSeriesData], names: list[str]) -> ModelLikelihood:
    config = experiment.config
    return ModelLikelihood(
        topology=experiment.topology,
        data=series,
        mode=config.inference.forward_mode,
        observation_model=experiment._likelihood_observation_model(series),
        param_names=names,
        initial_fractions=config.dynamics.initial_fractions,
        exposure_fn=experiment.exposure if config.pk.model != "none" else None,
        base_rate_set=experiment.rate_set,
    )
