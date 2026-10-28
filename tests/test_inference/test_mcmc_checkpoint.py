"""An interrupted emcee run resumes from its checkpoint with the same draws."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("emcee")

from umimic.inference.checkpoint import CheckpointMismatch, MCMCCheckpoint  # noqa: E402
from umimic.inference.mcmc import MCMCSampler  # noqa: E402


class _Gaussian:
    """A two-parameter likelihood cheap enough to sample in a unit test."""

    param_names = ["a", "b"]
    n_params = 2

    def theta_to_params(self, theta):
        return dict(zip(self.param_names, theta))

    def __call__(self, theta):
        return float(-0.5 * np.sum((np.asarray(theta) - [1.0, 2.0]) ** 2 / 0.1))


class _PositivePrior:
    def log_prior(self, params):
        return 0.0 if all(value > 0 for value in params.values()) else -np.inf

    def sample(self, rng):
        return {"a": rng.uniform(0.5, 1.5), "b": rng.uniform(1.5, 2.5)}


class _Stop(BaseException):
    """Stands in for a worker that dies mid-run."""


_SETTINGS = {"n_samples": 30, "n_chains": 6, "n_warmup": 20}


def _sample(**options):
    return MCMCSampler(_Gaussian(), _PositivePrior(), rng=7).sample(**_SETTINGS, **options)


def _stop_at(phase, step):
    def progress(current, done, total):
        if (current, done) == (phase, step):
            raise _Stop

    return progress


def _same_draws(left, right):
    for name in left.samples:
        np.testing.assert_array_equal(left.samples[name], right.samples[name])
    np.testing.assert_array_equal(left.log_posterior_trace, right.log_posterior_trace)
    np.testing.assert_array_equal(left.log_likelihood_trace, right.log_likelihood_trace)
    assert left.diagnostics["acceptance_fraction"] == right.diagnostics["acceptance_fraction"]


def test_checkpointing_does_not_change_the_draws(tmp_path):
    plain = _sample()
    saved = _sample(checkpoint=MCMCCheckpoint(tmp_path / "run.npz", every=7))
    _same_draws(plain, saved)
    assert saved.diagnostics["resumed_from"] is None


@pytest.mark.parametrize(("phase", "step", "resumed_at"), [("warmup", 12, 10), ("production", 13, 10)])
def test_interrupted_run_resumes_with_the_same_draws(tmp_path, phase, step, resumed_at):
    path = tmp_path / "run.npz"
    with pytest.raises(_Stop):
        _sample(checkpoint=MCMCCheckpoint(path, every=5), progress=_stop_at(phase, step))
    resumed = _sample(checkpoint=MCMCCheckpoint(path, every=5))
    _same_draws(_sample(), resumed)
    assert resumed.diagnostics["resumed_from"] == {"phase": phase, "steps_done": resumed_at}
    with np.load(path, allow_pickle=False) as archive:
        assert "meta" in archive.files


def test_a_checkpoint_from_another_run_is_not_reused(tmp_path):
    path = tmp_path / "run.npz"
    with pytest.raises(_Stop):
        _sample(checkpoint=MCMCCheckpoint(path, {"data": "x"}, every=5), progress=_stop_at("production", 7))
    fresh = _sample(checkpoint=MCMCCheckpoint(path, {"data": "y"}, every=5))
    assert fresh.diagnostics["resumed_from"] is None
    assert "different run" in fresh.diagnostics["checkpoint_discarded"]
    _same_draws(_sample(), fresh)


def test_a_strict_checkpoint_refuses_to_restart_quietly(tmp_path):
    path = tmp_path / "run.npz"
    with pytest.raises(_Stop):
        _sample(checkpoint=MCMCCheckpoint(path, {"data": "x"}, every=5), progress=_stop_at("warmup", 6))
    assert MCMCCheckpoint(path).peek() == {"phase": "warmup", "steps_done": 5}
    with pytest.raises(CheckpointMismatch, match="different run"):
        _sample(checkpoint=MCMCCheckpoint(path, {"data": "y"}, every=5, strict=True))
    # Without a saved file, strict mode simply starts the run.
    fresh = _sample(checkpoint=MCMCCheckpoint(tmp_path / "new.npz", strict=True))
    _same_draws(_sample(), fresh)


def test_a_save_refused_for_a_moment_is_retried(tmp_path, monkeypatch):
    import umimic.inference.checkpoint as module

    real_replace, refusals = module.os.replace, []

    def flaky_replace(source, target):
        if len(refusals) < 2:
            refusals.append(target)
            raise PermissionError("held open by a scanner")
        real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", flaky_replace)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    result = _sample(checkpoint=MCMCCheckpoint(tmp_path / "run.npz", every=10))
    assert len(refusals) == 2
    assert result.diagnostics["checkpoint_save_failures"] == 0
    assert (tmp_path / "run.npz").is_file()


def test_a_save_that_keeps_failing_does_not_stop_the_run(tmp_path, monkeypatch):
    import umimic.inference.checkpoint as module

    def refused(source, target):
        raise PermissionError("held open")

    monkeypatch.setattr(module.os, "replace", refused)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    result = _sample(checkpoint=MCMCCheckpoint(tmp_path / "run.npz", every=10))
    _same_draws(_sample(), result)
    # Warmup saves at 10 and 20, production at 10, 20 and 30.
    assert result.diagnostics["checkpoint_save_failures"] == 5


def test_progress_reports_both_phases():
    seen = []
    _sample(progress=lambda phase, done, total: seen.append((phase, done, total)))
    assert seen[0] == ("warmup", 1, 20)
    assert seen[19] == ("warmup", 20, 20)
    assert seen[-1] == ("production", 30, 30)
    assert len(seen) == 50
