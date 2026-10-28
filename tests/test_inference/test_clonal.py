"""Tests for clone-size based birth/death estimation."""

import numpy as np
import pytest

from umimic.dynamics.rates import RateSet
from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.gillespie import GillespieSimulator
from umimic.pk.exposure import ExposureProfile
from umimic.inference.clonal import (
    clone_size_logpmf,
    clone_size_params,
    fit_clone_sizes,
    moments_to_rates,
    sample_clone_sizes,
)


class TestCloneSizeLaw:
    @pytest.mark.parametrize(
        "b,d,t",
        [(0.05, 0.01, 72.0), (0.04, 0.035, 100.0), (0.08, 0.06, 24.0)],
    )
    def test_inversion_is_exact(self, b, d, t):
        """(b, d) -> (alpha, beta) -> (b, d) round-trips to machine precision.

        This is the identifiability claim: two observables determine two rates
        with no approximation.
        """
        alpha, beta = clone_size_params(b, d, t)
        b_hat, d_hat = moments_to_rates(alpha, beta, t)
        assert b_hat == pytest.approx(b, rel=1e-9)
        assert d_hat == pytest.approx(d, rel=1e-9)

    def test_critical_case_uses_the_limit(self):
        """b == d must not divide by r = 0; alpha = beta = bt/(1+bt)."""
        b = 0.02
        t = 50.0
        alpha, beta = clone_size_params(b, b, t)
        expected = b * t / (1 + b * t)
        assert alpha == pytest.approx(expected)
        assert beta == pytest.approx(expected)

    def test_pmf_normalises_and_has_the_right_mean(self):
        b, d, t = 0.05, 0.02, 60.0
        sizes = np.arange(0, 20000)
        p = np.exp(clone_size_logpmf(sizes, b, d, t))
        assert p.sum() == pytest.approx(1.0, abs=1e-6)
        # E[Z] = exp((b-d) t)
        assert float((sizes * p).sum()) == pytest.approx(
            np.exp((b - d) * t), rel=1e-3
        )

    def test_extinction_tends_to_d_over_b(self):
        """The long-run extinction probability is d/b for a supercritical process."""
        b, d = 0.05, 0.015
        alpha, _ = clone_size_params(b, d, 5000.0)
        assert alpha == pytest.approx(d / b, rel=1e-6)

    def test_matches_gillespie(self):
        """The closed form agrees with the package's independent simulator."""
        b, d, t = 0.05, 0.02, 60.0
        topology = ModelTopology(
            active_states=[CellType.P], transitions=[],
            division_states=[CellType.P], death_states=[CellType.P],
        )
        rates = RateSet(birth_base=b, death_base={CellType.P: d}, transition_base={})
        ens = GillespieSimulator(
            rates, topology, ExposureProfile.constant(0.0), rng=np.random.default_rng(7)
        ).simulate_ensemble(np.array([1.0]), t, np.array([0.0, t]), n_trajectories=3000)
        gillespie = np.array([tr["P"][-1] for tr in ens.trajectories])

        alpha, _ = clone_size_params(b, d, t)
        # Extinction fraction: binomial standard error at n = 3000 is ~0.9%.
        assert np.mean(gillespie == 0) == pytest.approx(alpha, abs=0.03)
        assert gillespie.mean() == pytest.approx(np.exp((b - d) * t), rel=0.15)


class TestFitCloneSizes:
    def test_recovers_rates(self):
        b, d, t = 0.05, 0.01, 72.0
        sizes = sample_clone_sizes(b, d, t, 20000, np.random.default_rng(3))
        fit = fit_clone_sizes(sizes, t)
        assert fit.birth_rate == pytest.approx(b, rel=0.10)
        assert fit.death_rate == pytest.approx(d, rel=0.15)
        assert fit.death_birth_ratio == pytest.approx(d / b, rel=0.15)

    def test_dropout_biases_death_upward_unless_declared(self):
        """An undetected clone looks extinct, which inflates the death rate.

        This is the trap in barcode data: the zero class carries most of the
        information about d/b and is exactly where sequencing dropout lands.
        """
        b, d, t = 0.05, 0.01, 72.0
        rng = np.random.default_rng(5)
        sizes = sample_clone_sizes(b, d, t, 20000, rng)
        detection = 0.6
        observed = sizes.copy()
        observed[(observed > 0) & (rng.random(sizes.size) > detection)] = 0

        naive = fit_clone_sizes(observed, t)
        aware = fit_clone_sizes(observed, t, detection=detection)

        assert naive.death_rate > 2 * d           # badly biased
        assert aware.death_rate == pytest.approx(d, rel=0.20)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            fit_clone_sizes(np.array([]), 10.0)
        with pytest.raises(ValueError):
            fit_clone_sizes(np.array([1.5, 2.0]), 10.0)
        with pytest.raises(ValueError):
            fit_clone_sizes(np.array([1, 2]), 10.0, detection=0.0)
        with pytest.raises(ValueError):
            clone_size_params(0.05, 0.01, 0.0)

    def test_reports_extinct_count(self):
        b, d, t = 0.04, 0.03, 80.0
        sizes = sample_clone_sizes(b, d, t, 5000, np.random.default_rng(9))
        fit = fit_clone_sizes(sizes, t)
        assert fit.n_clones == 5000
        assert fit.n_extinct == int(np.sum(sizes == 0))
        assert 0.0 < fit.extinction_prob < 1.0


def test_histogram_fit_matches_the_per_clone_fit():
    from umimic.inference.clonal import fit_clone_size_histogram, fit_clone_sizes, sample_clone_sizes

    sizes = sample_clone_sizes(0.05, 0.02, 48.0, 3000, np.random.default_rng(3))
    per_clone = fit_clone_sizes(sizes, 48.0, detection=0.9)
    unique, counts = np.unique(sizes, return_counts=True)
    histogram = fit_clone_size_histogram(unique, counts, 48.0, detection=0.9)
    assert histogram.birth_rate == pytest.approx(per_clone.birth_rate, rel=1e-9)
    assert histogram.death_rate == pytest.approx(per_clone.death_rate, rel=1e-9)
    assert (histogram.n_clones, histogram.n_extinct) == (per_clone.n_clones, per_clone.n_extinct)


def test_observed_pmf_sums_to_one_with_imperfect_detection():
    from umimic.inference.clonal import observed_clone_size_pmf

    total = observed_clone_size_pmf(np.arange(0, 5000), 0.05, 0.02, 48.0, detection=0.7).sum()
    assert total == pytest.approx(1.0, abs=1e-9)
