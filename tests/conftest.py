"""Shared test fixtures for U-MIMIC test suite."""

import numpy as np
import pytest

from umimic.dynamics.states import CellType, ModelTopology
from umimic.dynamics.rates import RateSet, EmaxHill
from umimic.data.schemas import TimeSeriesData


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def two_state_topology():
    """Simple P-Q model topology."""
    return ModelTopology.two_state()


@pytest.fixture
def three_state_topology():
    """P-Q-A model topology."""
    return ModelTopology.three_state()


@pytest.fixture
def four_state_topology():
    """Full P-Q-A-R model topology."""
    return ModelTopology.four_state()


@pytest.fixture
def cytotoxic_rates():
    """Rate set for a cytotoxic drug."""
    return RateSet.cytotoxic_drug(
        b0=0.04, d0=0.01,
        emax_death=0.05, ec50_death=1.0, hill_death=1.5,
    )


@pytest.fixture
def cytostatic_rates():
    """Rate set for a cytostatic drug."""
    return RateSet.cytostatic_drug(
        b0=0.04, d0=0.01,
        emax_birth=0.8, ec50_birth=1.0, hill_birth=1.5,
    )


@pytest.fixture
def simple_rates():
    """Simple rate set with no drug modulation."""
    return RateSet(
        birth_base=0.04,
        death_base={CellType.P: 0.01, CellType.Q: 0.005},
        transition_base={
            (CellType.P, CellType.Q): 0.005,
            (CellType.Q, CellType.P): 0.003,
        },
    )


@pytest.fixture
def initial_state_2():
    """Initial state for 2-state model: 100 P cells."""
    return np.array([100.0, 0.0])


@pytest.fixture
def initial_state_3():
    """Initial state for 3-state model: 100 P cells."""
    return np.array([100.0, 0.0, 0.0])


@pytest.fixture
def sample_data():
    """Sample time-series data for testing."""
    times = np.array([0, 4, 8, 12, 16, 20, 24, 48, 72], dtype=float)
    counts = np.array([100, 115, 135, 160, 190, 220, 260, 500, 950], dtype=float)
    return TimeSeriesData.from_counts(times, counts, concentration=0.0, group_id="test")


@pytest.fixture
def sample_data_with_drug():
    """Sample time-series data for a treated condition."""
    times = np.array([0, 4, 8, 12, 16, 20, 24, 48, 72], dtype=float)
    counts = np.array([100, 108, 112, 115, 116, 115, 112, 95, 70], dtype=float)
    return TimeSeriesData.from_counts(times, counts, concentration=10.0, group_id="drug")
