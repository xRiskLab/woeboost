# -*- coding: utf-8 -*-
"""
test_learner.py.

Tests for the learner module.
"""

import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import pytest

from woeboost import learner as learner_module
from woeboost.learner import WoeLearner, _bin_stats, _detect_freethreading

# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit


# pylint: disable=invalid-name, unused-argument
@pytest.fixture
def sample_data():
    """Sample numeric training data."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    return X, y


@pytest.fixture
def categorical_data():
    """Sample categorical training data."""
    X = np.array([["A"], ["B"], ["A"], ["C"]])
    y = np.array([1, 0, 1, 0])
    return X, y


# Fixture for reusable data
# pylint: disable=redefined-outer-name
def test_categorical_binning(categorical_data):
    """Test binning for categorical features."""
    X, y = categorical_data
    learner = WoeLearner(feature_names=["f1"], categorical_features=["f1"], n_bins=10)
    learner.fit(X, y)

    assert list(learner.bins_["f1"]) == ["A", "B", "C"]
    assert learner.bin_counts_["f1"] == [2, 1, 1]


@pytest.mark.parametrize(
    "bin_strategy, n_bins_range, expected_exception",
    [
        # Quantile strategy: Only integer is allowed
        ("quantile", 20, None),  # Valid integer for quantile
        ("quantile", (20, 25), TypeError),  # Invalid tuple for quantile
        ("quantile", "scott", TypeError),  # Invalid string for quantile
        ("quantile", ["scott", 25], TypeError),  # Invalid list for quantile
        # Histogram strategy: Allows string, scalar, tuple, or list
        ("histogram", "scott", None),  # Valid string for histogram
        ("histogram", 25, None),  # Valid scalar for histogram
        ("histogram", (20, 25), None),  # Valid tuple for histogram
        ("histogram", [20, 35], None),  # Valid list for histogram
        ("histogram", [5, (10, 15)], ValueError),  # Invalid list with tuple
        # Unsupported strategy
        ("unsupported", 20, ValueError),  # Unsupported strategy
    ],
    ids=[
        "valid-quantile-integer",
        "invalid-quantile-tuple",
        "invalid-quantile-string",
        "invalid-quantile-list",
        "valid-histogram-string",
        "valid-histogram-scalar",
        "valid-histogram-tuple",
        "valid-histogram-list",
        "invalid-histogram-list-with-tuple",
        "unsupported-strategy",
    ],
)

# pylint: disable=unused-variable
def test_bin_strategy_and_range_validation(bin_strategy, n_bins_range, expected_exception):
    """
    Test that `n_bins_range` is validated correctly based on `bin_strategy`.

    - For `quantile`: `n_bins_range` must be an integer.
    - For `histogram`: `n_bins_range` can be a string, scalar, tuple, or list of strings/integers.
    - Unsupported strategies should raise a ValueError or TypeError.
    """
    X = np.random.rand(100, 1)
    y = np.random.randint(0, 2, 100)

    with pytest.raises(expected_exception) if expected_exception else nullcontext():
        learner = WoeLearner(feature_names=["f1"], n_bins=n_bins_range, bin_strategy=bin_strategy)
        learner.fit(X, y)


@pytest.mark.parametrize(
    "subsample, expected_exception",
    [
        (0.1, None),  # Valid float subsample (0 < subsample <= 1)
        (1, None),  # Valid integer subsample
        (0, ValueError),  # Invalid: subsample = 0
        (-0.1, ValueError),  # Invalid: negative subsample
        (1.5, ValueError),  # Invalid: subsample > 1
        ("invalid", ValueError),  # Invalid: non-numeric subsample
        (None, None),  # Valid: None should be allowed (no subsampling)
    ],
    ids=[
        "valid-float",
        "valid-integer",
        "invalid-zero",
        "invalid-negative",
        "invalid-greater-than-one",
        "invalid-non-numeric",
        "valid-none",
    ],
)

# pylint: disable=unused-variable
def test_subsample_validation(subsample, expected_exception):
    """
    Test the `subsample` parameter validation in WoeLearner.

    - Valid values: float in (0, 1], positive integers, or None.
    - Invalid values: zero, negative values, strings, and floats > 1.
    """
    # Arrange
    X = np.random.rand(100, 2)
    y = np.random.randint(0, 2, 100)

    with pytest.raises(expected_exception) if expected_exception else nullcontext():
        learner = WoeLearner(
            feature_names=["f1", "f2"],
            categorical_features=None,
            n_bins="scott",
            subsample=subsample,
            verbosity=30,
        )
        learner.fit(X, y)


def test_nan_handling():
    """NaN values are excluded from bin statistics (they get zero evidence at transform)."""
    X = np.array([[1], [2], [np.nan], [4]])
    y = np.array([0, 1, 0, 1])
    learner = WoeLearner(feature_names=["f1"], n_bins=2, bin_strategy="quantile")
    learner.fit(X, y)

    # Edges come from the non-NaN values [1, 2, 4]: bins [1, 2) and [2, 4)
    assert len(learner.bins_["f1"]) == 3  # 2 bins -> 3 edges
    assert learner.bin_counts_["f1"] == [1, 1]  # 1.0 and 2.0; NaN and the right edge (4.0) excluded
    assert learner.transform(X)[2, 0] == 0.0  # NaN maps to zero evidence


@pytest.mark.parametrize(
    "monotonicity, target_values, expected_monotonicity",
    [
        ({"f1": "increasing"}, [1, 2, 3, 4], True),  # Enforce increasing trend
        ({"f1": "decreasing"}, [4, 3, 2, 1], True),  # Enforce decreasing trend
    ],
    ids=["increasing", "decreasing"],
)
def test_monotonicity_enforcement(monotonicity, target_values, expected_monotonicity):
    """Test monotonicity constraints."""
    X = np.array([[1], [2], [3], [4]])
    y = np.array(target_values)
    learner = WoeLearner(feature_names=["f1"], monotonicity=monotonicity, n_bins=10)
    learner.fit(X, y)

    bin_averages = learner.bin_averages_["f1"]
    monotonic = (
        all(bin_averages[i] <= bin_averages[i + 1] for i in range(len(bin_averages) - 1))
        if monotonicity["f1"] == "increasing"
        else all(bin_averages[i] >= bin_averages[i + 1] for i in range(len(bin_averages) - 1))
    )
    assert monotonic == expected_monotonicity


def test_transform(sample_data):
    """Test feature transformation."""
    X_train, y_train = sample_data
    learner = WoeLearner(feature_names=["f1", "f2"], n_bins=10)
    learner.fit(X_train, y_train)

    transformed = learner.transform(X_train)
    assert transformed.shape == X_train.shape


def test_predict(sample_data):
    """Test prediction of cumulative WOE."""
    X_train, y_train = sample_data
    learner = WoeLearner(feature_names=["f1", "f2"], n_bins=10)
    learner.fit(X_train, y_train)

    prediction = learner.predict(X_train)
    assert len(prediction) == len(y_train)


def test_predict_score(sample_data):
    """Test prediction of deciban scores."""
    X_train, y_train = sample_data
    learner = WoeLearner(feature_names=["f1", "f2"], n_bins=10)
    learner.fit(X_train, y_train)

    scores = learner.predict_score(X_train)
    assert len(scores) == len(y_train)


def test_non_numeric_data():
    """Test handling of non-numeric data."""
    X = np.array([[1, 2], ["a", "b"]])
    y = np.array([0, 1])
    learner = WoeLearner(feature_names=["f1", "f2"], n_bins=10)

    # Test if we can fit without raising an exception
    learner.fit(X, y)

    # Test if we can transform without raising an exception
    transformed = learner.transform(X)
    assert transformed.shape == X.shape


def test_n_threads_sets_n_tasks(monkeypatch):
    """Test that setting `n_threads` sets `n_tasks` and emits DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        learner = WoeLearner(feature_names=["f1"], n_threads=2)

        assert learner.n_tasks == 2, "n_threads did not map to n_tasks"
        assert learner.n_threads == 2, "n_threads attribute not set"

        assert any(issubclass(warning.category, DeprecationWarning) for warning in w), (
            "Expected DeprecationWarning not raised"
        )


def test_n_threads_does_not_override_n_tasks():
    """Test that `n_threads` does not override explicitly passed `n_tasks`."""
    learner = WoeLearner(feature_names=["f1"], n_threads=2, n_tasks=5)
    assert learner.n_tasks == 5, "n_threads should not override n_tasks when both are set"
    assert learner.n_threads == 2, "n_threads should still be set for backward compatibility"


def test_n_threads_always_exists():
    """Ensure `n_threads` always exists, even if not passed."""
    learner = WoeLearner(feature_names=["f1"])
    assert hasattr(learner, "n_threads"), "n_threads should always be defined"
    assert learner.n_threads is None, "n_threads should be None if not explicitly set"


def test_parallel_vs_sequential_results():
    """Test that parallel and sequential learners produce nearly identical results."""
    X = np.tile(np.array([[1, 2, 3, 4]]), (1000, 1))
    y = np.tile(np.array([0, 1, 0, 1]), 250)

    feature_names = ["f1", "f2", "f3", "f4"]

    # Sequential learner (n_tasks=None)
    learner_seq = WoeLearner(feature_names=feature_names, n_bins=5, n_tasks=1)
    learner_seq.fit(X, y)
    transformed_seq = learner_seq.transform(X)

    # Parallel learner (n_tasks > 1)
    learner_parallel = WoeLearner(
        feature_names=feature_names, n_bins=5, n_tasks=4, executor_cls=ThreadPoolExecutor
    )
    learner_parallel.fit(X, y)
    transformed_parallel = learner_parallel.transform(X)

    # Assert the transformed results are nearly the same
    np.testing.assert_allclose(
        transformed_seq,
        transformed_parallel,
        rtol=1e-5,
        err_msg="Parallel and sequential results differ",
    )


def test_bin_stats_matches_bin_boundaries():
    """Bins are [left, right); values outside the edges and NaN are not counted."""
    X = np.array([0.0, 0.5, 1.0, 1.5, 2.0, np.nan, -1.0])
    y = np.array([1.0, 3.0, 5.0, 7.0, 100.0, 100.0, 100.0])
    counts, averages = _bin_stats(X, y, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(counts, [2, 2])
    np.testing.assert_allclose(averages, [2.0, 6.0])


def test_bin_stats_empty_and_duplicate_edges():
    """Empty bins average to 0 and duplicate edges produce empty bins."""
    X = np.array([0.0, 0.0, 0.5])
    y = np.array([1.0, 2.0, 3.0])
    counts, averages = _bin_stats(X, y, np.array([0.0, 0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(counts, [0, 3, 0])
    np.testing.assert_allclose(averages, [0.0, 2.0, 0.0])


def test_parallel_by_default(monkeypatch):
    """Without `n_tasks`/`executor_cls`, features are processed on a thread pool."""
    created = []

    class SpyExecutor(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            created.append(kwargs.get("max_workers"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(learner_module, "ThreadPoolExecutor", SpyExecutor)
    monkeypatch.setattr(learner_module.os, "cpu_count", lambda: 8)
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(200, 3)), rng.integers(0, 2, 200)

    learner = WoeLearner(feature_names=["a", "b", "c"]).fit(X, y)
    learner.transform(X)

    assert learner.n_tasks is None, "n_tasks should be resolved per call, not stored"
    assert created == [3, 3], "Expected one pool for fit and one for transform"


def test_sequential_when_n_tasks_is_one(monkeypatch):
    """`n_tasks=1` never creates an executor."""

    def fail(*args, **kwargs):
        raise AssertionError("Executor should not be created")

    monkeypatch.setattr(learner_module, "ThreadPoolExecutor", fail)
    X, y = np.random.default_rng(0).normal(size=(50, 3)), np.arange(50) % 2
    WoeLearner(feature_names=["a", "b", "c"], n_tasks=1).fit(X, y).transform(X)


def test_default_parallel_matches_sequential_fit():
    """Parallel binning (with subsampling) fits the same bins as sequential binning."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 6))
    y = rng.integers(0, 2, 1000)
    names = [f"f{i}" for i in range(6)]
    kwargs = {"feature_names": names, "subsample": 0.7, "random_state": 42}

    seq = WoeLearner(n_tasks=1, **kwargs).fit(X, y)
    par = WoeLearner(**kwargs).fit(X, y)

    for name in names:
        np.testing.assert_array_equal(seq.bins_[name], par.bins_[name])
        np.testing.assert_allclose(seq.bin_averages_[name], par.bin_averages_[name])
    np.testing.assert_allclose(seq.transform(X), par.transform(X))


def test_detect_freethreading_follows_runtime_gil():
    """Free-threading is reported only when the GIL is actually disabled."""
    is_gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)
    assert _detect_freethreading() is (not is_gil_enabled())


if __name__ == "__main__":
    pytest.main()
