# -*- coding: utf-8 -*-
"""
test_learner.py.

Tests for the learner module.
"""

from contextlib import nullcontext

import numpy as np
import pytest

from woeboost.learner import WoeLearner

# Fixture for reusable data
# pylint: disable=invalid-name
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

    assert list(learner.bins_["f1"]) == ['A', "B", "C"]
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
    ]
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

    with pytest.raises(expected_exception) if (exc := expected_exception) else nullcontext():
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
    ]
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

    with pytest.raises(expected_exception) if (exc := expected_exception) else nullcontext():
        learner = WoeLearner(
            feature_names=["f1", "f2"],
            categorical_features=None,
            n_bins="scott",
            subsample=subsample,
            verbosity=30,
        )
        learner.fit(X, y)

def test_nan_handling():
    """Test handling of NaN values."""
    X = np.array([[1], [2], [np.nan], [4]])
    y = np.array([0, 1, 0, 1])
    learner = WoeLearner(feature_names=["f1"], n_bins=2, bin_strategy="quantile")
    learner.fit(X, y)

    assert len(learner.bins_["f1"]) == 3  # 2 bins + 1 NaN
    assert learner.bin_counts_["f1"][-1] == 1  # NaN bin count


@pytest.mark.parametrize(
    "monotonicity, target_values, expected_monotonicity",
    [
        ({"f1": "increasing"}, [1, 2, 3, 4], True),  # Enforce increasing trend
        ({"f1": "decreasing"}, [4, 3, 2, 1], True),  # Enforce decreasing trend
    ],
    ids=["increasing", "decreasing"]
)
def test_monotonicity_enforcement(monotonicity, target_values, expected_monotonicity):
    """Test monotonicity constraints."""
    X = np.array([[1], [2], [3], [4]])
    y = np.array(target_values)
    learner = WoeLearner(feature_names=["f1"], monotonicity=monotonicity, n_bins=10)
    learner.fit(X, y)

    bin_averages = learner.bin_averages_["f1"]
    monotonic = all(
        bin_averages[i] <= bin_averages[i + 1] for i in range(len(bin_averages) - 1)
    ) if monotonicity["f1"] == "increasing" else all(
        bin_averages[i] >= bin_averages[i + 1] for i in range(len(bin_averages) - 1)
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

if __name__ == "__main__":
    pytest.main()
