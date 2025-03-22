# -*- coding: utf-8 -*-
"""
test_classifier.py.

Tests for the classifier module.

NOTE: Test for 'subsample_range' is handled in learner.py.
"""

from contextlib import nullcontext

import numpy as np
import pytest

from woeboost.classifier import WoeBoostClassifier, WoeBoostConfig
from woeboost.learner import WoeLearner


# Fixtures
# pylint: disable=invalid-name
@pytest.fixture
def sample_data():
    """Sample numeric training data."""
    X = np.random.rand(100, 3)  # 3 features
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def config():
    """Default WoeBoostConfig for testing."""
    return WoeBoostConfig(
        estimator=None,
        n_estimators=100,
        bin_strategy="histogram",
        n_bins_range="scott",
        subsample_range=(0.8, 1.0),
        verbosity=30,
    )


# Tests
# pylint: disable=redefined-outer-name
def test_initialization(config):
    """Test classifier initialization."""
    clf = WoeBoostClassifier(config=config)
    assert clf.config.n_estimators == 100
    assert clf.config.bin_strategy == "histogram"
    assert clf.config.subsample_range == (0.8, 1.0)


@pytest.mark.parametrize(
    "n_bins_range, expected_exception",
    [
        ((3, 8), None),  # Valid tuple for quantile
        ("scott", None),  # Valid string for histogram
        ([10, 20], None),  # Valid list for histogram
        (["invalid"], ValueError),  # Invalid list content
        ((-1, -1), ValueError),  # Invalid range
    ],
    ids=[
        "valid-quantile-tuple",
        "valid-histogram-string",
        "valid-histogram-list",
        "invalid-list-content",
        "invalid-negative-range",
    ],
)

# pylint: disable=unused-variable
def test_n_bins_range_validation(n_bins_range, expected_exception, sample_data):
    """Test validation of `n_bins_range`."""
    X, y = sample_data

    with pytest.raises(expected_exception) if (exc := expected_exception) else nullcontext():
        config = WoeBoostConfig(n_bins_range=n_bins_range, estimator=None)
        clf = WoeBoostClassifier(config=config)
        clf.fit(X, y)


def test_training(sample_data):
    """Test basic training with valid data."""
    X, y = sample_data
    clf = WoeBoostClassifier(
        config=WoeBoostConfig(
            estimator=WoeLearner(feature_names=["f1", "f2", "f3"]),
            n_estimators=10,
            verbosity=30,
        )
    )
    clf.fit(X, y)

    assert len(clf.estimators) == clf.n_estimators
    assert clf.loss_history


def test_predict_proba(sample_data):
    """Test probability predictions."""
    X, y = sample_data
    clf = WoeBoostClassifier(
        config=WoeBoostConfig(
            estimator=WoeLearner(feature_names=["f1", "f2", "f3"]),
            n_estimators=10,
            verbosity=30,
        )
    )
    clf.fit(X, y)

    probas = clf.predict_proba(X)
    assert probas.shape == (X.shape[0], 2)
    assert np.all(probas >= 0) and np.all(probas <= 1)


def test_predict(sample_data):
    """Test class label predictions."""
    X, y = sample_data
    clf = WoeBoostClassifier(
        config=WoeBoostConfig(
            estimator=WoeLearner(feature_names=["f1", "f2", "f3"]),
            n_estimators=10,
            verbosity=30,
        )
    )
    clf.fit(X, y)

    preds = clf.predict(X)
    assert preds.shape == (X.shape[0],)
    assert np.all(np.isin(preds, [0, 1]))


def test_predict_score(sample_data):
    """Test score predictions in decibans."""
    X, y = sample_data
    clf = WoeBoostClassifier(
        config=WoeBoostConfig(
            estimator=WoeLearner(feature_names=["f1", "f2", "f3"]),
            n_estimators=10,
            verbosity=30,
        )
    )
    clf.fit(X, y)

    scores = clf.predict_score(X)
    assert scores.shape == (X.shape[0],)


def test_invalid_monotonicity(sample_data):
    """Test handling of invalid monotonicity constraints."""
    X, y = sample_data
    clf = WoeBoostClassifier(
        config=WoeBoostConfig(
            estimator=WoeLearner(
                feature_names=["f1", "f2", "f3"], monotonicity={"f4": "increasing"}
            ),
            n_estimators=10,
            verbosity=30,
        )
    )
    clf.fit(X, y)


if __name__ == "__main__":
    pytest.main()
