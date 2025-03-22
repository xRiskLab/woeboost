# -*- coding: utf-8 -*-
"""
test_explainer.py.

Tests for the explainer module.
"""

import random

import matplotlib
import numpy as np
import pandas as pd
import pytest

from woeboost.classifier import WoeBoostClassifier
from woeboost.explainer import EvidenceAnalyzer, PDPAnalyzer, WoeInferenceMaker

# Prevent displaying plots during tests
matplotlib.use("Agg")


# Fixtures
# pylint: disable=invalid-name, redefined-outer-name
@pytest.fixture
def sample_data():
    """Fixture for generating sample data."""
    X = pd.DataFrame(
        {
            "feature_1": random.choices(["A", "B", "C"], k=100),
            "feature_2": np.random.rand(100),
            "feature_3": np.random.randint(0, 100, 100),
        }
    )
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Fixture for training a simple WoeBoostClassifier."""
    X, y = sample_data
    model = WoeBoostClassifier(estimator=None, n_estimators=10, verbosity=30)
    model.fit(X, y)
    assert model.estimators, "The model was not trained correctly."
    return model


# pylint: disable=broad-exception-caught
def test_pdp_plot(sample_data, trained_model):
    """Test PDP plotting for a single feature."""
    X, _ = sample_data
    analyzer = PDPAnalyzer(trained_model, X)

    try:
        analyzer.plot_pdp("feature_1")
        analyzer.plot_pdp("feature_2")
        analyzer.plot_pdp("feature_3")
    except Exception as e:
        pytest.fail(f"PDP plot failed with exception: {e}")


def test_2way_pdp_calculation(sample_data, trained_model):
    """Test 2-way PDP calculation for two features."""
    X, _ = sample_data
    analyzer = PDPAnalyzer(trained_model, X)
    (grid_x, grid_y), pdp_values, (cat_names_1, cat_names_2) = analyzer.calculate_2way_pdp(
        "feature_1", "feature_2", num_points=10
    )

    assert grid_x.shape == grid_y.shape == pdp_values.shape
    assert cat_names_1 is not None or cat_names_2 is not None


def test_2way_pdp_plot(sample_data, trained_model):
    """Test 2-way PDP plotting."""
    X, _ = sample_data
    analyzer = PDPAnalyzer(trained_model, X)

    try:
        analyzer.plot_2way_pdp(
            feature1="feature_1",
            feature2="feature_2",
            num_points=10,
            plot_type="contourf",
        )
        analyzer.plot_2way_pdp(
            feature1="feature_2",
            feature2="feature_3",
            num_points=10,
            plot_type="contourf",
        )
    except Exception as e:
        pytest.fail(f"2-Way PDP plot failed with exception: {e}")


# Tests for EvidenceAnalyzer
def test_evidence_contribution_calculation(sample_data, trained_model):
    """Test calculation of evidence contributions."""
    X, _ = sample_data
    analyzer = EvidenceAnalyzer(trained_model, X)
    contributions = analyzer.calculate_contributions(mode="cumulative")

    assert len(contributions) == len(trained_model.estimators)


def test_evidence_contribution_plot(sample_data, trained_model):
    """Test plotting of evidence contributions."""
    X, _ = sample_data
    analyzer = EvidenceAnalyzer(trained_model, X)
    contributions = analyzer.calculate_contributions(mode="cumulative")

    try:
        analyzer.plot_contributions(contributions, mode="cumulative")
    except Exception as e:
        pytest.fail(f"Evidence contribution plot failed with exception: {e}")


def test_plot_decision_boundary(sample_data, trained_model):
    """Test the decision boundary plotting functionality."""
    X, y = sample_data
    analyzer = EvidenceAnalyzer(trained_model, X)

    feature_1 = "feature_1"
    feature_2 = "feature_2"

    try:
        # Plot the decision boundary
        analyzer.plot_decision_boundary(
            feature1=feature_1,
            feature2=feature_2,
            X=X,
            y=y,
            iteration_range=(0, 5),
            grid_size=(2, 3),
            cmap="viridis",
        )
    except Exception as e:
        pytest.fail(f"plot_decision_boundary failed with exception: {e}")


# Tests for WoeInferenceMaker
def test_bin_report(sample_data, trained_model):
    """Test bin report generation."""
    X, y = sample_data
    estimator = trained_model.estimators[0]
    woe_inference_maker = WoeInferenceMaker(estimator)
    report = woe_inference_maker.generate_bin_report("feature_1", X, y)

    assert "bin" in report.columns
    assert "woe" in report.columns
    assert len(report) > 0


def test_infer_woe_score(sample_data, trained_model):
    """Test inference of WOE scores."""
    X, y = sample_data
    estimator = trained_model.estimators[0]
    woe_inference_maker = WoeInferenceMaker(estimator)
    woe_inference_maker.fit(X, y)
    scores = woe_inference_maker.infer_woe_score("feature_1", X)

    assert len(scores) == len(X)
    assert isinstance(scores, np.ndarray)


def test_transform_to_woe_scores(sample_data, trained_model):
    """Test transformation of input data into WOE scores."""
    X, y = sample_data
    estimator = trained_model.estimators[0]
    woe_inference_maker = WoeInferenceMaker(estimator)
    woe_inference_maker.fit(X, y)
    transformed = woe_inference_maker.transform(X)

    assert transformed.shape == X.shape
    assert all(col in transformed.columns for col in X.columns)


def test_predict_proba(sample_data, trained_model):
    """Test prediction of probabilities."""
    X, y = sample_data
    estimator = trained_model.estimators[0]
    woe_inference_maker = WoeInferenceMaker(estimator)
    woe_inference_maker.fit(X, y)
    probabilities = woe_inference_maker.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)


if __name__ == "__main__":
    pytest.main()
