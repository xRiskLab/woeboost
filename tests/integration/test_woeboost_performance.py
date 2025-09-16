#!/usr/bin/env python3
"""
Test script to measure WoeBoost performance with different threading configurations.
This script tests the actual WoeBoost binning operations that could benefit from concurrency.
"""

import sys
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from concurrent.futures import ThreadPoolExecutor
import pytest

# Import WoeBoost components
from woeboost.learner import WoeLearner

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def sample_data():
    """Create synthetic test data for WoeBoost testing."""
    X, y = make_classification(
        n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )

    # Convert to DataFrame with meaningful column names
    feature_names = [f"feature_{i}" for i in range(20)]
    df = pd.DataFrame(X, columns=feature_names)

    return df, y, feature_names


def create_test_data(n_samples=10000, n_features=20):
    """Create synthetic test data for WoeBoost testing."""
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=15, n_redundant=5, random_state=42
    )

    # Convert to DataFrame with meaningful column names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)

    return df, y


def test_woeboost_sequential(sample_data):
    """Test WoeBoost with sequential processing (n_tasks=1)."""
    data, target, feature_names = sample_data
    n_bins = 10

    print("Testing WoeBoost sequential processing...")

    start_time = time.time()

    learner = WoeLearner(
        feature_names=feature_names,
        n_bins=n_bins,
        n_tasks=1,  # Sequential processing
        verbosity=0,  # Suppress logging
    )

    learner.fit(data, target)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Basic assertions
    assert learner is not None
    assert elapsed_time > 0
    assert len(learner.bins_) == len(feature_names)


def test_woeboost_concurrent(sample_data):
    """Test WoeBoost with concurrent processing."""
    data, target, feature_names = sample_data
    n_bins = 10
    n_tasks = 4

    print(f"Testing WoeBoost concurrent processing (n_tasks={n_tasks})...")

    start_time = time.time()

    learner = WoeLearner(
        feature_names=feature_names,
        n_bins=n_bins,
        n_tasks=n_tasks,  # Concurrent processing
        verbosity=0,  # Suppress logging
    )

    learner.fit(data, target)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Basic assertions
    assert learner is not None
    assert elapsed_time > 0
    assert len(learner.bins_) == len(feature_names)


def test_woeboost_with_executor(sample_data):
    """Test WoeBoost with custom ThreadPoolExecutor."""
    data, target, feature_names = sample_data
    n_bins = 10
    max_workers = 4

    print(f"Testing WoeBoost with custom ThreadPoolExecutor (max_workers={max_workers})...")

    start_time = time.time()

    learner = WoeLearner(
        feature_names=feature_names,
        n_bins=n_bins,
        n_tasks=max_workers,
        executor_cls=lambda max_workers=max_workers: ThreadPoolExecutor(max_workers=max_workers),
        verbosity=0,  # Suppress logging
    )

    learner.fit(data, target)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Basic assertions
    assert learner is not None
    assert elapsed_time > 0
    assert len(learner.bins_) == len(feature_names)


def compare_models(model1, model2, tolerance=1e-10):
    """Compare two WoeBoost models to ensure they produce the same results."""
    print("Comparing models for consistency...")

    # Compare bins
    for feature in model1.bins_:
        if feature in model2.bins_:
            bins1 = np.array(model1.bins_[feature])
            bins2 = np.array(model2.bins_[feature])
            if not np.allclose(bins1, bins2, rtol=tolerance):
                print(f"✗ Bins differ for {feature}")
                return False
            else:
                print(f"✓ Bins match for {feature}")

    # Compare bin averages
    for feature in model1.bin_averages_:
        if feature in model2.bin_averages_:
            avgs1 = np.array(model1.bin_averages_[feature])
            avgs2 = np.array(model2.bin_averages_[feature])
            if not np.allclose(avgs1, avgs2, rtol=tolerance):
                print(f"✗ Bin averages differ for {feature}")
                return False
            else:
                print(f"✓ Bin averages match for {feature}")

    print("✓ All models are consistent!")
    return True


def main():
    """Main test function."""
    print(f"Python version: {sys.version}")
    print(f"Free-threading enabled: {getattr(sys, '_is_freethreaded', False)}")
    print("=" * 60)

    # Create test data
    print("Creating test data...")
    data, target = create_test_data(n_samples=50000, n_features=30)
    feature_names = list(data.columns)

    print(f"Data shape: {data.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Target distribution: {np.bincount(target)}")
    print()

    # Test different configurations
    configurations = [
        ("Sequential (n_tasks=1)", lambda: test_woeboost_sequential(data, target, feature_names)),
        (
            "Concurrent (n_tasks=2)",
            lambda: test_woeboost_concurrent(data, target, feature_names, n_tasks=2),
        ),
        (
            "Concurrent (n_tasks=4)",
            lambda: test_woeboost_concurrent(data, target, feature_names, n_tasks=4),
        ),
        (
            "Concurrent (n_tasks=8)",
            lambda: test_woeboost_concurrent(data, target, feature_names, n_tasks=8),
        ),
        (
            "Custom Executor (4 workers)",
            lambda: test_woeboost_with_executor(data, target, feature_names, max_workers=4),
        ),
    ]

    results = []

    for config_name, test_func in configurations:
        print(f"\n{config_name}")
        print("-" * 40)

        try:
            model, elapsed_time = test_func()
            results.append((config_name, elapsed_time, model))
            print(f"Time: {elapsed_time:.4f} seconds")

            # Test prediction
            predictions = model.predict(data)
            print(f"Predictions shape: {predictions.shape}")
            print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")

        except Exception as e:
            print(f"Error: {e}")
            results.append((config_name, float("inf"), None))

    # Compare results
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    if len(results) >= 2:
        baseline_time = results[0][1]  # Sequential time
        print(f"Baseline (Sequential): {baseline_time:.4f} seconds")
        print()

        for config_name, elapsed_time, model in results[1:]:
            if elapsed_time != float("inf"):
                speedup = baseline_time / elapsed_time
                time_saved = baseline_time - elapsed_time
                print(f"{config_name}:")
                print(f"  Time: {elapsed_time:.4f} seconds")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Time saved: {time_saved:.4f} seconds")
                print()

    # Compare model consistency
    print("MODEL CONSISTENCY CHECK")
    print("-" * 30)

    valid_models = [(name, model) for name, _, model in results if model is not None]

    if len(valid_models) >= 2:
        # Compare first two valid models
        name1, model1 = valid_models[0]
        name2, model2 = valid_models[1]

        print(f"Comparing {name1} vs {name2}:")
        compare_models(model1, model2)

    print("\nTest completed!")


if __name__ == "__main__":
    main()
