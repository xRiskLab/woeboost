#!/usr/bin/env python3
"""
Test WoeBoost-style binning operations with free-threaded Python.
This simulates the CPU-intensive operations that would benefit from free-threading.
"""

import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def features_data():
    """Create test data for WoeBoost-style binning operations."""
    X, y = make_classification(
        n_samples=1000, n_features=5, n_redundant=0, n_informative=5, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    # Convert to dictionary format expected by the tests
    return {col: df[col].values for col in df.columns}


def check_freethreading():
    """Check if free-threading is enabled."""
    print(f"Python version: {sys.version}")
    print(f"Free-threading enabled: {getattr(sys, '_is_freethreaded', 'Unknown')}")
    print("=" * 60)


def simulate_histogram_binning(data, n_bins=10):
    """Simulate histogram binning operation (CPU-intensive)."""
    # This simulates the type of computation in WoeBoost's histogram binning
    min_val = min(data)
    max_val = max(data)
    bin_width = (max_val - min_val) / n_bins

    bin_counts = [0] * n_bins
    bin_sums = [0.0] * n_bins

    # Simulate the binning computation
    for value in data:
        if value == max_val:
            bin_idx = n_bins - 1
        else:
            bin_idx = int((value - min_val) / bin_width)
            bin_idx = max(0, min(bin_idx, n_bins - 1))

        bin_counts[bin_idx] += 1
        bin_sums[bin_idx] += value

    # Calculate bin averages (simulating WOE calculation)
    bin_averages = []
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_averages.append(bin_sums[i] / bin_counts[i])
        else:
            bin_averages.append(0.0)

    return bin_counts, bin_averages


def simulate_quantile_binning(data, n_bins=10):
    """Simulate quantile binning operation (CPU-intensive)."""
    # Sort data for quantile calculation
    sorted_data = sorted(data)
    n = len(sorted_data)

    bin_edges = []
    for i in range(n_bins + 1):
        quantile = i / n_bins
        idx = int(quantile * (n - 1))
        bin_edges.append(sorted_data[idx])

    bin_counts = [0] * n_bins
    bin_sums = [0.0] * n_bins

    # Assign data to bins
    for value in data:
        bin_idx = next(
            (i for i in range(n_bins) if value >= bin_edges[i] and value <= bin_edges[i + 1]),
            0,
        )
        bin_counts[bin_idx] += 1
        bin_sums[bin_idx] += value

    # Calculate bin averages
    bin_averages = []
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_averages.append(bin_sums[i] / bin_counts[i])
        else:
            bin_averages.append(0.0)

    return bin_counts, bin_averages


def process_single_feature(args):
    """Process a single feature (simulating WoeBoost's feature processing)."""
    feature_data, feature_name, binning_method, n_bins = args

    if binning_method == "histogram":
        bin_counts, bin_averages = simulate_histogram_binning(feature_data, n_bins)
    else:  # quantile
        bin_counts, bin_averages = simulate_quantile_binning(feature_data, n_bins)

    return feature_name, bin_counts, bin_averages


def test_sequential_binning(features_data, binning_method="histogram", n_bins=10):  # pylint: disable=redefined-outer-name
    """Test binning operations sequentially."""
    print(f"Sequential binning test: {len(features_data)} features, {binning_method} method")
    start_time = time.time()

    results = []
    # sourcery skip: no-loop-in-tests
    for feature_name, feature_data in features_data.items():
        result = process_single_feature((feature_data, feature_name, binning_method, n_bins))
        results.append(result)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Sequential binning time: {duration:.4f}s")
    assert len(results) == len(features_data)
    assert duration > 0


def test_concurrent_binning(features_data, binning_method="histogram", n_bins=10, max_workers=4):  # pylint: disable=redefined-outer-name
    """Test binning operations concurrently."""
    print(
        f"Concurrent binning test: {len(features_data)} features, {binning_method} method, {max_workers} workers"
    )
    start_time = time.time()

    # Prepare arguments for each feature
    args_list = [
        (feature_data, feature_name, binning_method, n_bins)
        for feature_name, feature_data in features_data.items()
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_feature, args) for args in args_list]
        results = [future.result() for future in futures]

    end_time = time.time()
    duration = end_time - start_time
    print(f"Concurrent binning time: {duration:.4f}s")
    assert len(results) == len(features_data)
    assert duration > 0


def create_test_data(n_features=20, n_samples=10000):
    """Create test data similar to WoeBoost's input."""
    features_data = {}  # pylint: disable=redefined-outer-name

    for i in range(n_features):
        # Create different types of distributions
        if i % 3 == 0:
            # Normal distribution
            data = [random.gauss(0, 1) for _ in range(n_samples)]
        elif i % 3 == 1:
            # Uniform distribution
            data = [random.uniform(-5, 5) for _ in range(n_samples)]
        else:
            # Exponential distribution
            data = [random.expovariate(1) for _ in range(n_samples)]

        features_data[f"feature_{i}"] = data

    return features_data


def test_binning_methods(features_data, n_bins=10, max_workers=4):  # pylint: disable=redefined-outer-name
    """Test both histogram and quantile binning methods."""
    methods = ["histogram", "quantile"]
    results = {}

    # sourcery skip: no-loop-in-tests
    for method in methods:
        print(f"\nTesting {method} binning:")
        print("-" * 30)

        # Sequential timing
        start_time = time.time()
        seq_results = []
        # sourcery skip: no-loop-in-tests
        for feature_name, feature_data in features_data.items():
            result = process_single_feature((feature_data, feature_name, method, n_bins))
            seq_results.append(result)
        seq_time = time.time() - start_time

        # Concurrent timing
        start_time = time.time()
        args_list = [
            (feature_data, feature_name, method, n_bins)
            for feature_name, feature_data in features_data.items()
        ]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_feature, args) for args in args_list]
            [future.result() for future in futures]  # pylint: disable=expression-not-assigned
        conc_time = time.time() - start_time

        speedup = seq_time / conc_time if conc_time > 0 else 0
        time_saved = seq_time - conc_time

        results[method] = {
            "sequential_time": seq_time,
            "concurrent_time": conc_time,
            "speedup": speedup,
            "time_saved": time_saved,
        }

        print(f"Sequential time: {seq_time:.4f} seconds")
        print(f"Concurrent time: {conc_time:.4f} seconds")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Time saved: {time_saved:.4f} seconds")

    # Assert that we got results for both methods
    assert len(results) == 2
    assert "histogram" in results
    assert "quantile" in results


def test_thread_scaling(features_data, method="histogram", n_bins=10):
    """Test how performance scales with different thread counts."""
    print(f"\nThread scaling test ({method} binning):")
    print("-" * 40)

    thread_counts = [1, 2, 4, 8]
    results = []

    for n_threads in thread_counts:
        start_time = time.time()
        args_list = [
            (feature_data, feature_name, method, n_bins)
            for feature_name, feature_data in features_data.items()
        ]
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(process_single_feature, args) for args in args_list]
            [future.result() for future in futures]
        time_taken = time.time() - start_time

        results.append((n_threads, time_taken))
        print(f"{n_threads} threads: {time_taken:.4f} seconds")

    # Assert that we got results for all thread counts
    assert len(results) == len(thread_counts)
    # Assert that all durations are positive
    # sourcery skip: no-loop-in-tests
    for thread_count, duration in results:
        assert duration > 0


def main():
    """Main test function."""
    check_freethreading()

    print("Creating test data...")
    features_data = create_test_data(n_features=15, n_samples=20000)
    print(
        f"Created {len(features_data)} features with {len(list(features_data.values())[0])} samples each"
    )
    print()

    # Test both binning methods
    print("BINNING METHODS COMPARISON")
    print("=" * 40)
    method_results = test_binning_methods(features_data, n_bins=10, max_workers=4)  # pylint: disable=assignment-from-no-return

    # Test thread scaling
    print("\nTHREAD SCALING ANALYSIS")
    print("=" * 30)
    scaling_results = test_thread_scaling(features_data, method="histogram", n_bins=10)  # pylint: disable=assignment-from-no-return

    # Summary
    print("\nSUMMARY")
    print("=" * 20)
    for method, results in method_results.items():
        print(f"{method.capitalize()} binning:")
        print(f"  Speedup: {results['speedup']:.2f}x")
        print(f"  Time saved: {results['time_saved']:.4f} seconds")
        print()

    # Find optimal thread count
    if scaling_results:
        best_threads = min(scaling_results, key=lambda x: x[1])
        print(f"Optimal thread count: {best_threads[0]} threads ({best_threads[1]:.4f} seconds)")

    print("\nTest completed!")


if __name__ == "__main__":
    main()
