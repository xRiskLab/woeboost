#!/usr/bin/env python3
"""
Test script to measure WoeBoost performance with free-threaded Python.
This script tests the binning operations that could benefit from true concurrency.

uv run --python 3.13t --with requirements test_requirements.txt python test_freethreaded.py
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

# Check if we're running with free-threading
print(f"Python version: {sys.version}")
print(f"Free-threading enabled: {getattr(sys, '_is_freethreaded', False)}")


@pytest.fixture
def data():
    """Create test data for binning operations."""
    return create_test_data(n_samples=1000, n_features=5)


@pytest.fixture
def columns():
    """Return column names for testing."""
    return [f"feature_{i}" for i in range(5)]


def create_test_data(n_samples=10000, n_features=20):
    """Create synthetic test data for binning operations."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(15, n_features - 1),
        n_redundant=0,
        random_state=42,
    )

    # Convert to DataFrame with meaningful column names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    return df


def simple_binning_operation(data, column, n_bins=10):
    """Simulate a simple binning operation that could benefit from concurrency."""
    # This simulates the type of computation that happens during WOE binning
    values = data[column].values
    bin_edges = np.linspace(values.min(), values.max(), n_bins + 1)
    binned = np.digitize(values, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)

    # Simulate some computation
    bin_stats = []
    for i in range(n_bins):
        mask = binned == i
        if np.any(mask):
            bin_values = values[mask]
            bin_stats.append(
                {
                    "bin": i,
                    "count": len(bin_values),
                    "mean": np.mean(bin_values),
                    "std": np.std(bin_values),
                    "min": np.min(bin_values),
                    "max": np.max(bin_values),
                }
            )

    return bin_stats


def bin_single_column(args):
    """Wrapper function for threading."""
    data, column, n_bins = args
    return simple_binning_operation(data, column, n_bins)


def test_sequential_binning(data, columns, n_bins=10):
    """Test binning operations sequentially."""
    start_time = time.time()
    results = []

    for column in columns:
        result = simple_binning_operation(data, column, n_bins)
        results.append((column, result))

    end_time = time.time()
    duration = end_time - start_time
    print(f"Sequential binning time: {duration:.4f}s")
    assert len(results) == len(columns)
    assert duration > 0


def test_concurrent_binning(data, columns, n_bins=10, max_workers=4):
    """Test binning operations concurrently."""
    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare arguments for each column
        args_list = [(data, column, n_bins) for column in columns]

        # Submit all tasks
        future_to_column = {executor.submit(bin_single_column, args): args[1] for args in args_list}

        # Collect results
        # sourcery skip: no-loop-in-tests
        for future, column in future_to_column.items():
            result = future.result()
            results.append((column, result))

    end_time = time.time()
    duration = end_time - start_time
    print(f"Concurrent binning time: {duration:.4f}s")
    assert len(results) == len(columns)
    assert duration > 0


def main():
    """Main test function."""
    print("Creating test data...")
    data = create_test_data(n_samples=50000, n_features=50)
    columns = [col for col in data.columns if col != "target"]

    print(f"Testing with {len(columns)} features and {len(data)} samples")
    print(f"Data shape: {data.shape}")

    # Test sequential processing
    print("\nTesting sequential binning...")
    seq_results, seq_time = test_sequential_binning(data, columns[:10])  # Test first 10 columns
    print(f"Sequential time: {seq_time:.4f} seconds")

    # Test concurrent processing
    print("\nTesting concurrent binning...")
    conc_results, conc_time = test_concurrent_binning(data, columns[:10], max_workers=4)
    print(f"Concurrent time: {conc_time:.4f} seconds")

    # Calculate speedup
    if seq_time > 0:
        speedup = seq_time / conc_time
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"Time saved: {seq_time - conc_time:.4f} seconds")
    else:
        print("\nCould not calculate speedup (sequential time was 0)")

    # Verify results are the same
    print("\nVerifying results consistency...")
    for i, (col, _) in enumerate(seq_results):
        if i < len(conc_results):
            seq_col, seq_res = seq_results[i]
            conc_col, conc_res = conc_results[i]
            if seq_col == conc_col and len(seq_res) == len(conc_res):
                print(f"✓ {seq_col}: Results match")
            else:
                print(f"✗ {seq_col}: Results differ")

    print("\nTest completed!")


if __name__ == "__main__":
    main()
