#!/usr/bin/env python3
"""
Test script to verify if free-threading is actually working in Python 3.13.
This tests CPU-intensive operations that should benefit from free-threading.
"""

import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def check_freethreading():
    """Check if free-threading is enabled and working."""
    print(f"Python version: {sys.version}")
    print(f"Free-threading enabled: {getattr(sys, '_is_freethreaded', 'Unknown')}")

    # Check for free-threading indicators
    if hasattr(sys, "_is_freethreaded"):
        print(f"sys._is_freethreaded: {sys._is_freethreaded}")  # pylint: disable=protected-access
    else:
        print("sys._is_freethreaded not available")

    # Check if we're using the free-threaded build
    if "freethreaded" in sys.version.lower():
        print("✓ Free-threaded build detected in version string")
    else:
        print("✗ No free-threaded indicator in version string")

    print()


def cpu_intensive_workload(n):
    """CPU-intensive workload that should benefit from free-threading."""
    result = 0
    for i in range(n):
        result += math.sqrt(i) * math.sin(i) * math.cos(i)
    return result


def test_cpu_intensive_sequential(n_tasks=4, work_per_task=1000000):
    """Test CPU-intensive tasks sequentially."""
    print(f"Sequential CPU test: {n_tasks} tasks, {work_per_task} iterations each")
    start_time = time.time()

    results = []
    # sourcery skip: no-loop-in-tests
    for _ in range(n_tasks):
        result = cpu_intensive_workload(work_per_task)
        results.append(result)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Sequential execution time: {duration:.4f}s")
    assert len(results) == n_tasks
    assert duration > 0
    return duration


def test_cpu_intensive_concurrent(n_tasks=4, work_per_task=1000000, max_workers=4):
    """Test CPU-intensive tasks concurrently."""
    print(
        f"Concurrent CPU test: {n_tasks} tasks, {work_per_task} iterations each, {max_workers} workers"
    )
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(cpu_intensive_workload, work_per_task) for _ in range(n_tasks)]
        results = [future.result() for future in futures]

    end_time = time.time()
    duration = end_time - start_time
    print(f"Concurrent execution time: {duration:.4f}s")
    assert len(results) == n_tasks
    assert duration > 0
    return duration


def test_thread_count_scaling(n_tasks=4, work_per_task=500000):
    """Test how performance scales with different thread counts."""
    print(f"Thread scaling test: {n_tasks} tasks, {work_per_task} iterations each")
    print()

    thread_counts = [1, 2, 4, 8]
    results = []

    # sourcery skip: no-loop-in-tests
    for n_threads in thread_counts:
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [
                executor.submit(cpu_intensive_workload, work_per_task) for _ in range(n_tasks)
            ]
            [future.result() for future in futures]
        end_time = time.time()

        time_taken = end_time - start_time
        results.append((n_threads, time_taken))
        print(f"{n_threads} threads: {time_taken:.4f} seconds")

    print()
    # Assert that we got results for all thread counts
    assert len(results) == len(thread_counts)
    # Assert that all durations are positive
    for thread_count, duration in results:
        assert duration > 0


def test_gil_impact():
    """Test to see if GIL is still present by running CPU-intensive tasks."""
    print("GIL Impact Test")
    print("-" * 20)

    # Test with different work sizes
    work_sizes = [100000, 500000, 1000000]

    for work_size in work_sizes:
        print(f"\nWork size: {work_size}")

        # Sequential timing
        start_time = time.time()
        seq_results = []
        for i in range(4):
            result = cpu_intensive_workload(work_size)
            seq_results.append(result)
        seq_time = time.time() - start_time

        # Concurrent timing
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_intensive_workload, work_size) for _ in range(4)]
            conc_results = [future.result() for future in futures]
        conc_time = time.time() - start_time

        if seq_time > 0:
            speedup = seq_time / conc_time
            print(f"Speedup: {speedup:.2f}x")

            if speedup > 1.5:  # Significant speedup suggests free-threading
                print("✓ Significant speedup - free-threading likely working")
            elif speedup < 0.8:  # Degradation suggests GIL overhead
                print("✗ Performance degradation - GIL likely still present")
            else:
                print("? Moderate speedup - unclear if free-threading is working")
        print()

        # Assert that we got results
        assert len(seq_results) == 4
        assert len(conc_results) == 4
        assert seq_time > 0
        assert conc_time > 0


def main():
    """Main test function."""
    print("=" * 60)
    print("FREE-THREADING VERIFICATION TEST")
    print("=" * 60)

    # Check free-threading status
    check_freethreading()

    # Test CPU-intensive operations
    print("CPU-INTENSIVE PERFORMANCE TEST")
    print("-" * 40)

    # Test with moderate workload
    seq_time = test_cpu_intensive_sequential(4, 500000)
    conc_time = test_cpu_intensive_concurrent(4, 500000, 4)

    print(f"Sequential time: {seq_time:.4f} seconds")
    print(f"Concurrent time: {conc_time:.4f} seconds")

    if seq_time > 0:
        speedup = seq_time / conc_time
        print(f"Speedup: {speedup:.2f}x")
        print(f"Time saved: {seq_time - conc_time:.4f} seconds")

    print()

    # Test thread scaling
    print("THREAD SCALING TEST")
    print("-" * 25)
    test_thread_count_scaling(4, 300000)

    # Test GIL impact
    test_gil_impact()

    print("=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
