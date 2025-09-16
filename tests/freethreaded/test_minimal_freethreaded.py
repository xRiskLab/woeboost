#!/usr/bin/env python3
"""
Minimal test script for free-threaded Python without project dependencies.
This tests basic threading performance without numpy/pandas compatibility issues.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
import math

# Check if we're running with free-threading
print(f"Python version: {sys.version}")
print(f"Free-threading enabled: {getattr(sys, '_is_freethreaded', False)}")
print("=" * 60)


def cpu_intensive_task(n):
    """CPU-intensive task that would benefit from free-threading."""
    result = 0
    for i in range(n):
        result += math.sqrt(i) * math.sin(i) * math.cos(i)
    return result


def io_simulation_task(duration):
    """I/O simulation task that releases GIL."""
    time.sleep(duration)
    return f"Completed after {duration}s"


def test_sequential_cpu_tasks(n_tasks=4, work_per_task=1000000):
    """Test CPU-intensive tasks sequentially."""
    print(f"Testing {n_tasks} CPU-intensive tasks sequentially...")
    start_time = time.time()

    results = []
    for i in range(n_tasks):
        result = cpu_intensive_task(work_per_task)
        results.append(result)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Sequential execution time: {duration:.4f}s")
    assert len(results) == n_tasks
    assert duration > 0


def test_concurrent_cpu_tasks(n_tasks=4, work_per_task=1000000, max_workers=4):
    """Test CPU-intensive tasks concurrently."""
    print(f"Testing {n_tasks} CPU-intensive tasks concurrently (max_workers={max_workers})...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(cpu_intensive_task, work_per_task) for _ in range(n_tasks)]
        results = [future.result() for future in futures]

    end_time = time.time()
    duration = end_time - start_time
    print(f"Concurrent execution time: {duration:.4f}s")
    assert len(results) == n_tasks
    assert duration > 0


def test_sequential_io_tasks(n_tasks=4, duration_per_task=0.1):
    """Test I/O simulation tasks sequentially."""
    print(f"Testing {n_tasks} I/O simulation tasks sequentially...")
    start_time = time.time()

    results = []
    for i in range(n_tasks):
        result = io_simulation_task(duration_per_task)
        results.append(result)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Sequential I/O execution time: {duration:.4f}s")
    assert len(results) == n_tasks
    assert duration > 0


def test_concurrent_io_tasks(n_tasks=4, duration_per_task=0.1, max_workers=4):
    """Test I/O simulation tasks concurrently."""
    print(f"Testing {n_tasks} I/O simulation tasks concurrently (max_workers={max_workers})...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(io_simulation_task, duration_per_task) for _ in range(n_tasks)]
        results = [future.result() for future in futures]

    end_time = time.time()
    duration = end_time - start_time
    print(f"Concurrent I/O execution time: {duration:.4f}s")
    assert len(results) == n_tasks
    assert duration > 0


def main():
    """Main test function."""
    print("Testing free-threaded Python performance...")
    print()

    # Test CPU-intensive tasks
    print("CPU-INTENSIVE TASKS")
    print("-" * 30)

    seq_cpu_results, seq_cpu_time = test_sequential_cpu_tasks(n_tasks=4, work_per_task=500000)
    conc_cpu_results, conc_cpu_time = test_concurrent_cpu_tasks(
        n_tasks=4, work_per_task=500000, max_workers=4
    )

    print(f"Sequential CPU time: {seq_cpu_time:.4f} seconds")
    print(f"Concurrent CPU time: {conc_cpu_time:.4f} seconds")

    if seq_cpu_time > 0:
        cpu_speedup = seq_cpu_time / conc_cpu_time
        print(f"CPU Speedup: {cpu_speedup:.2f}x")
        print(f"CPU Time saved: {seq_cpu_time - conc_cpu_time:.4f} seconds")
    print()

    # Test I/O simulation tasks
    print("I/O SIMULATION TASKS")
    print("-" * 30)

    seq_io_results, seq_io_time = test_sequential_io_tasks(n_tasks=4, duration_per_task=0.1)
    conc_io_results, conc_io_time = test_concurrent_io_tasks(
        n_tasks=4, duration_per_task=0.1, max_workers=4
    )

    print(f"Sequential I/O time: {seq_io_time:.4f} seconds")
    print(f"Concurrent I/O time: {conc_io_time:.4f} seconds")

    if seq_io_time > 0:
        io_speedup = seq_io_time / conc_io_time
        print(f"I/O Speedup: {io_speedup:.2f}x")
        print(f"I/O Time saved: {seq_io_time - conc_io_time:.4f} seconds")
    print()

    # Test with different thread counts
    print("THREAD SCALING TEST")
    print("-" * 30)

    thread_counts = [1, 2, 4, 8]
    for n_threads in thread_counts:
        _, time_taken = test_concurrent_cpu_tasks(
            n_tasks=4, work_per_task=200000, max_workers=n_threads
        )
        print(f"{n_threads} threads: {time_taken:.4f} seconds")

    print()
    print("Test completed!")


if __name__ == "__main__":
    main()
