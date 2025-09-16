#!/usr/bin/env python3
"""
Comprehensive test runner for WoeBoost.
Supports unit tests, integration tests, and free-threaded tests.
"""

import sys
import subprocess
import argparse


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"💥 {description} failed with error: {e}")
        return False


def run_unit_tests():
    """Run unit tests."""
    cmd = ["uv", "run", "pytest", "tests/unit", "-v", "-m", "unit"]
    return run_command(cmd, "Unit Tests")


def run_integration_tests():
    """Run integration tests."""
    cmd = ["uv", "run", "pytest", "tests/integration", "-v", "-m", "integration"]
    return run_command(cmd, "Integration Tests")


def run_all_standard_tests():
    """Run all standard tests (unit + integration)."""
    cmd = ["uv", "run", "pytest", "tests/unit", "tests/integration", "-v"]
    return run_command(cmd, "All Standard Tests")


def run_freethreaded_tests():
    """Run free-threaded tests."""
    cmd = ["python", "tests/freethreaded/run_freethreaded_tests.py"]
    return run_command(cmd, "Free-threaded Tests")


def run_coverage_tests():
    """Run tests with coverage."""
    cmd = [
        "uv",
        "run",
        "pytest",
        "tests/unit",
        "tests/integration",
        "--cov=woeboost",
        "--cov-report=html",
        "--cov-report=term",
    ]
    return run_command(cmd, "Coverage Tests")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="WoeBoost Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--freethreaded", action="store_true", help="Run free-threaded tests only")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    parser.add_argument("--all", action="store_true", help="Run all standard tests")
    parser.add_argument(
        "--everything", action="store_true", help="Run everything (standard + free-threaded)"
    )

    args = parser.parse_args()

    print("WoeBoost Test Runner")
    print("=" * 50)

    # If no specific tests requested, run all standard tests
    if not any(
        [args.unit, args.integration, args.freethreaded, args.coverage, args.all, args.everything]
    ):
        args.all = True

    results = []

    if args.unit:
        results.append(("Unit Tests", run_unit_tests()))

    if args.integration:
        results.append(("Integration Tests", run_integration_tests()))

    if args.all:
        results.append(("All Standard Tests", run_all_standard_tests()))

    if args.coverage:
        results.append(("Coverage Tests", run_coverage_tests()))

    if args.freethreaded:
        results.append(("Free-threaded Tests", run_freethreaded_tests()))

    if args.everything:
        results.append(("All Standard Tests", run_all_standard_tests()))
        results.append(("Free-threaded Tests", run_freethreaded_tests()))

    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")

    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False

    if all_passed:
        print("\n🎉 All requested tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
