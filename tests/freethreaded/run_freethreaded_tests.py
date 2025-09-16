#!/usr/bin/env python3
"""
Test runner for free-threaded Python tests.
This script runs WoeBoost performance tests with free-threaded Python builds.
"""

import sys
import subprocess
from pathlib import Path


def get_freethreaded_python_paths():
    """Get available free-threaded Python paths."""
    # Common locations for uv-installed Python versions
    uv_python_dir = Path.home() / ".local" / "share" / "uv" / "python"

    freethreaded_pythons = []

    if uv_python_dir.exists():
        for python_dir in uv_python_dir.iterdir():
            if "freethreaded" in str(python_dir):
                # Look for python executable
                python_exe = python_dir / "bin" / "python3.13t"
                if not python_exe.exists():
                    python_exe = python_dir / "bin" / "python3.14"
                if not python_exe.exists():
                    python_exe = python_dir / "bin" / "python3"

                if python_exe.exists():
                    freethreaded_pythons.append((str(python_exe), str(python_dir.name)))

    return freethreaded_pythons


def run_test_with_python(python_path, test_script, description):
    """Run a test script with a specific Python version."""
    print(f"\n{'=' * 60}")
    print(f"Running {description}")
    print(f"Python: {python_path}")
    print(f"Test: {test_script}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            [python_path, test_script],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
        else:
            print(f"❌ {description} failed with return code {result.returncode}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"💥 {description} failed with error: {e}")
        return False


def main():
    """Main test runner function."""
    print("WoeBoost Free-Threaded Python Test Runner")
    print("=" * 50)

    # Get available free-threaded Python versions
    freethreaded_pythons = get_freethreaded_python_paths()

    if not freethreaded_pythons:
        print("❌ No free-threaded Python versions found!")
        print("Please install free-threaded Python with: uv python install 3.14.0a5+freethreaded")
        return 1

    print(f"Found {len(freethreaded_pythons)} free-threaded Python version(s):")
    for python_path, version in freethreaded_pythons:
        print(f"  - {version}: {python_path}")

    # Test scripts to run
    test_scripts = [
        ("test_freethreading_verification.py", "Free-threading verification test"),
        ("test_minimal_freethreaded.py", "Minimal free-threading performance test"),
        ("test_woeboost_freethreaded.py", "WoeBoost-style binning with free-threading"),
    ]

    # Run tests with each Python version
    results = {}

    for python_path, version in freethreaded_pythons:
        print(f"\n🧪 Testing with {version}")
        version_results = {}

        for test_script, description in test_scripts:
            success = run_test_with_python(python_path, test_script, description)
            version_results[test_script] = success

        results[version] = version_results

    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")

    for version, version_results in results.items():
        print(f"\n{version}:")
        for test_script, success in version_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {test_script}: {status}")

    # Overall success
    all_passed = all(all(version_results.values()) for version_results in results.values())

    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
