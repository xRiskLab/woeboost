#!/bin/bash
# WoeBoost Free-Threaded Python Test Runner
# This script runs WoeBoost performance tests with free-threaded Python builds.

set -e

echo "WoeBoost Free-Threaded Python Test Runner"
echo "=========================================="

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first."
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if free-threaded Python is available
echo "🔍 Checking for free-threaded Python versions..."

# Try to find free-threaded Python versions
FREETHREADED_PYTHONS=()

# Check for Python 3.14+freethreaded
if uv python list | grep -q "3.14.*freethreaded"; then
    FREETHREADED_PYTHONS+=("3.14.0a5+freethreaded")
fi

# Check for Python 3.13+freethreaded
if uv python list | grep -q "3.13.*freethreaded"; then
    FREETHREADED_PYTHONS+=("3.13.2+freethreaded")
fi

if [ ${#FREETHREADED_PYTHONS[@]} -eq 0 ]; then
    echo "❌ No free-threaded Python versions found!"
    echo "Please install free-threaded Python with:"
    echo "  uv python install 3.14.0a5+freethreaded"
    echo "  uv python install 3.13.2+freethreaded"
    exit 1
fi

echo "✅ Found free-threaded Python versions: ${FREETHREADED_PYTHONS[*]}"

# Run tests with each Python version
for python_version in "${FREETHREADED_PYTHONS[@]}"; do
    echo ""
    echo "🧪 Testing with Python $python_version"
    echo "======================================"
    
    # Run the test runner with the specific Python version
    if uv run --python "$python_version" --with freethreaded python tests/freethreaded/run_freethreaded_tests.py; then
        echo "✅ Tests passed with Python $python_version"
    else
        echo "❌ Tests failed with Python $python_version"
        exit 1
    fi
done

echo ""
echo "🎉 All free-threaded tests completed successfully!"
