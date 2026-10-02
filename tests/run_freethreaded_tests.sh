#!/bin/bash
# WoeBoost Free-Threaded Python Test Runner
# Runs the unit and integration suites on installed free-threaded Python builds with the GIL disabled.

set -e

echo "WoeBoost Free-Threaded Python Test Runner"
echo "=========================================="

if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first."
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Extension modules without free-threading support (e.g., pandas 2.x) re-enable the GIL on import
export PYTHON_GIL=0

FREETHREADED_PYTHONS=()
for version in 3.14t 3.13t; do
    if uv python find "$version" &> /dev/null; then
        FREETHREADED_PYTHONS+=("$version")
    fi
done

if [ ${#FREETHREADED_PYTHONS[@]} -eq 0 ]; then
    echo "❌ No free-threaded Python versions found!"
    echo "Please install free-threaded Python with:"
    echo "  uv python install 3.14t"
    exit 1
fi

echo "✅ Found free-threaded Python versions: ${FREETHREADED_PYTHONS[*]}"

for python_version in "${FREETHREADED_PYTHONS[@]}"; do
    echo ""
    echo "🧪 Testing with Python $python_version"
    echo "======================================"

    run=(uv run --python "$python_version" --isolated --with-editable . --with pytest --with faker)

    "${run[@]}" python -c \
        "from woeboost import WoeLearner; assert WoeLearner().is_freethreaded, 'GIL is enabled'"
    "${run[@]}" python -m pytest tests/unit tests/integration -q -p no:cacheprovider

    echo "✅ Tests passed with Python $python_version"
done

echo ""
echo "🎉 All free-threaded tests completed successfully!"
