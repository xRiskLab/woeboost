# WoeBoost Test Suite

This directory contains the comprehensive test suite for WoeBoost, organized into different categories for better maintainability and execution control.

## Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_classifier.py   # WoeBoostClassifier unit tests
│   ├── test_explainer.py    # WoeExplainer unit tests
│   └── test_learner.py      # WoeLearner unit tests
├── integration/             # Integration tests
│   └── test_woeboost_performance.py  # Performance and integration tests
├── run_freethreaded_tests.sh  # Runs unit + integration tests on free-threaded Python
└── __init__.py
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Scope**: Individual classes, methods, and functions
- **Dependencies**: Minimal, mocked external dependencies
- **Speed**: Fast execution (< 1 second per test)
- **Examples**: 
  - WoeLearner binning logic
  - WoeExplainer calculation methods
  - WoeBoostClassifier prediction methods

### Integration Tests (`tests/integration/`)
- **Purpose**: Test component interactions and end-to-end workflows
- **Scope**: Multiple components working together
- **Dependencies**: Real data, actual libraries
- **Speed**: Moderate execution (1-10 seconds per test)
- **Examples**:
  - Full WoeBoost training and prediction pipeline
  - Performance benchmarks
  - Threading performance tests

### Free-threaded Runs (`run_freethreaded_tests.sh`)
- **Purpose**: Run the unit and integration suites on free-threaded Python builds (3.14t, 3.13t)
- **Scope**: Thread safety of parallel binning and transformation without the GIL
- **Dependencies**: A free-threaded Python installed with `uv python install 3.14t`

## Running Tests

### Using the Test Runner

```bash
# Run all standard tests
python run_tests.py --all

# Run specific categories
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --freethreaded

# Run with coverage
python run_tests.py --coverage

# Run everything
python run_tests.py --everything
```

### Using pytest directly

```bash
# Run unit tests
uv run pytest tests/unit -v

# Run integration tests
uv run pytest tests/integration -v

# Run with markers
uv run pytest -m unit -v
uv run pytest -m integration -v

# Run with coverage
uv run pytest --cov=woeboost --cov-report=html tests/unit tests/integration
```

## Test Markers

The test suite uses pytest markers for categorization:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow running tests

## Free-threaded Python Testing

`tests/run_freethreaded_tests.sh` runs the unit and integration suites on every installed free-threaded
Python (`uv python install 3.14t`) with the GIL disabled:

```bash
./tests/run_freethreaded_tests.sh
```

`PYTHON_GIL=0` is required because extension modules that do not declare free-threading support
(e.g., pandas 2.x) re-enable the GIL on import.

## Continuous Integration

The test suite is designed to work with CI/CD pipelines:

- **Unit tests**: Run on every commit (fast feedback)
- **Integration tests**: Run on pull requests (comprehensive testing)
- **Free-threaded tests**: Run on schedule or manual trigger (experimental)

## Adding New Tests

### Unit Tests
1. Create test file in `tests/unit/`
2. Use `@pytest.mark.unit` marker
3. Mock external dependencies
4. Keep tests fast and focused

### Integration Tests
1. Create test file in `tests/integration/`
2. Use `@pytest.mark.integration` marker
3. Use real data and dependencies
4. Test complete workflows

## Performance Benchmarks

The test suite includes performance benchmarks to ensure WoeBoost maintains good performance:

- **Threading performance**: Tests concurrent processing efficiency
- **Memory usage**: Monitors memory consumption during training
- **Speed benchmarks**: Ensures operations complete within expected timeframes
- **Free-threading benefits**: Measures performance improvements with free-threaded Python

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `pythonpath` is set correctly in `pyproject.toml`
2. **Free-threaded tests failing**: Check if free-threaded Python is installed
3. **Slow tests**: Use `-m "not slow"` to skip slow tests during development
4. **Coverage issues**: Ensure all test files are in the correct directories

### Debug Mode

```bash
# Run tests with verbose output
uv run pytest tests/unit -v -s

# Run specific test with debugging
uv run pytest tests/unit/test_learner.py::test_specific_function -v -s

# Run with pdb debugging
uv run pytest tests/unit/test_learner.py::test_specific_function --pdb
```
