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
├── freethreaded/            # Free-threaded Python tests
│   ├── test_freethreaded.py
│   ├── test_freethreading_verification.py
│   ├── test_woeboost_freethreaded.py
│   ├── test_minimal_freethreaded.py
│   ├── run_freethreaded_tests.py
│   ├── requirements-freethreaded.txt
│   ├── pyproject-freethreaded.toml
│   └── README_FREETHREADED.md
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

### Free-threaded Tests (`tests/freethreaded/`)
- **Purpose**: Test performance with free-threaded Python builds
- **Scope**: CPU-intensive operations and threading performance
- **Dependencies**: Free-threaded Python 3.14+ builds
- **Speed**: Variable (performance tests can take longer)
- **Examples**:
  - Free-threading verification
  - Binning performance with free-threading
  - Thread scaling analysis

## Running Tests

### Quick Commands

```bash
# Run all standard tests (unit + integration)
make test

# Run specific test categories
make test-unit
make test-integration
make test-freethreaded

# Run with coverage
make test-coverage

# Run everything
make test-all
```

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
uv run pytest -m freethreaded -v

# Run with coverage
uv run pytest --cov=woeboost --cov-report=html tests/unit tests/integration
```

## Test Markers

The test suite uses pytest markers for categorization:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.freethreaded` - Free-threaded Python tests
- `@pytest.mark.slow` - Slow running tests

## Free-threaded Python Testing

See [tests/freethreaded/README_FREETHREADED.md](freethreaded/README_FREETHREADED.md) for detailed information about free-threaded Python testing.

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

### Free-threaded Tests
1. Create test file in `tests/freethreaded/`
2. Use `@pytest.mark.freethreaded` marker
3. Test CPU-intensive operations
4. Compare performance with/without free-threading

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
