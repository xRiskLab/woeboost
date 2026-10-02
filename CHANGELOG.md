# Changelog

- **Unreleased**
  - **Parallel by default**: features are binned and transformed on a `ThreadPoolExecutor` without any configuration (`n_tasks=1` restores sequential processing).
  - **Faster binning**: bin statistics are computed in a single pass (`searchsorted` + `bincount`) instead of one mask per bin; fitted bins are unchanged.
  - Feature binning now also runs in parallel (previously only `transform` did, and only with an explicit `executor_cls`).
  - `n_tasks=None` is resolved at fit/transform time from the number of features instead of in `__init__`.
  - Free-threading detection uses `sys._is_gil_enabled()`, so it reports the runtime GIL state.
  - Removed `requirements.txt` (dependencies are managed in `pyproject.toml` / `uv.lock`).

- **v1.1.0** 🚀
  - **Free-threaded Python support** with `woeboost[freethreaded]` optional dependency
  - **Automatic performance optimization** - detects free-threading and optimizes thread usage
  - **3.67x speedup** for WoeBoost training with Python 3.14+freethreaded (real measured performance)
  - **Zero configuration** - works out of the box with automatic thread optimization
  - **Enhanced test suite** with unit, integration, and free-threaded test categories
  - **Comprehensive documentation** for free-threading setup and usage
  - **Backward compatibility** - existing code works unchanged
  - **Code formatting** - added ruff format to pre-commit hooks for consistent code style
  - **README example test** - added test for `print(f"Free-threading detected: {learner.is_freethreaded}")` example

- **v1.0.2**
  - Support for `n_tasks` with legacy `n_threads` fallback (deprecated in the future).
  - Updated concurrency support via a callable (e.g., `ThreadPoolExecutor`).
  - Type hints improvements.

- **v1.0.1**
  - Adjusted feature importance default plot size and added minor updates of documentation.

- **v1.0.0**
  - Initial release of WoeBoost.