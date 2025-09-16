# WoeBoost
**Author:** [xRiskLab](https://github.com/xRiskLab)<br>
**License:** [MIT License](https://opensource.org/licenses/MIT) (2025)

![Title](https://raw.githubusercontent.com/xRiskLab/woeboost/main/docs/ims/woeboost.png)

<div align="center">
  <img src="https://github.com/xRiskLab/woeboost/workflows/CI/badge.svg" alt="CI"/>
  <img src="https://github.com/xRiskLab/woeboost/actions/workflows/freethreaded.yml/badge.svg" alt="Free-threaded"/>
  <img src="https://github.com/xRiskLab/woeboost/actions/workflows/compatibility.yml/badge.svg" alt="Compatibility"/>
  <img src="https://img.shields.io/pypi/v/woeboost" alt="PyPI Version"/> 
  <img src="https://img.shields.io/github/license/xRiskLab/woeboost" alt="License"/> 
  <img src="https://img.shields.io/github/contributors/xRiskLab/woeboost" alt="Contributors"/> 
  <img src="https://img.shields.io/github/issues/xRiskLab/woeboost" alt="Issues"/> 
  <img src="https://img.shields.io/github/forks/xRiskLab/woeboost" alt="Forks"/> 
  <img src="https://img.shields.io/github/stars/xRiskLab/woeboost" alt="Stars"/>
</div><br>

**WoeBoost** is a Python 🐍 package designed to bridge the gap between the predictive power of gradient boosting and the interpretability required in high-stakes domains such as finance, healthcare, and law. It introduces an interpretable, evidence-driven framework for scoring tasks, inspired by the principles of **Weight of Evidence (WOE)** and the ideas of **Alan M. Turing**.

## 🔑 Key Features

- **🌟 Gradient Boosting with Explainability**: Combines the strength of gradient boosting with the interpretability of WOE-based scoring systems.
- **📊 Calibrated Scores**: Produces well-calibrated scores essential for decision-making in regulated environments.
- **🤖 AutoML-like Enhancements**:
  - Infers monotonic relationships automatically (`infer_monotonicity`).
  - Supports early stopping for efficient training (`enable_early_stopping`).
- **🔧 Support for Missing Values & Categorical Inputs**: Handles various data types seamlessly while maintaining interpretability.
- **🛠️ Diagnostic Toolkit**:
  - Partial dependence plots.
  - Feature importance analysis.
  - Decision boundary visualization.
- **📈 WOE Inference Maker**: Provides classical WOE calculations and bin-level insights.

## ⚙️ How It Works

1. **🔍 Initialization**: Starts with prior log odds, representing baseline probabilities.
2. **📈 Iterative Updates**: Each boosting iteration calculates residual per each binned feature and sums residuals into total evidence (WOE), updating predictions.
3. **🔗 Evidence Accumulation**: Combines evidence from all iterations, producing a cumulative and interpretable scoring model.

## 🧐 Why WoeBoost?

- **💡 Interpretability**: Every model step adheres to principles familiar to risk managers and data scientists, ensuring transparency and trust.
- **✅ Alignment with Regulatory Requirements**: Calibrated and interpretable results meet the demands of high-stakes applications.
- **⚡ Flexibility**: Works seamlessly with diverse data types and supports concurrency for feature binning with Python's `concurrent.futures`.

## Installation ⤵

### Standard Installation

Install the package using pip:

```bash
pip install woeboost
```

### Free-Threaded Python Support (Experimental)

For significant performance improvements with free-threaded Python builds:

```bash
# Install with free-threaded dependencies
pip install woeboost[freethreaded]

# Or install free-threaded Python first, then WoeBoost
uv python install 3.14.0a5+freethreaded
pip install woeboost[freethreaded]
```

**Benefits of free-threaded Python:**
- **3.67× faster training** - real measured performance improvement
- **Automatic thread optimization** (8 threads vs 4 with GIL)
- **No code changes required** - WoeBoost auto-detects free-threading
- **Same results, faster computation** - identical convergence, 3.67× speedup

```python
from woeboost import WoeLearner

# Automatically detects free-threading and optimizes thread count
learner = WoeLearner(n_tasks=8)  # Uses more tasks with free-threading
print(f"Free-threading detected: {learner.is_freethreaded}")
```

## 🧪 Free-Threaded Python Support

WoeBoost includes experimental support for free-threaded Python builds, providing significant performance improvements for CPU-bound operations:

- **3.67× speedup** for WoeBoost training with Python 3.14+freethreaded
- **Optimal performance at 8 threads** (vs 4 with GIL)
- **Tested on Python 3.14.0a5+freethreaded** (experimental builds)

### Running Free-Threaded Tests

```bash
# Install free-threaded Python
uv python install 3.14.0a5+freethreaded

# Run free-threaded tests
./tests/run_freethreaded_tests.sh
```

See [tests/README_FREETHREADED.md](tests/README_FREETHREADED.md) for detailed information.

## 💻 Example Usage

Below we provide two examples of using WoeBoost.

### Training and Inference with WoeBoost classifier

```python
from woeboost import WoeBoostClassifier

# Initialize the classifier
woe_model = WoeBoostClassifier(infer_monotonicity=True)

# Fit the model
woe_model.fit(X_train, y_train)

# Predict probabilities and scores
probas = woe_model.predict_proba(X_test)[:, 1]
preds = woe_model.predict(X_test)
scores = woe_model.predict_score(X_test)
```

### Preparation of WOE inputs for logistic regression

```python
from woeboost import WoeBoostClassifier

# Initialize the classifier
woe_model = WoeBoostClassifier(infer_monotonicity=True)

# Fit the model
woe_model.fit(X_train, y_train)

X_woe_train = woe_model.transform(X_train)
X_woe_test = woe_model.transform(X_test)
```

## 📚 Documentation

- **[`Technical Note`](https://github.com/xRiskLab/woeboost/blob/main/docs/technical_note.md)**: Overview of the WoeBoost modules.
- **[`learner.py`](https://github.com/xRiskLab/woeboost/blob/main/docs/learner.md)**: Core module implementing a base learner.
- **[`Development Guide`](DEVELOPMENT.md)**: Setup, testing, and contributing guidelines.
- **[`classifier.py`](https://github.com/xRiskLab/woeboost/blob/main/docs/classifier.md)**: Module for building a boosted classification model.
- **[`explainer.py`](https://github.com/xRiskLab/woeboost/blob/main/docs/explainer.md)**: Module for explaining the model predictions.

## 📝 Changelog

For a changelog, see [CHANGELOG](CHANGELOG.md).

## 📄 License

This project is licensed under the MIT License - see the **[LICENSE](https://github.com/xRiskLab/woeboost/blob/main/LICENSE.md)** file for details.
