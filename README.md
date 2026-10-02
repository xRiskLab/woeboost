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
- **⚡ Flexibility**: Works seamlessly with diverse data types and bins and transforms features in parallel by default, with free-threaded Python support.

## Installation ⤵

### Standard Installation

Install the package using pip:

```bash
pip install woeboost
```

## ⚡ Parallelism

Features are binned and transformed in parallel on a thread pool by default — no configuration needed.
The number of workers is chosen from the number of features (up to 4, or up to 8 on free-threaded Python).

```python
from concurrent.futures import ThreadPoolExecutor

from woeboost import WoeLearner

learner = WoeLearner()  # parallel by default
learner = WoeLearner(n_tasks=8)  # explicit number of workers
learner = WoeLearner(n_tasks=1)  # sequential
learner = WoeLearner(n_tasks=8, executor_cls=ThreadPoolExecutor)  # any executor taking max_workers
print(f"Free-threading detected: {learner.is_freethreaded}")
```

Results are identical to sequential processing. Measured on an Apple Silicon Mac (n=200,000, 20 features, 30 boosting rounds):

| Python | `n_tasks=1` | default |
|--------|-------------|---------|
| 3.14 (GIL) | 11.1s | 5.1s |
| 3.14t (free-threaded) | 11.0s | 4.0s |

NumPy releases the GIL inside its kernels, so most of the speedup is available on the standard build as well.

### Free-Threaded Python (Experimental)

On a free-threaded build, WoeBoost uses more workers. Note that the GIL is re-enabled at runtime if an extension
module does not declare free-threading support (`is_freethreaded` reflects the runtime state); set `PYTHON_GIL=0` to override.

```bash
uv python install 3.14t
PYTHON_GIL=0 uv run --python 3.14t --with woeboost python your_script.py

# Run free-threaded tests
./tests/run_freethreaded_tests.sh
```

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
- **[`classifier.py`](https://github.com/xRiskLab/woeboost/blob/main/docs/classifier.md)**: Module for building a boosted classification model.
- **[`explainer.py`](https://github.com/xRiskLab/woeboost/blob/main/docs/explainer.md)**: Module for explaining the model predictions.

## 📝 Changelog

For a changelog, see [CHANGELOG](CHANGELOG.md).

## 📄 License

This project is licensed under the MIT License - see the **[LICENSE](https://github.com/xRiskLab/woeboost/blob/main/LICENSE.md)** file for details.
