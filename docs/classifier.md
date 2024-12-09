# 🎛️ `classifier.py`

The `classifier.py` module implements the **WoeBoostClassifier**, the training orchestrator. Building upon the `WoeLearner` from `learner.py`, it trains a sequence of estimators, applies evidence-based learning, and supports high interpretability in decision-making contexts.

> The `WoeBoostClassifier` combines the strengths of gradient boosting and Weight of Evidence (WOE) principles for interpretable and high-performing scoring models.

---

## **Key Features**

- **Boosted Learning Framework**:
  - Iteratively improves predictions by minimizing the binomial log-likelihood loss.
  - Uses residual-based WOE learning for updating evidence.
- **Early Stopping**:
  - Stops training when improvements fall below a specified tolerance (`tol`).
- **Flexible Binning**:
  - Supports deterministic and randomized histogram-based and quantile-based binning strategies.
- **Monotonicity Constraints**:
  - Allows monotonicity constraints for each feature, inferred automatically if enabled.
- **Deciban Scoring**:
  - Outputs interpretable scores in decibans, ensuring interpretability.

> Different levels of verbosity can be set in accordance with Python's `logging` module (default is `logging.WARNING`).

---

## **Class: `WoeBoostClassifier`**

The `WoeBoostClassifier` class implements the main functionality of the boosted WOE-based scoring system.

### **Constructor Parameters**

| Parameter             | Type                        | Default            | Description                                                                 |
|-----------------------|-----------------------------|--------------------|-----------------------------------------------------------------------------|
| `config`              | `WoeBoostConfig`           | `None`             | Configuration object encapsulating model parameters.                        |
| `estimator`           | `BaseEstimator`            | `None`             | Instance of `WoeLearner` or compatible model.                               |
| `n_estimators`        | `int`                      | `1000`             | Number of boosting iterations.                                              |
| `bin_strategy`        | `str`                      | `'histogram'`      | Strategy for binning continuous features (`'histogram'` or `'quantile'`).   |
| `n_bins_range`        | `Optional[Union[str, tuple]]` | `None`            | Range or strategy for bin count selection during training.                  |
| `subsample_range`     | `tuple`                    | `(1.0, 1.0)`       | Range of subsample proportions or sizes for training iterations.            |
| `random_state`        | `int`                      | `None`             | Seed for reproducibility.                                                   |
| `early_stopping`      | `bool`                     | `True`             | Enables early stopping based on tolerance.                                  |
| `tol`                 | `float`                    | `1e-5`             | Tolerance threshold for early stopping.                                     |
| `infer_monotonicity`  | `bool`                     | `False`            | Automatically infers monotonicity constraints for features.                 |
| `early_stopping_metric` | `Callable`               | `log_loss`         | Metric used for early stopping (default is log loss).                       |
| `verbosity`           | `int`                      | `logging.WARNING`  | Logging verbosity level (`DEBUG`, `INFO`, etc.).                            |

---

### **Attributes**

| Attribute             | Type                    | Description                                                                 |
|-----------------------|-------------------------|-----------------------------------------------------------------------------|
| `estimators`          | `List[BaseEstimator]`  | List of trained `WoeLearner` instances from each boosting iteration.        |
| `base_score`          | `float`                | A priori log-odds of the target variable, used as the initial evidence.     |
| `loss_history`        | `List[Dict[str, float]]` | Record of training and validation loss for each iteration.                  |
| `monotonicity`        | `dict`                 | Monotonicity constraints for features, inferred or user-specified.          |
| `metadata`            | `dict`                 | Metadata about features, including names and categorical features.        |

---

## **Key Methods**

### `fit(X, y, valid=None)`
Trains the model with the provided dataset.

- **Parameters**:
  - `X`: Feature matrix (`pd.DataFrame` or `np.ndarray`).
  - `y`: Target variable (`np.ndarray` or `pd.Series`).
  - `valid`: Optional tuple (`X_val`, `y_val`) for validation.
- **Returns**: Trained instance of `WoeBoostClassifier`.

---

### `predict_proba(X)`
Predicts probabilities for each class.

- **Parameters**:
  - `X`: Feature matrix (`pd.DataFrame` or `np.ndarray`).
- **Returns**: Class probabilities (`np.ndarray`).

---

### `predict(X)`
Predicts binary class labels.

- **Parameters**:
  - `X`: Feature matrix (`pd.DataFrame` or `np.ndarray`).
- **Returns**: Binary class predictions (`np.ndarray`).

---

### `predict_score(X)`
Calculates the final deciban score for the input data.

- **Parameters**:
  - `X`: Feature matrix (`pd.DataFrame` or `np.ndarray`).
- **Returns**: Deciban scores (`np.ndarray`).

---

### `predict_scores(X)`
Generates feature-level scores in decibans for interpretability.

- **Parameters**:
  - `X`: Feature matrix (`pd.DataFrame` or `np.ndarray`).
- **Returns**: Feature-level deciban scores (`np.ndarray`).

---

### `transform(X)`
Computes cumulative evidence (WOE) for logistic regression.

- **Parameters**:
  - `X`: Feature matrix (`pd.DataFrame` or `np.ndarray`).
- **Returns**: Evidence array (`np.ndarray`).

---

### Class: `WoeBoostConfig`

A configuration class that simplifies the setup of `WoeBoostClassifier`. Use this class to manage and validate parameters.

#### Example Initialization

```python
from classifier import WoeBoostConfig

config = WoeBoostConfig(
    estimator=None,  # or WoeLearner() instance
    n_estimators=500,
    bin_strategy='quantile',
    random_state=42,
    early_stopping=True,
    tol=1e-4
)
```

## **Code Examples**
### Example: Training, Tranforming, and Prediction

```python
from classifier import WoeBoostClassifier

# Instantiate the classifier
classifier = WoeBoostClassifier(
    n_estimators=500,
    bin_strategy='quantile',
    random_state=42,
    infer_monotonicity=True
)

# Fit the model
classifier.fit(X_train, y_train, valid=(X_val, y_val))

# Transform to WOE scores
X_train_woe = classifier.transform(X_test)

# Predict probabilities
probabilities = classifier.predict_proba(X_test)

# Predict labels
predictions = classifier.predict(X_test)

# Predict deciban score for each sample
scores = classifier.predict_score(X_test)
```

### Example: Feature Scores in Decibans

```python
# Feature-level deciban scores
feature_scores = classifier.predict_scores(X_test)
print("Feature-level deciban scores:")
print(feature_scores[:5])
```
**The distribution of scores can be visualized as follows:**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Create credit scores
preds = -woe_boosting_model.predict_score(X.loc[ix_test]) + 40

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
common_params = dict(edgecolor='black', bins='auto', grid=False, alpha=0.7, ax=ax)
pd.Series(feature_scores[y_train == 0]).hist(color='dodgerblue', **common_params)
pd.Series(feature_scores[y_train == 1]).hist(color='salmon', **common_params)
plt.legend(['Good', 'Bad'])
plt.title('Score distribution')
plt.show()
```

![Score Distribution Example](https://raw.githubusercontent.com/xRiskLab/woeboost/main/docs/ims/classifier_predict_score.png)

## Related Modules
- **[`learner.py`](https://github.com/xRiskLab/woeboost/blob/main/docs/learner.md)**: Implements the base learner, `WoeLearner`, used by WoeBoostClassifier.

## References
For an overview of modules, check **[the Technical Note](https://github.com/xRiskLab/woeboost/blob/main/docs/technical_note.md)**.

## License
This project is licensed under the MIT License - see the **[LICENSE](https://github.com/xRiskLab/woeboost/blob/main/LICENSE.md)** file for details.