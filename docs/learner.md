# 🕵️ `learner.py`

The `learner.py` module implements the core Weight of Evidence (WOE) learner, responsible for binning features, enforcing monotonicity, and transforming input features into interpretable WOE scores. It serves as the foundation of the WoeBoost framework, allowing for transparent evidence-based modeling.

> `WoeLearner` is a truly "weak" learner such that it does not make any assumptions about the data, simply collecting evidence across features through residuals passed to it from `WoeBoostClassifier` (classifier.py).
---

## **Key Features**

### 1. **Feature Binning**
   - Supports **quantile** and **histogram-based binning**.
   - Handles categorical features by treating levels as unique bins.
   - Handles missing values by assigning log-odds of the average event rate.
   - Automatically detects categorical features and bins them based on unique categories.

### 2. **Monotonicity Constraints**
   - Enforces monotonicity constraints for numerical features using **Isotonic Regression**.
   - Supports both `increasing` and `decreasing` constraints, ensuring interpretability.

### 3. **WOE Transformation**
   - Transforms input features into WOE scores, representing logarithmic evidence.
   - Produces interpretable scores that can be summed across features and iterations.

### 4. **Parallel Processing**
   - Accelerates binning and transformation operations using multi-threading by setting `n_threads`.

> Different levels of verbosity can be set in accordance with Python's `logging` module (default is `logging.WARNING`).

---

## **Class: `WoeLearner`**

The `WoeLearner` class implements the core functionality of the Weight of Evidence (WOE) learner, responsible for binning features, enforcing monotonicity, and transforming input features into interpretable WOE scores.

### **Constructor Parameters**

| Parameter               | Type               | Default       | Description                                                                 |
|-------------------------|--------------------|---------------|-----------------------------------------------------------------------------|
| `feature_names`         | `List[str]`       | `None`        | List of feature names to bin and transform.                                |
| `n_bins`                | `int`             | `None`        | Number of bins or method for binning (e.g., `'scott'` for histograms).     |
| `subsample`             | `int` or `float`  | `None`        | Proportion or number of samples to use for binning.                        |
| `base_score`            | `float`           | `None`        | Initial score (intercept).                                                 |
| `bin_strategy`          | `str`             | `'histogram'` | Binning strategy (`'histogram'` or `'quantile'`).                          |
| `random_state`          | `int`             | `None`        | Seed for random number generation.                                         |
| `categorical_features`  | `List[str]`       | `None`        | List of categorical features.                                              |
| `monotonicity`          | `dict`            | `None`        | Dictionary of monotonicity constraints (`'increasing'` or `'decreasing'`). |
| `infer_monotonicity`    | `bool`            | `False`       | Whether to automatically infer monotonicity.                               |
| `n_threads`             | `int`             | `None`        | Number of threads for parallel operations.                                 |
| `verbosity`             | `int`             | `logging.WARNING` | Logging level (`DEBUG`, `INFO`, etc.).                                    |

---

### **Attributes**

| Attribute           | Type  | Description                                                              |
|---------------------|-------|--------------------------------------------------------------------------|
| `bins_`             | `dict`| Bin edges for each feature.                                              |
| `bin_averages_`     | `dict`| Average `y` values in each bin for each feature.                         |
| `bin_counts_`       | `dict`| Count of samples in each bin for each feature.                          |
| `monotonicity`      | `dict`| Monotonicity constraints for each feature.                              |

---

## **Key Methods**

### **`fit(X, y)`**
Fits the WoeLearner model to the input features and target variable.

#### Parameters:
- `X` (`np.ndarray` or `pd.DataFrame`): Input features (2D array-like structure).
- `y` (`np.ndarray` or `pd.Series`): Target variable (1D array-like).

#### Code Example:
```python
learner = WoeLearner(feature_names=["age", "income"], n_bins=10, bin_strategy="quantile")
learner.fit(X_train, y_train)
```

### **`transform(X)`**
Transforms the input features into Weight of Evidence (WOE) scores.

#### Parameters:
- `X` (`np.ndarray` or `pd.DataFrame`): Input features.
#### Returns:
`np.ndarray`: Transformed features as WOE scores.
Code Example:

```python
X_woe = learner.transform(X_test)
```

### **`predict(X)`**
Predicts cumulative WOE scores for the input features.

#### Parameters:
- `X` (`np.ndarray` or `pd.DataFrame`): Input features.
#### Returns:
`np.ndarray`: Predicted cumulative WOE scores.
Code Example:

```python
predictions = learner.predict(X_test)
```
### **`predict_score(X)`**
Predicts deciban scores (scaled logarithmic evidence) for the input features.

#### Parameters:
- `X` (`np.ndarray` or `pd.DataFrame`): Input features.
#### Returns:
`np.ndarray`: Predicted deciban scores.
Code Example:

```python
scores = learner.predict_score(X_test)
```

## **Code Examples**
### Example: Fitting and Transforming

```python
from woeboost.learner import WoeLearner

# Define data
X_train = np.array([[25, 30000], [45, 70000], [35, 40000]])
y_train = np.array([1, 0, 1])

# Initialize and fit the WoeLearner
learner = WoeLearner(
    feature_names=["tenure", "income"], n_bins=5, bin_strategy="histogram"
)
learner.fit(X_train, y_train)

# Transform the features into WOE scores
X_woe = learner.transform(X_train)
print(f"WOE Transformed Data: \t {X_woe[:5]}")
```

### Example: Predicting Scores

```python
# Predict WOE scores
woe_scores = learner.predict(X_train)
print(f"WOE Scores: {woe_scores[:5]}")

# Predict deciban scores
deciban_scores = learner.predict_score(X_train)
print(f"Deciban Scores: {deciban_scores[:5]}")
```

## Related Modules
- [`classifier.py`](https://github.com/xRiskLab/woeboost/blob/main/docs/classifier.md): Uses `WoeLearner` to build the boosted classification model.

## **References**
For an overview of modules, check **[the Technical Note](https://github.com/xRiskLab/woeboost/blob/main/docs/technical_note.md)**.

## License
This project is licensed under the MIT License - see the **[LICENSE](https://github.com/xRiskLab/woeboost/blob/main/LICENSE.md)** file for details.