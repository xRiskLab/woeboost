# 📜 Technical Note

WoeBoost draws on the ideas expressed by **Alan M. Turing** and the credit scoring applications of the **Weight of Evidence (WOE)** technique. WOE is an inference framework based on likelihood ratios.

WOE can be understood as a **logarithmic factor** by which we adjust our initial beliefs to maximize the likelihood of data. In its **odds form**, WOE functions as a **Bayes factor**, updating the prior odds to posterior odds.

Conceptually, the algorithm combines the principles of **logistic regression**  estimation and the **gradient boosting** framework. This makes it both interpretable and powerful, blending traditional statistical and machine learning methods.

### 🧑‍🎓 Training

At each boosting iteration, we calculate the residuals on the probability scale:

$$\text{residual}_i = y_i - p_i$$

Here:

* $y_i$ is the true label.
* $p_i$ is the probability of the positive class.

> In the first iteration, we begin with the prior log odds converted to $p_i$. This implies that all evidence added after the initial pass can be understood as WOE factors per each feature.

In the subsequent iterations, we calculate average residuals per each bin and feature.

$$\text{WOE}_j^{(t)} = 1/N \times\ \sum_{b} {residual_i}_{j,b}^{(t)}$$

After this the evidence from all features is added into a single logit score. This is similar to fitting logistic regression on top of WOE scores, but here weights are unity.

$$Score_i = \text{prior\_log\_odds} + \sum_{j=1}^{k} \text{WOE}_j$$

This logic is handled in [`classifier.py`](https://github.com/xRiskLab/woeboost/blob/main/docs/classifier.md) module.

### 💭 Inference

To get the final score, we sum scores from all iterations and convert them to a probability similar to how standard gradient boosting frameworks work.

First we calculate the logit score by summing the prior log odds and WOE scores:

$$\text{Score} = \text{prior\_log\_odds} + \sum_{i=1}^{n}\sum_{j=1}^{k} \text{WOE}_i$$

Where:
* i is feature index.
* j is bin index.

Then we convert the logit score to a probability of positive class using the sigmoid function:

$$\text{P(Y=1)} = \frac{\exp(\text{Score})}{1 + \exp(\text{Score})}$$

Here:

* $prior\_log\_odds$ represents the initial odds before any evidence is introduced.
* Each WOE (Weight of Evidence) is calculated based on the gradient boosting residuals and represents the evidence contributed by a specific feature value.
* The sigmoid function converts the log-odds into a probability between 0 and 1.

This logic is handled in **[`classifier.py`](https://github.com/xRiskLab/woeboost/blob/main/docs/classifier.md)** module.

### 🪣 Binning

Binning is used to discretize inputs, which are required for using the WOE framework.

* For *numerical features*, we can leverage quantile and histogram binning schemes from NumPy. For histogram-based bins, we can apply monotonicity constraints using scikit-learn's *IsotonicRegression*.
* For *categorical features* we take average residuals per each category to avoid any inconsistent aggregation of distinct categories.
* Missing data points are skipped in binning and are replaced with 0 to produce average log-odds given no evidence.

This logic is handled in **[`learner.py`](https://github.com/xRiskLab/woeboost/blob/main/docs/learner.md)** module.

## 📂 Module Overview

WoeBoost is organized into the following key modules:

1. **[`learner.py`](https://github.com/xRiskLab/woeboost/blob/main/docs/learner.md)**  
   - Implements the core **`WoeLearner`**, the foundational class for WoeBoost's training logic.  
   - Handles residual calculations, WOE binning, and iterative evidence accumulation.  
   - Provides extensible methods for advanced customization of the boosting process.  

2. **[`classifier.py`](https://github.com/xRiskLab/woeboost/blob/main/docs/classifier.md)**  
   - Contains the **`WoeBoostClassifier`**, a ready-to-use class for training interpretable models.  
   - Integrates the **`WoeLearner`** to offer a streamlined API for model fitting, prediction, and scoring.  
   - Features AutoML-like enhancements such as monotonicity inference and early stopping.  

3. **[`explainer.py`](https://github.com/xRiskLab/woeboost/blob/main/docs/explainer.md)**  
   - Provides tools for interpreting and visualizing WoeBoost models, including:  
        - **PDPAnalyzer:**  
          Analyze and visualize **Partial Dependence Plots (PDP)** for features to understand their influence on predictions.
          
          **Use Case:** Explore how a single feature or interaction between two features affects predictions.
        
        - **Evidence Analyzer:**  
          Calculate and visualize **evidence contributions** for features over iterations, helping understand how evidence accumulates during boosting.
          
          > **Use Case:** Diagnose feature importance and explain how features contribute to the model's cumulative WOE scores.

        - **WoeInferenceMaker:**  
          Perform **classical WOE calculations** and transform features into interpretable WOE scores for analysis or use in external models like logistic regression.
          
          > **Use Case:** Create interpretable feature transformations and analyze WOE-based feature contributions.
