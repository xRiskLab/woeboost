# -*- coding: utf-8 -*-
"""
classifier.py.

░██╗░░░░░░░██╗░█████╗░███████╗██████╗░░█████╗░░█████╗░░██████╗████████╗
░██║░░██╗░░██║██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝╚══██╔══╝
░╚██╗████╗██╔╝██║░░██║█████╗░░██████╦╝██║░░██║██║░░██║╚█████╗░░░░██║░░░
░░████╔═████║░██║░░██║██╔══╝░░██╔══██╗██║░░██║██║░░██║░╚═══██╗░░░██║░░░
░░╚██╔╝░╚██╔╝░╚█████╔╝███████╗██████╦╝╚█████╔╝╚█████╔╝██████╔╝░░░██║░░░
░░░╚═╝░░░╚═╝░░░╚════╝░╚══════╝╚═════╝░░╚════╝░░╚════╝░╚═════╝░░░░╚═╝░░░

Copyright (c) 2024 xRiskLab / deburky

Licensed under the MIT License. You may obtain a copy of the License at:
    https://opensource.org/licenses/MIT
"""

import logging
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from rich.logging import RichHandler
from scipy.special import expit as sigmoid
from scipy.special import logit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import log_loss

from .learner import WoeLearner


@dataclass
class WoeBoostConfig:  # pylint: disable=too-many-instance-attributes
    """
    Configuration class for WoeBoostClassifier.

    Attributes
    ----------
    estimator : BaseEstimator, optional
        An instance of a WoeLearner or a compatible estimator.

    n_estimators : int
        Number of boosting iterations.

    bin_strategy : str
        Strategy for binning continuous features. Options are 'histogram' or 'quantile'.

    n_bins_range : Optional[Union[str, tuple]]
        Range of `n_bins` values to randomly choose from at each iteration.

    subsample_range : tuple
        Range of `subsample` values to randomly choose from at each iteration.

    random_state : Optional[int]
        Random state for reproducibility in bin randomization.

    early_stopping : bool
        Enables automatic stopping based on loss reduction if True.

    tol : Optional[float]
        Tolerance threshold for early stopping.

    infer_monotonicity : bool
        If `True`, infers monotonicity constraints for each feature.

    early_stopping_metric : Callable
        Metric to use for early stopping. Default is log loss.

    verbosity : Optional[int]
        Verbosity level for logging.
    """

    estimator: BaseEstimator
    n_estimators: int = 1000
    bin_strategy: str = "histogram"
    n_bins_range: Optional[Union[str, tuple]] = None
    subsample_range: tuple = (1.0, 1.0)
    random_state: Optional[int] = None
    early_stopping: bool = True
    tol: Optional[float] = None
    infer_monotonicity: bool = False
    early_stopping_metric: Callable = log_loss
    verbosity: Optional[int] = logging.WARNING

    def __post_init__(self):
        """
        Perform post-initialization steps.

        Sets default values for `n_bins_range` and validates `subsample_range`
        and `n_estimators`.
        """
        if self.n_bins_range is None:
            self.n_bins_range = "scott" if self.bin_strategy == "histogram" else (3, 8)
        if not isinstance(self.subsample_range, tuple) or len(self.subsample_range) != 2:
            raise ValueError("subsample_range must be a tuple of size 2.")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0.")


    @classmethod
    def from_dict(cls, param_dict: dict) -> "WoeBoostConfig":
        """
        Create a WoeBoostConfig instance from a dictionary.

        Parameters:
        -----------
        param_dict : dict
            Dictionary containing configuration parameters.

        Returns:
        --------
        WoeBoostConfig : An instance of WoeBoostConfig.
        """
        return cls(**param_dict)


class WoeBoostClassifier(BaseEstimator, ClassifierMixin):  # pylint: disable=too-many-instance-attributes
    """
    Weight of Evidence (WOE) Gradient Boosting Classifier.

    The model optimizes binomial log-likelihood using the logic of WOE.

    Parameters
    ----------
    config : Optional[WoeBoostConfig]
        Configuration for the classifier, encapsulated in a WoeBoostConfig instance.

    Other Parameters
    ----------------
    Refer to WoeBoostConfig for descriptions of the parameters.

    Attributes
    ----------
    estimators : List[BaseEstimator]
        List of trained estimators from each boosting iteration.

    base_score : float
        The a priori log odds of the theory used as the base score.

    loss_history : list of dict
        A record of training and validation loss for each iteration.

    monotonicity : dict
        A dictionary of inferred monotonicity constraints for each feature.

    Methods
    -------
    fit(X, y)
        Trains the model with early stopping based on tolerance.

    predict_proba(X)
        Returns class probability estimates for the input data.

    predict(X)
        Predicts binary class labels for input data.

    predict_score(X)
        Predicts score for input data in decibans.
    """

    def __init__(
        self,
        config: Optional[WoeBoostConfig] = None,
        estimator: Optional[BaseEstimator] = None,
        n_estimators: int = 1000,
        bin_strategy: str = "histogram",
        n_bins_range: Optional[Union[str, tuple]] = None,
        subsample_range: tuple = (1.0, 1.0),
        random_state: Optional[int] = None,
        early_stopping: bool = True,
        tol: Optional[float] = None,
        infer_monotonicity: bool = False,
        early_stopping_metric: Callable = log_loss,
        verbosity: Optional[int] = logging.WARNING,
    ):  # pylint: disable=too-many-locals, too-many-arguments,  too-many-positional-arguments
        """
        Initialize the WoeBoostClassifier.

        Parameters:
        -----------
        config : Optional[WoeBoostConfig]
            Configuration for the classifier, encapsulated in a WoeBoostConfig instance.
        estimator : Optional[BaseEstimator]
            An instance of a WoeLearner or a compatible estimator. If None, a default
            WoeLearner is instantiated. To use custom logging verbosity or other specific
            configurations for WoeLearner like multi-threading, instantiate it separately.
        n_estimators : int
            Number of boosting iterations.
        bin_strategy : str
            Strategy for binning continuous features. Options are 'histogram' or 'quantile'.
        n_bins_range : Optional[Union[str, tuple]]
            Range of `n_bins` values to randomly choose from at each iteration.
        subsample_range : tuple
            Range of `subsample` values to randomly choose from at each iteration.
        random_state : Optional[int]
            Random state for reproducibility in bin randomization.
        early_stopping : bool
            Enables automatic stopping based on loss reduction if True.
        tol : Optional[float]
            Tolerance threshold for early stopping.
        infer_monotonicity : bool
            If `True`, infers monotonicity constraints for each feature.
        early_stopping_metric : Callable
            Metric to use for early stopping. Default is log loss.
        verbosity : Optional[int]
            Verbosity level for logging.
        """
        # Handle configuration
        if config is not None:
            self.config = config
        else:
            self.config = WoeBoostConfig(
                estimator=estimator,
                n_estimators=n_estimators,
                bin_strategy=bin_strategy,
                n_bins_range=n_bins_range,
                subsample_range=subsample_range,
                random_state=random_state,
                early_stopping=early_stopping,
                tol=tol,
                infer_monotonicity=infer_monotonicity,
                early_stopping_metric=early_stopping_metric,
                verbosity=verbosity
            )

        # Map configuration fields to attributes
        self.estimator = self.config.estimator
        self.n_estimators = self.config.n_estimators
        self.bin_strategy = self.config.bin_strategy
        self.n_bins_range = self.config.n_bins_range
        self.subsample_range = self.config.subsample_range
        self.random_state = self.config.random_state
        self.early_stopping = self.config.early_stopping
        self.tol = self.config.tol
        self.infer_monotonicity = self.config.infer_monotonicity
        self.early_stopping_metric = self.config.early_stopping_metric
        self.verbosity = self.config.verbosity

        # Initialize other attributes
        self.estimators: List[BaseEstimator] = []
        self.base_score = None
        self.loss_history = []
        self.monotonicity = None
        self.metadata = None
        self._configure_logger()  # Configure logger

        # Validate n_bins_range based on bin_strategy
        valid_types = {"quantile": tuple, "histogram": (str, tuple, list)}

        if self.bin_strategy in {"histogram"} and isinstance(self.n_bins_range, (list, tuple)):
            for value in self.n_bins_range:
                if not isinstance(value, (str, int, tuple)) or (
                    isinstance(value, tuple) and len(value) != 2
                ):
                    self.logger.error(
                        "[bold red]Invalid value in n_bins_range for histogram:[/bold red] %s\n"
                        "[bold cyan]Expected:[/bold cyan] strings (e.g., 'scott'), "
                        "integers, or tuples of size 2 (e.g., (1, 5)).",
                        value,
                    )
                    raise ValueError(
                        "n_bins_range for histogram must contain strings (e.g., 'scott'), integers,"
                        f" or tuples of size 2 (e.g., (1, 5)). Got {value}."
                    )
        elif not isinstance(self.n_bins_range, valid_types.get(self.bin_strategy, ())):
            valid_type_desc = {
                "quantile": "a tuple like (4, 10)",
                "histogram": "a string (e.g., 'scott'), a tuple (e.g., (4, 10)),"
                " or a list containing valid values",
            }
            self.logger.error(
                "[bold red]Invalid n_bins_range for bin_strategy '%s':[/bold red] %s\n"
                "[bold cyan]Expected:[/bold cyan] %s",
                self.bin_strategy,
                self.n_bins_range,
                valid_type_desc.get(self.bin_strategy, "valid"),
            )
            raise ValueError(
                f"When bin_strategy='{self.bin_strategy}', "
                f"n_bins_range must be {valid_type_desc.get(self.bin_strategy, 'valid')}. "
                f"Got {self.n_bins_range}."
            )

        # Set default tolerance if early_stopping is enabled
        if self.early_stopping:
            self.tol = tol if tol is not None else 1e-5
        elif tol is not None:
            self.logger.error(
                "[bold red]Invalid parameter:[/bold red] tol=%s\n"
                "[bold cyan]Expected:[/bold cyan] tol can only be set when early_stopping=True.",
                tol
            )
            # Raise an error if tol is provided but early_stopping is False
            raise ValueError("`tol` can only be set when `early_stopping=True`.")

    def _configure_logger(self):
        """Configure logger and handlers for the class."""
        self.logger = logging.getLogger("woeboost")

        # Avoid duplicate handlers
        if not self.logger.hasHandlers():
            # Use RichHandler with custom formatting
            console_handler = RichHandler(markup=True, rich_tracebacks=True, show_path=False)
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(console_handler)

        # Set logger level and enforce handler levels
        self.logger.setLevel(self.verbosity)
        for handler in self.logger.handlers:
            handler.setLevel(self.verbosity)

    def _infer_feature_metadata(self, X: Union[pd.DataFrame, np.ndarray]) -> dict:  # pylint: disable=invalid-name
        """
        Infer metadata such as feature names and categorical features.

        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            The input dataset.

        Returns:
        --------
        metadata : dict
            A dictionary with inferred feature_names and categorical_features.
        """
        metadata = {}
        if isinstance(X, pd.DataFrame):
            metadata["feature_names"] = X.columns.tolist()
            metadata["categorical_features"] = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        else:
            metadata["feature_names"] = [f"feature_{i}" for i in range(X.shape[1])]
            metadata["categorical_features"] = []

        return metadata

    def _infer_monotonicity(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> dict:  # pylint: disable=invalid-name
        """
        Infer monotonicity constraints based on the initial dataset (X, y).

        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            The input dataset.
        y : np.ndarray
            The response vector.
        """
        def calculate_correlation(feature_data, y):
            mask = ~np.isnan(feature_data) & ~np.isnan(y)
            feature_data, y_clean = feature_data[mask], y[mask]
            if len(feature_data) > 1:
                correlation = np.corrcoef(feature_data, y_clean)[0, 1]
                return (
                    "increasing"
                    if correlation > 0
                    else "decreasing" if correlation < 0 else None
                )
            return None

        # Lazily infer feature names and categorical features if estimator is None
        if self.estimator.feature_names is None:
            if isinstance(X, pd.DataFrame):
                self.estimator.feature_names = X.columns.tolist()
            else:
                self.estimator.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        if self.estimator.categorical_features is None:
            if isinstance(X, pd.DataFrame):
                self.estimator.categorical_features = X.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()
            else:
                self.estimator.categorical_features = []

        monotonicity = {}
        if isinstance(X, pd.DataFrame):
            for feature in self.estimator.feature_names:
                if feature not in self.estimator.categorical_features:
                    monotonicity[feature] = calculate_correlation(X[feature].to_numpy(), y)
        else:  # Assume NumPy array
            for i, feature in enumerate(self.estimator.feature_names):
                if feature not in self.estimator.categorical_features:
                    monotonicity[feature] = calculate_correlation(X[:, i], y)

        return {k: v for k, v in monotonicity.items() if v}

    def _update_estimator_params(self, estimator: BaseEstimator) -> BaseEstimator:
        """
        Update the parameters of the estimator for each pass.

        Parameters:
        -----------
        estimator : The estimator to update.
        """
        param_updates = {
            "bin_strategy": (self.bin_strategy if hasattr(estimator, "bin_strategy") else None),
            "n_bins": (
                random.choice(self.n_bins_range)
                if isinstance(self.n_bins_range, list)
                else (
                    random.randint(*self.n_bins_range)
                    if isinstance(self.n_bins_range, tuple)
                    else self.n_bins_range if isinstance(self.n_bins_range, str) else None
                )
            ),
            "subsample": (
                (
                    random.uniform(*self.subsample_range)
                    if isinstance(self.subsample_range[0], float)
                    else random.randint(*self.subsample_range)
                )
                if hasattr(estimator, "subsample")
                else None
            ),
            "base_score": self.base_score if hasattr(estimator, "base_score") else None,
            "random_state": (self.random_state if hasattr(estimator, "random_state") else None),
            "monotonicity": self.monotonicity if hasattr(estimator, "monotonicity") else None,
        }
        estimator.set_params(**{k: v for k, v in param_updates.items() if v is not None})
        return estimator

    # pylint: disable=too-many-locals, invalid-name
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        valid: Optional[tuple] = None,
    ) -> "WoeBoostClassifier":
        """
        Fits the inputs into the WoeBoost Classifier.

        Parameters:
        -----------
        X : The input features.
        y : The response vector.
        valid : Optional[tuple]
            Validation data (X_val, y_val).
        """
        # Infer metadata
        self.metadata = self._infer_feature_metadata(X)
        feature_names = self.metadata["feature_names"]
        categorical_features = self.metadata["categorical_features"]

        # Configure the estimator if not provided
        if self.estimator is None:
            self.estimator = WoeLearner(
                feature_names=feature_names,
                categorical_features=categorical_features,
                n_bins=self.n_bins_range,
                bin_strategy=self.bin_strategy,
                random_state=self.random_state
            )

        # Validate metadata in the estimator
        if self.estimator.feature_names is None:
            self.estimator.feature_names = feature_names
        if self.estimator.categorical_features is None:
            self.estimator.categorical_features = categorical_features

        self.logger.info(
            "Starting [bold pale_green3]WoeBoostClassifier[/bold pale_green3] "
            "training with %d estimators.",
            self.n_estimators,
        )

        # Infer monotonicity request
        if self.infer_monotonicity and self.monotonicity is None:
            self.monotonicity = self._infer_monotonicity(X, y)
        if self.infer_monotonicity:
            max_len = max(len(feature) for feature in self.monotonicity.keys())
            formatted_constraints = "\n".join(
                f"[#8787d7]{feature.ljust(max_len)}[/#8787d7] : " f"[#d787d7]{constraint}[/#d787d7]"
                for feature, constraint in self.monotonicity.items()
            )
            self.logger.info(
                "[bold #d7d787]Inferred Monotonicity Constraints:[/bold #d7d787]\n%s",
                formatted_constraints,
            )

        # Initialize base score
        self.base_score = logit(y.mean())
        self.logger.info(
            "[bold pale_green3]Base score[/bold pale_green3] initialized to prior log-odds: %.4f",
            self.base_score,
        )
        # Bring in the a priori log-odds of the theory
        evidence = np.full(y.shape, self.base_score)

        # pylint: disable=invalid-name
        if valid is not None:
            X_val, y_val = valid
            val_evidence = np.full(y_val.shape, self.base_score)

        for iteration in range(self.n_estimators):
            p = np.clip(sigmoid(evidence), 1e-15, 1 - 1e-15)
            residual = y - p
            if self.random_state is not None:
                random.seed(self.random_state + iteration)
            estimator = self._update_estimator_params(clone(self.estimator))
            estimator.fit(X, residual)
            self.estimators.append(estimator)
            evidence += estimator.predict(X)
            train_loss = self.early_stopping_metric(y, p)

            if valid is not None:
                val_evidence += estimator.predict(X_val)
                p_val = sigmoid(val_evidence)
                val_loss = self.early_stopping_metric(y_val, p_val)
                stop_loss = val_loss
                self.loss_history.append({"train_loss": train_loss, "val_loss": val_loss})
            else:
                stop_loss = train_loss
                self.loss_history.append({"train_loss": train_loss})

            self.logger.info(
                "Iteration %2d \t Loss: %.4f [train]%s",
                iteration + 1,
                train_loss,
                f" \t Loss: {val_loss:.4f} [val]" if valid else "",
                extra={"markup": False},
            )
            if self.early_stopping and iteration > 0:
                # Check previous loss for relative change
                previous_loss = (
                    self.loss_history[-2]["val_loss"]
                    if valid
                    else self.loss_history[-2]["train_loss"]
                )
                loss_change = (previous_loss - stop_loss) / previous_loss
                if loss_change < self.tol:
                    self.n_estimators = iteration + 1
                    self.logger.info(
                        "[bold pale_green3]Early stopping triggered after %d iterations "
                        "(tol=%.0e).[/bold pale_green3]",
                        iteration + 1,
                        self.tol,
                    )
                    break
        self.logger.info(
            "[bold #4dabf7]Training completed with %d estimators.[/bold #4dabf7]", self.n_estimators
        )
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Return cumulative evidence (WOE) for for logistic regression use.

        Parameters:
        -----------
        X : The input features.

        Returns:
        --------
        np.ndarray : Cumulative evidence (WOE) for each feature.
        """
        # Initialize cumulative evidence as a zero matrix
        cumulative_evidence = np.zeros_like(X, dtype=float)

        # Sum WOE values across all estimators
        for estimator in self.estimators:
            cumulative_evidence += estimator.transform(X)

        return cumulative_evidence

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Predict probabilities based on the trained model.

        Parameters:
        -----------
        X : The input features.
        """
        evidence = np.full(X.shape[0], self.base_score)

        for estimator in self.estimators:
            evidence += estimator.predict(X)

        probabilities = sigmoid(evidence)
        return np.column_stack((1 - probabilities, probabilities))

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Predict the class labels based on the trained model.

        Parameters:
        -----------
        X : The input features.
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def predict_score(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Predict final score based on the trained model.

        The scores are measured in decibans.

        Parameters:
        -----------
        X : The input features.
        """
        score = np.full(X.shape[0], self.base_score)

        for estimator in self.estimators:
            score += estimator.predict_score(X)

        return np.round(score, decimals=1)

    def predict_scores(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Predict feature scores based on the trained model.

        The scores are measured in decibans.

        Parameters:
        -----------
        X : The input features.

        Returns:
        --------
        np.ndarray
            Predicted feature-level scores (in decibans) for each instance.
            Shape: (n_samples, n_features).
        """
        # Ensure X is converted to a NumPy array
        X = np.asarray(X)

        # Initialize cumulative evidence per feature
        n_features = X.shape[1]
        cumulative_evidence = np.full((X.shape[0], n_features), self.base_score / n_features)

        # Sum WOE values across all estimators, adjusted to decibans
        for estimator in self.estimators:
            cumulative_evidence += 10 * estimator.transform(X) / np.log(10)

        return cumulative_evidence
