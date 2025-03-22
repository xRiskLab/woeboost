# -*- coding: utf-8 -*-
"""
learner.py.

░██╗░░░░░░░██╗░█████╗░███████╗██████╗░░█████╗░░█████╗░░██████╗████████╗
░██║░░██╗░░██║██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝╚══██╔══╝
░╚██╗████╗██╔╝██║░░██║█████╗░░██████╦╝██║░░██║██║░░██║╚█████╗░░░░██║░░░
░░████╔═████║░██║░░██║██╔══╝░░██╔══██╗██║░░██║██║░░██║░╚═══██╗░░░██║░░░
░░╚██╔╝░╚██╔╝░╚█████╔╝███████╗██████╦╝╚█████╔╝╚█████╔╝██████╔╝░░░██║░░░
░░░╚═╝░░░╚═╝░░░╚════╝░╚══════╝╚═════╝░░╚════╝░░╚════╝░╚═════╝░░░░╚═╝░░░

Copyright (c) 2025 xRiskLab / deburky

Licensed under the MIT License.

You may obtain a copy of the License at:
    https://opensource.org/licenses/MIT
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from rich.logging import RichHandler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import check_X_y


# pylint: disable=invalid-name
def parse_feature(args):
    """Parse a single feature for binning and transformation."""
    i, X_column, feature_name, categorical_features, bins_, bin_averages_ = args

    if feature_name in categorical_features:
        transformed_feature = _transform_categorical_feature(
            X_column, bins_[feature_name], bin_averages_[feature_name]
        )
    else:
        transformed_feature = _transform_numerical_feature(
            X_column, bins_[feature_name], bin_averages_[feature_name]
        )

    return i, transformed_feature


def encode_categories_once(feature_data):
    """Encode categories once and return unique categories and codes."""
    categories, codes = np.unique(feature_data, return_inverse=True)
    return categories, codes


def _transform_categorical_feature(X_column, categories, avg_per_category):
    """Transform a categorical feature using the average target rate per category."""
    category_to_index = {cat: idx for idx, cat in enumerate(categories)}
    indices = np.array([category_to_index.get(val, -1) for val in X_column])
    avg_per_category_array = np.array(avg_per_category, dtype=float)
    return np.where(indices >= 0, avg_per_category_array[indices], 0.0)


def _transform_numerical_feature(X_column, bin_edges, bin_avgs):
    """Transform a numerical feature using the average target rate per bin."""
    bin_avgs = np.array(bin_avgs)
    binned_indices = np.digitize(X_column, bin_edges) - 1
    binned_indices = np.clip(binned_indices, 0, len(bin_avgs) - 1)
    result = bin_avgs[binned_indices]

    if np.issubdtype(X_column.dtype, np.number):
        nan_mask = np.isnan(X_column)
        result[nan_mask] = 0.0

    return result


class WoeLearner(BaseEstimator, ClassifierMixin):  # pylint: disable=too-many-instance-attributes
    """
    A Weight of Evidence (WOE) Learner for binning and transforming features.

    Parameters
    ----------
    feature_names : List[str]
        List of feature names to bin and transform.

    n_bins : int, default=None
        The number of bins or method for binning (e.g., 'scott' for histogram binning).

    subsample : int or float, default=None
        The number or proportion of samples to use for binning. If float, it must be
        between 0 and 1.

    base_score : float, default=None
        The base score (intercept). If None, the base score is set to 0.5.

    bin_strategy : str, default='histogram'
        The binning strategy to use. Options are:
            - 'histogram': Use histogram-based binning.
            - 'quantile': Use quantile-based binning.

    random_state : int, default=None
        The random seed used to generate random numbers, ensuring reproducibility.

    categorical_features : List[str], default=None
        A list of feature names to treat as categorical variables. These features
        are encoded and binned based on unique categories.

    monotonicity : dict, default=None
        A dictionary specifying the monotonicity constraints for each feature.
        Key-value pairs should be feature names and 'increasing'/'decreasing', e.g.,
        {'feature1': 'increasing'}.

    infer_monotonicity : bool, default=False
        Whether to infer monotonicity constraints automatically based on the data.

    n_tasks : int, default=None
        The number of threads or processes to use for parallel operations. If None,
        operations will be executed sequentially.

    executor_cls : Callable[..., ThreadPoolExecutor], default=None
        The type of executor to use for parallel operations.

    verbosity : int, default=logging.WARNING
        The logging level for controlling verbosity. Options are standard logging
        levels like logging.DEBUG, logging.INFO, and logging.WARNING.

    Attributes
    ----------
    bins_ : dict
        A dictionary containing the bin edges for each feature.

    bin_averages_ : dict
        A dictionary containing the average `y` values in each bin for each feature.

    bin_counts_ : dict
        A dictionary containing the counts of samples in each bin for each feature.

    monotonicity : dict
        A dictionary specifying the monotonicity constraints for each feature.

    Methods
    -------
    fit(X, y)
        Fit the WOE Learner to the input features and target variable.

    transform(X)
        Transform the input features using the binning and learned WOE values.

    predict(X)
        Predict the cumulative WOE values for the input features.

    predict_score(X)
        Predict the score in decibans for the input features.
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        n_bins: Optional[int] = None,
        subsample: Optional[int] = None,
        base_score: Optional[float] = None,
        bin_strategy: str = "histogram",
        random_state: Optional[int] = None,
        categorical_features: Optional[List[str]] = None,
        monotonicity: Optional[dict] = None,
        infer_monotonicity: bool = False,
        n_tasks: Optional[int] = None,
        executor_cls: Optional[Callable[..., ThreadPoolExecutor]] = None,
        verbosity: int = logging.WARNING,
        n_threads: Optional[int] = None,  # legacy parameter
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        """
        Initialize the WoeLearner model.

        Parameters:
        -----------
        feature_names : List[str], default=None
            List of feature names to bin and transform.
        n_bins : int, default=None
            The number of bins or method for binning (e.g., 'scott' for histogram binning).
            If None, the model will raise a ValueError during initialization.
        subsample : int or float, default=None
            The number or proportion of samples to use for binning. If float, it must be
            between 0 and 1.
        base_score : float, default=None
            The base score (intercept). Inferred automatically from data. Defaults to None.
        bin_strategy : str, default='histogram'
            The binning strategy to use. Options are:
                - 'histogram': Use histogram-based binning.
                - 'quantile': Use quantile-based binning.
        random_state : int, default=None
            The random seed used to generate random numbers, ensuring reproducibility.
        categorical_features : List[str], default=None
            A list of feature names to treat as categorical variables. These features
            are encoded and binned based on unique categories.
        monotonicity : dict, default=None
            A dictionary specifying the monotonicity constraints for each feature.
            Key-value pairs should be feature names and 'increasing'/'decreasing', e.g.,
            {'feature1': 'increasing'}.
        infer_monotonicity : bool, default=False
            Whether to infer monotonicity constraints automatically based on the data.
        n_tasks : int, default=None
            The number of threads or processes to use for parallel operations. If None,
            operations will be executed sequentially.
        executor_cls : Callable[..., ThreadPoolExecutor], default=None
            The type of executor to use for parallel operations.
        verbosity : int, default=logging.WARNING
            The logging level for controlling verbosity. Options are standard logging
            levels like logging.DEBUG, logging.INFO, and logging.WARNING.
        """
        self.feature_names: list[str] = feature_names or []
        self.n_bins: Optional[int] = n_bins
        self.subsample: Optional[Union[int, float]] = subsample
        self.base_score: Optional[float] = base_score
        self.bin_strategy: str = bin_strategy
        self.random_state: Optional[int] = random_state
        self.categorical_features: list[str] = (
            categorical_features if categorical_features is not None else []
        )
        self.monotonicity: dict[str, str] = monotonicity if monotonicity is not None else {}
        self.infer_monotonicity: bool = infer_monotonicity

        self.n_threads: Optional[int] = n_threads

        if n_threads is not None:
            warnings.warn(
                "`n_threads` is deprecated and will be removed in a future version. "
                "Please use `n_tasks` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if n_tasks is None:
                n_tasks = n_threads  # Fallback

        self.n_tasks: Optional[int] = n_tasks if n_tasks is not None else 1
        self.executor_cls: Optional[Callable[..., ThreadPoolExecutor]] = executor_cls
        self.verbosity: int = verbosity

        self.warning_issued: bool = False  # Flag to prevent duplicate warnings
        self._configure_logger()  # Configure logger

        self.bins_: dict[str, np.ndarray] = {}
        self.bin_averages_: dict[str, list[float]] = {}
        self.bin_counts_: dict[str, np.ndarray] = {}

    def _configure_logger(self):
        """Configure logger and handlers for the class."""
        self.logger = logging.getLogger("woe_learner")

        # Avoid duplicate handlers
        if not self.logger.hasHandlers():
            # Use RichHandler with custom formatting
            console_handler = RichHandler(markup=True, rich_tracebacks=True)
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(console_handler)

        # Set logger level and enforce handler levels
        self.logger.setLevel(self.verbosity if self.verbosity is not None else logging.WARNING)
        for handler in self.logger.handlers:
            handler.setLevel(self.verbosity if self.verbosity is not None else logging.WARNING)

    def _enforce_monotonicity(self, feature_name: str, increasing: bool = True) -> None:
        """
        Enforce monotonicity in the bin averages for a specific feature using isotonic regression.

        Parameters:
        -----------
        feature_name : str
            The name of the feature for which to enforce monotonicity.
        increasing : bool, default=True
            Whether to enforce an increasing or decreasing trend.
        """
        # Issue warning only once for all features
        if self.bin_strategy in {"quantile"}:
            # Print warning only once
            if not self.warning_issued:
                self.logger.warning(
                    "Monotonicity constraints are disabled for quantile binning.\n"
                    "Use [bold pale_green3]histogram[/bold pale_green3] bin_strategy "
                    "to enforce monotonicity."
                )
                self.warning_issued = True  # Set the flag to prevent further warnings
            return  # Skip for quantile binning

        if feature_name in self.bin_averages_:
            bin_averages = self.bin_averages_[feature_name]
            bin_counts = self.bin_counts_.get(feature_name, [1] * len(bin_averages))

            # Use isotonic regression to enforce monotonicity
            iso_reg = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
            corrected_bin_averages = iso_reg.fit_transform(
                range(len(bin_averages)), bin_averages, sample_weight=bin_counts
            )
            # Update bin averages with monotonic values
            self.bin_averages_[feature_name] = corrected_bin_averages

    def _quantile_binning(
        self, X: np.ndarray, y: np.ndarray, n_bins: int
    ) -> tuple[np.ndarray, list[int], list[float]]:  # pylint: disable=invalid-name
        """
        Perform quantile-based binning.

        Parameters:
        -----------
        X : The input features in a 2D NumPy array.
        y : The target variable.
        n_bins : The number of bins to use.
        """
        if np.isnan(X).any():
            self.logger.warning(
                "[bold pale_green3]NaN values detected in input data.[/bold pale_green3]\n"
                "These observations will be equated to sample log odds.\n"
                "Consider preprocessing the data to impute or remove NaNs."
            )
            bin_edges = np.nanquantile(X, np.linspace(0, 1, n_bins + 1))  # Handle NaNs
        else:
            bin_edges = np.quantile(X, np.linspace(0, 1, n_bins + 1))

        bin_averages = []
        bin_counts = []

        for j in range(len(bin_edges) - 1):
            mask = (X >= bin_edges[j]) & (X < bin_edges[j + 1])
            bin_counts.append(np.sum(mask))
            if np.any(mask):  # Handle empty bins
                bin_averages.append(y[mask].mean())
            else:
                bin_averages.append(0)

        return bin_edges, bin_counts, bin_averages

    def _histogram_binning(
        self, X: np.ndarray, y: np.ndarray, n_bins_or_method: Union[int, str] = "scott"
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:  # pylint: disable=invalid-name
        """
        Perform histogram-based binning.

        Parameters:
        -----------
        X : The input features in a 2D NumPy array.
        y : The target variable.
        n_bins_or_method : The number of bins or method for histograms.
        """
        if np.isnan(X).any():
            self.logger.warning(
                "[bold pale_green3]NaN values detected in input data.[/bold pale_green3]\n"
                "These observations will be equated to sample log odds.\n"
                "Consider preprocessing the data to impute or remove NaNs."
            )
            counts, bin_edges = np.histogram(X[~np.isnan(X)], bins=n_bins_or_method)  # Handle NaNs
        else:
            counts, bin_edges = np.histogram(X, bins=n_bins_or_method)

        bin_averages = []
        bin_counts = []

        for j in range(len(bin_edges) - 1):
            mask = (X >= bin_edges[j]) & (X < bin_edges[j + 1])
            bin_counts.append(np.sum(mask))
            if np.any(mask):  # Handle empty bins
                bin_averages.append(y[mask].mean())
            else:
                bin_averages.append(0)

        return bin_edges, counts, bin_averages

    def _bin_features(self, X: np.ndarray, y: np.ndarray) -> None:  # pylint: disable=invalid-name, too-many-locals, too-many-branches, too-many-positional-arguments, too-many-statements
        """
        Create bins for each feature in the input data.

        Parameters:
        -----------
        X : The input features in a 2D NumPy array.
        y : The target variable.
        """

        def _subsample_data(feature_data, y):
            """
            Subsample the input data for binning.

            Parameters:
            -----------
            feature_data : np.ndarray
                The input feature data.
            y : np.ndarray
                The target variable.

            Returns:
            --------
            tuple : (np.ndarray, np.ndarray)
                Subsampled feature data and target variable.

            Raises:
            -------
            ValueError : If `subsample` is not valid (float in (0, 1], positive integer, or None).
            """
            if self.subsample is None:
                subsample_size = len(feature_data)
            elif isinstance(self.subsample, float):
                if not 0.0 < self.subsample <= 1.0:
                    self.logger.error(
                        "Invalid `subsample` value: %.2f. Must be in the range (0, 1].",
                        self.subsample,
                    )
                    raise ValueError("`subsample` as a float must be in the range (0, 1].")
                subsample_size = int(len(feature_data) * self.subsample)
            elif isinstance(self.subsample, int):
                if self.subsample <= 0:
                    self.logger.error(
                        "Invalid `subsample` value: %d. Must be a positive integer.",
                        self.subsample,
                    )
                    raise ValueError("`subsample` as an integer must be positive.")
                subsample_size = self.subsample
            else:
                self.logger.error(
                    "Invalid `subsample` type: %s. Must be float, int, or None.",
                    type(self.subsample).__name__,
                )
                raise ValueError(
                    "`subsample` must be a float in (0, 1], a positive integer, or None."
                )

            if subsample_size > len(feature_data):
                self.logger.error(
                    "Calculated `subsample_size` (%d) exceeds dataset size (%d).",
                    subsample_size,
                    len(feature_data),
                )
                raise ValueError("Subsample size cannot exceed the size of the dataset.")

            sample_indices = np.random.choice(len(feature_data), subsample_size, replace=False)
            return feature_data[sample_indices], y[sample_indices]

        def _process_categorical_feature(feature_name, feature_data, y_sampled):
            """Process categorical feature: encode categories and calculate bin stats."""
            categories, codes = encode_categories_once(feature_data)
            bin_averages = np.zeros(len(categories))
            bin_counts = np.zeros(len(categories), dtype=int)

            for idx in range(len(categories)):
                mask = codes == idx
                bin_counts[idx] = np.sum(mask)
                bin_averages[idx] = y_sampled[mask].mean() if bin_counts[idx] > 0 else 0.0

            self.bins_[feature_name] = categories
            self.bin_averages_[feature_name] = bin_averages.tolist()
            self.bin_counts_[feature_name] = bin_counts.tolist()

            self.logger.info(
                "Processed categorical feature: [bold slate_blue1]%s"
                "[/bold slate_blue1] with %d categories.",
                feature_name,
                len(categories),
            )

        def _process_numerical_feature(feature_name, feature_data, y_sampled):
            """Process numerical feature: bin data and calculate bin stats."""
            if not np.issubdtype(feature_data.dtype, np.number):
                raise ValueError(
                    f"Feature {feature_name} is not numeric and cannot be "
                    "processed as a numerical feature."
                )
            if self.bin_strategy == "histogram":
                bin_edges, bin_counts, bin_averages = self._histogram_binning(
                    feature_data, y_sampled, self.n_bins or "scott"
                )
            elif self.bin_strategy == "quantile":
                bin_edges, bin_counts, bin_averages = self._quantile_binning(
                    feature_data, y_sampled, self.n_bins
                )
            else:
                self.logger.error("Unsupported bin_strategy: %s", self.bin_strategy)
                raise ValueError(f"Unsupported bin_strategy: {self.bin_strategy}")

            self.bins_[feature_name] = bin_edges
            self.bin_counts_[feature_name] = bin_counts
            self.bin_averages_[feature_name] = bin_averages
            self.logger.info(
                "Processed numerical feature: [bold slate_blue1]%s"
                "[/bold slate_blue1] with %d bins.",
                feature_name,
                len(bin_edges) - 1,
            )

        np.random.seed(self.random_state)
        for i, feature in enumerate(self.feature_names):
            # Get feature data
            feature_data = X[:, i]
            is_categorical = feature in self.categorical_features
            try:  # Automatically detect categorical features if missing
                feature_data = feature_data.astype(float)
                is_categorical = False
            except ValueError:
                # Conversion to float failed, flag as categorical
                feature_data = feature_data.astype(str)
                is_categorical = True
                if feature not in self.categorical_features:
                    self.categorical_features.append(feature)
                    self.logger.info(
                        "Automatically detected and added feature [bold slate_blue1]%s"
                        "[/bold slate_blue1] as categorical based on failed float conversion.",
                        feature,
                    )
            # Handle subsampling
            if self.subsample is not None:
                feature_data, y_sampled = _subsample_data(feature_data, y)
            else:
                y_sampled = y

            # Process features based on type
            if is_categorical:
                logging.info(
                    "Processing categorical feature: [bold slate_blue1]%s[/bold slate_blue1]",
                    feature,
                )
                _process_categorical_feature(feature, feature_data, y_sampled)
            else:
                _process_numerical_feature(feature, feature_data, y_sampled)

            # Enforce monotonicity for numerical features
            if not is_categorical and feature in self.monotonicity:
                increasing = self.monotonicity[feature] == "increasing"
                self._enforce_monotonicity(feature_name=feature, increasing=increasing)

        self.logger.info(
            "[bold pale_green3]Feature binning completed for all features.[/bold pale_green3]"
        )

    def _collect_evidence(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Collect the evidence for each bin in the input features.

        Parameters:
        -----------
        X : The input features in a 2D NumPy array.
        """
        transformed_features = np.zeros(X.shape, dtype=float)

        bins = dict(self.bins_.items())
        bin_avgs = dict(self.bin_averages_.items())
        cat_feats = list(self.categorical_features)
        feat_names = list(self.feature_names)

        args_list = [
            (i, X[:, i], feat_names[i], cat_feats, bins, bin_avgs) for i in range(len(feat_names))
        ]

        # Parallel processing using Ray
        if self.executor_cls and self.n_tasks and self.n_tasks > 1:
            with self.executor_cls(max_workers=self.n_tasks) as executor:
                results = executor.map(parse_feature, args_list)
        else:
            results = map(parse_feature, args_list)

        for i, transformed_feature in results:
            transformed_features[:, i] = transformed_feature

        return transformed_features

    # pylint: disable=invalid-name
    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> WoeLearner:
        """
        Fit the WoeLearner model to the input data.

        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Input features (2D array-like structure).
        y : np.ndarray or pd.Series
            Target variable (1D array-like).
        """
        # Validate and convert X and y
        X, y = check_X_y(X, y, dtype=None, ensure_2d=True, ensure_all_finite=False)

        self.logger.info(
            "[bold pale_green3]Starting WoeLearner with %d features.[/bold pale_green3]",
            X.shape[1],
        )

        # Bin features
        self._bin_features(X, y)
        self.logger.info(
            "[bold pale_green3]Data contains %d samples and %d features.[/bold pale_green3]",
            X.shape[0],
            X.shape[1],
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Calculate WOE in favor of the theory for each input feature.

        Residuals are calculated relative to the average event rate,
        expressed in log odds form. These residuals natively represent
        logarithmic evidence (WOE) cumulatively added to the prior
        log odds during each boosting pass.

        Parameters:
        -----------
        X : The input features.
        """
        X = np.asarray(X)
        return self._collect_evidence(X)

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Predict WOE in favor of the theory for the input features.

        Parameters:
        -----------
        X : The input features.
        """
        return self.transform(X).sum(axis=1)

    def predict_score(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Predict score in favor of the theory for the input features.

        The scores are measured in decibans.

        Parameters:
        -----------
        X : The input features.
        """
        return 10 * self.predict(X) / np.log(10)  # 10log(8/3) = 4.3
