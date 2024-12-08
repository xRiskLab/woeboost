# -*- coding: utf-8 -*-
"""
learner.py.

░██╗░░░░░░░██╗░█████╗░███████╗██████╗░░█████╗░░█████╗░░██████╗████████╗
░██║░░██╗░░██║██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝╚══██╔══╝
░╚██╗████╗██╔╝██║░░██║█████╗░░██████╦╝██║░░██║██║░░██║╚█████╗░░░░██║░░░
░░████╔═████║░██║░░██║██╔══╝░░██╔══██╗██║░░██║██║░░██║░╚═══██╗░░░██║░░░
░░╚██╔╝░╚██╔╝░╚█████╔╝███████╗██████╦╝╚█████╔╝╚█████╔╝██████╔╝░░░██║░░░
░░░╚═╝░░░╚═╝░░░╚════╝░╚══════╝╚═════╝░░╚════╝░░╚════╝░╚═════╝░░░░╚═╝░░░

Copyright (c) 2024 xRiskLab / deburky

Licensed under the MIT License.

You may obtain a copy of the License at:
    https://opensource.org/licenses/MIT
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from rich.logging import RichHandler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import check_X_y


class WoeLearner(
    BaseEstimator, ClassifierMixin
):  # pylint: disable=too-many-instance-attributes
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

    n_threads : int, default=None
        The number of threads to use for parallel operations. If None, operations
        will be executed sequentially.

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
        bin_strategy: str = 'histogram',
        random_state: Optional[int] = None,
        categorical_features: Optional[List[str]] = None,
        monotonicity: Optional[dict] = None,
        infer_monotonicity: bool = False,
        n_threads: Optional[int] = None,
        verbosity: Optional[int] = logging.WARNING,
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
        n_threads : int, default=None
            The number of threads to use for parallel operations. If None, operations
            will be executed sequentially.
        verbosity : int, default=logging.WARNING
            The logging level for controlling verbosity. Options are standard logging
            levels like logging.DEBUG, logging.INFO, and logging.WARNING.
        """
        self.feature_names = feature_names
        self.n_bins = n_bins
        self.subsample = subsample
        self.base_score = base_score
        self.bin_strategy = bin_strategy
        self.random_state = random_state
        self.categorical_features = (
            categorical_features if categorical_features is not None else []
        )
        self.monotonicity = monotonicity if monotonicity is not None else {}
        self.infer_monotonicity = infer_monotonicity
        self.n_threads = n_threads
        self.verbosity = verbosity

        self.warning_issued = False  # Flag to prevent multiple warnings
        self._configure_logger()  # Configure logger

        self.bins_ = {}
        self.bin_averages_ = {}
        self.bin_counts_ = {}

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
        self.logger.setLevel(self.verbosity)
        for handler in self.logger.handlers:
            handler.setLevel(self.verbosity)

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
            bin_counts = self.bin_counts_.get(
                feature_name, [1] * len(bin_averages))

            # Use isotonic regression to enforce monotonicity
            iso_reg = IsotonicRegression(
                increasing=increasing, out_of_bounds="clip")
            corrected_bin_averages = iso_reg.fit_transform(
                range(len(bin_averages)), bin_averages, sample_weight=bin_counts
            )
            # Update bin averages with monotonic values
            self.bin_averages_[feature_name] = corrected_bin_averages

    def _quantile_binning(self, X: np.ndarray, y: np.ndarray, n_bins):  # pylint: disable=invalid-name
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
            bin_edges = np.nanquantile(
                X, np.linspace(0, 1, n_bins + 1))  # Handle NaNs
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

    def _histogram_binning(self, X: np.ndarray, y: np.ndarray, n_bins_or_method='scott'):  # pylint: disable=invalid-name
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
            counts, bin_edges = np.histogram(
                X[~np.isnan(X)], bins=n_bins_or_method)  # Handle NaNs
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
                        self.subsample
                    )
                    raise ValueError("`subsample` as a float must be in the range (0, 1].")
                subsample_size = int(len(feature_data) * self.subsample)
            elif isinstance(self.subsample, int):
                if self.subsample <= 0:
                    self.logger.error(
                        "Invalid `subsample` value: %d. Must be a positive integer.",
                        self.subsample
                    )
                    raise ValueError("`subsample` as an integer must be positive.")
                subsample_size = self.subsample
            else:
                self.logger.error(
                    "Invalid `subsample` type: %s. Must be float, int, or None.",
                    type(self.subsample).__name__
                )
                raise ValueError(
                    "`subsample` must be a float in (0, 1], a positive integer, or None."
                )

            if subsample_size > len(feature_data):
                self.logger.error(
                    "Calculated `subsample_size` (%d) exceeds dataset size (%d).",
                    subsample_size,
                    len(feature_data)
                )
                raise ValueError("Subsample size cannot exceed the size of the dataset.")

            sample_indices = np.random.choice(len(feature_data), subsample_size, replace=False)
            return feature_data[sample_indices], y[sample_indices]

        def _process_categorical_feature(feature_name, feature_data, y_sampled):
            """Process categorical feature: encode categories and calculate bin stats."""
            categories, avg_per_category = np.unique(feature_data, return_inverse=True)
            self.bins_[feature_name] = categories
            self.bin_averages_[feature_name] = [
                y_sampled[avg_per_category == j].mean() for j in range(len(categories))
            ]
            self.bin_counts_[feature_name] = [
                np.sum(avg_per_category == j) for j in range(len(categories))
            ]
            self.logger.info(
                "Processed categorical feature: [bold slate_blue1]%s"
                "[/bold slate_blue1] with %d unique categories.",
                feature_name, len(categories),
            )

        def _process_numerical_feature(feature_name, feature_data, y_sampled):
            """Process numerical feature: bin data and calculate bin stats."""
            if not np.issubdtype(feature_data.dtype, np.number):
                raise ValueError(
                    f"Feature {feature_name} is not numeric and cannot be "
                    "processed as a numerical feature."
                )
            if self.bin_strategy in {"histogram"}:
                bin_edges, bin_counts, bin_averages = (
                    self._histogram_binning(feature_data, y_sampled, self.n_bins)
                )
            elif self.bin_strategy in {"quantile"}:
                bin_edges, bin_counts, bin_averages = (
                    self._quantile_binning(feature_data, y_sampled, self.n_bins)
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
                feature_name, len(bin_edges) - 1
            )

        np.random.seed(self.random_state)
        for i, feature in enumerate(self.feature_names):
            # Get feature data
            feature_data = X[:, i]
            is_categorical = feature in self.categorical_features
            try:  # Automatically detect categorical features if missing
                # Try converting to float to check if it's numerical
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

    def _collect_evidence(self, X: np.ndarray) -> np.ndarray:    # pylint: disable=invalid-name
        """
        Collect the evidence for each bin in the input features.

        Parameters:
        -----------
        X : The input features in a 2D NumPy array.
        """
        # Initialize an empty array with the shape of X
        transformed_features = np.zeros(X.shape, dtype=float)

        def parse_feature(i):
            feature_data = X[:, i]
            if self.feature_names[i] in self.categorical_features:
                categories = self.bins_[self.feature_names[i]]
                avg_per_category = self.bin_averages_[self.feature_names[i]]
                category_map = dict(zip(categories, avg_per_category))
                transformed_feature = np.array(
                    [category_map.get(str(x), 0.0) for x in feature_data]
                )
            else:
                bin_edges = self.bins_[self.feature_names[i]]
                bin_averages = np.array(self.bin_averages_[self.feature_names[i]])
                binned_indices = np.digitize(feature_data, bin_edges) - 1
                binned_indices = np.clip(binned_indices, 0, len(bin_averages) - 1)
                transformed_feature = bin_averages[binned_indices]
                if np.issubdtype(feature_data.dtype, np.number):  # Check if numerical
                    nan_mask = np.isnan(feature_data)
                    transformed_feature[nan_mask] = 0.0
            return transformed_feature

        # Check if threading was requested
        if self.n_threads and self.n_threads > 1:
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                results = list(
                    executor.map(parse_feature, range(len(self.feature_names)))
                )
            # Combine the results into an array
            for i, transformed_feature in enumerate(results):
                transformed_features[:, i] = transformed_feature
        else:
            # Process sequentially if n_threads is None
            for i in range(len(self.feature_names)):
                transformed_features[:, i] = parse_feature(i)

        return transformed_features

    # pylint: disable=invalid-name
    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> "WoeLearner":
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
        X, y = check_X_y(X, y, dtype=None, ensure_2d=True, force_all_finite=False)

        self.logger.info(
            "[bold pale_green3]Starting WoeLearner with %d features.[/bold pale_green3]",
            X.shape[1]
        )

        # Bin features
        self._bin_features(X, y)
        self.logger.info(
            "[bold pale_green3]Data contains %d samples and %d features.[/bold pale_green3]",
            X.shape[0], X.shape[1]
        )

        return self

    def transform(self, X: np.array) -> np.array:  # pylint: disable=invalid-name
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

    def predict(self, X: np.array) -> np.array:  # pylint: disable=invalid-name
        """
        Predict WOE in favor of the theory for the input features.

        Parameters:
        -----------
        X : The input features.
        """
        return self.transform(X).sum(axis=1)

    def predict_score(self, X: np.array) -> np.array:  # pylint: disable=invalid-name
        """
        Predict score in favor of the theory for the input features.

        The scores are measured in decibans.

        Parameters:
        -----------
        X : The input features.
        """
        return 10 * self.predict(X) / np.log(10)  # 10log(8/3) = 4.3
