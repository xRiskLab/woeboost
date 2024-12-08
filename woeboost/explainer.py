# -*- coding: utf-8 -*-
"""
explainer.py.

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
from functools import wraps
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmap import Colormap
from rich.logging import RichHandler
from scipy.special import expit as sigmoid
from sklearn.base import TransformerMixin, BaseEstimator


def configure_logger(name: str, verbosity: int = logging.WARNING) -> logging.Logger:
    """
    Configure and return a logger with RichHandler.

    Parameters:
    ----------
    name : str
        The name of the logger, typically the class name.
    verbosity : int
        Logging verbosity level (e.g., logging.INFO, logging.DEBUG).

    Returns:
    -------
    logger : logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():  # Prevent adding duplicate handlers
        handler = RichHandler(markup=True, rich_tracebacks=True, show_path=False)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    logger.setLevel(verbosity)
    for handler in logger.handlers:
        handler.setLevel(verbosity)

    return logger


def experimental_feature(func: callable) -> callable:
    """
    Decorate to mark a function as experimental.

    Print a warning only once per function during the session.
    """
    warned_functions = (
        set()
    )  # Keep track of functions that have already displayed the warning

    @wraps(func)
    def wrapper(*args, **kwargs):
        if func.__name__ not in warned_functions:
            print(
                f"WARNING: '{func.__name__}' is an experimental feature and may be unstable."
            )
            warned_functions.add(func.__name__)
        return func(*args, **kwargs)

    return wrapper


class PDPAnalyzer:
    """
    A class to calculate and visualize Partial Dependence Plots (PDP) for features.

    Attributes:
    ----------
    model : object
        A trained WoeBoostClassifier or compatible model.
    df : DataFrame
        The dataframe containing the features to analyze.
    verbosity : int
        Logging verbosity level.
    """

    def __init__(
        self, model: object, df: pd.DataFrame, verbosity: int = logging.WARNING
    ) -> None:
        """
        Initialize the PDPAnalyzer.

        Parameters:
        ----------
        model : trained WoeBoostClassifier
            The model to evaluate.
        df : DataFrame
            The dataframe containing the features to analyze.
        verbosity : int
            Logging verbosity level.
        """
        self.model = model
        self.df = df
        self.verbosity = verbosity
        self.logger = configure_logger("PDPAnalyzer", verbosity)

    def _configure_logger(self) -> None:
        """Configure logger for PDPAnalyzer."""
        self.logger = logging.getLogger("PDPAnalyzer")

        if not self.logger.hasHandlers():
            handler = RichHandler(markup=True, rich_tracebacks=True, show_path=False)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

        self.logger.setLevel(self.verbosity)
        for handler in self.logger.handlers:
            handler.setLevel(self.verbosity)

    # pylint: disable=too-many-locals
    def calculate_pdp(
        self, feature: str, fixed_features: Optional[dict] = None, num_points: int = 100
    ) -> tuple[np.ndarray, list[float], list[float], list[float]]:
        """
        Calculate Partial Dependence for a single feature.

        For categorical features, uses unique categories instead of interpolated values.
        """
        self.logger.info("Calculating PDP for feature: %s", feature)
        df_temp = self.df.copy()

        # Handle categorical and numerical features differently
        if feature in self.model.estimator.categorical_features:
            feature_range = list(df_temp[feature].value_counts().index)
            is_categorical = True
        else:
            feature_range = np.linspace(df_temp[feature].min(), df_temp[feature].max(), num_points)
            is_categorical = False

        pdp_values = []
        pdp_values_upper = []
        pdp_values_lower = []

        for val in feature_range:
            df_temp[feature] = val
            if fixed_features:
                for fixed_feature, fixed_value in fixed_features.items():
                    df_temp[fixed_feature] = fixed_value
            predictions = self.model.predict_proba(df_temp)[:, 1]
            mean_pred = predictions.mean()
            pdp_values.append(mean_pred)
            std_dev = predictions.std()
            se = std_dev / np.sqrt(len(predictions))  # Standard error

            pdp_values_upper.append(mean_pred + 1.96 * se)
            pdp_values_lower.append(mean_pred - 1.96 * se)

        self.logger.debug(
            "[bold pale_green3]PDP[/bold pale_green3] calculation completed for feature: %s",
            feature,
        )
        return feature_range, pdp_values, pdp_values_upper, pdp_values_lower, is_categorical

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def plot_pdp(
        self,
        feature: str,
        fixed_features: Optional[dict] = None,
        num_points: int = 100,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
    ) -> None:
        """Plot Partial Dependence for a single feature."""
        feature_range, pdp_values, pdp_values_upper, pdp_values_lower, is_categorical = (
            self.calculate_pdp(feature, fixed_features, num_points)
        )

        if ax is None:
            _, ax = plt.subplots()

        if is_categorical:
            # Bar chart for categorical features
            ax.bar(
                feature_range, pdp_values, color="#00d9ff",
                edgecolor="black", alpha=0.7, label="PDP"
            )
            ax.errorbar(
                feature_range, pdp_values,
                yerr=[
                    [p - b for p, b in zip(pdp_values, pdp_values_lower)],
                    [a - p for a, p in zip(pdp_values_upper, pdp_values)],
                ],
                fmt="none", ecolor="black",
                capsize=5, label="95% CI",
            )
            ax.set_xticks(range(len(feature_range)))
            ax.set_xticklabels(feature_range, rotation=45, ha="right")
        else:
            # Line plot for numerical features
            ax.plot(feature_range, pdp_values, label="PDP", color="#00d9ff", linewidth=1.5)
            ax.fill_between(
                feature_range, pdp_values_upper, pdp_values_lower,
                color="#d4f5ff", alpha=0.5, label="95% Confidence Interval"
            )

        ax.set_xlabel(feature)
        ax.set_ylabel("Partial Dependence (Predicted Probability)")
        ax.set_title(title or f"Partial Dependence Plot for {feature}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(which="major", linewidth=0.2)
        ax.grid(which="minor", linewidth=0.2)
        ax.minorticks_on()
        plt.legend()
        if ax is None:
            plt.show()

    def calculate_2way_pdp(
        self, feature1: str, feature2: str, num_points: int = 50
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, tuple[list[str], list[str]]]:  # pylint: disable=too-many-locals
        """Calculate 2-way Partial Dependence for two features."""
        self.logger.info(
            "Calculating [bold pale_green3]2-way PDP[/bold pale_green3] for features: %s, %s",
            feature1,
            feature2,
        )

        # Determine if features are categorical
        is_categorical_1 = feature1 in self.model.estimator.categorical_features
        is_categorical_2 = feature2 in self.model.estimator.categorical_features

        cat_names_1, cat_names_2 = None, None

        if is_categorical_1:
            feature1_range = list(self.df[feature1].value_counts().index)
            cat_names_1 = feature1_range
        else:
            feature1_range = np.linspace(
                self.df[feature1].min(), self.df[feature1].max(), num_points
            )

        if is_categorical_2:
            feature2_range = list(self.df[feature2].value_counts().index)
            cat_names_2 = feature2_range
        else:
            feature2_range = np.linspace(
                self.df[feature2].min(), self.df[feature2].max(), num_points
            )

        grid_x, grid_y = np.meshgrid(feature1_range, feature2_range)
        flat_grid_x, flat_grid_y = grid_x.ravel(), grid_y.ravel()

        # Ensure df_temp has the same number of rows as the grid points
        n_grid_points = len(flat_grid_x)
        df_temp = pd.DataFrame(
            np.tile(self.df.to_numpy(), (n_grid_points // len(self.df) + 1, 1))[:n_grid_points],
            columns=self.df.columns,
        )

        # Assign the grid values to the respective features
        df_temp[feature1] = flat_grid_x
        df_temp[feature2] = flat_grid_y

        predictions = self.model.predict_proba(df_temp)[:, 1]
        pdp_values = predictions.reshape(grid_x.shape)

        self.logger.info(
            "[bold pale_green3]2-way PDP[/bold pale_green3] calculation completed for: %s, %s",
            feature1,
            feature2,
        )
        return (grid_x, grid_y), pdp_values, (cat_names_1, cat_names_2)

    # pylint: disable=too-many-positional-arguments
    def plot_2way_pdp(
        self, feature1, feature2, num_points=50,
        plot_type="contourf", cmap="colorcet:cet_d10",
        title=None,
        ax=None,  # pylint: disable=too-many-arguments
    ):  # pylint: disable=invalid-name, too-many-arguments
        """
        Plot 2-way Partial Dependence for two features.

        Parameters:
        ----------
        feature1 : str
            The first feature for 2D PDP.
        feature2 : str
            The second feature for 2D PDP.
        num_points : int, default=50
            Number of points to evaluate in the feature range for each feature.
        plot_type : str, default="contourf"
            Type of plot: "contourf", "hexbin", or "hist2d".
        cmap : str, default="colorcet:cet_d10"
            The colormap to use for the plot.
        title : str, optional
            The title of the plot.
        ax : matplotlib.axes._axes.Axes, optional
            Axis object to plot on. If None, a new figure and axis are created.
        """
        cm = Colormap(cmap).to_mpl()
        (grid_x, grid_y), pdp_values, (cat_names_1, cat_names_2) = self.calculate_2way_pdp(
            feature1, feature2, num_points
        )

        # Convert categorical features to numeric indices
        if cat_names_1 is not None:
            mapping_1 = {name: i for i, name in enumerate(cat_names_1)}
            grid_x = np.vectorize(mapping_1.get)(grid_x)

        if cat_names_2 is not None:
            mapping_2 = {name: i for i, name in enumerate(cat_names_2)}
            grid_y = np.vectorize(mapping_2.get)(grid_y)

        if ax is None:
            _, ax = plt.subplots()

        if plot_type == "contourf":
            contour = ax.contourf(grid_x, grid_y, pdp_values, levels=50, cmap=cm)
            plt.colorbar(contour, ax=ax, label="Partial Dependence")
        elif plot_type == "hist2d":
            hist = ax.hist2d(
                grid_x.flatten(), grid_y.flatten(), bins=num_points,
                weights=pdp_values.flatten(), cmap=cm
            )
            plt.colorbar(hist[3], ax=ax, label="Partial Dependence")
        elif plot_type == "hexbin":
            hexbin = ax.hexbin(
                grid_x.flatten(), grid_y.flatten(), C=pdp_values.flatten(),
                gridsize=num_points, cmap=cm
            )
            plt.colorbar(hexbin, ax=ax, label="Partial Dependence")
        else:
            raise ValueError("Invalid plot_type. Choose from 'contourf', 'hist2d', or 'hexbin'.")

        # Update axis labels for categorical features
        if cat_names_1 is not None:
            ax.set_xticks(range(len(cat_names_1)))
            ax.set_xticklabels(cat_names_1, rotation=45, ha="right")

        if cat_names_2 is not None:
            ax.set_yticks(range(len(cat_names_2)))
            ax.set_yticklabels(cat_names_2)

        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_title(title or f"2-Way PDP:\n{feature1}\nvs {feature2}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # If a new axis was created, show the plot
        if ax is None:
            plt.show()


class EvidenceAnalyzer:
    """
    A class to calculate and visualize evidence contributions for WoeBoost models.

    Attributes:
    ----------
    model : object
        A trained WoeBoostClassifier.
    df : DataFrame
        The dataframe containing the features to analyze.
    """

    def __init__(
        self, model: object, df: pd.DataFrame, verbosity: int = logging.WARNING
    ) -> None:
        """
        Initialize the EvidenceAnalyzer.

        Parameters:
        ----------
        model : trained WoeBoostClassifier
            The model to evaluate.
        df : DataFrame
            The dataframe containing the features to analyze.
        verbosity : int
            Logging verbosity level.
        """
        self.model = model
        self.df = df
        self.logger = configure_logger("EvidenceAnalyzer", verbosity)

    def calculate_contributions(self, mode: str = "cumulative") -> list[pd.Series]:
        """
        Calculate evidence contributions over iterations.

        Parameters:
        -----------
        mode : str, optional
            Mode of contributions, either 'cumulative' or 'iteration'. Default is 'cumulative'.

        Returns:
        --------
        contributions_per_pass : list of pd.Series
            List of contributions for each pass (one series per iteration).
        """
        self.logger.info(
            "Calculating [bold pale_green3]contributions[/bold pale_green3] in %s mode.",
            mode,
        )
        cumulative_logits = pd.DataFrame(
            0, index=self.df.index, columns=self.df.columns
        )
        contributions_per_pass = []

        for _, estimator in enumerate(self.model.estimators):
            self.logger.debug("Calculating contributions for pass %s", _ + 1)
            # Transform logits for the current iteration
            iteration_logits = estimator.transform(self.df)
            iteration_df = pd.DataFrame(
                iteration_logits, columns=self.df.columns, index=self.df.index
            )
            if mode == "cumulative":
                # Add current iteration logits to cumulative logits
                cumulative_logits += iteration_df
                contributions = (
                    np.abs(cumulative_logits).sum() / np.abs(cumulative_logits).sum().sum()
                )
            elif mode == "iteration":
                # Calculate iteration-based contributions
                contributions = (
                    np.abs(iteration_df).sum() / np.abs(iteration_df).sum().sum()
                )
            else:
                self.logger.error("Invalid mode provided: %s", mode)
                raise ValueError("Mode must be either 'cumulative' or 'iteration'")

            # Append the contributions for this iteration to the list
            contributions_per_pass.append(contributions)

        self.logger.info(
            "[bold pale_green3]Contributions[/bold pale_green3] calculation "
            "completed in %s mode.",
            mode,
        )
        return contributions_per_pass

    def plot_contributions(
        self,
        contributions_per_pass: list[pd.Series],
        mode: str = "cumulative",
        iteration: Optional[int] = None,
    ) -> None:
        """
        Plot evidence contributions for a specific iteration or cumulatively.

        Parameters:
        -----------
        contributions_per_pass : list of pd.Series
            List of contributions for each pass (one series per iteration).
        mode : str, optional
            Mode of contributions, either 'cumulative' or 'iteration'. Default is 'cumulative'.
        iteration : int, optional
            Specific iteration to plot. If None, plots the last iteration for cumulative mode.
        """
        self.logger.info(
            "Plotting [bold pale_green3]contributions[/bold pale_green3] in %s mode.",
            mode,
        )
        if mode == "cumulative":
            # Plot the last iteration if no specific iteration is provided
            iteration = (
                iteration if iteration is not None else len(contributions_per_pass) - 1
            )

        elif mode == "iteration" and iteration is None:
            raise ValueError(
                "For iteration-based mode, a specific iteration must be provided."
            )

        contributions = contributions_per_pass[iteration]
        contributions_df = pd.DataFrame(contributions, columns=["contribution"])
        aggregated = contributions_df.sort_values(by="contribution", ascending=True)

        # Plot contributions
        _, ax = plt.subplots(figsize=(8, 6), dpi=200)
        aggregated.plot(
            kind="barh", ax=ax, width=0.8, color="#02ff00",
            edgecolor="black", linewidth=1.0, legend=False
        )
        ax.set_title(
            (
                f"{'Cumulative' if mode == 'cumulative' else 'Iteration-based'} "
                f"Evidence Contributions (Pass {iteration + 1})"
            ),
            pad=15,
        )
        ax.set_xlabel("Normalized Evidence")

        # Disable spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add tight layout
        plt.tight_layout()
        if ax is None:
            plt.show()

        self.logger.info(
            "[bold pale_green3]Contributions[/bold pale_green3] plot completed."
        )

    # pylint: disable=too-many-positional-arguments, too-many-branches
    @experimental_feature
    def plot_decision_boundary(
        self,
        feature1: str,
        feature2: str,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        iteration_range: Optional[tuple[int, int]] = None,
        grid_size: tuple[int, int] = (5, 5),
        cmap: str = "colorcet:cet_d10",
        output_file: Optional[str] = None,
    ) -> (
        None
    ):  # pylint: disable=too-many-arguments, invalid-name, too-many-locals, too-many-statements
        """
        Plot the decision boundary for WoeBoost iterations (experimental feature).

        Parameters:
        ----------
        feature1 : str
            Name of the first feature for the x-axis.
        feature2 : str
            Name of the second feature for the y-axis.
        iteration_range : tuple, optional
            Range of iterations to visualize as (start, end). Default is (0, 25).
        grid_size : tuple, optional
            Grid size for subplots (rows, cols). Default is (5, 5).
        cmap : str, default="colorcet:cet_d10"
            The colormap to use for the plot. See the full list of colormaps at:
            https://cmap-docs.readthedocs.io/en/latest/catalog/.
        step_size : float, optional
            Step size for the grid. Default is 0.05.
        output_file : str, optional
            If specified, saves the plot to the given file. Default is None.
        """
        self.logger.info(
            "Generating [bold pale_green3]decision boundary[/bold pale_green3] plot "
            "for features: %s, %s",
            feature1, feature2
        )
        # Convert NumPy arrays to DataFrames if necessary
        if isinstance(X, np.ndarray):
            if not hasattr(self, "feature_names"):
                self.logger.error(
                    "If X is a numpy array, the model must have 'feature_names' defined."
                )
                raise ValueError(
                    "If X is a numpy array, the model must have 'feature_names' defined."
                )
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        # Determine the iteration range
        start, end = iteration_range or (0, 25)
        rows, cols = grid_size

        is_categorical_1 = feature1 in self.model.metadata['categorical_features']
        is_categorical_2 = feature2 in self.model.metadata['categorical_features']

        if is_categorical_1:
            # Handle categorical feature 1
            cat_names_1 = list(X[feature1].value_counts().index)
            feature_1_range = np.arange(len(cat_names_1))  # Use indices for grid creation
        else:
            feature_1_range = np.linspace(X[feature1].min(), X[feature1].max(), num=200)

        if is_categorical_2:
            # Handle categorical feature 2
            cat_names_2 = list(X[feature2].value_counts().index)
            feature_2_range = np.arange(len(cat_names_2))  # Use indices for grid creation
        else:
            feature_2_range = np.linspace(X[feature2].min(), X[feature2].max(), num=200)

        # Create the grid for plotting
        feature_1, feature_2 = np.meshgrid(
            feature_1_range,
            feature_2_range,
        )

        # Combine the features into a grid (for predictions or plotting)
        grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T

        # If categorical, map grid indices back to category names for predictions
        if is_categorical_1:
            # Map grid indices to category names for the first feature
            grid[:, 0] = [
                cat_names_1[int(idx)] if isinstance(idx, (int, np.integer)) else idx
                for idx in grid[:, 0]
            ]

        if is_categorical_2:
            # Map grid indices to category names for the second feature
            grid[:, 1] = [
                cat_names_2[int(idx)] if isinstance(idx, (int, np.integer)) else idx
                for idx in grid[:, 1]
            ]

        # Initialize the base predictions (log odds)
        cumulative_preds = np.full(grid.shape[0], self.model.base_score)

        # Add contributions from estimators prior to `start`
        if start > 0:
            grid_df = pd.DataFrame(
                np.zeros((grid.shape[0], len(self.df.columns))), columns=self.df.columns
            )
            grid_df[feature1] = grid[:, 0]
            grid_df[feature2] = grid[:, 1]
            for estimator in self.model.estimators[:start]:
                grid_preds_woe = estimator.transform(grid_df)
                # Replace all values not belonging to feature1 and feature2 with 0
                mask = np.isin(grid_df.columns, [feature1, feature2])
                grid_preds_woe[:, ~mask] = 0
                cumulative_preds += grid_preds_woe.sum(axis=1)

        # Set up the plot
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), dpi=200)
        axes = axes.ravel()  # Flatten axes for easy iteration

        cm = Colormap(cmap).to_mpl()  # Use the Colormap module

        i = -1  # Initialize to a default value before the loop
        for i, estimator in enumerate(self.model.estimators[start:end]):
            if i >= len(axes):
                break

            # Prepare the grid for predictions
            grid_df = pd.DataFrame(
                np.zeros((grid.shape[0], len(self.df.columns))), columns=self.df.columns
            )
            grid_df[feature1] = grid[:, 0]
            grid_df[feature2] = grid[:, 1]

            # Add contributions from the current estimator
            grid_preds_woe = estimator.transform(grid_df)
            # Replace all values not belonging to feature1 and feature2 with 0
            mask = np.isin(grid_df.columns, [feature1, feature2])
            grid_preds_woe[:, ~mask] = 0
            cumulative_preds += grid_preds_woe.sum(axis=1)

            # Reshape predictions for plotting
            grid_probas = sigmoid(cumulative_preds).reshape(feature_1.shape)

            # Plot the decision boundary
            ax = axes[i]
            ax.contourf(
                feature_1, feature_2, grid_probas, alpha=1.0, cmap=cm, levels=50
            )  # Plot the decision boundary

            # Scatter the data points on top
            for unique_label in np.unique(y):
                # Mask for points belonging to the current class
                class_mask = y == unique_label
                class_desc = "-" if unique_label == 0 else "+"
                color = cm(0) if unique_label == 0 else cm(255)

                # Extract raw feature values for the current class
                raw_x_vals = X[feature1][class_mask].values
                raw_y_vals = X[feature2][class_mask].values

                # Scatter plot for aligned points
                ax.scatter(
                    raw_x_vals,
                    raw_y_vals,
                    c=[color],
                    marker="o",
                    edgecolors="k",
                    linewidth=0.3,
                    alpha=0.8,
                    s=40,  # Size of points
                    label=class_desc,
                )
            if is_categorical_1:
                ax.set_xticks(range(len(cat_names_1)))
                ax.set_xticklabels(cat_names_1, rotation=45, ha="right")
            if is_categorical_2:
                ax.set_yticks(range(len(cat_names_2)))
                ax.set_yticklabels(cat_names_2)
            ax.set_title(f"Iteration {start + i + 1}")
            ax.set_xlim(feature_1.min(), feature_1.max())
            ax.set_ylim(feature_2.min(), feature_2.max())
            ax.set_xlabel(feature1)
            ax.set_ylabel(feature2)

        # Remove unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Add a super title
        _shift = 1 + 1e-5
        plt.suptitle("WoeBoost Decision Boundary", fontsize=16, y=_shift)
        plt.tight_layout()

        # Save the plot if an output file is specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {output_file}")
        else:
            if ax is None:
                plt.show()

        self.logger.info(
            "Finished generating [bold pale_green3]decision boundary[/bold pale_green3] plot "
            "for features: %s, %s",
            feature1, feature2
        )


class WoeInferenceMaker(BaseEstimator, TransformerMixin):
    """
    A class to infer WOE scores and predict probabilities.

    Parameters:
    ----------
    model : WoeLearner
        A trained WoeLearner model.
    """

    def __init__(self, model: object, verbosity: int = logging.WARNING) -> None:
        """Initialize the WoeInferenceMaker."""
        self.model = model
        self.verbosity = verbosity
        self.bin_reports = {}  # Store bin reports for each feature
        self.logger = configure_logger("WoeInferenceMaker", verbosity)

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.

        Parameters:
        ----------
        deep : bool, default=True
            If True, will return parameters of sub-objects that are estimators.

        Returns:
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {"model": self.model, "verbosity": self.verbosity}

    def set_params(self, **params) -> "WoeInferenceMaker":
        """
        Set parameters for this estimator.

        Parameters:
        ----------
        **params : dict
            Parameters to set.

        Returns:
        -------
        self : WoeInferenceMaker
            The instance with updated parameters.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> None:  # pylint: disable=invalid-name
        """
        Fit the inference maker by generating bin reports for all features.

        Parameters:
        ----------
        X : pd.DataFrame
            Input data used to generate bin reports.
        y : pd.Series or np.ndarray
            Target variable for generating bin reports.
        """
        y = np.asarray(y)
        self.logger.info("Fitting WoeInferenceMaker with provided data.")

        for feature_name in self.model.feature_names:
            self.logger.info("Generating bin report for feature: %s", feature_name)
            bin_report = self.generate_bin_report(feature_name, X, y)
            self.bin_reports[feature_name] = bin_report

        self.logger.info("Fit process completed. Bin reports are stored.")
        return self

    # pylint: disable=invalid-name, too-many-locals
    def generate_bin_report(
        self,
        feature_name: str,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> pd.DataFrame:
        """
        Generate a bin report for a specific feature.

        Parameters:
        ----------
        feature_name : str
            The name of the feature to analyze.
        X : pd.DataFrame or np.ndarray
            Input data to analyze.
        y : pd.Series or np.ndarray
            Target variable for calculating WOE values.

        Returns:
        -------
        pd.DataFrame
            Bin statistics including WOE values.
        """
        X = pd.DataFrame(X, columns=self.model.feature_names) if isinstance(X, np.ndarray) else X
        y = np.asarray(y)

        feature_names = list(self.model.feature_names)
        if feature_name not in feature_names:
            raise ValueError(f"Feature {feature_name} is not in the trained model.")

        feature_data = X[feature_name]
        bin_edges = self.model.bins_[feature_name]
        bin_averages = self.model.bin_averages_[feature_name]

        (
            bins,
            edge_min,
            edge_max,
            avg_res,
            avg_y,
            num_events,
            num_non_events,
            woe_values,
        ) = ([], [], [], [], [], [], [], [])

        total_events = (y == 1).sum()
        total_non_events = (y == 0).sum()

        # Check if feature is categorical
        if feature_name in self.model.categorical_features:
            categories = bin_edges
            for i, category in enumerate(categories):
                mask = feature_data == category
                bins.append(i)
                edge_min.append(category)
                edge_max.append(category)
                avg_res.append(bin_averages[i])

                y_in_bin = y[mask]
                avg_y.append(y_in_bin.mean() if len(y_in_bin) > 0 else 0)
                events_in_bin = (y_in_bin == 1).sum()
                non_events_in_bin = (y_in_bin == 0).sum()
                num_events.append(events_in_bin)
                num_non_events.append(non_events_in_bin)

                woe_cond_prob = np.log(
                    (events_in_bin / total_events)
                    / (non_events_in_bin / total_non_events)
                    if events_in_bin > 0 and non_events_in_bin > 0
                    else ((events_in_bin + 0.5) / total_events)
                    / ((non_events_in_bin + 0.5) / total_non_events)
                )
                woe_values.append(woe_cond_prob)

        else:
            for i in range(len(bin_edges) - 1):
                start, end = bin_edges[i], bin_edges[i + 1]
                if i == len(bin_edges) - 2:
                    mask = (feature_data >= start) & (feature_data <= end)
                else:
                    mask = (feature_data >= start) & (feature_data < end)
                bins.append(i)
                edge_min.append(start)
                edge_max.append(end)
                avg_res.append(bin_averages[i])

                y_in_bin = y[mask]
                avg_y.append(y_in_bin.mean() if len(y_in_bin) > 0 else 0)
                events_in_bin = (y_in_bin == 1).sum()
                non_events_in_bin = (y_in_bin == 0).sum()
                num_events.append(events_in_bin)
                num_non_events.append(non_events_in_bin)

                woe_cond_prob = np.log(
                    (events_in_bin / total_events)
                    / (non_events_in_bin / total_non_events)
                    if events_in_bin > 0 and non_events_in_bin > 0
                    else ((events_in_bin + 0.5) / total_events)
                    / ((non_events_in_bin + 0.5) / total_non_events)
                )
                woe_values.append(woe_cond_prob)

        return pd.DataFrame(
            {
                "bin": bins,
                "edge_min": edge_min,
                "edge_max": edge_max,
                "average_res": avg_res,
                "average_y": avg_y,
                "number_events": num_events,
                "number_non_events": num_non_events,
                "woe": woe_values,
            }
        )

    def infer_woe_score(self, feature_name: str, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Infer WOE scores for a specific feature using the pre-generated bin report.

        Parameters:
        ----------
        feature_name : str
            The name of the feature for which to infer WOE scores.
        X : pd.DataFrame or np.ndarray
            Input data to infer WOE scores.

        Returns:
        -------
        np.ndarray
            An array of WOE scores corresponding to the input data.
        """
        X = pd.DataFrame(X, columns=self.model.feature_names) if isinstance(X, np.ndarray) else X

        if feature_name not in self.bin_reports:
            self.logger.error("Bin report for feature %s is not available.", feature_name)
            raise ValueError(f"Bin report for feature {feature_name} is not available.")

        bin_report = self.bin_reports[feature_name]
        feature_data = X[feature_name]

        woe_scores = np.zeros(len(feature_data))

        if feature_data.dtype.name in {"object", "category"} or isinstance(
            feature_data.iloc[0], str
        ):
            category_to_woe = {row["edge_min"]: row["woe"] for _, row in bin_report.iterrows()}
            woe_scores = feature_data.map(category_to_woe).fillna(0).to_numpy()
        else:
            for _, row in bin_report.iterrows():
                if row.name == len(bin_report) - 1:
                    mask = (feature_data >= row["edge_min"]) & (feature_data <= row["edge_max"])
                else:
                    mask = (feature_data >= row["edge_min"]) & (feature_data < row["edge_max"])
                woe_scores[mask] = row["woe"]

            nan_mask = feature_data.isna()
            woe_scores[nan_mask] = 0.0

        return woe_scores

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:  # pylint: disable=invalid-name
        """
        Transform the input data into WOE scores for all features.

        Parameters:
        ----------
        X : pd.DataFrame or np.ndarray
            Input data to transform.

        Returns:
        -------
        pd.DataFrame
            A DataFrame of WOE scores for all features.
        """
        X = pd.DataFrame(X, columns=self.model.feature_names) if isinstance(X, np.ndarray) else X
        woe_scores = {
            feature_name: self.infer_woe_score(feature_name, X)
            for feature_name in self.model.feature_names
        }
        return pd.DataFrame(woe_scores, index=X.index)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Predict probabilities using the logistic transformation.

        Parameters:
        ----------
        X : pd.DataFrame or np.ndarray
            Input data for prediction.

        Returns:
        -------
        np.ndarray
            Predicted probabilities.
        """
        woe_scores = self.transform(X)
        log_odds = self.model.base_score + woe_scores.sum(axis=1)
        probabilities = sigmoid(log_odds.values if isinstance(log_odds, pd.Series) else log_odds)
        return np.column_stack([1 - probabilities, probabilities])
