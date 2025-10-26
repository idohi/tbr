"""
Object-oriented interface for Time-Based Regression analysis.

This module provides the TBRAnalysis class, which wraps the functional API
with a stateful interface for storing configuration, fitted parameters, and
analysis results.

Examples
--------
>>> from tbr.core.model import TBRAnalysis
>>> import pandas as pd
>>> import numpy as np
>>>
>>> # Create sample data
>>> data = pd.DataFrame({
...     'date': pd.date_range('2023-01-01', periods=90),
...     'control': np.random.normal(1000, 50, 90),
...     'test': np.random.normal(1020, 55, 90)
... })
>>>
>>> # Initialize and fit model
>>> model = TBRAnalysis(level=0.80, threshold=0.0)
>>> model.fit(
...     data=data,
...     time_col='date',
...     control_col='control',
...     test_col='test',
...     pretest_start='2023-01-01',
...     test_start='2023-02-15',
...     test_end='2023-03-01'
... )
>>>
>>> # Access results
>>> final_effect = model.summaries_.iloc[-1]['estimate']
>>> print(f"Treatment Effect: {final_effect:.2f}")

Notes
-----
Configuration parameters are stored in __init__. Analysis is performed via
the fit() method. Fitted results are accessed via underscore-suffixed
attributes (results_, summaries_, params_), which validate fitted state
before access.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


class TBRAnalysis:
    """
    Time-Based Regression Analysis with stateful interface.

    Wraps the functional TBR API to store configuration, fitted parameters,
    and analysis results.

    Parameters
    ----------
    level : float, default=0.80
        Credibility level for credible intervals (e.g., 0.80 for 80% credible interval).
        Must be between 0 and 1 exclusive.
    threshold : float, default=0.0
        Threshold for probability calculation. Typically 0.0 for testing
        positive effects. Can be any finite float value.
    test_end_inclusive : bool, default=False
        Whether to include the test_end boundary in the test period.

        - False (default): Exclusive end boundary (data < test_end)
        - True: Inclusive end boundary (data <= test_end)

    Attributes
    ----------
    level : float
        Credibility level for credible intervals.
    threshold : float
        Threshold for probability calculation.
    test_end_inclusive : bool
        Whether test_end is inclusive.
    fitted_ : bool
        Whether the model has been fitted.
    results_ : pd.DataFrame
        TBR DataFrame with predictions, effects, and uncertainties.
        Available after calling fit().
    summaries_ : pd.DataFrame
        Incremental summaries with daily progression of effects.
        Available after calling fit().
    params_ : dict
        Regression model parameters (alpha, beta, sigma, variances, etc.).
        Available after calling fit().

    Examples
    --------
    Basic workflow:

    >>> model = TBRAnalysis(level=0.80, threshold=0.0)
    >>> model.fit(data, 'date', 'control', 'test',
    ...           pretest_start='2023-01-01',
    ...           test_start='2023-02-15',
    ...           test_end='2023-03-01')
    >>> print(model.summaries_.iloc[-1])

    Custom configuration:

    >>> model = TBRAnalysis(level=0.95, threshold=5.0, test_end_inclusive=True)
    >>> model.fit(data, 'date', 'control', 'test',
    ...           pretest_start='2023-01-01',
    ...           test_start='2023-02-15',
    ...           test_end='2023-02-15')  # Same-day analysis

    Notes
    -----
    Configuration parameters are stored in __init__. Call fit() to perform
    analysis. Access fitted results via underscore-suffixed attributes
    (results_, summaries_, params_), which validate fitted state before access.

    See Also
    --------
    tbr.functional.perform_tbr_analysis : Functional API for TBR analysis
    """

    def __init__(
        self,
        level: float = 0.80,
        threshold: float = 0.0,
        test_end_inclusive: bool = False,
    ) -> None:
        """
        Initialize TBR analysis with configuration parameters.

        Parameters
        ----------
        level : float, default=0.80
            Credibility level for credible intervals.
        threshold : float, default=0.0
            Threshold for probability calculation.
        test_end_inclusive : bool, default=False
            Whether to include test_end boundary in analysis.

        Raises
        ------
        ValueError
            If level is not between 0 and 1 exclusive.
        TypeError
            If parameters have incorrect types.
        """
        # Validate configuration parameters
        if not isinstance(level, (int, float)):
            raise TypeError(f"level must be numeric, got {type(level).__name__}")

        if not (0 < level < 1):
            raise ValueError(f"level must be between 0 and 1 exclusive, got {level}")

        if not isinstance(threshold, (int, float)):
            raise TypeError(
                f"threshold must be numeric, got {type(threshold).__name__}"
            )

        if not isinstance(test_end_inclusive, bool):
            raise TypeError(
                f"test_end_inclusive must be bool, got {type(test_end_inclusive).__name__}"
            )

        # Store configuration
        self.level = float(level)
        self.threshold = float(threshold)
        self.test_end_inclusive = test_end_inclusive

        # Initialize state (will be set by fit())
        self._fitted = False
        self._results: Optional[pd.DataFrame] = None
        self._summaries: Optional[pd.DataFrame] = None
        self._params: Optional[Dict[str, Any]] = None
        self._fit_info: Optional[Dict[str, Any]] = None

    def fit(
        self,
        data: pd.DataFrame,
        time_col: str,
        control_col: str,
        test_col: str,
        pretest_start: Union[pd.Timestamp, int, float],
        test_start: Union[pd.Timestamp, int, float],
        test_end: Union[pd.Timestamp, int, float],
    ) -> "TBRAnalysis":
        """
        Fit TBR model to data and store results.

        Performs Time-Based Regression analysis using the functional API,
        storing results and fitted parameters for later access via properties.

        Parameters
        ----------
        data : pd.DataFrame
            Time series data with time, control, and test columns.
        time_col : str
            Name of the time column (datetime64[ns], int64, or float64).
        control_col : str
            Name of control group metric column.
        test_col : str
            Name of test group metric column.
        pretest_start : Union[pd.Timestamp, int, float]
            Start time of pretest period (inclusive).
        test_start : Union[pd.Timestamp, int, float]
            Start time of test period (inclusive).
        test_end : Union[pd.Timestamp, int, float]
            End time of test period (inclusive/exclusive based on test_end_inclusive).

        Returns
        -------
        TBRAnalysis
            Returns self for method chaining.

        Raises
        ------
        TypeError
            If input types are invalid.
        ValueError
            If input validation fails or insufficient data for analysis.

        Examples
        --------
        Basic fitting:

        >>> model = TBRAnalysis(level=0.80, threshold=0.0)
        >>> model.fit(data, 'date', 'control', 'test',
        ...           pretest_start='2023-01-01',
        ...           test_start='2023-02-15',
        ...           test_end='2023-03-01')
        >>> print(f"Final effect: {model.summaries_.iloc[-1]['estimate']:.2f}")

        Method chaining:

        >>> results = (TBRAnalysis(level=0.95)
        ...            .fit(data, 'date', 'control', 'test',
        ...                 pretest_start='2023-01-01',
        ...                 test_start='2023-02-15',
        ...                 test_end='2023-03-01')
        ...            .results_)

        Notes
        -----
        Uses the stored configuration (level, threshold, test_end_inclusive)
        from initialization. Call fit() to perform analysis before accessing
        results_, summaries_, or params_ properties.
        """
        # Lazy imports to minimize loading overhead
        from tbr.functional import perform_tbr_analysis
        from tbr.utils.preprocessing import split_time_series_by_periods
        from tbr.utils.validation import (
            validate_dataframe_not_empty,
            validate_metric_columns,
            validate_no_nulls,
            validate_required_columns,
            validate_time_boundaries_type,
            validate_time_column_type,
            validate_time_periods,
        )

        # ===== Input Validation =====
        # Validate DataFrame type and not empty
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"data must be a pandas DataFrame, got {type(data).__name__}"
            )

        validate_dataframe_not_empty(data, "data")

        # Validate column name types
        if not isinstance(time_col, str):
            raise TypeError(f"time_col must be a string, got {type(time_col).__name__}")
        if not isinstance(control_col, str):
            raise TypeError(
                f"control_col must be a string, got {type(control_col).__name__}"
            )
        if not isinstance(test_col, str):
            raise TypeError(f"test_col must be a string, got {type(test_col).__name__}")

        # Validate required columns exist
        validate_required_columns(data, [time_col, control_col, test_col], "data")

        # Validate time column type
        validate_time_column_type(data, time_col, "data")

        # Validate metric columns are numeric
        validate_metric_columns(data, control_col, test_col)

        # Validate no nulls in required columns
        validate_no_nulls(data, [time_col, control_col, test_col], "data")

        # Validate time boundaries type consistency
        validate_time_boundaries_type(
            pretest_start, test_start, test_end, data[time_col].dtype
        )

        # Validate time periods ordering
        validate_time_periods(
            pretest_start, test_start, test_end, self.test_end_inclusive
        )

        # Perform TBR analysis using functional API with stored configuration
        tbr_dataframe, daily_summaries = perform_tbr_analysis(
            data=data,
            time_col=time_col,
            control_col=control_col,
            test_col=test_col,
            pretest_start=pretest_start,
            test_start=test_start,
            test_end=test_end,
            level=self.level,
            threshold=self.threshold,
            test_end_inclusive=self.test_end_inclusive,
        )

        # Extract model parameters from summaries (all rows have same parameters)
        summary_row = daily_summaries.iloc[0]

        # Extract pretest data to calculate pretest_sum_x_squared_deviations
        _, pretest_df, test_df, _ = split_time_series_by_periods(
            aggregated_data=data,
            time_col=time_col,
            pretest_start=pretest_start,
            test_start=test_start,
            test_end=test_end,
            test_end_inclusive=self.test_end_inclusive,
        )

        # Calculate pretest_sum_x_squared_deviations from pretest control values
        pretest_control = pretest_df[control_col].values
        pretest_x_mean = float(np.mean(pretest_control))
        pretest_sum_x_squared_deviations = float(
            np.sum((pretest_control - pretest_x_mean) ** 2)
        )

        # Store results
        self._results = tbr_dataframe
        self._summaries = daily_summaries

        # Store parameters dictionary
        self._params = {
            "alpha": float(summary_row["alpha"]),
            "beta": float(summary_row["beta"]),
            "sigma": float(summary_row["sigma"]),
            "var_alpha": float(summary_row["var_alpha"]),
            "var_beta": float(summary_row["var_beta"]),
            "cov_alpha_beta": float(summary_row["alpha_beta_cov"]),
            "degrees_freedom": int(summary_row["t_dist_df"]),
            "pretest_x_mean": pretest_x_mean,
            "pretest_sum_x_squared_deviations": pretest_sum_x_squared_deviations,
        }

        # Store fit information
        self._fit_info = {
            "time_col": time_col,
            "control_col": control_col,
            "test_col": test_col,
            "pretest_start": pretest_start,
            "test_start": test_start,
            "test_end": test_end,
            "n_pretest": len(pretest_df),
            "n_test": len(test_df),
        }

        # Mark as fitted
        self._fitted = True

        # Return self for method chaining
        return self

    def predict(
        self,
        control_values: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """
        Generate counterfactual predictions using the fitted TBR model.

        Predicts what the test group values would have been without treatment,
        using the regression relationship learned from the pretest period.

        Parameters
        ----------
        control_values : Union[pd.Series, np.ndarray, list], optional
            Control group values to generate predictions for. If None (default),
            uses control values from the test period of the fitted data.
            Can be a numpy array, pandas Series, or Python list.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - pred: Predicted values (counterfactual)
            - predsd: Prediction standard deviations (uncertainty)

        Raises
        ------
        AttributeError
            If the model has not been fitted yet.
        TypeError
            If control_values has invalid type.
        ValueError
            If control_values has invalid shape, is empty, or contains non-finite values.

        Examples
        --------
        Predict using fitted test period data:

        >>> model = TBRAnalysis(level=0.80)
        >>> model.fit(data, 'date', 'control', 'test', ...)
        >>> predictions = model.predict()
        >>> print(predictions[['pred', 'predsd']].head())

        Predict for new control values:

        >>> new_control = np.array([1000, 1050, 1100])
        >>> predictions = model.predict(control_values=new_control)

        Predict using a Python list:

        >>> predictions = model.predict(control_values=[1000, 1050, 1100])

        Notes
        -----
        Predictions are generated using the fitted regression model:
        pred = alpha + beta * control_value

        Prediction standard deviation includes both model and residual uncertainty:
        predsd = sqrt(sigma^2 * (1 + 1/n + (x* - x̄)^2 / Σ(xi - x̄)^2))
        """
        # Check if model is fitted
        if not self._fitted:
            raise AttributeError(
                "This TBRAnalysis instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using predict()."
            )

        # Lazy imports
        from tbr.core.prediction import generate_counterfactual_predictions

        assert self._results is not None
        assert self._params is not None

        # Use test period control values if not provided
        if control_values is None:
            test_period = self._results[self._results["period"] == 1]
            control_values = test_period["x"].values
        else:
            # Convert to numpy array if needed
            if isinstance(control_values, pd.Series):
                control_values = control_values.values
            elif isinstance(control_values, list):
                control_values = np.array(control_values)
            elif not isinstance(control_values, np.ndarray):
                try:
                    control_values = np.array(control_values)
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"control_values must be array-like (numpy array, pandas Series, or list), "
                        f"got {type(control_values).__name__}"
                    ) from e

        # Validate control values type and dimensions
        if not np.issubdtype(control_values.dtype, np.number):
            raise TypeError(
                f"control_values must contain numeric values, "
                f"got dtype '{control_values.dtype}'"
            )

        if control_values.ndim != 1:
            raise ValueError(
                f"control_values must be 1-dimensional, got {control_values.ndim}-dimensional "
                f"array with shape {control_values.shape}"
            )

        if len(control_values) == 0:
            raise ValueError("control_values cannot be empty")

        if not np.all(np.isfinite(control_values)):
            n_invalid = np.sum(~np.isfinite(control_values))
            raise ValueError(
                f"control_values must contain only finite values, "
                f"found {n_invalid} non-finite value(s)"
            )

        # Create test period DataFrame for predictions
        assert self._fit_info is not None
        test_period_data = pd.DataFrame(
            {
                self._fit_info["time_col"]: range(len(control_values)),
                self._fit_info["control_col"]: control_values,
            }
        )

        # Generate predictions using core functionality
        predictions = generate_counterfactual_predictions(
            alpha=self._params["alpha"],
            beta=self._params["beta"],
            sigma=self._params["sigma"],
            n_pretest=self._params["degrees_freedom"] + 2,
            pretest_x_mean=self._params["pretest_x_mean"],
            pretest_sum_x_squared_deviations=self._params[
                "pretest_sum_x_squared_deviations"
            ],
            test_period_data=test_period_data,
            control_col=self._fit_info["control_col"],
            time_col=self._fit_info["time_col"],
        )

        # Return only pred and predsd columns
        return predictions[["pred", "predsd"]].copy()

    def summarize(self, incremental: bool = False) -> pd.DataFrame:
        """
        Get summary statistics from the TBR analysis.

        Returns either the final cumulative summary or incremental day-by-day
        summaries for the test period.

        Parameters
        ----------
        incremental : bool, default=False
            If False (default), returns single-row final summary.
            If True, returns day-by-day incremental summaries.

        Returns
        -------
        pd.DataFrame
            Summary statistics DataFrame containing:
            - estimate: Treatment effect estimate
            - lower, upper: Credible interval bounds
            - se: Standard error
            - prob: Posterior probability of exceeding threshold
            - precision: Precision (inverse variance)
            - level: Credibility level used
            - thres: Threshold used
            - Additional model parameters (alpha, beta, sigma, etc.)

        Raises
        ------
        AttributeError
            If the model has not been fitted yet.

        Examples
        --------
        Get final summary:

        >>> model = TBRAnalysis(level=0.80, threshold=0.0)
        >>> model.fit(data, 'date', 'control', 'test', ...)
        >>> summary = model.summarize()
        >>> print(f"Effect: {summary['estimate'].iloc[0]:.2f}")
        >>> print(f"CI: [{summary['lower'].iloc[0]:.2f}, {summary['upper'].iloc[0]:.2f}]")

        Get incremental summaries:

        >>> incremental_summaries = model.summarize(incremental=True)
        >>> print(incremental_summaries[['estimate', 'lower', 'upper']])

        Notes
        -----
        The summary statistics are computed from the incremental summaries
        stored during fit(). The final summary is the last row of the
        incremental summaries.
        """
        # Check if model is fitted
        if not self._fitted:
            raise AttributeError(
                "This TBRAnalysis instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using summarize()."
            )

        assert self._summaries is not None

        # Return incremental or final summary
        if incremental:
            return self._summaries.copy()
        else:
            # Return only the final summary (last row)
            return self._summaries.iloc[[-1]].copy()

    def analyze_subinterval(
        self,
        start_day: int,
        end_day: int,
        ci_level: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Analyze treatment effect for a custom subinterval of the test period.

        Computes the treatment effect estimate and credible interval for a
        specific range of days within the test period.

        Parameters
        ----------
        start_day : int
            Starting day of the subinterval (1-indexed, inclusive).
            Day 1 is the first day of the test period.
        end_day : int
            Ending day of the subinterval (1-indexed, inclusive).
        ci_level : float, optional
            Credibility level for credible interval (must be between 0 and 1).
            If None, uses the level specified during initialization.

        Returns
        -------
        dict
            Dictionary containing:
            - estimate: Treatment effect for the subinterval
            - lower: Lower bound of credible interval
            - upper: Upper bound of credible interval
            - se: Standard error of the estimate
            - ci_level: Credibility level used
            - start_day: Starting day (as provided)
            - end_day: Ending day (as provided)
            - n_days: Number of days in the subinterval

        Raises
        ------
        AttributeError
            If the model has not been fitted yet.
        TypeError
            If start_day, end_day, or ci_level have invalid types.
        ValueError
            If start_day or end_day are invalid, start_day > end_day, days exceed
            test period, or ci_level is not between 0 and 1.

        Examples
        --------
        Analyze first week of test period:

        >>> model = TBRAnalysis(level=0.80)
        >>> model.fit(data, 'date', 'control', 'test', ...)
        >>> week1_effect = model.analyze_subinterval(start_day=1, end_day=7)
        >>> print(f"Week 1 effect: {week1_effect['estimate']:.2f}")
        >>> print(f"Week 1 CI: [{week1_effect['lower']:.2f}, {week1_effect['upper']:.2f}]")

        Analyze with custom credibility level:

        >>> week2_effect = model.analyze_subinterval(start_day=8, end_day=14, ci_level=0.95)

        Notes
        -----
        Subinterval analysis uses the compute_interval_estimate_and_ci function
        from the analysis module to calculate effects for specific day ranges
        with proper variance calculations.
        """
        # Check if model is fitted
        if not self._fitted:
            raise AttributeError(
                "This TBRAnalysis instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using analyze_subinterval()."
            )

        # Lazy imports
        from tbr.analysis.subinterval import compute_interval_estimate_and_ci
        from tbr.utils.validation import validate_probability_level

        assert self._results is not None
        assert self._params is not None

        # Validate day parameter types
        if not isinstance(start_day, (int, np.integer)):
            raise TypeError(
                f"start_day must be an integer, got {type(start_day).__name__}"
            )

        if not isinstance(end_day, (int, np.integer)):
            raise TypeError(f"end_day must be an integer, got {type(end_day).__name__}")

        # Convert numpy integers to Python int
        start_day = int(start_day)
        end_day = int(end_day)

        # Validate day parameter values
        if start_day < 1:
            raise ValueError(
                f"start_day must be a positive integer (>= 1), got {start_day}"
            )

        if end_day < 1:
            raise ValueError(
                f"end_day must be a positive integer (>= 1), got {end_day}"
            )

        if start_day > end_day:
            raise ValueError(f"start_day ({start_day}) must be <= end_day ({end_day})")

        # Check if days are within test period
        n_test_days = len(self._results[self._results["period"] == 1])

        if start_day > n_test_days:
            raise ValueError(
                f"start_day ({start_day}) exceeds test period length ({n_test_days} days)"
            )

        if end_day > n_test_days:
            raise ValueError(
                f"end_day ({end_day}) exceeds test period length ({n_test_days} days)"
            )

        # Use model's level if not provided, otherwise validate ci_level
        if ci_level is None:
            ci_level = self.level
        else:
            if not isinstance(ci_level, (int, float)):
                raise TypeError(
                    f"ci_level must be numeric, got {type(ci_level).__name__}"
                )
            validate_probability_level(ci_level, "ci_level")

        # Compute subinterval estimate
        result = compute_interval_estimate_and_ci(
            tbr_df=self._results,
            tbr_summary=self._summaries,
            start_day=start_day,
            end_day=end_day,
            ci_level=ci_level,
        )

        # Calculate standard error from precision (half-width of CI)
        se = result["precision"]

        # Return results as dictionary
        return {
            "estimate": result["estimate"],
            "lower": result["lower"],
            "upper": result["upper"],
            "se": se,
            "ci_level": ci_level,
            "start_day": start_day,
            "end_day": end_day,
            "n_days": end_day - start_day + 1,
        }

    @property
    def fitted_(self) -> bool:
        """
        Whether the model has been fitted.

        Returns
        -------
        bool
            True if fit() has been called successfully, False otherwise.
        """
        return self._fitted

    @property
    def results_(self) -> pd.DataFrame:
        """
        TBR DataFrame with predictions, effects, and uncertainties.

        This DataFrame contains the complete time series with all TBR calculations:
        - Original data (time, control, test values)
        - Period indicators (pretest=0, test=1, cooldown=3)
        - Counterfactual predictions (pred, predsd)
        - Effects (dif, cumdif, cumsd, estsd)

        Returns
        -------
        pd.DataFrame
            Complete TBR analysis results DataFrame.

        Raises
        ------
        AttributeError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = TBRAnalysis()
        >>> model.fit(data, 'date', 'control', 'test', ...)
        >>> results = model.results_
        >>> test_period_results = results[results['period'] == 1]
        """
        if not self._fitted:
            raise AttributeError(
                "This TBRAnalysis instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before accessing results_."
            )
        assert self._results is not None  # Guaranteed by _fitted check
        return self._results

    @property
    def summaries_(self) -> pd.DataFrame:
        """
        Incremental summaries with daily progression of cumulative effects.

        This DataFrame contains day-by-day summaries for the test period:
        - estimate: Cumulative treatment effect
        - precision: 1/variance of the estimate
        - lower, upper: Credible interval bounds
        - se: Standard error of the estimate
        - level: Credibility level used
        - threshold: Threshold used for probability calculation
        - prob: Posterior probability of exceeding threshold
        - Model parameters (alpha, beta, sigma, variances, covariances)

        Returns
        -------
        pd.DataFrame
            Incremental summaries for each day of the test period.

        Raises
        ------
        AttributeError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = TBRAnalysis()
        >>> model.fit(data, 'date', 'control', 'test', ...)
        >>> summaries = model.summaries_
        >>> final_effect = summaries.iloc[-1]['estimate']
        >>> is_significant = summaries.iloc[-1]['lower'] > 0
        """
        if not self._fitted:
            raise AttributeError(
                "This TBRAnalysis instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before accessing summaries_."
            )
        assert self._summaries is not None  # Guaranteed by _fitted check
        return self._summaries

    @property
    def params_(self) -> Dict[str, Any]:
        """
        Regression model parameters from TBR analysis.

        Returns
        -------
        dict
            Dictionary containing regression parameters:
            - alpha: Intercept coefficient
            - beta: Slope coefficient
            - sigma: Residual standard error
            - var_alpha: Variance of alpha
            - var_beta: Variance of beta
            - cov_alpha_beta: Covariance between alpha and beta
            - degrees_freedom: Degrees of freedom for t-distribution
            - pretest_x_mean: Mean of control in pretest period
            - pretest_sum_x_squared_deviations: Sum of squared deviations

        Raises
        ------
        AttributeError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = TBRAnalysis()
        >>> model.fit(data, 'date', 'control', 'test', ...)
        >>> params = model.params_
        >>> print(f"Beta coefficient: {params['beta']:.4f}")
        >>> print(f"Residual std error: {params['sigma']:.4f}")
        """
        if not self._fitted:
            raise AttributeError(
                "This TBRAnalysis instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before accessing params_."
            )
        assert self._params is not None  # Guaranteed by _fitted check
        return self._params

    def __repr__(self) -> str:
        """
        Return string representation of TBRAnalysis instance.

        Returns
        -------
        str
            String representation showing configuration and fitted status.
        """
        fitted_str = "fitted" if self._fitted else "not fitted"
        return (
            f"TBRAnalysis(level={self.level}, threshold={self.threshold}, "
            f"test_end_inclusive={self.test_end_inclusive}, {fitted_str})"
        )

    def __str__(self) -> str:
        """
        Return user-friendly string representation.

        Returns
        -------
        str
            Human-readable string with key information.
        """
        if not self._fitted:
            return f"TBRAnalysis (not fitted)\n  level={self.level}\n  threshold={self.threshold}"

        assert self._summaries is not None  # Guaranteed by _fitted check
        n_test_days = len(self._summaries)
        final_effect = self._summaries.iloc[-1]["estimate"]
        final_lower = self._summaries.iloc[-1]["lower"]
        final_upper = self._summaries.iloc[-1]["upper"]

        return (
            f"TBRAnalysis (fitted)\n"
            f"  Configuration:\n"
            f"    level={self.level}\n"
            f"    threshold={self.threshold}\n"
            f"  Results:\n"
            f"    Test period days: {n_test_days}\n"
            f"    Final effect estimate: {final_effect:.2f}\n"
            f"    {int(self.level*100)}% CI: [{final_lower:.2f}, {final_upper:.2f}]"
        )
