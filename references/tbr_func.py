"""
TBR (Time-Based Regression) Python Package - Functional Implementation

This module provides a pure functional Python implementation of Time-Based
Regression analysis for measuring ad effectiveness through geo experiments,
replacing the R GeoexperimentsResearch package functionality.

All functions are designed to be independent, testable, and reusable.
"""

import pandas as pd
from typing import List, Tuple, Union, Dict, Optional
import datetime
import numpy as np
import statsmodels.api as sm
import constants
from scipy import stats


def validate_required_columns(df: pd.DataFrame, required_cols: List[str], df_name: str) -> None:
    """
    Validate that DataFrame contains all required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_cols : List[str]
        List of required column names
    df_name : str
        Name of the DataFrame for error messages

    Raises
    ------
    ValueError
        If any required columns are missing
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {df_name}: {missing_cols}")


def validate_no_nulls(df: pd.DataFrame, cols: List[str], df_name: str) -> None:
    """
    Validate that specified columns contain no null values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    cols : List[str]
        List of column names to check for nulls
    df_name : str
        Name of the DataFrame for error messages

    Raises
    ------
    ValueError
        If null values are found
    """
    null_counts = df[cols].isnull().sum()
    if null_counts.any():
        null_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Null values found in {df_name}: {null_cols}")


def merge_data_with_assignments(
    data: pd.DataFrame,
    assignments: pd.DataFrame,
    geo_col: str,
    date_col: str,
    metric_col: str,
    assignment_col: str
) -> pd.DataFrame:
    """
    Merge time series data with geo assignments.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data with columns: date, geo, metric
    assignments : pd.DataFrame
        Assignment data with columns: geo, assignment
    geo_col : str
        Name of the geo column
    date_col : str
        Name of the date column
    metric_col : str
        Name of the metric column
    assignment_col : str
        Name of the assignment column

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns: date, geo, metric, assignment

    Raises
    ------
    ValueError
        If required columns are missing or data validation fails
    """
    # Validate input DataFrames
    validate_required_columns(data, [date_col, geo_col, metric_col], "data")
    validate_required_columns(assignments, [geo_col, assignment_col], "assignments")

    # Check for nulls in required columns
    validate_no_nulls(data, [date_col, geo_col, metric_col], "data")
    validate_no_nulls(assignments, [geo_col, assignment_col], "assignments")

    # Validate metric column is numeric
    if not pd.api.types.is_numeric_dtype(data[metric_col]):
        raise ValueError(f"Metric column '{metric_col}' must be numeric")

    # Check for empty DataFrames
    if data.empty:
        raise ValueError("Data DataFrame cannot be empty")
    if assignments.empty:
        raise ValueError("Assignments DataFrame cannot be empty")

    # Check for duplicate geos in assignments
    if assignments[geo_col].duplicated().any():
        raise ValueError("Duplicate geos found in assignments DataFrame")

    # Merge data with assignments (inner join to keep only assigned geos)
    merged_df = data.merge(
        assignments[[geo_col, assignment_col]],
        on=geo_col,
        how='inner'
    )

    # Check that merge was successful
    if merged_df.empty:
        raise ValueError("No matching geos found between data and assignments")

    return merged_df


def aggregate_metrics_by_group(
    merged_data: pd.DataFrame,
    date_col: str,
    assignment_col: str,
    metric_col: str,
    control_group: str,
    test_group: str
) -> pd.DataFrame:
    """
    Aggregate metrics by date and assignment group.

    Parameters
    ----------
    merged_data : pd.DataFrame
        Merged data with columns: date, geo, metric, assignment
    date_col : str
        Name of the date column
    assignment_col : str
        Name of the assignment column
    metric_col : str
        Name of the metric column
    control_group : str
        Value representing control group in assignment column
    test_group : str
        Value representing test group in assignment column

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with columns: date, control, test

    Raises
    ------
    ValueError
        If required groups are not found or validation fails
    """
    # Validate input
    validate_required_columns(merged_data, [date_col, assignment_col, metric_col], "merged_data")
    validate_no_nulls(merged_data, [date_col, assignment_col, metric_col], "merged_data")

    # Check that required groups exist
    unique_groups = set(merged_data[assignment_col].unique())
    required_groups = {control_group, test_group}
    missing_groups = required_groups - unique_groups
    if missing_groups:
        raise ValueError(f"Missing required groups in data: {missing_groups}")

    # Filter to only include required groups
    filtered_data = merged_data[merged_data[assignment_col].isin([control_group, test_group])]

    # Aggregate by date and assignment
    aggregated = (
        filtered_data
        .groupby([date_col, assignment_col], as_index=False)
        [metric_col]
        .sum()
    )

    # Pivot to wide format (control and test as columns)
    pivoted = aggregated.pivot(
        index=date_col,
        columns=assignment_col,
        values=metric_col
    ).fillna(0.0).reset_index()

    # Ensure both control and test columns exist
    if control_group not in pivoted.columns:
        pivoted[control_group] = 0.0
    if test_group not in pivoted.columns:
        pivoted[test_group] = 0.0

    # Rename columns to standard names and reorder
    pivoted = pivoted.rename(columns={
        control_group: constants.CONTROL_VAL,
        test_group: constants.TEST_VAL
    })

    # Ensure column order
    pivoted = pivoted[[date_col, control_group, test_group]]
    # Remove name from columns index
    pivoted.columns.name = None

    return pivoted


def parse_date_string(date_str: Union[str, pd.Timestamp], param_name: str, make_exclusive: bool) -> pd.Timestamp:
    """
    Parse date string or timezone-aware datetime to a UTC datetime object.

    Parameters
    ----------
    date_str : Union[str, pd.Timestamp]
        Date as string (YYYY-MM-DD format) or timezone-aware datetime object
    param_name : str
        Parameter name for error messages
    make_exclusive : bool, default False
        If True, adds 1 day to make the date exclusive (useful for end dates).
        If False, keeps the date as-is (useful for start dates).

    Returns
    -------
    pd.Timestamp
        Parsed date object with UTC timezone. If make_exclusive=True,
        the date is shifted to the next day at 00:00:00 UTC.

    Raises
    ------
    ValueError
        If date format is invalid or timezone is not specified
    """
    if isinstance(date_str, pd.Timestamp):
        if date_str.tzinfo is None:
            raise ValueError(f"{param_name} must be timezone-aware, got naive datetime")
        parsed_date = date_str.tz_convert('UTC')
    elif isinstance(date_str, str):
        try:
            parsed_date = pd.to_datetime(date_str).tz_localize('UTC')
        except ValueError:
            raise ValueError(f"{param_name} must be in YYYY-MM-DD format, got: {date_str}")
    else:
        raise ValueError(f"{param_name} must be string or timezone-aware datetime, got: {type(date_str)}")

    # Add 1 day for exclusive end dates
    if make_exclusive:
        parsed_date = parsed_date + pd.Timedelta(days=1)

    return parsed_date


def validate_date_periods(
    pretest_start: Union[str, datetime.date],
    test_start: Union[str, datetime.date],
    test_end: Union[str, datetime.date]
) -> Tuple[datetime.date, datetime.date, datetime.date]:
    """
    Validate and parse date parameters for TBR analysis.

    Parameters
    ----------
    pretest_start : Union[str, datetime.date]
        Start date of pretest period
    test_start : Union[str, datetime.date]
        Start date of test period
    test_end : Union[str, datetime.date]
        End date of test period

    Returns
    -------
    Tuple[datetime.date, datetime.date, datetime.date]
        Parsed and validated dates (pretest_start, test_start, test_end)

    Raises
    ------
    ValueError
        If dates are invalid or in wrong order
    """
    # Parse dates
    pretest_start_date = parse_date_string(pretest_start, "pretest_start", make_exclusive=False)
    test_start_date = parse_date_string(test_start, "test_start", make_exclusive=False)
    test_end_date = parse_date_string(test_end, "test_end", make_exclusive=True)

    # Validate date order
    if not (pretest_start_date < test_start_date < test_end_date):
        raise ValueError(f"Dates must be in order: pretest_start < test_start < test_end, "
                         f"got: {pretest_start_date} < {test_start_date} < {test_end_date} "
                         f"(Note: test_end shown as next day due to exclusive boundary handling)")

    return pretest_start_date, test_start_date, test_end_date


def split_by_periods(
    aggregated_data: pd.DataFrame,
    date_col: str,
    control_col: str,
    test_col: str,
    pretest_start: Union[str, datetime.date],
    test_start: Union[str, datetime.date],
    test_end: Union[str, datetime.date]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split aggregated time series data into pretest, test and cooldown periods.

    Parameters
    ----------
    aggregated_data : pd.DataFrame
        Aggregated data with columns: date, control, test
    date_col : str
        Name of the date column
    pretest_start : Union[str, datetime.date]
        Start date of pretest period
    test_start : Union[str, datetime.date]
        Start date of test period
    test_end : Union[str, datetime.date]
        End date of test period

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (baseline_data, pretest_data, test_data, cooldown_data) - DataFrames for each period

    Raises
    ------
    ValueError
        If date validation fails or no data found in periods
    """
    # Validate dates
    pretest_start_date, test_start_date, test_end_date = validate_date_periods(
        pretest_start, test_start, test_end
    )

    # Validate input data
    validate_required_columns(aggregated_data,
                              [date_col, control_col, test_col],
                              "aggregated_data"
                              )

    # Convert date column to datetime if it's string
    data_copy = aggregated_data.copy()
    if data_copy[date_col].dtype == 'object':
        data_copy[date_col] = pd.to_datetime(data_copy[date_col]).dt.tz_localize('UTC')

    # Split into periods - pandas handles datetime vs date comparison well
    baseline_mask = (
        (data_copy[date_col] < pd.to_datetime(pretest_start_date))
    )
    pretest_mask = (
        (data_copy[date_col] >= pd.to_datetime(pretest_start_date)) &
        (data_copy[date_col] < pd.to_datetime(test_start_date))
    )
    test_mask = (
        (data_copy[date_col] >= pd.to_datetime(test_start_date)) &
        (data_copy[date_col] < pd.to_datetime(test_end_date))  # Exclusive end date
    )
    cooldown_mask = (
        (data_copy[date_col] >= pd.to_datetime(test_end_date))  # Exclusive end date
    )

    baseline_data = data_copy[baseline_mask].copy()
    pretest_data = data_copy[pretest_mask].copy()
    test_data = data_copy[test_mask].copy()
    cooldown_data = data_copy[cooldown_mask].copy()

    return baseline_data, pretest_data, test_data, cooldown_data


def create_time_series_for_tbr(
    data: pd.DataFrame,
    assignments: pd.DataFrame,
    geo_col: str,
    date_col: str,
    metric_col: str,
    assignment_col: str,
    control_group: str,
    test_group: str,
    pretest_start: Union[str, datetime.date],
    test_start: Union[str, datetime.date],
    test_end: Union[str, datetime.date]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete pipeline to create time series data for TBR analysis.

    This function orchestrates all the data preparation steps:
    1. Merge data with assignments
    2. Aggregate by groups and dates
    3. Split into pretest and test periods

    Parameters
    ----------
    data : pd.DataFrame
        Raw time series data
    assignments : pd.DataFrame
        Geo assignment data
    geo_col : str
        Name of geo column
    date_col : str
        Name of date column
    metric_col : str
        Name of metric column
    assignment_col : str
        Name of assignment column
    control_group : str
        Control group identifier
    test_group : str
        test group identifier
    pretest_start : Union[str, datetime.date]
        Pretest start date
    test_start : Union[str, datetime.date]
        Test start date
    test_end : Union[str, datetime.date]
        Test end date

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (full_aggregated_data, baseline_data, pretest_data, test_data, cooldown_data)

    Raises
    ------
    ValueError
        If any validation step fails
    """
    # Step 1: Merge data with assignments
    merged_data = merge_data_with_assignments(
        data, assignments, geo_col, date_col, metric_col, assignment_col
    )

    # Step 2: Aggregate by groups
    aggregated_data = aggregate_metrics_by_group(
        merged_data, date_col, assignment_col, metric_col, control_group, test_group
    )

    # Step 3: Split by periods
    baseline_data, pretest_data, test_data, cooldown_data = split_by_periods(
        aggregated_data,
        date_col,
        control_col=control_group,
        test_col=test_group,
        pretest_start=pretest_start,
        test_start=test_start,
        test_end=test_end
    )

    return aggregated_data, baseline_data, pretest_data, test_data, cooldown_data


def fit_tbr_regression_model(
    control_test_df: pd.DataFrame,
    start_pretest_date: str,
    start_test_date: str,
    control_col: str,
    test_col: str
) -> Dict[str, float]:
    """
    Fits a TBR regression model using statsmodels OLS.

    This function fits a linear regression model of the form:
    test = α + β * control + ε

    Parameters:
    - control_test_df: DataFrame with 'date', control, and test columns
    - start_pretest_date: Start date for pretest period (YYYY-MM-DD)
    - start_test_date: Start date for test period (YYYY-MM-DD)
    - control_col: Name of the control column
    - test_col: Name of the test column (can be flexible)

    Parameters returned:
    - 'alpha': Intercept (α)
    - 'beta': Slope coefficient (β)
    - 'sigma': Residual standard deviation (σ)
    - 'var_alpha': Variance of intercept estimate
    - 'var_beta': Variance of slope estimate
    - 'cov_alpha_beta': Covariance between α and β estimates
    - 'degrees_freedom': Residual degrees of freedom
    - 'n_pretest': Number of pretest observations
    - 'x_mean': Mean of control values (x̄)

    Args:
        control_test_df: DataFrame with control and test columns
        start_pretest_date: Start date for pretest period
        start_test_date: Start date for test period (end of pretest)

    Returns:
        Dict with regression parameters needed for TBR analysis
    """
    # Input validation
    if control_test_df.empty:
        raise ValueError("Input DataFrame is empty")

    required_cols = ['date', control_col, test_col]
    missing_cols = [col for col in required_cols if col not in control_test_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert dates to datetime if they aren't already
    df = control_test_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize('UTC')

    start_pretest = pd.to_datetime(start_pretest_date).tz_localize('UTC')
    start_test = pd.to_datetime(start_test_date).tz_localize('UTC')

    # Filter to pretest period only
    pretest_df = df[
        (df['date'] >= start_pretest) &
        (df['date'] < start_test)
    ].copy()

    if len(pretest_df) < 3:
        raise ValueError(
            f"Insufficient pretest data: {len(pretest_df)} observations. Need at least 3."
        )

    # Check for missing or invalid values
    if pretest_df[[control_col, test_col]].isnull().any().any():
        raise ValueError("Pretest data contains null values")

    if not np.isfinite(pretest_df[[control_col, test_col]]).all().all():
        raise ValueError("Pretest data contains infinite or NaN values")

    # Extract x (control) and y (test) for regression
    x = pretest_df[control_col].values
    y = pretest_df[test_col].values
    n = len(x)

    # Check for constant control values
    if np.var(x) == 0:
        raise ValueError(
            "Control group values are constant in pretest period - cannot fit regression"
        )

    # Prepare data for statsmodels (add constant for intercept)
    X = sm.add_constant(x)

    # Fit OLS regression using statsmodels
    model = sm.OLS(y, X).fit()

    # Extract all parameters directly from statsmodels
    alpha = model.params[0]  # Intercept
    beta = model.params[1]   # Slope

    # Extract variances from standard errors
    var_alpha = model.bse[0] ** 2  # Variance of intercept
    var_beta = model.bse[1] ** 2   # Variance of slope

    # Extract covariance from covariance matrix
    cov_matrix = model.cov_params()
    cov_alpha_beta = cov_matrix[0, 1]  # Covariance between intercept and slope

    # Extract other statistics
    sigma = np.sqrt(model.scale)  # Residual standard deviation
    degrees_freedom = int(model.df_resid)  # Degrees of freedom

    # Compute additional statistics needed for TBR
    x_mean = np.mean(x)

    # Validation of computed statistics
    if not np.isfinite([alpha, beta, sigma, var_alpha, var_beta, cov_alpha_beta]).all():
        raise ValueError("Computed regression parameters contain invalid values")

    if sigma <= 0:
        raise ValueError(f"Invalid residual standard deviation: {sigma}")

    if var_alpha <= 0 or var_beta <= 0:
        raise ValueError("Computed coefficient variances are non-positive")

    # Return all parameters as a simple dictionary
    return {
        'alpha': float(alpha),
        'beta': float(beta),
        'sigma': float(sigma),
        'var_alpha': float(var_alpha),
        'var_beta': float(var_beta),
        'cov_alpha_beta': float(cov_alpha_beta),
        'degrees_freedom': int(degrees_freedom),
        'n_pretest': int(n),
        'x_mean': float(x_mean)
    }


def calculate_model_variance(
    x_values: np.ndarray,
    x_mean: float,
    sigma: float,
    n_pretest: int,
    sum_x_squared_deviations: Optional[float] = None,
    var_beta: Optional[float] = None
) -> np.ndarray:
    """
    Calculate model variance for fitted values using TBR formula.

    Implements the TBR model variance formula for MODEL UNCERTAINTY ONLY:
    V[ŷ*] = σ² · (1/n + (x* - x̄)²/Σ(xi - x̄)²)

    This captures only the uncertainty in the fitted model, not the residual noise.
    For prediction variance which includes residual noise, use calculate_prediction_variance().

    Parameters
    ----------
    x_values : np.ndarray
        Control values for which to calculate model variance
    x_mean : float
        Mean of control values from pretest period (x̄)
    sigma : float
        Residual standard deviation (σ)
    n_pretest : int
        Number of pretest observations
    sum_x_squared_deviations : Optional[float], optional
        Σ(xi - x̄)². If not provided, calculated from var_beta and sigma
    var_beta : Optional[float], optional
        Variance of slope coefficient. Used to calculate sum_x_squared_deviations if not provided

    Returns
    -------
    np.ndarray
        Model variances for each x value (model uncertainty only)

    Notes
    -----
    Either sum_x_squared_deviations OR var_beta must be provided.
    If both are provided, sum_x_squared_deviations takes precedence.
    """
    # Input validation
    if len(x_values) == 0:
        raise ValueError("x_values cannot be empty")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    if n_pretest < 3:
        raise ValueError("n_pretest must be at least 3")

    # Calculate sum_x_squared_deviations if not provided
    if sum_x_squared_deviations is None:
        if var_beta is None:
            raise ValueError("Either sum_x_squared_deviations or var_beta must be provided")
        if var_beta <= 0:
            raise ValueError("var_beta must be positive")
        sum_x_squared_deviations = sigma**2 / var_beta

    if sum_x_squared_deviations <= 0:
        raise ValueError("sum_x_squared_deviations must be positive")

    # Apply TBR model variance formula (MODEL UNCERTAINTY ONLY)
    # V[ŷ*] = σ² · (1/n + (x* - x̄)²/Σ(xi - x̄)²)
    x_deviations_squared = (x_values - x_mean) ** 2

    model_variances = sigma**2 * (
        1.0/n_pretest +
        x_deviations_squared / sum_x_squared_deviations
    )

    return model_variances


def calculate_prediction_variance(
    x_values: np.ndarray,
    x_mean: float,
    sigma: float,
    n_pretest: int,
    sum_x_squared_deviations: Optional[float] = None,
    var_beta: Optional[float] = None
) -> np.ndarray:
    """
    Calculate prediction variance including both model uncertainty and residual noise.

    Implements the TBR prediction variance formula:
    V[y*] = σ² + V[ŷ*] = σ² + σ² · (1/n + (x* - x̄)²/Σ(xi - x̄)²)

    This can be simplified to:
    V[y*] = σ² · (1 + 1/n + (x* - x̄)²/Σ(xi - x̄)²)

    Parameters
    ----------
    x_values : np.ndarray
        Control values for which to calculate prediction variance
    x_mean : float
        Mean of control values from pretest period (x̄)
    sigma : float
        Residual standard deviation (σ)
    n_pretest : int
        Number of pretest observations
    sum_x_squared_deviations : Optional[float], optional
        Σ(xi - x̄)². If not provided, calculated from var_beta and sigma
    var_beta : Optional[float], optional
        Variance of slope coefficient. Used to calculate sum_x_squared_deviations if not provided

    Returns
    -------
    np.ndarray
        Prediction variances for each x value (model uncertainty + residual noise)

    Notes
    -----
    Either sum_x_squared_deviations OR var_beta must be provided.
    If both are provided, sum_x_squared_deviations takes precedence.
    """
    # Calculate model uncertainty component
    model_variances = calculate_model_variance(
        x_values=x_values,
        x_mean=x_mean,
        sigma=sigma,
        n_pretest=n_pretest,
        sum_x_squared_deviations=sum_x_squared_deviations,
        var_beta=var_beta
    )

    # Add residual variance: V[y*] = σ² + V[ŷ*]
    prediction_variances = sigma**2 + model_variances

    return prediction_variances


def generate_counterfactual_predictions(
    alpha: float,
    beta: float,
    sigma: float,
    x_mean: float,
    n_pretest: int,
    var_beta: float,
    test_period_data: pd.DataFrame,
    control_col: str
) -> pd.DataFrame:
    """
    Generate counterfactual predictions and their standard deviations for test period.

    Parameters
    ----------
    alpha : float
        Regression intercept coefficient
    beta : float
        Regression slope coefficient
    sigma : float
        Residual standard deviation from regression model
    x_mean : float
        Mean of control values during pretest period
    n_pretest : int
        Number of observations in pretest period
    var_beta : float
        Variance of the slope coefficient estimate
    test_period_data : pd.DataFrame
        Data for test period with control values
    control_col : str
        Name of control column

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, control, pred, predsd
        - pred: counterfactual predictions (ŷ*)
        - predsd: prediction standard deviations (V[ŷ*]^0.5)
    """
    # Input validation
    if test_period_data.empty:
        raise ValueError("test_period_data cannot be empty")

    if control_col not in test_period_data.columns:
        raise ValueError(f"Column '{control_col}' not found in test_period_data")

    # Get control values for test period
    x_test = test_period_data[control_col].values

    # Calculate counterfactual predictions: ŷ* = α + β * x*
    predictions = alpha + beta * x_test

    # Calculate prediction variances
    prediction_variances = calculate_prediction_variance(
        x_values=x_test,
        x_mean=x_mean,
        sigma=sigma,
        n_pretest=int(n_pretest),
        var_beta=var_beta
    )

    # Calculate prediction standard deviations
    prediction_std_devs = np.sqrt(prediction_variances)

    # Create result DataFrame
    result_df = test_period_data[['date', control_col]].copy()
    result_df['pred'] = predictions
    result_df['predsd'] = prediction_std_devs

    return result_df


def calculate_cumulative_standard_deviation(
    test_x_values: np.ndarray,
    sigma: float,
    var_alpha: float,
    var_beta: float,
    cov_alpha_beta: float
) -> np.ndarray:
    """
    Calculate the standard deviation of the cumulative causal effect for TBR test period using vectorized operations.

    This function implements the TBR formula for cumulative variance:
    V[Δr(T)] = T · σ² + T² · v
    where v = Var(α̂) + 2·x̄_T·Cov(α̂,β̂) + x̄_T²·Var(β̂)

    Parameters
    ----------
    test_x_values : np.ndarray
        Array of control group values during test period
    sigma : float
        Residual standard deviation from regression model
    var_alpha : float
        Variance of intercept estimate
    var_beta : float
        Variance of slope estimate
    cov_alpha_beta : float
        Covariance between intercept and slope estimates

    Returns
    -------
    np.ndarray
        Array of cumulative standard deviations for each time point
    """
    n = len(test_x_values)
    T_values = np.arange(1, n + 1)  # [1, 2, 3, ..., n]

    # Calculate cumulative means efficiently using vectorized operations
    cumsum_x = np.cumsum(test_x_values)
    x_mean_cumulative = cumsum_x / T_values

    # Vectorized calculation of v for all time points
    v_values = (
            var_alpha + 2 * x_mean_cumulative * cov_alpha_beta + (x_mean_cumulative ** 2) * var_beta
    )

    # Vectorized calculation of cumulative variance
    cum_variance = T_values * (sigma ** 2) + (T_values ** 2) * v_values

    # Vectorized square root
    return np.sqrt(cum_variance)


def compute_interval_estimate_and_ci(tbr_df, tbr_summary, start_day, end_day, ci_level):
    """
    Compute the cumulative effect estimate and credible interval for a subinterval of the test period.

    Parameters:
        tbr_df (pd.DataFrame): TBR daily output with columns 'y', 'pred', 'period', 'estsd'
        tbr_summary (pd.DataFrame): TBR summary with 'sigma' and 't_dist_df'
        start_day (int): Start day of subinterval (1-indexed within test period)
        end_day (int): End day of subinterval (inclusive)
        ci_level (float): Credible interval level (default 0.80)

    Returns:
        dict with keys: 'estimate', 'precision', 'lower', 'upper'
    """
    # Filter for test period
    test_df = tbr_df[tbr_df['period'] == 1].reset_index(drop=True)

    # Slice the subinterval (remember start_day is 1-indexed)
    interval_df = test_df.iloc[start_day - 1: end_day]

    # Estimate of cumulative effect (sum of differences)
    estimate = (interval_df['y'] - interval_df['pred']).sum()

    # Posterior variance = sum of estsd^2 + n * sigma^2
    sum_estsd_sq = np.sum(interval_df['estsd'] ** 2)
    n_days = end_day - start_day + 1
    sigma = float(tbr_summary.iloc[-1]['sigma'])
    dof = int(tbr_summary.iloc[-1]['t_dist_df'])

    posterior_variance = sum_estsd_sq + n_days * sigma**2
    se = np.sqrt(posterior_variance)

    # t-multiplier
    alpha = 1 - ci_level
    t_mult = stats.t.ppf(1 - alpha / 2, dof)

    # Precision (half-width)
    precision = t_mult * se

    return {
        'estimate': estimate,
        'precision': precision,
        'lower': estimate - precision,
        'upper': estimate + precision
    }


def create_tbr_baseline_dataframe(
    baseline_data: pd.DataFrame,
    control_col: str,
    test_col: str
) -> pd.DataFrame:
    """
    Create a TBR baseline dataframe segment from the baseline data.
    Parameters
    ----------
    baseline_data : pd.DataFrame
        Baseline period data with date, control, and test columns
    control_col : str
        Name of control column
    test_col : str
        Name of test column

    Returns
    -------
    pd.DataFrame
        Baseline period TBR dataframe segment
    """
    baseline_df = baseline_data.copy()
    if baseline_df.empty:
        return None
    baseline_df['period'] = -1
    baseline_df['y'] = baseline_df[test_col]
    baseline_df['x'] = baseline_df[control_col]
    baseline_df['pred'] = np.nan
    baseline_df['estsd'] = np.nan
    baseline_df['predsd'] = np.nan
    baseline_df['dif'] = np.nan
    baseline_df['cumdif'] = np.nan
    baseline_df['cumsd'] = np.nan

    return baseline_df


def create_tbr_pretest_dataframe(
    pretest_data: pd.DataFrame,
    alpha: float,
    beta: float,
    sigma: float,
    x_mean: float,
    n_pretest: int,
    var_beta: float,
    control_col: str,
    test_col: str
) -> pd.DataFrame:
    """
    Process the pretest period data to create a TBR dataframe segment.

    Parameters
    ----------
    pretest_data : pd.DataFrame
        Pretest period data with date, control, and test columns
    alpha : float
        Regression intercept coefficient
    beta : float
        Regression slope coefficient
    sigma : float
        Residual standard deviation from regression model
    x_mean : float
        Mean of control values during pretest period
    n_pretest : int
        Number of observations in pretest period
    var_beta : float
        Variance of the slope coefficient estimate
    control_col : str
        Name of control column
    test_col : str
        Name of test column

    Returns
    -------
    pd.DataFrame
        Pretest period TBR dataframe segment
    """
    pretest_df = pretest_data.copy()
    pretest_df['period'] = 0
    pretest_df['y'] = pretest_df[test_col]
    pretest_df['x'] = pretest_df[control_col]

    # Calculate fitted values for pretest period
    pretest_df['pred'] = alpha + beta * pretest_df['x']

    # Calculate fitted value standard deviations (estsd) for pretest
    # This uses the formula for variance of fitted values: V[ŷ] = σ²(1/n + (x-x̄)²/Σ(xi-x̄)²)
    # For fitted values, we want only model uncertainty, not residual noise
    fitted_variances = calculate_model_variance(
        x_values=pretest_df['x'].values,
        x_mean=x_mean,
        sigma=sigma,
        n_pretest=n_pretest,
        var_beta=var_beta
    )
    pretest_df['estsd'] = np.sqrt(fitted_variances)

    # Pretest period doesn't have prediction standard deviation in the same sense because the focus is on model fitting
    # rather than prediction, emphasizing the standard deviation of the fitted model.
    pretest_df['predsd'] = 0.0

    # Calculate residuals for pretest
    pretest_df['dif'] = pretest_df['y'] - pretest_df['pred']

    # Pretest doesn't have cumulative metrics
    pretest_df['cumdif'] = np.nan
    pretest_df['cumsd'] = 0.0

    return pretest_df


def create_tbr_test_dataframe(
    test_data: pd.DataFrame,
    alpha: float,
    beta: float,
    sigma: float,
    x_mean: float,
    n_pretest: int,
    var_beta: float,
    var_alpha: float,
    cov_alpha_beta: float,
    control_col: str,
    test_col: str
) -> pd.DataFrame:
    """
    Process the test period data to create a TBR dataframe segment.
    Note: test_data should include both test period (period=1) and cooldown period (period=3) if exists.
    The period values should already be assigned when the data reaches this function.

    Parameters
    ----------
    test_data : pd.DataFrame
        Test period data with date, control, test, and period columns.
        Should include both test period (period=1) and cooldown period (period=3) if exists.
    alpha : float
        Regression intercept coefficient
    beta : float
        Regression slope coefficient
    sigma : float
        Residual standard deviation from regression model
    x_mean : float
        Mean of control values during pretest period
    n_pretest : int
        Number of observations in pretest period
    var_beta : float
        Variance of the slope coefficient estimate
    var_alpha : float
        Variance of the intercept coefficient estimate
    cov_alpha_beta : float
        Covariance between intercept and slope coefficient estimates
    control_col : str
        Name of control column
    test_col : str
        Name of test column

    Returns
    -------
    pd.DataFrame
        Test period TBR dataframe segment
    """
    test_predictions = generate_counterfactual_predictions(
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        x_mean=x_mean,
        n_pretest=n_pretest,
        var_beta=var_beta,
        test_period_data=test_data,
        control_col=control_col
    )

    test_df = test_data.copy()
    test_df['y'] = test_df[test_col]
    test_df['x'] = test_df[control_col]
    test_df['pred'] = test_predictions['pred']
    test_df['predsd'] = test_predictions['predsd']

    # Calculate effects (difference from counterfactual)
    test_df['dif'] = test_df['y'] - test_df['pred']

    # Calculate cumulative effects continuously across test period
    test_df['cumdif'] = test_df['dif'].cumsum()

    # Calculate cumulative standard deviations for the entire period
    cumsd_values = calculate_cumulative_standard_deviation(
        test_df['x'].values,
        sigma,
        var_alpha,
        var_beta,
        cov_alpha_beta
    )
    test_df['cumsd'] = cumsd_values

    # Test period doesn't have fitted value standard deviations
    test_df['estsd'] = np.nan

    return test_df


def validate_and_prepare_period_dataframe(df: pd.DataFrame, name: str, required_cols: List[str]) -> pd.DataFrame:
    """
    Validate and prepare a period dataframe for TBR analysis.
    Parameters
    ----------
    df : pd.DataFrame
        Period dataframe with date, control, and test columns
    name : str
        Name of the period
    required_cols : List[str]
        List of required columns for the period

    Returns
    -------
    pd.DataFrame
        Prepared period dataframe with date, control, and test columns

    Raises
    ------
    ValueError
        If input validation fails or required data is missing
    """
    if df.empty:
        return df
    df = df.sort_values('date').reset_index(drop=True)
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {name}: {missing_cols}")

    return df


def create_tbr_dataframe(
    baseline_data: pd.DataFrame,
    pretest_data: pd.DataFrame,
    test_data: pd.DataFrame,
    alpha: float,
    beta: float,
    sigma: float,
    x_mean: float,
    n_pretest: int,
    var_alpha: float,
    var_beta: float,
    cov_alpha_beta: float,
    control_col: str,
    test_col: str
) -> pd.DataFrame:
    """
    Create the main TBR dataframe with all required columns for analysis.

    This function combines pretest and test period data with predictions to create
    the comprehensive TBR dataframe that matches the R package output format.
    Note: test_data should include both test period (period=1) and cooldown period (period=3) if exists.
    The period values should already be assigned when the data reaches this function.

    Note: This function automatically sorts input data by date in ascending order
    to ensure correct cumulative calculations.

    Parameters
    ----------
    baseline_data : pd.DataFrame
        Baseline period data with date, control, and test columns
    pretest_data : pd.DataFrame
        Pretest period data with date, control, and test columns
    test_data : pd.DataFrame
        Test period data with date, control, test, and period columns.
        Should include both test period (period=1) and cooldown period (period=3) if exists.
    alpha : float
        Regression intercept coefficient
    beta : float
        Regression slope coefficient
    sigma : float
        Residual standard deviation from regression model
    x_mean : float
        Mean of control values during pretest period
    n_pretest : int
        Number of observations in pretest period
    var_alpha : float
        Variance of the intercept coefficient estimate
    var_beta : float
        Variance of the slope coefficient estimate
    cov_alpha_beta : float
        Covariance between intercept and slope coefficient estimates
    control_col : str
        Name of control column
    test_col : str
        Name of test column

    Returns
    -------
    pd.DataFrame
        Complete TBR dataframe with columns:
        - date: Date
        - period: 0 for pretest, 1 for test, 3 for cooldown
        - y: Test values
        - x: Control values
        - pred: Predictions (fitted values for pretest, counterfactuals for test and cooldown)
        - predsd: Prediction standard deviations
        - dif: y - pred (residuals for pretest, effects for test and cooldown)
        - cumdif: Cumulative sum of dif (continuous across test and cooldown periods)
        - cumsd: Cumulative standard deviation (continuous across test and cooldown periods)
        - estsd: Standard deviation of fitted values (only for pretest)
    """
    # Input validation
    if pretest_data.empty or test_data.empty:
        raise ValueError("Both pretest_data and test_data must be non-empty")

    required_cols = ['date', control_col, test_col]
    baseline_data = validate_and_prepare_period_dataframe(baseline_data, 'baseline_data', required_cols)
    pretest_data = validate_and_prepare_period_dataframe(pretest_data, 'pretest_data', required_cols)
    test_data = validate_and_prepare_period_dataframe(test_data, 'test_data', required_cols)

    # Process baseline period
    baseline_df = create_tbr_baseline_dataframe(
        baseline_data=baseline_data,
        control_col=control_col,
        test_col=test_col
    )

    # Process pretest period
    pretest_df = create_tbr_pretest_dataframe(
        pretest_data=pretest_data,
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        x_mean=x_mean,
        n_pretest=n_pretest,
        var_beta=var_beta,
        control_col=control_col,
        test_col=test_col
    )

    # Process test data (includes test period and cooldown period if exists)
    test_df = create_tbr_test_dataframe(
        test_data=test_data,
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        x_mean=x_mean,
        n_pretest=n_pretest,
        var_beta=var_beta,
        var_alpha=var_alpha,
        cov_alpha_beta=cov_alpha_beta,
        control_col=control_col,
        test_col=test_col
    )

    # Combine all periods
    tbr_df = pd.concat([baseline_df, pretest_df, test_df], ignore_index=True)

    # Order columns
    output_cols = ['date', 'period', 'y', 'x', 'pred', 'predsd', 'dif', 'cumdif', 'cumsd', 'estsd']
    tbr_df = tbr_df[output_cols]

    return tbr_df


def create_tbr_summary(
    tbr_dataframe: pd.DataFrame,
    alpha: float,
    beta: float,
    sigma: float,
    var_alpha: float,
    var_beta: float,
    cov_alpha_beta: float,
    degrees_freedom: int,
    level: float,
    threshold: float,
    model_name: str
) -> pd.DataFrame:
    """
    Create TBR summary statistics DataFrame matching R package output format.

    This function generates a single-row summary DataFrame containing all key
    statistics for the TBR analysis, including the cumulative effect estimate,
    credible intervals, and model parameters.

    Parameters
    ----------
    tbr_dataframe : pd.DataFrame
        Complete TBR dataframe with all periods and statistics
    alpha : float
        Regression intercept coefficient
    beta : float
        Regression slope coefficient
    sigma : float
        Residual standard deviation from regression model
    var_alpha : float
        Variance of intercept estimate
    var_beta : float
        Variance of slope estimate
    cov_alpha_beta : float
        Covariance between intercept and slope estimates
    degrees_freedom : int
        Residual degrees of freedom from regression
    level : float
        Credibility level for confidence intervals
    threshold : float
        Threshold for probability calculation
    model_name : str
        Name of the TBR model

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with TBR summary statistics matching R output format

    Raises
    ------
    ValueError
        If input validation fails or required data is missing
    """
    # Input validation
    if tbr_dataframe.empty:
        raise ValueError("TBR dataframe cannot be empty")

    required_cols = ['period', 'cumdif', 'cumsd']
    missing_cols = [col for col in required_cols if col not in tbr_dataframe.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in TBR dataframe: {missing_cols}")

    if not (0 <= level <= 1):
        raise ValueError(f"Level must be between 0 and 1, got: {level}")

    if degrees_freedom <= 0:
        raise ValueError(f"Degrees of freedom must be positive, got: {degrees_freedom}")

    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got: {sigma}")

    # Extract test period data (period == 1)
    test_period_data = tbr_dataframe[tbr_dataframe['period'] == 1].copy()

    if test_period_data.empty:
        raise ValueError("No test period data found (period == 1)")

    # Calculate core summary statistics
    # estimate: Final cumulative effect from test period
    estimate = test_period_data['cumdif'].iloc[-1]

    # se: Final cumulative standard deviation from test period
    se = test_period_data['cumsd'].iloc[-1]

    # Calculate credible interval using t-distribution
    alpha_level = 1 - level  # Probability outside interval
    t_critical = stats.t.ppf(1 - alpha_level/2, df=degrees_freedom)

    # Credible interval bounds
    margin_of_error = t_critical * se
    lower = estimate - margin_of_error
    upper = estimate + margin_of_error

    # precision: Half-width of credible interval
    precision = margin_of_error

    # prob: Posterior probability that true cumulative effect exceeds threshold
    #
    # Statistical reasoning:
    # The cumulative effect Δr(T) follows a t-distribution: Δr(T) ~ t_ν(estimate, se)
    # where estimate is the location parameter and se is the scale parameter.
    #
    # To find P(Δr(T) > threshold), we standardize the distribution:
    # Let τ = (Δr(T) - estimate) / se, then τ ~ t_ν(0, 1) (standard t-distribution)
    #
    # We want: P(Δr(T) > threshold) = P((Δr(T) - estimate)/se > (threshold - estimate)/se)
    #                                = P(τ > t_stat) = 1 - CDF_t(t_stat)
    #
    # where t_stat = (threshold - estimate) / se is the standardized threshold
    t_stat = (threshold - estimate) / se if se > 0 else 0
    prob = 1 - stats.t.cdf(t_stat, df=degrees_freedom)

    # Ensure probability is between 0 and 1
    prob = max(0.0, min(1.0, prob))

    # Create summary dictionary
    summary_data = {
        'estimate': float(estimate),
        'precision': float(precision),
        'lower': float(lower),
        'upper': float(upper),
        'se': float(se),
        'level': float(level),
        'thres': float(threshold),
        'prob': float(prob),
        'model': str(model_name),
        'alpha': float(alpha),
        'beta': float(beta),
        'alpha_beta_cov': float(cov_alpha_beta),
        'var_alpha': float(var_alpha),
        'var_beta': float(var_beta),
        'sigma': float(sigma),
        't_dist_df': float(degrees_freedom)
    }

    # Create single-row DataFrame with specified dtypes
    summary_df = pd.DataFrame([summary_data])

    # Ensure correct dtypes match the specification
    dtype_mapping = {
        'estimate': 'float64',
        'precision': 'float64',
        'lower': 'float64',
        'upper': 'float64',
        'se': 'float64',
        'level': 'float64',
        'thres': 'float64',
        'prob': 'float64',
        'model': 'object',
        'alpha': 'float64',
        'beta': 'float64',
        'alpha_beta_cov': 'float64',
        'var_alpha': 'float64',
        'var_beta': 'float64',
        'sigma': 'float64',
        't_dist_df': 'float64'
    }

    summary_df = summary_df.astype(dtype_mapping)

    return summary_df


def create_incremental_tbr_summaries(
    tbr_dataframe: pd.DataFrame,
    alpha: float,
    beta: float,
    sigma: float,
    var_alpha: float,
    var_beta: float,
    cov_alpha_beta: float,
    degrees_freedom: int,
    level: float,
    threshold: float,
    model_name: str
) -> pd.DataFrame:
    """
    Create incremental TBR summary statistics for each test period day.

    This function generates summary statistics for incremental test periods:
    - Day 1: Summary for first day only
    - Day 2: Summary for first two days (cumulative)
    - Day 3: Summary for first three days (cumulative)
    - ...and so on

    This allows analysis of how treatment effects evolve over time during the test period.

    Parameters
    ----------
    tbr_dataframe : pd.DataFrame
        Complete TBR dataframe with all periods and statistics
    alpha : float
        Regression intercept coefficient
    beta : float
        Regression slope coefficient
    sigma : float
        Residual standard deviation from regression model
    var_alpha : float
        Variance of intercept estimate
    var_beta : float
        Variance of slope estimate
    cov_alpha_beta : float
        Covariance between intercept and slope estimates
    degrees_freedom : int
        Residual degrees of freedom from regression
    level : float
        Credibility level for confidence intervals
    threshold : float
        Threshold for probability calculation
    model_name : str
        Name of the TBR model

    Returns
    -------
    pd.DataFrame
        Multi-row DataFrame with incremental TBR summary statistics.
        Each row represents cumulative statistics up to that test day.
        Includes an additional 'test_day' column indicating the incremental period.

    Raises
    ------
    ValueError
        If input validation fails or no test period data is found

    """
    # Input validation
    if tbr_dataframe.empty:
        raise ValueError("TBR dataframe cannot be empty")

    required_cols = ['period', 'cumdif', 'cumsd']
    missing_cols = [col for col in required_cols if col not in tbr_dataframe.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in TBR dataframe: {missing_cols}")

    if not (0 <= level <= 1):
        raise ValueError(f"Level must be between 0 and 1, got: {level}")

    if degrees_freedom <= 0:
        raise ValueError(f"Degrees of freedom must be positive, got: {degrees_freedom}")

    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got: {sigma}")

    # Extract test period data (period == 1)
    test_period_data = tbr_dataframe[tbr_dataframe['period'] == 1].copy()

    if test_period_data.empty:
        raise ValueError("No test period data found (period == 1)")

    # Get pretest data for combining with incremental test periods
    pretest_data = tbr_dataframe[tbr_dataframe['period'] == 0].copy()

    num_test_days = len(test_period_data)
    incremental_summaries = []

    # Generate summary for each incremental test period
    for day_idx in range(num_test_days):
        # Create subset of test data up to current day (inclusive)
        test_subset = test_period_data.iloc[:day_idx + 1].copy()

        # Combine pretest data with current test subset
        incremental_df = pd.concat([pretest_data, test_subset], ignore_index=True)

        # Generate summary for this incremental period
        summary = create_tbr_summary(
            tbr_dataframe=incremental_df,
            alpha=alpha,
            beta=beta,
            sigma=sigma,
            var_alpha=var_alpha,
            var_beta=var_beta,
            cov_alpha_beta=cov_alpha_beta,
            degrees_freedom=degrees_freedom,
            level=level,
            threshold=threshold,
            model_name=model_name
        )

        # Add test day identifier
        summary['test_day'] = day_idx + 1

        incremental_summaries.append(summary)

    # Combine all incremental summaries
    result_df = pd.concat(incremental_summaries, ignore_index=True)

    # Reorder columns to put test_day first for clarity
    cols = ['test_day'] + [col for col in result_df.columns if col != 'test_day']
    result_df = result_df[cols]

    return result_df


def prepare_test_data_extended(
    test_data: pd.DataFrame,
    cooldown_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare test and cooldown data as one continuous period.

    Parameters
    ----------
    test_data : pd.DataFrame
        Test period data with date, control, and test columns
    cooldown_data : pd.DataFrame
        Cooldown period data with date, control, and test columns

    Returns
    -------
    pd.DataFrame
        Combined and sorted DataFrame with period values assigned
    """
    # Add period column to test data
    test_data_with_period = test_data.copy()
    test_data_with_period['period'] = 1

    # Add period column to cooldown data if it exists
    if not cooldown_data.empty:
        cooldown_data_with_period = cooldown_data.copy()
        cooldown_data_with_period['period'] = 3

        # Combine test and cooldown data and sort by date
        test_data_extended = pd.concat(
            [test_data_with_period, cooldown_data_with_period],
            ignore_index=True
        ).sort_values('date').reset_index(drop=True)
    else:
        test_data_extended = test_data_with_period

    return test_data_extended


def perform_tbr_analysis(
    data: pd.DataFrame,
    assignments: pd.DataFrame,
    geo_col: str,
    date_col: str,
    metric_col: str,
    assignment_col: str,
    control_group: str,
    test_group: str,
    pretest_start: Union[str, pd.Timestamp],
    test_start: Union[str, pd.Timestamp],
    test_end: Union[str, pd.Timestamp],
    level: float,
    threshold: float,
    model_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full TBR analysis pipeline.

    This function orchestrates the entire TBR analysis process:
    1. Create time series data for TBR analysis
    2. Fit the TBR regression model
    3. Prepare test and cooldown data as one continuous period
    4. Create the TBR dataframe
    5. Create incremental TBR summaries

    Parameters
    ----------
    data : pd.DataFrame
        Raw time series data
    assignments : pd.DataFrame
        Geo assignment data
    geo_col : str
        Name of geo column
    date_col : str
        Name of date column
    metric_col : str
        Name of metric column
    assignment_col : str
        Name of assignment column
    control_group : str
        Control group identifier
    test_group : str
        Test group identifier
    pretest_start : Union[str, datetime.date]
        Pretest start date
    test_start : Union[str, datetime.date]
        Test start date
    test_end : Union[str, datetime.date]
        Test end date
    level : float
        Credibility level for confidence intervals
    threshold : float
        Threshold for probability calculation
    model_name : str
        Name of the TBR model

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        TBR dataframe and incremental TBR summary statistics
    """
    # Step 1: Create time series data for TBR analysis
    aggregated_data, baseline_data, pretest_data, test_data, cooldown_data = create_time_series_for_tbr(
        data=data,
        assignments=assignments,
        geo_col=geo_col,
        date_col=date_col,
        metric_col=metric_col,
        assignment_col=assignment_col,
        control_group=control_group,
        test_group=test_group,
        pretest_start=pretest_start,
        test_start=test_start,
        test_end=test_end
    )

    # Step 2: Fit the TBR regression model
    model = fit_tbr_regression_model(
        control_test_df=aggregated_data,
        start_pretest_date=pretest_start,
        start_test_date=test_start,
        control_col=control_group,
        test_col=test_group
    )

    # Step 3: Prepare test and cooldown data as one continuous period
    test_data_extended = prepare_test_data_extended(test_data, cooldown_data)

    # Step 4: Create TBR dataframe
    tbr_dataframe = create_tbr_dataframe(
        baseline_data=baseline_data,
        pretest_data=pretest_data,
        test_data=test_data_extended,
        alpha=model['alpha'],
        beta=model['beta'],
        sigma=model['sigma'],
        x_mean=model['x_mean'],
        n_pretest=int(model['n_pretest']),
        var_alpha=model['var_alpha'],
        var_beta=model['var_beta'],
        cov_alpha_beta=model['cov_alpha_beta'],
        control_col=control_group,
        test_col=test_group
    )

    # Step 5: Create incremental TBR summaries
    tbr_daily_summary = create_incremental_tbr_summaries(
        tbr_dataframe=tbr_dataframe,
        alpha=model['alpha'],
        beta=model['beta'],
        sigma=model['sigma'],
        var_alpha=model['var_alpha'],
        var_beta=model['var_beta'],
        cov_alpha_beta=model['cov_alpha_beta'],
        degrees_freedom=int(model['degrees_freedom']),
        level=level,
        threshold=threshold,
        model_name=model_name
    )

    return tbr_dataframe, tbr_daily_summary
